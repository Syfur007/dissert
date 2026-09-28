"""
orchestration/runner.py — sweep driver.

Expands a resolved experiment config across seed x fold combinations and
runs each through ``train.run_training()``, wrapped in a manifest
(start/finish/save) and a Runs-table ledger row. A combination whose
manifest already reports ``status: "done"`` is skipped unless *force* is
set — the idempotent-skip generalises the per-fold ``try/except`` already in
train.py's K-Fold CLI loop (train.py:376-384) to the full seed x fold grid,
across process restarts (a manifest on disk survives a killed process; an
in-memory try/except does not).

One failed combination does not stop the sweep — same "log it, keep going"
behaviour train.py's existing per-fold loop and search.py's existing
per-trial loop already have, generalised here across both axes at once.

A sweep this size (9 combinations for a non-CV config at the default 3
seeds x 3 repeats, 45 for a 5-fold one) routinely outlives a single
session on session-limited compute, so *budget* threads a shared
``orchestration.budget.WallClockBudget`` through the grid: a combination
whose projected duration no longer fits in the remaining time is not
started at all, and one that stops itself mid-training is recorded as
``interrupted`` rather than ``done`` so the next session retries it.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .ledger import LedgerWriter
from .manifest import build_manifest
from .runid import config_hash as compute_config_hash
from .runid import experiment_paths
from .runid import run_id as compute_run_id
from dissert.training.determinism import (
    get_recorded_manifest_extras,
    get_recorded_nondeterminism,
    reset_recorded_nondeterminism,
)

TrainFn = Callable[..., float]


def _manifest_path(
    output_dir: str, experiment_name: str, config_hash_: str, seed: int,
    fold: Optional[int], repeat: Optional[int] = None,
) -> str:
    # Co-located with that fold's checkpoints — the same directory
    # train.py's CheckpointManager writes best.pth/last.pth into.
    checkpoints_dir = experiment_paths(
        output_dir, experiment_name, config_hash_, seed, fold, repeat,
    )["checkpoints"]
    return os.path.join(checkpoints_dir, "manifest.json")


def _existing_status(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("status")
    except Exception:
        # A partially-written or corrupt manifest is treated as "not done"
        # rather than raising — a sweep re-run should retry it, not crash.
        return None


def _with_seed(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.setdefault("training", {})["seed"] = seed
    return cfg


def run_sweep(
    resolved_config: Dict[str, Any],
    seeds: Sequence[int],
    folds: Sequence[Optional[int]] = (None,),
    repeats: Sequence[Optional[int]] = (None,),
    train_fn: Optional[TrainFn] = None,
    ledger_dir: str = "outputs/ledger",
    force: bool = False,
    budget: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Run *train_fn* (defaults to ``train.run_training``) once per
    ``seeds x repeats x folds`` combination.

    Args:
        resolved_config: an already-validated config dict (see
            ``dissert.config.schema.validate_config`` / ``dissert.config.loader.load_config``).
        seeds: seeds to sweep.
        folds: fold indices to sweep; ``(None,)`` (the default) means a
            single non-CV run per seed.
        repeats: repeat indices to sweep per seed; ``(None,)`` (the
            default) means a single, non-repeated run per seed — same
            "index or None" convention as *folds*. Each repeat re-seeds
            *identically* to the same seed (train_fn does this, not this
            loop) and gets its own independent
            checkpoints/logs/tensorboard/plots/eval tree — the point is to
            measure/average out whatever noise survives fixed seeding
            (hardware/kernel non-determinism), not to vary anything.
        train_fn: injectable for testing; defaults to a lazy import of
            ``dissert.cli.train.run_training`` (kept lazy so importing this
            module never drags in torch/train.py's full dependency chain).
        force: re-run a combination even if its manifest already says
            ``status: "done"``.
        budget: shared ``orchestration.budget.WallClockBudget`` or ``None``
            (the default — no budget checks at all, the historical
            behaviour). When given it is both *passed down* to *train_fn*
            (so a run stops itself at an epoch boundary) and *checked here*
            before starting each combination, against the longest
            combination observed so far. Without that second level, a sweep
            would happily start a fresh run with three minutes left and
            spend the rest of the session on setup for zero completed
            epochs.

    Returns:
        One result dict per combination:
        ``{"run_id", "status", "best_metric", "error"}``.

        ``status`` is ``done``/``failed`` from the run itself, or one of two
        sweep-level values that are never written to a manifest:
        ``skipped-done`` (already complete, idempotent skip) and
        ``skipped-budget`` (not started — no time left). ``interrupted``
        means the run stopped itself on *budget*; it *is* written to the
        manifest, and — unlike ``done`` — is deliberately not skipped on a
        later call, so the next session resumes it.
    """
    if train_fn is None:
        from dissert.cli.train import run_training as train_fn  # local: see docstring

    h = compute_config_hash(resolved_config)
    ledger = LedgerWriter(ledger_dir)
    results: List[Dict[str, Any]] = []

    output_dir = resolved_config.get("output_dir", "outputs/experiments")
    experiment_name = resolved_config.get("logging", {}).get("experiment_name", "experiment")

    # Durations of the combinations this call actually ran, used to project
    # whether the next one fits in the remaining budget.
    run_seconds: List[float] = []

    for seed in seeds:
        for repeat in repeats:
            for fold in folds:
                rid = compute_run_id(h, seed=seed, fold=fold, repeat=repeat)
                mpath = _manifest_path(output_dir, experiment_name, h, seed, fold, repeat)

                # Only "done" is skipped — an "interrupted" manifest must be
                # picked up and continued, which is what makes a sweep
                # resumable across sessions. checkpoint.resume already
                # defaults to True (dissert/config/schema.py), and a
                # never-started combination just hits train.py's existing
                # "No checkpoint at ... Starting from scratch" path.
                if not force and _existing_status(mpath) == "done":
                    results.append(
                        {"run_id": rid, "status": "skipped-done", "best_metric": None, "error": None}
                    )
                    continue

                if budget is not None:
                    projected = max(run_seconds) if run_seconds else 0.0
                    if budget.exhausted_by(projected):
                        # `continue`, not `break`: the caller gets a complete
                        # result list showing exactly which combinations were
                        # gated off, rather than a list that simply ends.
                        results.append(
                            {"run_id": rid, "status": "skipped-budget",
                             "best_metric": None, "error": None}
                        )
                        continue

                run_config = _with_seed(resolved_config, seed)
                manifest = build_manifest(rid, run_config, seed=seed, fold=fold, repeat=repeat)
                manifest.start()
                reset_recorded_nondeterminism()

                status, best_metric, error = "failed", None, None
                started = time.monotonic()
                try:
                    # `budget` is passed only when there is one, so an
                    # injected train_fn written against the original
                    # (config, fold, run_id, repeat) signature keeps working
                    # unchanged — accepting `budget` is required only of a
                    # train_fn that is actually run under a budget.
                    extra_kwargs = {"budget": budget} if budget is not None else {}
                    best_metric = train_fn(
                        run_config, fold=fold, run_id=rid, repeat=repeat, **extra_kwargs,
                    )
                    status = "done"
                except Exception as exc:  # noqa: BLE001 — one bad run must not kill the sweep
                    error = str(exc)
                    status = "failed"
                finally:
                    run_seconds.append(time.monotonic() - started)
                    for note in get_recorded_nondeterminism():
                        manifest.record_nondeterminism(note)
                    extras = get_recorded_manifest_extras()
                    for key, value in extras.items():
                        manifest.record(key, value)

                    # The run stopped itself at an epoch boundary because
                    # the budget was nearly spent (train.run_training sets
                    # this flag via the determinism side-channel). Healthy
                    # and resumable — not "done", so a later sweep picks it
                    # back up, and not "failed", so nothing treats it as an
                    # error. Extras are cleared per-combination by
                    # reset_recorded_nondeterminism() above, so this can't
                    # leak from one run into the next.
                    if status == "done" and extras.get("stopped_on_budget"):
                        status = "interrupted"

                    manifest.record(
                        "resumable",
                        os.path.exists(os.path.join(os.path.dirname(mpath), "last.pth")),
                    )
                    manifest.finish(status=status, error=error)
                    manifest.save(mpath)

                    log_cfg = run_config.get("logging", {})
                    model_cfg = run_config.get("model", {})
                    dataset_cfg = run_config.get("dataset", {})
                    chk_cfg = run_config.get("checkpoint", {})
                    git = manifest.data["git"]

                    ledger.append_run_row(
                        run_id=rid,
                        config_hash=h,
                        experiment_name=log_cfg.get("experiment_name", ""),
                        model_name=model_cfg.get("name", ""),
                        dataset_name=dataset_cfg.get("name", ""),
                        seed=seed,
                        repeat=repeat if repeat is not None else "",
                        fold=fold if fold is not None else "",
                        status=status,
                        start_time=manifest.data["start_time"],
                        end_time=manifest.data["end_time"],
                        gpu_hours=manifest.data.get("gpu_hours") or "",
                        best_metric=best_metric if best_metric is not None else "",
                        monitor_metric=chk_cfg.get("monitor_metric", ""),
                        git_commit=git.get("commit") or "",
                        git_dirty=git.get("dirty"),
                        manifest_path=mpath,
                        python_version=manifest.data["hardware"].get("python_version", ""),
                        torch_version=manifest.data["hardware"].get("torch_version", ""),
                        opencv_version=manifest.data["hardware"].get("opencv_version", ""),
                    )

                results.append(
                    {"run_id": rid, "status": status, "best_metric": best_metric, "error": error}
                )

    return results
