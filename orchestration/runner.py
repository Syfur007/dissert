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
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from .ledger import LedgerWriter
from .manifest import build_manifest
from .runid import config_hash as compute_config_hash
from .runid import run_id as compute_run_id
from training.determinism import (
    get_recorded_manifest_extras,
    get_recorded_nondeterminism,
    reset_recorded_nondeterminism,
)

TrainFn = Callable[..., float]


def _manifest_path(artifacts_dir: str, rid: str) -> str:
    return os.path.join(artifacts_dir, "runs", rid, "manifest.json")


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
    train_fn: Optional[TrainFn] = None,
    artifacts_dir: str = "artifacts",
    ledger_dir: str = "artifacts/ledger",
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Run *train_fn* (defaults to ``train.run_training``) once per
    ``seeds x folds`` combination.

    Args:
        resolved_config: an already-validated config dict (see
            ``orchestration.schema.validate_config`` / ``utils.config.load_config``).
        seeds: seeds to sweep.
        folds: fold indices to sweep; ``(None,)`` (the default) means a
            single non-CV run per seed.
        train_fn: injectable for testing; defaults to a lazy import of
            ``train.run_training`` (kept lazy so importing this module never
            drags in torch/train.py's full dependency chain).
        force: re-run a combination even if its manifest already says
            ``status: "done"``.

    Returns:
        One result dict per combination:
        ``{"run_id", "status", "best_metric", "error"}``.
    """
    if train_fn is None:
        from train import run_training as train_fn  # local: see docstring

    h = compute_config_hash(resolved_config)
    ledger = LedgerWriter(ledger_dir)
    results: List[Dict[str, Any]] = []

    for seed in seeds:
        for fold in folds:
            rid = compute_run_id(h, seed=seed, fold=fold)
            mpath = _manifest_path(artifacts_dir, rid)

            if not force and _existing_status(mpath) == "done":
                results.append(
                    {"run_id": rid, "status": "skipped-done", "best_metric": None, "error": None}
                )
                continue

            run_config = _with_seed(resolved_config, seed)
            manifest = build_manifest(rid, run_config, seed=seed, fold=fold)
            manifest.start()
            reset_recorded_nondeterminism()

            status, best_metric, error = "failed", None, None
            try:
                best_metric = train_fn(run_config, fold=fold, run_id=rid)
                status = "done"
            except Exception as exc:  # noqa: BLE001 — one bad run must not kill the sweep
                error = str(exc)
                status = "failed"
            finally:
                for note in get_recorded_nondeterminism():
                    manifest.record_nondeterminism(note)
                for key, value in get_recorded_manifest_extras().items():
                    manifest.record(key, value)
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
                )

            results.append(
                {"run_id": rid, "status": status, "best_metric": best_metric, "error": error}
            )

    return results
