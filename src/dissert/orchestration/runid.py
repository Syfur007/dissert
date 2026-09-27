"""
Deterministic run identity: config_hash + run_id.

Two runs of the *same* experiment (same resolved config) at different seeds
or different K-Fold indices must share one config_hash — the hash identifies
*what* is being run, not *which repetition*. run_id then re-adds seed/fold to
build the actual per-run identifier used for the manifest/checkpoint/ledger
paths.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
from typing import Any, Dict, Optional


def config_hash(resolved_config: Dict[str, Any]) -> str:
    """SHA1 hex digest over *resolved_config*, canonicalised via
    ``json.dumps(sort_keys=True)`` so key order never affects the hash.

    Strips everything that's bookkeeping/presentation rather than part of
    the atomic, reproducible experiment setup — two configs differing only
    in these fields describe the *same* experiment and must hash the same:

    - ``training.seed`` — identifies a repetition, not the setup.
    - ``logging`` (whole section) — ``experiment_name`` is a label;
      ``log_interval``/``save_overlays``/``overlay_save_every``/
      ``overlay_n_samples`` only affect which side-artifacts get written.
    - ``output_dir`` — where to write, not what to run.
    - ``checkpoint.checkpoint_path``/``resume``/``periodic_save_every`` —
      resume/save-cadence bookkeeping. ``checkpoint.monitor_metric``/
      ``mode`` are deliberately *not* stripped — they change which
      checkpoint is selected as "best", i.e. the actual result.
    - ``stats`` (whole section) — declares which significance-testing
      family a result feeds into; doesn't change the run itself.

    (``fold`` is never part of the config dict in this repo — it's passed
    as a separate runtime argument to ``run_training(config, fold=...)`` —
    so there is nothing to strip for it here.)
    """
    stripped = copy.deepcopy(resolved_config)
    training = stripped.get("training")
    if isinstance(training, dict):
        training.pop("seed", None)
    stripped.pop("logging", None)
    stripped.pop("output_dir", None)
    stripped.pop("stats", None)
    checkpoint = stripped.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint.pop("checkpoint_path", None)
        checkpoint.pop("resume", None)
        checkpoint.pop("periodic_save_every", None)
    canonical = json.dumps(stripped, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def run_id(config_hash_: str, seed: int, fold: Optional[int] = None, repeat: Optional[int] = None) -> str:
    """``R-{hash[:7]}-s{seed}-r{repeat}-f{fold}``. For a non-CV run
    (``fold is None``), the fold segment reads ``f-`` rather than the
    literal string "None". The repeat segment is omitted entirely when
    ``repeat is None`` (a single, non-repeated run — the historical
    format), same convention as ``fold``."""
    fold_part = fold if fold is not None else "-"
    repeat_part = f"-r{repeat}" if repeat is not None else ""
    return f"R-{config_hash_[:7]}-s{seed}{repeat_part}-f{fold_part}"


def experiment_id(config_hash_: str, seed: int, repeat: Optional[int] = None) -> str:
    """The atomic on-disk unit within one experiment_name: one
    (config_hash, seed[, repeat]) pair, shared by every fold of that run.
    Unlike the old ``{experiment_name}-s{seed}`` scheme, the hash is now
    part of the identifier itself — a config change (anything
    ``config_hash`` doesn't strip) lands in a fresh directory instead of
    silently reusing an old one, so ``--resume`` only ever continues a
    byte-identical setup. See ``experiment_paths``.

    ``repeat`` identifies the Nth full re-run of this exact (config_hash,
    seed) — same seed re-seeded identically each time, run to measure/
    average out whatever residual noise survives fixed seeding (hardware/
    kernel non-determinism — see training/determinism.py), as distinct
    from ``seed`` itself (which varies model-init/data-order stochasticity
    on purpose). Omitted from the identifier (and therefore the path) when
    ``repeat is None`` — a single, non-repeated run — so existing single-
    run layouts are unaffected."""
    repeat_part = f"-r{repeat}" if repeat is not None else ""
    return f"{config_hash_[:7]}-s{seed}{repeat_part}"


def experiment_paths(
    base_dir: str,
    experiment_name: str,
    config_hash_: str,
    seed: int,
    fold: Optional[int] = None,
    repeat: Optional[int] = None,
) -> Dict[str, str]:
    """Resolve every path under one experiment's directory,
    ``{base_dir}/{experiment_name}/{experiment_id}/``. Content type is the
    top-level split (checkpoints/logs/tensorboard/plots/eval); fold is a
    subdirectory of checkpoints/tensorboard/plots when this is a K-Fold run
    (``fold`` given) and omitted entirely for a non-CV run (``fold=None``).
    ``repeat`` (Nth identical re-run of this (config_hash, seed) — see
    ``experiment_id``) is folded into ``root`` itself via ``experiment_id``,
    so every repeat gets its own complete, independent
    checkpoints/logs/tensorboard/plots/eval tree, exactly like every seed
    does today.

    ``fold_splits``, ``seed_combined_report`` and ``combined_report`` are
    deliberately *not* under ``root`` — they're shared across every
    seed/repeat of this (experiment_name, config_hash), not scoped to one
    run:

    - ``fold_splits``: all seeds *and* repeats of the same config hash
      must train/validate on the same fold partition (only model-init/
      training randomness — and, for repeats, only residual hardware
      non-determinism — should vary, not the data split itself) — see
      ``datasets.datamodule.KFoldDataModule``.
    - ``seed_combined_report``: the repeat-averaged eval report for one
      seed (mean ± std over that seed's ``n_repeats`` independent runs —
      see ``utils.report.aggregate_repeat_reports``), sitting adjacent to
      every ``{hash7}-s{seed}-r{repeat}/`` directory sharing that seed.
      Feeds ``combined_report`` below in place of a raw single-run
      report.json when repeats are in use.
    - ``combined_report``: the seed-averaged eval report, sitting adjacent
      to every ``{hash7}-s{seed}[-r{repeat}]/`` directory under this
      experiment_name.
    """
    exp_root = os.path.join(base_dir, experiment_name)
    root = os.path.join(exp_root, experiment_id(config_hash_, seed, repeat))

    def _fold_scoped(*parts: str) -> str:
        joined = os.path.join(root, *parts)
        return os.path.join(joined, f"fold{fold}") if fold is not None else joined

    return {
        "root": root,
        "checkpoints": _fold_scoped("checkpoints"),
        "logs": os.path.join(root, "logs"),
        "tensorboard": _fold_scoped("tensorboard"),
        "plots": _fold_scoped("plots"),
        "eval": os.path.join(root, "eval"),
        "run_meta": os.path.join(root, "run_meta.json"),
        "fold_splits": os.path.join(exp_root, f"{config_hash_[:7]}-fold_splits.json"),
        "seed_combined_report": os.path.join(exp_root, f"{config_hash_[:7]}-s{seed}.json"),
        "combined_report": os.path.join(exp_root, f"{experiment_name}.json"),
    }


def check_and_record_run_meta(
    run_meta_path: str,
    experiment_name: str,
    seed: int,
    config_hash_: str,
    repeat: Optional[int] = None,
    logger: Any = None,
) -> None:
    """Enforce atomicity by *check*, not by path uniqueness: on first use
    for this experiment_id, record its config_hash; on every later call
    (a later fold, a resume), warn — never block — if the current config's
    hash has drifted from what was first recorded here.

    With the hash now embedded in the directory name itself
    (``experiment_id``), a genuine config change can no longer collide
    with an old run's directory — it lands in a fresh one. This can now
    only fire on a truncated-hash collision (two different full hashes
    sharing the same first 7 hex chars), so it's kept as a defensive
    check rather than the primary drift guard it used to be.
    """
    if os.path.exists(run_meta_path):
        with open(run_meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("config_hash") != config_hash_ and logger is not None:
            logger.warning(
                f"run_meta.json config_hash mismatch at {run_meta_path}: "
                f"recorded {meta.get('config_hash')}, current {config_hash_} — "
                "truncated-hash collision between two different configs "
                "under the same experiment_name+seed; checkpoints/logs "
                "below are now shared across both."
            )
        return
    os.makedirs(os.path.dirname(run_meta_path), exist_ok=True)
    meta = {
        "experiment_name": experiment_name,
        "seed": seed,
        "repeat": repeat,
        "config_hash": config_hash_,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(run_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
