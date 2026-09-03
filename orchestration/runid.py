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
    """SHA1 hex digest over *resolved_config* with ``training.seed``
    excluded, canonicalised via ``json.dumps(sort_keys=True)`` so key order
    never affects the hash. (``fold`` is never part of the config dict in
    this repo — it's passed as a separate runtime argument to
    ``run_training(config, fold=...)`` — so there is nothing to strip for it
    here; only seed needs excluding.)
    """
    stripped = copy.deepcopy(resolved_config)
    training = stripped.get("training")
    if isinstance(training, dict):
        training.pop("seed", None)
    canonical = json.dumps(stripped, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def run_id(config_hash_: str, seed: int, fold: Optional[int] = None) -> str:
    """``R-{hash[:7]}-s{seed}-f{fold}``. For a non-CV run (``fold is None``),
    the fold segment reads ``f-`` rather than the literal string "None"."""
    fold_part = fold if fold is not None else "-"
    return f"R-{config_hash_[:7]}-s{seed}-f{fold_part}"


def experiment_id(experiment_name: str, seed: int) -> str:
    """The atomic on-disk unit: one (experiment_name, seed) pair, shared by
    every fold of that run. Deliberately excludes any config hash — see
    ``experiment_paths``'s ``run_meta.json`` note for why (keeps
    ``--resume`` working across ordinary config tweaks)."""
    return f"{experiment_name}-s{seed}"


def experiment_paths(
    base_dir: str, experiment_name: str, seed: int, fold: Optional[int] = None
) -> Dict[str, str]:
    """Resolve every path under one experiment's directory,
    ``{base_dir}/{experiment_id}/``. Content type is the top-level split
    (checkpoints/logs/tensorboard/plots/eval); fold is a subdirectory of
    checkpoints/tensorboard/plots when this is a K-Fold run (``fold`` given)
    and omitted entirely for a non-CV run (``fold=None``), matching the old
    fold_suffix-on-filename convention but applied to directories instead.
    """
    root = os.path.join(base_dir, experiment_id(experiment_name, seed))

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
        "fold_splits": os.path.join(root, "fold_splits.json"),
        "run_meta": os.path.join(root, "run_meta.json"),
    }


def check_and_record_run_meta(
    run_meta_path: str,
    experiment_name: str,
    seed: int,
    config_hash_: str,
    logger: Any = None,
) -> None:
    """Enforce atomicity by *check*, not by path uniqueness: on first use
    for this experiment_id, record its config_hash; on every later call
    (a later fold, a resume), warn — never block — if the current config's
    hash has drifted from what was first recorded here. A real
    reconfiguration under the same experiment_name/seed is then a visible
    warning instead of silently blending two different runs' checkpoints.
    """
    if os.path.exists(run_meta_path):
        with open(run_meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("config_hash") != config_hash_ and logger is not None:
            logger.warning(
                f"run_meta.json config_hash mismatch at {run_meta_path}: "
                f"recorded {meta.get('config_hash')}, current {config_hash_} — "
                "this experiment_name+seed was previously run with a different "
                "config; checkpoints/logs below are now shared across both."
            )
        return
    os.makedirs(os.path.dirname(run_meta_path), exist_ok=True)
    meta = {
        "experiment_name": experiment_name,
        "seed": seed,
        "config_hash": config_hash_,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(run_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
