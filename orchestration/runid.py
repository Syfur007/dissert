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
import hashlib
import json
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
