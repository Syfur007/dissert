"""
training/determinism.py — the one seed_everything() entry point, replacing
train.py's former local ``set_seed()`` and eval.py's former total absence of
seeding.

Also configures torch for deterministic execution and *records* (rather than
silently drops) any op torch cannot actually run deterministically, so a run
can be reported non-reproducible instead of assumed reproducible — this
matters most for Phase 6's Mamba model: on the ``mamba-ssm`` fused-kernel
path it is expected to genuinely trip this, and the point is that it shows
up in the manifest instead of being missed.
"""
from __future__ import annotations

import random
import warnings
from typing import Callable, List

import numpy as np
import torch

from datasets.datamodule import _make_worker_init_fn

_NONDETERMINISM_LOG: List[str] = []
_original_showwarning = warnings.showwarning
_hook_installed = False

# Substrings torch's own warn_only=True path uses when a non-deterministic
# op actually executes (as of torch 1.13; see torch/_utils_internal.py /
# torch/utils/deterministic.py for the exact wording torch emits).
_NONDETERMINISM_MARKERS = (
    "does not have a deterministic implementation",
    "nondeterministic",
)


def _capture_showwarning(message, category, filename, lineno, file=None, line=None):
    text = str(message)
    if any(marker in text.lower() for marker in _NONDETERMINISM_MARKERS):
        _NONDETERMINISM_LOG.append(text)
    _original_showwarning(message, category, filename, lineno, file, line)


def seed_everything(seed: int, deterministic: bool = True) -> Callable[[int], None]:
    """Seed python/numpy/torch(+cuda) and, when *deterministic*, configure
    torch to run deterministically wherever it can.

    Returns the same per-worker DataLoader seeding function
    ``datasets.datamodule`` already builds its loaders with
    (``_make_worker_init_fn``) — callers don't need to use the return value
    for training/eval, since every DataModule in this repo already derives
    its own worker seed from ``config["training"]["seed"]`` independently;
    it's exposed here so there is exactly one implementation of "seed a
    DataLoader worker" that both paths point at, not two that can drift.

    Non-deterministic ops are not raised on (``warn_only=True``): several
    ops this codebase already depends on have no deterministic CUDA kernel
    in torch 1.13, and raising would make training unusable rather than
    just imprecisely reproducible. Instead, every such warning is captured
    — call :func:`get_recorded_nondeterminism` after the run to fetch them
    (e.g. to attach to a run manifest) and :func:`reset_recorded_nondeterminism`
    before each new run so warnings don't leak across runs sharing a
    process (a K-Fold sweep, a search.py trial loop).
    """
    global _hook_installed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if deterministic:
        # `warn_only` kwarg present from torch 1.11 onward — confirmed
        # available at this repo's torch 1.13 pin (CHANGELOG.md Phase 0).
        torch.use_deterministic_algorithms(True, warn_only=True)

    if not _hook_installed:
        warnings.showwarning = _capture_showwarning
        _hook_installed = True

    return _make_worker_init_fn(seed)


def reset_recorded_nondeterminism() -> None:
    _NONDETERMINISM_LOG.clear()


def get_recorded_nondeterminism() -> List[str]:
    return list(_NONDETERMINISM_LOG)
