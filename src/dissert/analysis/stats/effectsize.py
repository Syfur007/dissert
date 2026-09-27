"""
stats/effectsize.py — Cliff's delta and paired median difference, per
spec §10's "Effect size" row: "Reported with every p" — every
significance-test call site is meant to report one of these alongside its
p-value, not a bare p-value on its own.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """δ = (#{a_i > b_j} - #{a_i < b_j}) / (len(a) * len(b)), in [-1, 1].
    0 = no stochastic dominance either way; +1 = every a strictly exceeds
    every b; -1 the reverse.

    O(n·m) direct pairwise comparison (not the O(n log n) rank-based
    shortcut) — the per-image score arrays this project compares are small
    (tens to a few hundred images), not the scale where that would matter.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("cliffs_delta: both inputs must be non-empty")
    greater = int((a[:, None] > b[None, :]).sum())
    less = int((a[:, None] < b[None, :]).sum())
    return float((greater - less) / (len(a) * len(b)))


def paired_median_diff(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Median (and mean, for reference) of the per-pair differences
    ``a_i - b_i`` — **not** ``median(a) - median(b)``, a different (and for
    this project's "same images, paired" comparisons, less appropriate)
    quantity. a and b must be paired: same length, same order.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"paired_median_diff: a and b must be paired, got {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("paired_median_diff: no scores given")
    diffs = a - b
    return {"median_diff": float(np.median(diffs)), "mean_diff": float(np.mean(diffs)), "n": int(len(a))}
