"""
metrics/calibration.py — Expected Calibration Error (ECE), new in this
phase (no prior implementation existed anywhere in this repo).

Uses equal-*mass* (quantile) binning rather than equal-width confidence
bins: with a small eval set and a model that's overwhelmingly confident
near 0/1 (typical for a converged segmentation model), most equal-width
bins near 0.5 would be empty while the two end bins hold nearly every pixel
— equal-mass binning keeps every bin populated regardless of how skewed the
confidence distribution is.
"""
from __future__ import annotations

from typing import List

import numpy as np


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-mass-binned ECE.

    Args:
        confidence: 1-D array in [0, 1] — how confident each prediction was
            in *its own* predicted class (i.e. already ``max(p, 1-p)`` for a
            binary problem, not the raw foreground probability).
        correct: 1-D 0/1 (or bool) array, same length — whether that
            prediction was correct.
        n_bins: number of equal-mass bins.
    """
    confidence = np.asarray(confidence).ravel()
    correct = np.asarray(correct).ravel().astype(float)
    n = len(confidence)
    if n == 0:
        return 0.0

    order = np.argsort(confidence)
    conf_sorted = confidence[order]
    correct_sorted = correct[order]
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        bin_conf = conf_sorted[lo:hi].mean()
        bin_acc = correct_sorted[lo:hi].mean()
        ece += (hi - lo) / n * abs(bin_acc - bin_conf)
    return float(ece)


def pixelwise_ece(
    probs_list: List[np.ndarray], gts_list: List[np.ndarray], n_bins: int = 10
) -> float:
    """ECE pooled across every pixel in the dataset (pooling first, rather
    than averaging a per-image ECE, is standard for pixel-level calibration
    and keeps a handful of tiny images from dominating the average the way
    a per-image mean would).

    Args:
        probs_list: soft foreground probabilities, one (H, W) (or (1, H, W))
            array per image, matching eval.py's ``probs_list``.
        gts_list: ground-truth masks, same shapes.
    """
    all_probs = np.concatenate([np.asarray(p).ravel() for p in probs_list])
    all_gts = np.concatenate([np.asarray(g).ravel() for g in gts_list]).astype(float)

    all_preds = (all_probs > 0.5).astype(float)
    # Confidence in the *predicted* class: a prediction of 0.99 background
    # (p=0.01) is just as confident as a prediction of 0.99 foreground —
    # using the raw foreground probability directly would score the former
    # as "very unconfident."
    confidence = np.where(all_preds == 1, all_probs, 1 - all_probs)
    correct = (all_preds == all_gts).astype(float)

    return expected_calibration_error(confidence, correct, n_bins=n_bins)
