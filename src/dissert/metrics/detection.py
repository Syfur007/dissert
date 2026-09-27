"""
metrics/detection.py — precision/recall/specificity/F2/accuracy (per-image),
plus two dataset-level aggregates over the lesion-free subset.

Ports utils/report.py's compute_extended_metrics() (same TP/FP/FN/TN
convention, same "0.0 when the denominator is 0" convention) as
precision_recall_specificity_f2_accuracy(), and adds the two aggregates the
per-image version can't express on its own: on a lesion-*present* image,
"true negative" pixels are trivially most of the image, which waters down
specificity as a lesion-detection sanity check — fpr_on_normals() and
specificity_on_lesion_free_subset() restrict to images with an entirely
empty ground truth ("normal" cases), where every predicted foreground pixel
is unambiguously a false positive.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    """(tp, fp, fn, tn) over every pixel of one (H, W) binary pair."""
    p_b = pred.astype(bool).ravel()
    g_b = gt.astype(bool).ravel()
    tp = int(np.logical_and(p_b, g_b).sum())
    fp = int(np.logical_and(p_b, ~g_b).sum())
    fn = int(np.logical_and(~p_b, g_b).sum())
    tn = int(np.logical_and(~p_b, ~g_b).sum())
    return tp, fp, fn, tn


def precision_recall_specificity_f2_accuracy(
    pred: np.ndarray, gt: np.ndarray, beta: float = 2.0
) -> dict:
    tp, fp, fn, tn = confusion_counts(pred, gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    beta2 = beta ** 2
    denom = beta2 * precision + recall
    f_beta = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f2": f_beta,
        "accuracy": accuracy,
    }


def fpr_on_normals(preds: List[np.ndarray], gts: List[np.ndarray]) -> Optional[float]:
    """Pixel-level false-positive rate over lesion-free images only
    (``gt`` entirely empty). ``None`` — not 0.0 — when the dataset has no
    lesion-free image, since the metric is undefined there, not perfect.
    """
    fp_total = 0
    neg_total = 0
    for p, g in zip(preds, gts):
        g_b = g.astype(bool)
        if g_b.sum() == 0:
            p_b = p.astype(bool)
            fp_total += int(p_b.sum())
            neg_total += p_b.size
    return (fp_total / neg_total) if neg_total > 0 else None


def specificity_on_lesion_free_subset(
    preds: List[np.ndarray], gts: List[np.ndarray]
) -> Optional[float]:
    """Specificity restricted to lesion-free images — equivalently
    ``1 - fpr_on_normals()``, reported separately since "specificity" and
    "FPR on normals" are each the conventional name in different corners of
    the medical-imaging literature. ``None`` when undefined (no lesion-free
    images in the dataset).
    """
    fpr = fpr_on_normals(preds, gts)
    return None if fpr is None else 1.0 - fpr
