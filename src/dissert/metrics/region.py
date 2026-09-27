"""
metrics/region.py — Dice and IoU for a single binary mask pair.

Empty-mask convention (see metrics.aggregate.EMPTY_MASK_CONVENTION): when
both prediction and ground truth are empty, Dice and IoU are defined as 1.0
— perfect agreement on "nothing here" — not undefined. This ports
utils/metrics.py's original get_binary_metrics() convention unchanged; only
the boundary metrics (hd95/asd/nsd, see boundary.py) change their empty-mask
behaviour in this phase.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Dice and IoU for one binary (H, W) pair, computed in a single pass
    (one intersection/union computation shared by both) — matches the
    original get_binary_metrics()'s efficiency.
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    pred_sum = pred_b.sum()
    gt_sum = gt_b.sum()
    total = pred_sum + gt_sum

    if total == 0:
        return 1.0, 1.0

    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()

    d = float(2.0 * intersection / total)
    i = float(intersection / union) if union > 0 else 0.0
    return d, i


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    return dice_iou(pred, gt)[0]


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    return dice_iou(pred, gt)[1]
