"""
analysis/failure_taxonomy.py — spec §12's mechanism-analysis module,
failure-taxonomy row: buckets every test image's prediction into one of a
fixed set of failure categories, for the failure gallery + taxonomy bar
chart. "Findings written even if null" (spec's own S12 exit criterion) —
``failure_counts`` reports every category's count explicitly, including
zero, rather than omitting empty buckets.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from metrics.detection import precision_recall_specificity_f2_accuracy
from metrics.region import dice_iou

FAILURE_CATEGORIES = (
    "success",
    "missed_lesion",
    "false_positive",
    "under_segmentation",
    "over_segmentation",
    "boundary_only",
)


def classify_failure(
    pred: np.ndarray,
    gt: np.ndarray,
    dice_threshold: float = 0.5,
    pr_gap_threshold: float = 0.3,
) -> str:
    """One of ``FAILURE_CATEGORIES`` for a single (H, W) binary
    prediction/ground-truth pair:

      - ``success``: both empty (correct "nothing here"), or Dice >=
        *dice_threshold* with precision/recall reasonably balanced.
      - ``missed_lesion``: ground truth has foreground, prediction is
        entirely empty (complete miss, the clinically worst case).
      - ``false_positive``: ground truth is entirely empty, prediction has
        foreground (spurious detection on a normal case).
      - ``under_segmentation``: some overlap, but recall trails precision
        by more than *pr_gap_threshold* (missed a substantial part of the
        true region).
      - ``over_segmentation``: some overlap, but precision trails recall
        by more than *pr_gap_threshold* (predicted well beyond the true
        region).
      - ``boundary_only``: some overlap, precision/recall roughly
        balanced, but Dice still falls below *dice_threshold* (a
        boundary-placement problem rather than a region-finding one).
    """
    gt_sum = int(gt.astype(bool).sum())
    pred_sum = int(pred.astype(bool).sum())

    if gt_sum == 0 and pred_sum == 0:
        return "success"
    if gt_sum > 0 and pred_sum == 0:
        return "missed_lesion"
    if gt_sum == 0 and pred_sum > 0:
        return "false_positive"

    d, _ = dice_iou(pred, gt)
    prf = precision_recall_specificity_f2_accuracy(pred, gt)
    precision, recall = prf["precision"], prf["recall"]

    if d >= dice_threshold and abs(precision - recall) <= pr_gap_threshold:
        return "success"
    if recall < precision - pr_gap_threshold:
        return "under_segmentation"
    if precision < recall - pr_gap_threshold:
        return "over_segmentation"
    return "boundary_only"


def failure_counts(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    dice_threshold: float = 0.5,
    pr_gap_threshold: float = 0.3,
) -> Dict[str, object]:
    """Classifies every (pred, gt) pair and tallies counts per category —
    spec's "failure counts" output. Every category in ``FAILURE_CATEGORIES``
    is present in the returned ``counts`` dict even at 0 (spec's "Findings
    written even if null").

    Returns ``{"counts": {category: n}, "fractions": {category: n/N},
    "per_image_category": [category, ...], "n": N}`` — per_image_category
    is index-aligned with *preds*/*gts*, the index list a failure gallery
    figure reads to pick which images to show per category.
    """
    if len(preds) != len(gts):
        raise ValueError(f"failure_counts: preds/gts length mismatch {len(preds)} vs {len(gts)}")
    if not preds:
        raise ValueError("failure_counts: no images given")

    counts = {c: 0 for c in FAILURE_CATEGORIES}
    per_image_category: List[str] = []
    for p, g in zip(preds, gts):
        category = classify_failure(p, g, dice_threshold, pr_gap_threshold)
        counts[category] += 1
        per_image_category.append(category)

    n = len(preds)
    return {
        "counts": counts,
        "fractions": {c: count / n for c, count in counts.items()},
        "per_image_category": per_image_category,
        "n": n,
    }


def gallery_indices(per_image_category: List[str], category: str, max_examples: int = 8) -> List[int]:
    """Indices (into the original preds/gts list) of up to *max_examples*
    images classified as *category* — what a failure-gallery figure panel
    for that category is built from.
    """
    if category not in FAILURE_CATEGORIES:
        raise ValueError(f"Unknown failure category '{category}'. Known: {FAILURE_CATEGORIES}")
    matches = [i for i, c in enumerate(per_image_category) if c == category]
    return matches[:max_examples]
