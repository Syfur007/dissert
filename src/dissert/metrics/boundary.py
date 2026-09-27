"""
metrics/boundary.py — HD95, ASD, NSD for a single binary mask pair.

**Behaviour change from utils/metrics.py's get_binary_metrics()**, not just a
port: when exactly one of prediction/ground-truth is empty, a boundary
distance is mathematically undefined (there is no "surface" on the empty
side to measure to/from) — this now returns ``None`` rather than the old ad
hoc ``999.0`` penalty. ``None`` means "exclude this sample from the dataset
average and count it" (see metrics.aggregate.compute_dataset_metrics's
``*_excluded_n`` fields), not "treat as zero" or "treat as a fixed penalty
distance" — a 999.0 constant silently distorts a mean/percentile in a way
that depends entirely on how many empty-vs-nonempty pairs happen to be in a
given eval run, which is exactly the kind of hidden convention this phase
exists to remove. When *both* masks are empty, the two sides trivially agree
(no boundary mismatch to measure), so HD95/ASD are 0.0 and NSD is 1.0 — this
part is unchanged from the original.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

# medpy.metric.binary imports `numpy.bool`, removed in numpy>=1.20; patched
# at package import time (metrics/__init__.py) before this module's own
# medpy import below ever runs.


def hd95(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    """95th-percentile (symmetric) Hausdorff distance, in pixels."""
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    pred_sum, gt_sum = pred_b.sum(), gt_b.sum()
    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return None
    from medpy.metric.binary import hd95 as _medpy_hd95
    return float(_medpy_hd95(pred_b, gt_b))


def asd(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    """Average (symmetric) surface distance, in pixels."""
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    pred_sum, gt_sum = pred_b.sum(), gt_b.sum()
    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return None
    from medpy.metric.binary import asd as _medpy_asd
    return float(_medpy_asd(pred_b, gt_b))


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of *mask*'s boundary voxels (mask minus its erosion)."""
    return mask & ~binary_erosion(mask)


def nsd(pred: np.ndarray, gt: np.ndarray, tolerance: float = 1.0) -> Optional[float]:
    """Normalised Surface Dice: fraction of surface voxels (on both sides)
    that lie within *tolerance* pixels of the other mask's surface. New
    metric (no prior implementation existed in this repo) — the 1.0-pixel
    default tolerance is a starting point, not a value derived from this
    project's data; tune per dataset/paper convention before citing.
    """
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return None

    pred_surface = _surface_voxels(pred_b)
    gt_surface = _surface_voxels(gt_b)

    # Distance transform of the complement of each surface gives, at every
    # voxel, the distance to the *nearest* surface voxel of that mask.
    dt_from_gt_surface = distance_transform_edt(~gt_surface)
    dt_from_pred_surface = distance_transform_edt(~pred_surface)

    pred_to_gt = dt_from_gt_surface[pred_surface]
    gt_to_pred = dt_from_pred_surface[gt_surface]

    within = np.concatenate([pred_to_gt <= tolerance, gt_to_pred <= tolerance])
    return float(within.mean()) if within.size > 0 else 1.0
