"""
metrics/ — the canonical segmentation-metrics package (Phase 2 of
IMPLEMENTATION_PLAN.md). Collapses what used to be three independently
drifted Dice/IoU/HD95/ASD implementations (utils/metrics.py,
training/trainer.py's rolling proxy, utils/report.py's extended metrics)
into this one module, used by training's canonical validation pass, eval.py,
and (later phases) attribution/robustness/statistics.

training/trainer.py's per-batch *rolling* Dice/IoU during training is the
one deliberate exception — it stays a separate, cheap, tensor-native
computation for live progress-bar/TensorBoard display and must never be
cited as a result; see its docstring in training/trainer.py.
"""
import numpy as np

# medpy.metric.binary (metrics.boundary's hd95/asd) imports `numpy.bool`,
# removed in numpy>=1.20. Patched once here, at package import time, before
# any submodule below can trigger medpy's import.
if not hasattr(np, "bool"):
    np.bool = bool

from .aggregate import EMPTY_MASK_CONVENTION, compute_dataset_metrics, write_per_image_parquet
from .boundary import asd, hd95, nsd
from .calibration import expected_calibration_error, pixelwise_ece
from .detection import (
    confusion_counts,
    fpr_on_normals,
    precision_recall_specificity_f2_accuracy,
    specificity_on_lesion_free_subset,
)
from .region import dice, dice_iou, iou

__all__ = [
    "EMPTY_MASK_CONVENTION",
    "compute_dataset_metrics",
    "write_per_image_parquet",
    "dice",
    "iou",
    "dice_iou",
    "hd95",
    "asd",
    "nsd",
    "confusion_counts",
    "precision_recall_specificity_f2_accuracy",
    "fpr_on_normals",
    "specificity_on_lesion_free_subset",
    "expected_calibration_error",
    "pixelwise_ece",
]
