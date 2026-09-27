"""
robustness/ — Phase 13 of IMPLEMENTATION_PLAN.md, spec §13's ROBUSTNESS
MODULE (photometric/geometric/acquisition corruption suites, severities
1-5, degradation curves + mean corruption error) plus spec §11's shortcut
audit (coordonly Dice, translation-shift degradation, frame jitter),
grouped here per IMPLEMENTATION_PLAN.md's Phase 13 section since both
reuse the same severity scale and geometric-transform machinery.

Corruptions apply at test time only, after resize, never during training.
"""
from __future__ import annotations

from .common import apply_corruption_to_batch, degradation_curve, evaluate_under_corruption, mean_corruption_error
from .corruptions import ACQUISITION, CORRUPTIONS, PHOTOMETRIC, SEVERITY_LEVELS
from .geometric import (
    GEOMETRIC_TRANSFORMS,
    frame_jitter_sensitivity,
    geometric_degradation_curve,
    off_centre_crop,
    rotate,
    scale,
    shortcut_audit,
    translate,
)

__all__ = [
    "SEVERITY_LEVELS",
    "CORRUPTIONS",
    "PHOTOMETRIC",
    "ACQUISITION",
    "apply_corruption_to_batch",
    "evaluate_under_corruption",
    "degradation_curve",
    "mean_corruption_error",
    "GEOMETRIC_TRANSFORMS",
    "translate",
    "rotate",
    "scale",
    "off_centre_crop",
    "geometric_degradation_curve",
    "shortcut_audit",
    "frame_jitter_sensitivity",
]
