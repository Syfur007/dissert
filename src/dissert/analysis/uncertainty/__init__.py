"""
uncertainty/ — Phase 12 of IMPLEMENTATION_PLAN.md, spec §12's UNCERTAINTY
MODULE: a deep ensemble over the existing seeds (zero extra training
cost), per-pixel predictive entropy and inter-seed variance, AUROC/
correlation of uncertainty as an error detector, and risk-coverage
(error-retention) curves.
"""
from __future__ import annotations

from .ensemble import inter_seed_variance, predict_ensemble_members, predictive_entropy
from .retention import error_detection_auroc, retention_curve, uncertainty_error_correlation

__all__ = [
    "predict_ensemble_members",
    "predictive_entropy",
    "inter_seed_variance",
    "error_detection_auroc",
    "uncertainty_error_correlation",
    "retention_curve",
]
