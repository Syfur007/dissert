"""
analysis/ — Phase 13 of IMPLEMENTATION_PLAN.md, spec §12's MECHANISM
ANALYSIS: gradient-based Effective Receptive Field (erf.py), linear
Centered Kernel Alignment between representations (cka.py), and the
per-image failure taxonomy + gallery indexing (failure_taxonomy.py).
"""
from __future__ import annotations

from .cka import cka_matrix, flatten_spatial_features, linear_cka
from .erf import compute_erf, erf_radius
from .failure_taxonomy import FAILURE_CATEGORIES, classify_failure, failure_counts, gallery_indices

__all__ = [
    "compute_erf",
    "erf_radius",
    "linear_cka",
    "cka_matrix",
    "flatten_spatial_features",
    "FAILURE_CATEGORIES",
    "classify_failure",
    "failure_counts",
    "gallery_indices",
]
