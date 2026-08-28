"""
analysis/cka.py — spec §12's mechanism-analysis module, CKA row: linear
Centered Kernel Alignment (Kornblith et al., 2019) between two sets of
per-stage representations — a similarity score in [0, 1] used to compare
what two model families' (or two stages') internal features actually
encode, independent of any specific basis/rotation/scale those features
happen to be represented in.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between activation matrices *x* ``(n, p)`` and *y*
    ``(n, q)`` — *n* matched samples (same images through both stages/
    models), *p*/*q* feature dimensions (need not match). Uses the
    Frobenius-norm identity ``||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)``
    (Kornblith et al. 2019, eq. 3) rather than forming the full ``n x n``
    Gram matrices HSIC's original formulation uses — identical result,
    ``O(n*p*q)`` instead of ``O(n^2*(p+q))``, which matters once *n* is a
    full test set's worth of feature maps.

    Both inputs are mean-centred (over the sample dimension) before the
    above — CKA is defined on centred features.

    Returns a value in [0, 1] (1.0 = identical up to rotation/isotropic
    scaling); raises ``ValueError`` if *x*/*y* don't share the sample
    dimension.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"linear_cka: expected 2D (n, features) arrays, got shapes {x.shape}, {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"linear_cka: sample-dimension mismatch {x.shape[0]} vs {y.shape[0]}")

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    numerator = np.linalg.norm(y.T @ x, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    if denom == 0:
        return 0.0
    return float(numerator / denom)


def cka_matrix(stage_features_a: List[np.ndarray], stage_features_b: List[np.ndarray]) -> np.ndarray:
    """Pairwise linear CKA between every stage of model/branch A and every
    stage of model/branch B — spec's "CKA per stage" row, assembled into
    one ``(len(stage_features_a), len(stage_features_b))`` matrix (e.g.
    the primary vs. auxiliary branch's 4 stages each, for a Mamba-family
    model, or two different model families' stage sequences).

    Each element of *stage_features_a*/*stage_features_b* is one stage's
    ``(n_samples, n_features)`` flattened-activation matrix (same
    *n_samples*, matched images, across every entry in both lists).
    """
    matrix = np.zeros((len(stage_features_a), len(stage_features_b)))
    for i, fa in enumerate(stage_features_a):
        for j, fb in enumerate(stage_features_b):
            matrix[i, j] = linear_cka(fa, fb)
    return matrix


def flatten_spatial_features(activation: np.ndarray) -> np.ndarray:
    """``(B, C, H, W)`` activation -> ``(B, C*H*W)`` — the flattening CKA
    needs (one feature vector per sample), kept as a named helper so every
    caller applies it identically rather than each writing its own
    ``.reshape``.
    """
    if activation.ndim != 4:
        raise ValueError(f"flatten_spatial_features: expected (B, C, H, W), got shape {activation.shape}")
    b = activation.shape[0]
    return activation.reshape(b, -1)
