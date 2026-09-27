"""
uncertainty/retention.py — spec §12's error-correspondence and retention
rows: "Correlation between uncertainty and per-pixel error; AUROC of
uncertainty as an error detector" and "Risk-coverage / error-retention:
Dice as a function of the fraction of most-uncertain images referred
away."
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def error_detection_auroc(per_pixel_uncertainty: np.ndarray, per_pixel_error: np.ndarray) -> float:
    """AUROC treating "is this pixel wrong" (``per_pixel_error``, boolean —
    typically ``pred != gt``) as the label and ``per_pixel_uncertainty``
    (predictive entropy or inter-seed variance, same shape) as the score —
    spec's "AUROC of uncertainty as an error detector" row: a useful
    uncertainty map should assign higher scores to pixels the model
    actually got wrong.

    Returns ``float('nan')`` when every pixel (or no pixel) is an error —
    AUROC is undefined with only one class present, not a silent 0.5 or 1.0.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(per_pixel_error).astype(bool).ravel()
    scores = np.asarray(per_pixel_uncertainty).ravel()
    if y.shape != scores.shape:
        raise ValueError(f"error_detection_auroc: shape mismatch {y.shape} vs {scores.shape}")
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, scores))


def uncertainty_error_correlation(per_pixel_uncertainty: np.ndarray, per_pixel_error: np.ndarray) -> float:
    """Pearson correlation between per-pixel uncertainty and per-pixel
    error (0/1) — spec's "Correlation between uncertainty and per-pixel
    error" row, a linear-agreement complement to the (rank-based, harder
    to threshold-hack) AUROC above. NaN when either side has zero variance
    (e.g. a perfect prediction — nothing to correlate against).
    """
    u = np.asarray(per_pixel_uncertainty, dtype=np.float64).ravel()
    e = np.asarray(per_pixel_error, dtype=np.float64).ravel()
    if u.shape != e.shape:
        raise ValueError(f"uncertainty_error_correlation: shape mismatch {u.shape} vs {e.shape}")
    if np.std(u) == 0 or np.std(e) == 0:
        return float("nan")
    return float(np.corrcoef(u, e)[0, 1])


def retention_curve(
    per_image_uncertainty: Sequence[float],
    per_image_metric: Sequence[float],
    fractions: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> List[Dict[str, float]]:
    """Risk-coverage / error-retention curve: images are ranked by
    *per_image_uncertainty* (e.g. mean predictive entropy per image), the
    most-uncertain ``fraction`` is "referred away" (dropped), and the
    mean of *per_image_metric* (typically Dice) is recomputed over the
    retained (least-uncertain) images — spec's "Dice as a function of the
    fraction of most-uncertain images referred away".

    A metric that tracks real error should show *retained_metric*
    increasing as *fraction* grows (referring away the images the model is
    least sure about should leave an easier, higher-scoring subset behind)
    — this function reports the curve; judging whether it actually rises
    is the caller's/reporting layer's job.

    Returns one dict per fraction: ``{"referred_fraction", "n_retained",
    "retained_metric"}``, in the order *fractions* was given.
    """
    u = np.asarray(per_image_uncertainty, dtype=np.float64)
    m = np.asarray(per_image_metric, dtype=np.float64)
    if u.shape != m.shape:
        raise ValueError(f"retention_curve: shape mismatch {u.shape} vs {m.shape}")
    n = len(u)
    if n == 0:
        raise ValueError("retention_curve: no images given")

    order = np.argsort(u)  # ascending uncertainty: index 0 = most certain
    metric_sorted = m[order]

    results = []
    for frac in fractions:
        if not 0.0 <= frac < 1.0:
            raise ValueError(f"retention_curve: fractions must be in [0, 1), got {frac}")
        n_retained = max(1, int(round(n * (1.0 - frac))))
        retained_metric = float(np.mean(metric_sorted[:n_retained]))
        results.append({"referred_fraction": float(frac), "n_retained": n_retained, "retained_metric": retained_metric})
    return results
