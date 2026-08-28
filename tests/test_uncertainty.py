"""
tests/test_uncertainty.py — Phase 12: uncertainty module
(uncertainty/ensemble.py, uncertainty/retention.py).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from models.registry import get_model
from uncertainty.ensemble import inter_seed_variance, predict_ensemble_members, predictive_entropy
from uncertainty.retention import error_detection_auroc, retention_curve, uncertainty_error_correlation


def _mk_unet():
    return get_model(
        name="mk_unet", channels=[4, 8, 16, 24, 32], depths=[1, 1, 1, 1, 1],
        kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3,
        num_classes=1, in_channels=3,
    )


# ---------------------------------------------------------------------------
# ensemble.py
# ---------------------------------------------------------------------------

def test_predictive_entropy_binary_matches_closed_form():
    p = torch.tensor([0.5, 0.0001, 0.9999, 0.9]).view(4, 1, 1, 1)
    ent = predictive_entropy(p, is_multiclass=False).flatten()
    assert ent[0].item() == pytest.approx(math.log(2), abs=1e-4)
    assert ent[1].item() < 0.01
    assert ent[2].item() < 0.01
    expected_p9 = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    assert ent[3].item() == pytest.approx(expected_p9, abs=1e-6)


def test_predictive_entropy_multiclass_uniform_is_log_c():
    p = torch.full((1, 4, 1, 1), 0.25)
    ent = predictive_entropy(p, is_multiclass=True)
    assert ent.item() == pytest.approx(math.log(4), abs=1e-5)


def test_inter_seed_variance_matches_numpy_population_variance():
    member_probs = torch.tensor([0.2, 0.5, 0.8]).view(3, 1, 1, 1, 1)
    var = inter_seed_variance(member_probs, is_multiclass=False)
    assert var.item() == pytest.approx(float(np.var([0.2, 0.5, 0.8])), abs=1e-6)


def test_inter_seed_variance_multiclass_sums_over_classes():
    # 2 members, 1 pixel, 3 classes — per-class variance summed
    member_probs = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]]).view(2, 1, 3, 1, 1)
    var = inter_seed_variance(member_probs, is_multiclass=True)
    expected = sum(np.var([a, b]) for a, b in zip([0.2, 0.3, 0.5], [0.4, 0.1, 0.5]))
    assert var.item() == pytest.approx(expected, abs=1e-6)


def test_predict_ensemble_members_real_models():
    torch.manual_seed(0)
    models = [_mk_unet() for _ in range(3)]
    images = torch.rand(2, 3, 32, 32)
    member_probs = predict_ensemble_members(models, images, is_multiclass=False)
    assert member_probs.shape == (3, 2, 1, 32, 32)
    assert torch.all(member_probs >= 0) and torch.all(member_probs <= 1)

    mean_probs = member_probs.mean(dim=0)
    ent = predictive_entropy(mean_probs, is_multiclass=False)
    var = inter_seed_variance(member_probs, is_multiclass=False)
    assert ent.shape == (2, 32, 32)
    assert var.shape == (2, 32, 32)
    assert torch.all(ent >= 0)
    assert torch.all(var >= 0)


# ---------------------------------------------------------------------------
# retention.py
# ---------------------------------------------------------------------------

def test_error_detection_auroc_perfectly_separable():
    uncertainty = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])
    error = np.array([0, 0, 1, 1, 0, 1])
    assert error_detection_auroc(uncertainty, error) == 1.0


def test_error_detection_auroc_nan_when_single_class():
    uncertainty = np.array([0.1, 0.2, 0.8])
    assert np.isnan(error_detection_auroc(uncertainty, np.zeros(3)))
    assert np.isnan(error_detection_auroc(uncertainty, np.ones(3)))


def test_error_detection_auroc_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        error_detection_auroc(np.zeros(3), np.zeros(4))


def test_uncertainty_error_correlation_matches_numpy():
    uncertainty = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])
    error = np.array([0, 0, 1, 1, 0, 1])
    corr = uncertainty_error_correlation(uncertainty, error)
    assert corr == pytest.approx(float(np.corrcoef(uncertainty, error)[0, 1]), abs=1e-9)


def test_uncertainty_error_correlation_nan_when_zero_variance():
    assert np.isnan(uncertainty_error_correlation(np.zeros(4), np.array([0, 1, 0, 1])))


def test_retention_curve_hand_computed():
    per_image_uncertainty = [0.1, 0.4, 0.2, 0.3]
    per_image_metric = [0.9, 0.5, 0.8, 0.6]
    curve = retention_curve(per_image_uncertainty, per_image_metric, fractions=(0.0, 0.5))
    assert curve[0]["n_retained"] == 4
    assert curve[0]["retained_metric"] == pytest.approx(float(np.mean(per_image_metric)))
    assert curve[1]["n_retained"] == 2
    assert curve[1]["retained_metric"] == pytest.approx(0.85)  # mean of the 2 least-uncertain (0.9, 0.8)


def test_retention_curve_rejects_fraction_out_of_range():
    with pytest.raises(ValueError):
        retention_curve([0.1], [0.5], fractions=(1.0,))
    with pytest.raises(ValueError):
        retention_curve([0.1], [0.5], fractions=(-0.1,))


def test_retention_curve_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        retention_curve([0.1, 0.2], [0.5])


def test_retention_curve_rejects_empty():
    with pytest.raises(ValueError):
        retention_curve([], [])
