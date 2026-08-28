"""
tests/test_robustness.py — Phase 13: robustness module
(robustness/corruptions.py, common.py, geometric.py).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.registry import get_model
from robustness.common import degradation_curve, evaluate_under_corruption, mean_corruption_error
from robustness.corruptions import CORRUPTIONS, SEVERITY_LEVELS
from robustness.geometric import (
    frame_jitter_sensitivity,
    geometric_degradation_curve,
    off_centre_crop,
    rotate,
    scale,
    shortcut_audit,
    translate,
)

H, W = 32, 32


class _FixedLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


class _IdentityLikeModel(nn.Module):
    """Predictions directly track input brightness — used to test the
    corruption/transform *pipeline* (does it actually alter what the
    model sees, does content/mask stay co-registered) independent of an
    untrained random-weight net's near-constant-output confound (observed
    directly elsewhere in this session's synthetic-data checks)."""

    def forward(self, x):
        return (x[:, :1] - 0.5) * 20.0


def _mk_unet():
    return get_model(
        name="mk_unet", channels=[4, 8, 16, 24, 32], depths=[1, 1, 1, 1, 1],
        kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3,
        num_classes=1, in_channels=3,
    )


# ---------------------------------------------------------------------------
# corruptions.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(CORRUPTIONS.keys()))
def test_every_corruption_preserves_shape_and_dtype(name):
    img = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    for severity in SEVERITY_LEVELS:
        out = CORRUPTIONS[name](img, severity)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_resolution_change_monotonically_more_destructive_with_severity():
    img = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    diffs = [np.abs(CORRUPTIONS["resolution_change"](img, s).astype(float) - img.astype(float)).mean() for s in SEVERITY_LEVELS]
    assert diffs == sorted(diffs)


def test_corruption_rejects_invalid_severity():
    img = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    with pytest.raises(ValueError):
        CORRUPTIONS["blur"](img, 6)
    with pytest.raises(ValueError):
        CORRUPTIONS["blur"](img, 0)


# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------

def test_evaluate_under_corruption_clean_baseline_perfect_alignment():
    model = _IdentityLikeModel()
    content = torch.zeros(1, 3, H, W)
    mask = torch.zeros(1, 1, H, W)
    content[:, :, 8:24, 8:24] = 1.0
    mask[:, :, 8:24, 8:24] = 1.0
    loader = _FixedLoader([(content, mask, [{}])])
    result = evaluate_under_corruption(model, loader, None, torch.device("cpu"))
    assert result["mean_dice"] == pytest.approx(1.0, abs=1e-6)


def test_blur_degrades_identity_like_model():
    model = _IdentityLikeModel()
    content = torch.zeros(1, 3, H, W)
    mask = torch.zeros(1, 1, H, W)
    content[:, :, 8:24, 8:24] = 1.0
    mask[:, :, 8:24, 8:24] = 1.0
    loader = _FixedLoader([(content, mask, [{}])])
    clean = evaluate_under_corruption(model, loader, None, torch.device("cpu"))
    blurred = evaluate_under_corruption(model, loader, lambda img: CORRUPTIONS["blur"](img, 5), torch.device("cpu"))
    assert blurred["mean_dice"] < clean["mean_dice"]


def test_degradation_curve_real_model_shape():
    torch.manual_seed(0)
    model = _mk_unet()
    images = torch.rand(2, 3, H, W)
    masks = (torch.rand(2, 1, H, W) > 0.5).float()
    loader = _FixedLoader([(images, masks, [{}, {}])])
    curve = degradation_curve(model, loader, "gaussian_noise", torch.device("cpu"))
    assert len(curve) == 6
    assert curve[0]["severity"] == 0
    assert [r["severity"] for r in curve[1:]] == list(SEVERITY_LEVELS)


def test_degradation_curve_rejects_unknown_corruption():
    model = _IdentityLikeModel()
    loader = _FixedLoader([(torch.zeros(1, 3, H, W), torch.zeros(1, 1, H, W), [{}])])
    with pytest.raises(ValueError):
        degradation_curve(model, loader, "not_a_real_corruption", torch.device("cpu"))


def test_mean_corruption_error_hand_computed():
    curve = [
        {"severity": 0, "mean_dice": 0.8},
        {"severity": 1, "mean_dice": 0.8},
        {"severity": 2, "mean_dice": 0.4},
    ]
    mce = mean_corruption_error(curve)
    expected = np.mean([(0.8 - 0.8) / 0.8, (0.8 - 0.4) / 0.8])
    assert mce == pytest.approx(expected)


def test_mean_corruption_error_nan_when_clean_is_zero():
    curve = [{"severity": 0, "mean_dice": 0.0}, {"severity": 1, "mean_dice": 0.0}]
    assert np.isnan(mean_corruption_error(curve))


def test_mean_corruption_error_requires_clean_baseline():
    with pytest.raises(ValueError):
        mean_corruption_error([{"severity": 1, "mean_dice": 0.5}])


# ---------------------------------------------------------------------------
# geometric.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("transform_fn,arg", [
    (translate, (0.0, 0.0)),
    (rotate, (0.0,)),
    (scale, (1.0,)),
])
def test_identity_parameters_are_near_noop(transform_fn, arg):
    content = torch.rand(1, 3, H, W)
    mask = (torch.rand(1, 1, H, W) > 0.5).float()
    c_out, m_out = transform_fn(content, mask, *arg)
    assert torch.allclose(c_out, content, atol=1e-5)
    assert torch.equal(m_out, mask)


@pytest.mark.parametrize("transform_fn,args", [
    (translate, (0.3, -0.15)),
    (rotate, (25.0,)),
    (scale, (1.4,)),
])
def test_mask_stays_strictly_binary(transform_fn, args):
    content = torch.rand(1, 3, H, W)
    mask = (torch.rand(1, 1, H, W) > 0.5).float()
    _c_out, m_out = transform_fn(content, mask, *args)
    assert set(torch.unique(m_out).tolist()).issubset({0.0, 1.0})


@pytest.mark.parametrize("transform_fn,args", [
    (translate, (0.2, -0.15)),
    (rotate, (25.0,)),
    (scale, (1.4,)),
])
def test_content_mask_stay_aligned_under_transform(transform_fn, args):
    content = torch.zeros(1, 3, H, W)
    mask = torch.zeros(1, 1, H, W)
    content[:, :, 8:16, 8:16] = 1.0
    mask[:, :, 8:16, 8:16] = 1.0
    c_out, m_out = transform_fn(content, mask, *args)
    fg = m_out[0, 0] > 0.5
    assert fg.sum() > 0
    assert c_out[0, :, fg].mean().item() > 0.7


def test_off_centre_crop_alignment():
    content = torch.zeros(1, 3, H, W)
    mask = torch.zeros(1, 1, H, W)
    content[:, :, 8:16, 8:16] = 1.0
    mask[:, :, 8:16, 8:16] = 1.0
    c_out, m_out = off_centre_crop(content, mask, 0.7, 0.1, 0.05)
    fg = m_out[0, 0] > 0.5
    if fg.sum() > 0:
        assert c_out[0, :, fg].mean().item() > 0.7


def test_scale_rejects_nonpositive_factor():
    content = torch.rand(1, 3, H, W)
    mask = torch.zeros(1, 1, H, W)
    with pytest.raises(ValueError):
        scale(content, mask, 0.0)
    with pytest.raises(ValueError):
        scale(content, mask, -1.0)


def test_geometric_degradation_curve_real_model_shape():
    torch.manual_seed(0)
    model = _mk_unet()
    images = torch.rand(2, 3, H, W)
    masks = (torch.rand(2, 1, H, W) > 0.5).float()
    loader = _FixedLoader([(images, masks, [{}, {}])])
    ds_cfg = {"channel_mode": "m1", "img_height": H, "img_width": W}
    curve = geometric_degradation_curve(model, loader, "rotate", ds_cfg, "colour", torch.device("cpu"))
    assert len(curve) == 6
    assert curve[0]["severity"] == 0


def test_geometric_degradation_curve_rejects_unknown_transform():
    model = _mk_unet()
    loader = _FixedLoader([(torch.zeros(1, 3, H, W), torch.zeros(1, 1, H, W), [{}])])
    ds_cfg = {"channel_mode": "m1", "img_height": H, "img_width": W}
    with pytest.raises(ValueError):
        geometric_degradation_curve(model, loader, "not_a_transform", ds_cfg, "colour", torch.device("cpu"))


def test_shortcut_audit_flags_above_threshold():
    model = _IdentityLikeModel()
    content = torch.zeros(1, 3, H, W)
    mask = torch.zeros(1, 1, H, W)
    content[:, :, 8:24, 8:24] = 1.0
    mask[:, :, 8:24, 8:24] = 1.0
    loader = _FixedLoader([(content, mask, [{}])])
    result = shortcut_audit(model, loader, torch.device("cpu"), threshold=0.5)
    assert result["coordonly_dice"] == pytest.approx(1.0, abs=1e-6)
    assert result["shortcut_flag"] is True


def test_shortcut_audit_does_not_flag_below_threshold():
    model = _IdentityLikeModel()
    content = torch.zeros(1, 3, H, W)
    mask = torch.zeros(1, 1, H, W)
    content[:, :, 8:24, 8:24] = 1.0
    mask[:, :, 8:24, 8:24] = 1.0
    loader = _FixedLoader([(content, mask, [{}])])
    result = shortcut_audit(model, loader, torch.device("cpu"), threshold=1.5)
    assert result["shortcut_flag"] is False


def test_frame_jitter_sensitivity_real_model_shape():
    torch.manual_seed(0)
    model = _mk_unet()
    images = torch.rand(2, 3, H, W)
    masks = (torch.rand(2, 1, H, W) > 0.5).float()
    loader = _FixedLoader([(images, masks, [{}, {}])])
    ds_cfg = {"channel_mode": "m1", "img_height": H, "img_width": W}
    result = frame_jitter_sensitivity(model, loader, ds_cfg, "colour", torch.device("cpu"), n_trials=3, seed=1)
    assert result["n_trials"] == 3
    assert len(result["per_trial_dice"]) == 3
