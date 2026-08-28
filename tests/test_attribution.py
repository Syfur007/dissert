"""
tests/test_attribution.py — Phase 11: attribution and explainability
module (attribution/{common,occlusion,shapley,integrated_grads,branch,
fusion_probe,segcam,sanity}.py).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from attribution.branch import ablate_auxiliary_stage, run_branch_ablation
from attribution.common import compute_training_mean_image, occlude_groups, predict_hard, resolve_group_slices
from attribution.fusion_probe import run_fusion_probe
from attribution.integrated_grads import agreement_score, run_integrated_gradients
from attribution.occlusion import run_channel_group_occlusion
from attribution.sanity import (
    label_randomization_sanity_check,
    parameter_randomization_sanity_check,
    randomize_model_,
)
from attribution.segcam import seg_grad_cam, seg_xres_cam
from attribution.shapley import run_exact_shapley, shapley_values_from_characteristic_function
from metrics.region import dice as _dice
from models.registry import get_model

H, W = 32, 32


class _FixedLoader:
    """Re-iterable, deterministic — every ``for ... in loader`` pass sees
    the identical data, matching a real (unshuffled) DataLoader's
    behaviour, which every cross-coalition/cross-stage comparison here
    depends on."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def _make_batches(n_batches, batch_size, channels=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        images = torch.rand(batch_size, channels, H, W, generator=g)
        masks = (torch.rand(batch_size, 1, H, W, generator=g) > 0.5).float()
        out.append((images, masks, [{}] * batch_size))
    return out


def _mk_unet(in_channels=3):
    return get_model(
        name="mk_unet", channels=[4, 8, 16, 24, 32], depths=[1, 1, 1, 1, 1],
        kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3,
        num_classes=1, in_channels=in_channels,
    )


def _mamba_unet(fusion="cbffm"):
    return get_model(
        name="mamba_unet", num_classes=1, in_channels=3, out_channels=1,
        channels=[4, 8, 16, 24, 32], depths=[1, 1, 1, 1, 1], kernel_sizes=[1, 3, 5],
        expansion_factor=2, d_state=8, vss_expand=2, scan_directions=2,
        pvm_groups=1, drop_path_rate=0.0, fusion=fusion, skip_policy="fused",
    )


# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------

def test_resolve_group_slices_m4():
    ds_cfg = {"channel_mode": "m4", "img_height": H, "img_width": W}
    slices = resolve_group_slices(ds_cfg, "colour")
    assert slices == {"rgb": slice(0, 3), "xy": slice(3, 5), "rtheta": slice(5, 8)}


def test_resolve_group_slices_grayscale_drops_ycbcr():
    ds_cfg = {"channel_mode": "m5", "img_height": H, "img_width": W}
    slices = resolve_group_slices(ds_cfg, "grayscale")
    assert "ycbcr" not in slices
    assert set(slices) == {"rgb", "xy", "rtheta"}


def test_compute_training_mean_image_shape_and_value():
    loader = _FixedLoader(_make_batches(2, 4, channels=3, seed=1))
    mean_image = compute_training_mean_image(loader, torch.device("cpu"))
    assert mean_image.shape == (3, H, W)
    all_images = torch.cat([b[0] for b in loader.batches], dim=0)
    assert torch.allclose(mean_image, all_images.mean(dim=0), atol=1e-6)


def test_occlude_groups_replaces_only_targeted_channels():
    images = torch.rand(2, 8, H, W)
    mean_image = torch.zeros(8, H, W)
    slices = {"rgb": slice(0, 3), "xy": slice(3, 5), "rtheta": slice(5, 8)}
    out = occlude_groups(images, mean_image, slices, ["xy"])
    assert torch.equal(out[:, 0:3], images[:, 0:3])       # rgb untouched
    assert torch.equal(out[:, 3:5], torch.zeros(2, 2, H, W))  # xy zeroed
    assert torch.equal(out[:, 5:8], images[:, 5:8])       # rtheta untouched
    # original tensor not mutated
    assert not torch.equal(images[:, 3:5], torch.zeros(2, 2, H, W))


def test_full_occlusion_equals_mean_image_directly():
    images = torch.rand(2, 8, H, W)
    mean_image = torch.rand(8, H, W)
    slices = {"rgb": slice(0, 3), "xy": slice(3, 5), "rtheta": slice(5, 8)}
    out = occlude_groups(images, mean_image, slices, list(slices.keys()))
    assert torch.allclose(out, mean_image.unsqueeze(0).expand(2, -1, -1, -1))


# ---------------------------------------------------------------------------
# occlusion.py + shapley.py (RQ2, exercised together for the efficiency check)
# ---------------------------------------------------------------------------

@pytest.fixture
def m4_setup():
    ds_cfg = {"channel_mode": "m4", "img_height": H, "img_width": W}
    modality = "colour"
    model = _mk_unet(in_channels=8)
    model.eval()
    train_loader = _FixedLoader(_make_batches(2, 4, channels=8, seed=2))
    test_loader = _FixedLoader(_make_batches(2, 3, channels=8, seed=3))
    device = torch.device("cpu")
    mean_image = compute_training_mean_image(train_loader, device)
    return model, ds_cfg, modality, device, test_loader, mean_image


def test_occlusion_reports_expected_shape(m4_setup):
    model, ds_cfg, modality, device, test_loader, mean_image = m4_setup
    result = run_channel_group_occlusion(model, test_loader, ds_cfg, modality, device, mean_image=mean_image)
    assert set(result.keys()) == {"rgb", "xy", "rtheta"}
    for group, r in result.items():
        assert r["n"] == 6  # 2 batches * 3 images
        assert 0.0 <= r["baseline_dice"] <= 1.0
        assert 0.0 <= r["occluded_dice"] <= 1.0
        assert r["dice_drop"] == pytest.approx(r["baseline_dice"] - r["occluded_dice"])


def test_occlusion_requires_mean_image_or_train_loader(m4_setup):
    model, ds_cfg, modality, device, test_loader, _mean_image = m4_setup
    with pytest.raises(ValueError):
        run_channel_group_occlusion(model, test_loader, ds_cfg, modality, device)


def test_shapley_matches_hand_computed_toy_game():
    v = {
        frozenset(): 0.0,
        frozenset(["A"]): 0.6,
        frozenset(["B"]): 0.3,
        frozenset(["A", "B"]): 1.0,
    }
    phi = shapley_values_from_characteristic_function(v, ["A", "B"])
    assert phi["A"] == pytest.approx(0.65)
    assert phi["B"] == pytest.approx(0.35)


def test_shapley_efficiency_property_additive_game():
    groups = ["A", "B", "C"]
    v = {frozenset(c): float(len(c)) for r in range(4) for c in itertools.combinations(groups, r)}
    phi = shapley_values_from_characteristic_function(v, groups)
    for g in groups:
        assert phi[g] == pytest.approx(1.0)


def test_shapley_efficiency_property_on_real_model(m4_setup):
    model, ds_cfg, modality, device, test_loader, mean_image = m4_setup
    occ_result = run_channel_group_occlusion(model, test_loader, ds_cfg, modality, device, mean_image=mean_image)
    shap_result = run_exact_shapley(model, test_loader, ds_cfg, modality, device, mean_image=mean_image)

    slices = resolve_group_slices(ds_cfg, modality)
    dices_none = []
    for images, masks, _meta in test_loader:
        occ = occlude_groups(images, mean_image, slices, list(slices.keys()))
        preds, _ = predict_hard(model, occ, False)
        for p, g in zip(preds.numpy(), masks.numpy()):
            dices_none.append(_dice(p, g.squeeze()))
    v_none = float(np.mean(dices_none))
    v_all = next(iter(occ_result.values()))["baseline_dice"]

    shap_sum = sum(shap_result.values())
    assert shap_sum == pytest.approx(v_all - v_none, abs=1e-6)


# ---------------------------------------------------------------------------
# integrated_grads.py
# ---------------------------------------------------------------------------

def test_integrated_gradients_mass_sums_to_one(m4_setup):
    model, ds_cfg, modality, device, test_loader, mean_image = m4_setup
    result = run_integrated_gradients(model, test_loader, ds_cfg, modality, device, mean_image=mean_image, n_steps=10)
    assert sum(result["per_channel_mass"]) == pytest.approx(1.0, abs=1e-5)
    assert set(result["per_group_mass"].keys()) == {"rgb", "xy", "rtheta"}
    assert result["n_images"] == 6


def test_agreement_score_perfect_when_rankings_match():
    occ = {"a": {"dice_drop": 0.5}, "b": {"dice_drop": 0.2}, "c": {"dice_drop": 0.1}}
    ig = {"per_group_mass": {"a": 0.9, "b": 0.3, "c": 0.05}}
    assert agreement_score(occ, ig) == pytest.approx(1.0)


def test_agreement_score_negative_when_rankings_invert():
    occ = {"a": {"dice_drop": 0.5}, "b": {"dice_drop": 0.2}, "c": {"dice_drop": 0.1}}
    ig = {"per_group_mass": {"a": 0.05, "b": 0.3, "c": 0.9}}
    assert agreement_score(occ, ig) == pytest.approx(-1.0)


def test_agreement_score_rejects_mismatched_groups():
    occ = {"a": {"dice_drop": 0.5}}
    ig = {"per_group_mass": {"b": 0.5}}
    with pytest.raises(ValueError):
        agreement_score(occ, ig)


# ---------------------------------------------------------------------------
# branch.py (Mamba-family only)
# ---------------------------------------------------------------------------

def test_ablate_auxiliary_stage_zeroes_only_target_stage():
    model = _mamba_unet()
    model.eval()
    with torch.no_grad(), ablate_auxiliary_stage(model, 2):
        captured = {}

        def probe(module, inp, out):
            captured["out"] = [o.clone() for o in out]

        h = model.auxiliary_encoder.register_forward_hook(probe)
        model(torch.rand(1, 3, H, W))
        h.remove()

    for i, o in enumerate(captured["out"]):
        if i == 2:
            assert torch.all(o == 0)
        else:
            assert not torch.all(o == 0)


def test_ablate_auxiliary_stage_hook_removed_after_context():
    model = _mamba_unet()
    n_before = len(model.auxiliary_encoder._forward_hooks)
    with ablate_auxiliary_stage(model, 0):
        assert len(model.auxiliary_encoder._forward_hooks) == n_before + 1
    assert len(model.auxiliary_encoder._forward_hooks) == n_before


def test_run_branch_ablation_covers_all_stages():
    model = _mamba_unet()
    model.eval()
    test_loader = _FixedLoader(_make_batches(2, 2, channels=3, seed=4))
    result = run_branch_ablation(model, test_loader, torch.device("cpu"), n_stages=4)
    assert set(result.keys()) == {0, 1, 2, 3}
    for stage, r in result.items():
        assert r["n"] == 4
        assert r["dice_drop"] == pytest.approx(r["baseline_dice"] - r["ablated_dice"])


def test_run_branch_ablation_rejects_non_mamba_model():
    model = _mk_unet()
    test_loader = _FixedLoader(_make_batches(1, 2, channels=3, seed=5))
    with pytest.raises(AttributeError):
        run_branch_ablation(model, test_loader, torch.device("cpu"))


# ---------------------------------------------------------------------------
# fusion_probe.py (CBFFM only)
# ---------------------------------------------------------------------------

def test_run_fusion_probe_gate_values_in_unit_interval():
    model = _mamba_unet(fusion="cbffm")
    test_loader = _FixedLoader(_make_batches(2, 2, channels=3, seed=6))
    result = run_fusion_probe(model, test_loader, torch.device("cpu"))
    assert set(result["per_stage_gate_mean"].keys()) == {0, 1, 2, 3}
    for stage, gates in result["per_stage_gate_mean"].items():
        assert len(gates) == 4
        assert all(0.0 <= g <= 1.0 for g in gates)
    assert len(result["gt_foreground_fraction"]) == 4


def test_run_fusion_probe_rejects_non_cbffm_model():
    model = _mamba_unet(fusion="add")
    test_loader = _FixedLoader(_make_batches(1, 2, channels=3, seed=7))
    with pytest.raises(ValueError):
        run_fusion_probe(model, test_loader, torch.device("cpu"))


def test_run_fusion_probe_rejects_model_without_fusion_stages():
    model = _mk_unet()
    test_loader = _FixedLoader(_make_batches(1, 2, channels=3, seed=8))
    with pytest.raises(AttributeError):
        run_fusion_probe(model, test_loader, torch.device("cpu"))


# ---------------------------------------------------------------------------
# segcam.py
# ---------------------------------------------------------------------------

def test_seg_grad_cam_shape_and_range():
    model = _mamba_unet()
    model.eval()
    image = torch.rand(1, 3, H, W)
    forced_mask = torch.ones(1, 1, H, W)
    cam = seg_grad_cam(model, model.decoder.final_conv, image, target_mask=forced_mask)
    assert cam.shape == (H, W)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
    assert cam.max() > 0.0


def test_seg_xres_cam_shape_and_range():
    model = _mamba_unet()
    model.eval()
    image = torch.rand(1, 3, H, W)
    forced_mask = torch.ones(1, 1, H, W)
    cam = seg_xres_cam(model, model.decoder.final_conv, image, target_mask=forced_mask)
    assert cam.shape == (H, W)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# sanity.py
# ---------------------------------------------------------------------------

def test_randomize_model_changes_output():
    model = _mk_unet()
    model.eval()
    image = torch.rand(1, 3, H, W)
    with torch.no_grad():
        before = model(image).clone()
    randomize_model_(model, fraction_top_down=1.0, seed=0)
    with torch.no_grad():
        after = model(image)
    assert not torch.allclose(before, after)


def test_randomize_model_partial_fraction_touches_fewer_modules():
    model = _mk_unet()
    n_full = len(randomize_model_(model, fraction_top_down=1.0, seed=0))
    model2 = _mk_unet()
    n_partial = len(randomize_model_(model2, fraction_top_down=0.25, seed=0))
    assert 0 < n_partial < n_full


def test_randomize_model_rejects_invalid_fraction():
    model = _mk_unet()
    with pytest.raises(ValueError):
        randomize_model_(model, fraction_top_down=0.0)
    with pytest.raises(ValueError):
        randomize_model_(model, fraction_top_down=1.5)


def test_parameter_randomization_sanity_check_passes_for_real_saliency():
    # Raw sigmoid output from an untrained model is near-constant (no
    # learned structure for cascading randomisation to disrupt — verified
    # directly: std ~3e-5 across the map), so it is a poor stand-in for
    # "real saliency" here; seg_grad_cam's gradient-based map (confirmed
    # non-degenerate above, full [0, 1] range) is used instead.
    model = _mamba_unet()
    model.eval()
    forced_mask = torch.ones(1, 1, H, W)

    def saliency_fn(m, image):
        return seg_grad_cam(m, m.decoder.final_conv, image, target_mask=forced_mask).numpy()

    image = torch.rand(1, 3, H, W)
    result = parameter_randomization_sanity_check(model, saliency_fn, image, fractions=(1.0,), seed=0)
    assert result["overall_pass"] is True
    assert result["per_fraction"][1.0]["ssim"] < result["ssim_pass_threshold"]


def test_parameter_randomization_sanity_check_fails_for_input_insensitive_saliency():
    model = _mk_unet()

    def constant_saliency_fn(m, image):
        return np.ones((H, W))

    image = torch.rand(1, 3, H, W)
    result = parameter_randomization_sanity_check(model, constant_saliency_fn, image, fractions=(1.0,), seed=0)
    assert result["overall_pass"] is False
    assert result["per_fraction"][1.0]["ssim"] == pytest.approx(1.0)


def test_label_randomization_sanity_check_identical_maps_fail():
    saliency = np.random.rand(H, W)
    result = label_randomization_sanity_check(saliency, saliency.copy())
    assert result["passes"] is False
    assert result["ssim"] == pytest.approx(1.0)


def test_label_randomization_sanity_check_different_maps_pass():
    rng = np.random.default_rng(0)
    a = rng.random((H, W))
    b = rng.random((H, W))
    result = label_randomization_sanity_check(a, b)
    assert result["passes"] is True
