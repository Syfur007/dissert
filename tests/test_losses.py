"""
tests/test_losses.py — Phase 7: declarative compound losses, redundancy
guard, boundary loss.
"""
from __future__ import annotations

import copy

import numpy as np
import pydantic
import pytest
import torch

from losses import get_loss
from losses.compound import CompoundLoss, REDUNDANT_TERM_FAMILIES, StructureLoss
from losses.schedules import apply_schedule, constant, linear_ramp
from losses.terms import (
    bce,
    boundary,
    ce,
    compute_or_load_distance_map,
    compute_signed_distance_map,
    dice,
    focal,
    tversky,
)
from orchestration.schema import validate_config


def _binary_batch(seed=0, size=16):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 1, size, size, generator=g)
    targets = (torch.rand(2, 1, size, size, generator=g) > 0.5).float()
    return logits, targets


def _multiclass_batch(seed=0, size=16, num_classes=4):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, num_classes, size, size, generator=g)
    targets = torch.randint(0, num_classes, (2, size, size), generator=g)
    return logits, targets


# ---------------------------------------------------------------------------
# Individual terms
# ---------------------------------------------------------------------------

def test_tversky_reduces_to_dice_at_half_half():
    logits, targets = _binary_batch()
    d = dice(logits, targets)
    t = tversky(logits, targets, alpha=0.5, beta=0.5)
    assert torch.allclose(d, t, atol=1e-6)


def test_tversky_asymmetric_penalises_differently():
    logits, targets = _binary_batch()
    fn_heavy = tversky(logits, targets, alpha=0.9, beta=0.1)
    fp_heavy = tversky(logits, targets, alpha=0.1, beta=0.9)
    assert not torch.allclose(fn_heavy, fp_heavy)


@pytest.mark.parametrize("term_fn", [bce, dice, tversky, focal])
def test_binary_terms_gradients_flow(term_fn):
    logits, targets = _binary_batch()
    logits.requires_grad_(True)
    loss = term_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_dice_empty_mask_convention_matches_metrics_package():
    """losses.terms.dice's empty-vs-empty convention should agree with
    metrics.region.dice_iou's (both 1.0 for total agreement on emptiness)
    — a loss and a metric disagreeing about this would be a confusing,
    silent inconsistency between what's optimised and what's reported."""
    empty_logits = torch.full((1, 1, 8, 8), -100.0)  # sigmoid(-100) == 0.0 exactly in float32
    empty_targets = torch.zeros(1, 1, 8, 8)
    loss = dice(empty_logits, empty_targets)
    assert loss.item() < 1e-3  # loss ~= 0 means dice ~= 1.0, matching metrics.region


def test_ce_and_multiclass_dice_shapes():
    logits, targets = _multiclass_batch()
    assert ce(logits, targets).ndim == 0
    assert dice(logits, targets).ndim == 0


# ---------------------------------------------------------------------------
# Boundary loss
# ---------------------------------------------------------------------------

def test_signed_distance_map_sign_convention():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    dmap = compute_signed_distance_map(mask)
    assert dmap[0, 0] > 0  # outside the foreground square -> positive
    assert dmap[7, 7] < 0  # well inside -> negative


def test_signed_distance_map_degenerate_masks_are_zero():
    all_empty = np.zeros((8, 8), dtype=np.uint8)
    all_full = np.full((8, 8), 255, dtype=np.uint8)
    assert (compute_signed_distance_map(all_empty) == 0).all()
    assert (compute_signed_distance_map(all_full) == 0).all()


def test_distance_map_disk_cache_round_trips(tmp_path):
    import cv2

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    mask_path = str(tmp_path / "mask.png")
    cv2.imwrite(mask_path, mask)

    cache_dir = str(tmp_path / "cache")
    d1 = compute_or_load_distance_map(mask_path, cache_dir=cache_dir)
    d2 = compute_or_load_distance_map(mask_path, cache_dir=cache_dir)  # cache hit
    assert np.array_equal(d1, d2)


def test_boundary_loss_uses_probabilities_not_logits():
    """boundary() takes probabilities — passing raw logits would silently
    compute something on the wrong scale; this pins the documented
    contract with a concrete before/after comparison."""
    probs = torch.sigmoid(torch.randn(1, 1, 8, 8))
    dmap = torch.randn(1, 1, 8, 8)
    val_probs = boundary(probs, dmap)
    val_raw = boundary(probs * 10, dmap)  # NOT valid probabilities
    assert not torch.allclose(val_probs, val_raw)


# ---------------------------------------------------------------------------
# CompoundLoss + presets
# ---------------------------------------------------------------------------

def test_compound_loss_matches_manual_weighted_sum():
    logits, targets = _binary_batch()
    compound = CompoundLoss([("bce", 0.3, None), ("dice", 0.7, None)])
    expected = 0.3 * bce(logits, targets) + 0.7 * dice(logits, targets)
    assert torch.allclose(compound(logits, targets), expected, atol=1e-6)


def test_compound_loss_structure_term_forwards_construction_kwargs():
    logits, targets = _binary_batch()
    compound = CompoundLoss(
        [("structure", 1.0, None)], term_kwargs={"structure": {"boundary_weight": 2.0}}
    )
    direct = StructureLoss(boundary_weight=2.0)
    assert torch.allclose(compound(logits, targets), direct(logits, targets), atol=1e-6)


def test_compound_loss_boundary_term_requires_distance_maps():
    logits, targets = _binary_batch()
    compound = CompoundLoss([("boundary", 1.0, None)])
    with pytest.raises(ValueError):
        compound(logits, targets)  # no distance_maps given

    dmap = torch.randn(2, 1, 16, 16)
    compound(logits, targets, distance_maps=dmap)  # now works


def test_compound_loss_rejects_empty_term_list():
    with pytest.raises(ValueError):
        CompoundLoss([])


def test_get_loss_combo_matches_manual_bce_dice_sum():
    logits, targets = _binary_batch()
    combo = get_loss("combo", num_classes=1, bce_weight=0.4, dice_weight=0.6)
    expected = 0.4 * bce(logits, targets) + 0.6 * dice(logits, targets)
    assert torch.allclose(combo(logits, targets), expected, atol=1e-6)


def test_get_loss_bce_autoswitches_to_ce_for_multiclass():
    """Matches the pre-Phase-7 training/losses.py behaviour: requesting
    'bce' for a multiclass problem auto-switches to cross-entropy rather
    than hitting a shape-mismatch error."""
    logits, targets = _multiclass_batch()
    loss = get_loss("bce", num_classes=4)
    value = loss(logits, targets)
    expected = ce(logits, targets)
    assert torch.allclose(value, expected, atol=1e-6)


def test_get_loss_structure_rejects_multiclass():
    with pytest.raises(ValueError):
        get_loss("structure", num_classes=4)


def test_get_loss_compound_with_term_list():
    logits, targets = _binary_batch()
    loss = get_loss("compound", num_classes=1, term_list=[("focal", 1.0, None)])
    assert torch.allclose(loss(logits, targets), focal(logits, targets), atol=1e-6)


def test_get_loss_unknown_name_raises():
    with pytest.raises(ValueError):
        get_loss("not_a_real_loss")


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def test_linear_ramp_endpoints_and_midpoint():
    assert linear_ramp(0, 10) == pytest.approx(0.0)
    assert linear_ramp(10, 10) == pytest.approx(1.0)
    assert linear_ramp(5, 10) == pytest.approx(0.5)


def test_linear_ramp_clamped_outside_range():
    assert linear_ramp(-5, 10) == pytest.approx(0.0)
    assert linear_ramp(50, 10) == pytest.approx(1.0)


def test_apply_schedule_defaults_to_constant_one():
    assert apply_schedule(None, epoch=3, max_epoch=10) == 1.0
    assert apply_schedule({}, epoch=3, max_epoch=10) == 1.0


def test_compound_loss_set_epoch_changes_scheduled_term_weight():
    logits, targets = _binary_batch()
    compound = CompoundLoss(
        [("bce", 1.0, None), ("dice", 1.0, {"type": "linear", "start": 0.0, "end": 1.0})]
    )
    compound.set_epoch(0, 4)
    val_start = compound(logits, targets)
    compound.set_epoch(4, 4)
    val_end = compound(logits, targets)
    # At epoch 0, dice contributes ~0; at epoch==max_epoch, it contributes fully.
    assert torch.allclose(val_start, bce(logits, targets), atol=1e-5)
    assert torch.allclose(val_end, bce(logits, targets) + dice(logits, targets), atol=1e-5)


# ---------------------------------------------------------------------------
# Redundancy guard (schema-level)
# ---------------------------------------------------------------------------

def test_redundant_term_families_contains_dice_tversky():
    assert {"dice", "tversky"} in REDUNDANT_TERM_FAMILIES


def test_schema_rejects_redundant_loss_terms_without_override(tiny_config):
    cfg = copy.deepcopy(tiny_config)
    cfg["training"]["loss_type"] = "compound"
    cfg["training"]["loss_terms"] = [
        {"name": "dice", "weight": 0.5},
        {"name": "tversky", "weight": 0.5},
    ]
    with pytest.raises(pydantic.ValidationError):
        validate_config(cfg)


def test_schema_accepts_redundant_loss_terms_with_override(tiny_config):
    cfg = copy.deepcopy(tiny_config)
    cfg["training"]["loss_type"] = "compound"
    cfg["training"]["loss_terms"] = [
        {"name": "dice", "weight": 0.5},
        {"name": "tversky", "weight": 0.5},
    ]
    cfg["training"]["loss_override_reason"] = "deliberate ablation comparing dice vs. tversky weighting"
    validated = validate_config(cfg)
    assert len(validated["training"]["loss_terms"]) == 2


def test_schema_accepts_non_redundant_compound_terms(tiny_config):
    cfg = copy.deepcopy(tiny_config)
    cfg["training"]["loss_type"] = "compound"
    cfg["training"]["loss_terms"] = [
        {"name": "bce", "weight": 0.5},
        {"name": "dice", "weight": 0.5},
    ]
    validated = validate_config(cfg)  # must not raise
    assert validated["training"]["loss_terms"][0]["name"] == "bce"
