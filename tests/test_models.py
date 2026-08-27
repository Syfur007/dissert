"""
tests/test_models.py — Phase 5: capacity control (build_width_matched) and
the model registry's budget guard.

The test the plan names for this phase: test_capacity_control_match.
"""
from __future__ import annotations

import pytest

from models.build import build_width_matched
from models.registry import ModelBudgetExceededError, get_model
from utils.metrics import count_parameters

_PRESETS = {
    "t": [4, 8, 16, 24, 32],
    "s": [8, 16, 32, 48, 80],
    "base": [16, 32, 64, 96, 160],
    "m": [32, 64, 128, 192, 320],
    "l": [64, 128, 256, 384, 512],
}


def _mk_unet_fn(channels):
    return get_model(
        name="mk_unet", channels=channels, depths=[1, 1, 1, 1, 1],
        kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3,
        num_classes=1, in_channels=3,
    )


def test_capacity_control_match():
    target_params = count_parameters(_mk_unet_fn(_PRESETS["base"]))
    result = build_width_matched(
        _mk_unet_fn,
        [_PRESETS["t"], _PRESETS["s"], _PRESETS["base"], _PRESETS["m"], _PRESETS["l"]],
        input_shape=(3, 64, 64),
        target_params=target_params,
        tol=0.1,
    )
    assert result["within_tolerance"] is True
    assert result["channels"] == _PRESETS["base"]
    assert result["relative_error"] == pytest.approx(0.0)


def test_capacity_control_reports_closest_when_no_match_within_tolerance():
    target_params = count_parameters(_mk_unet_fn(_PRESETS["s"]))
    # Deliberately exclude "s" so no candidate can be within tolerance.
    result = build_width_matched(
        _mk_unet_fn,
        [_PRESETS["t"], _PRESETS["base"], _PRESETS["m"], _PRESETS["l"]],
        input_shape=(3, 64, 64),
        target_params=target_params,
        tol=0.05,
    )
    assert result["within_tolerance"] is False
    assert result["channels"] == _PRESETS["t"]  # closest of the excluded set


def test_capacity_control_requires_a_target():
    with pytest.raises(ValueError):
        build_width_matched(_mk_unet_fn, [_PRESETS["t"]], input_shape=(3, 64, 64))


def test_capacity_control_requires_candidates():
    with pytest.raises(ValueError):
        build_width_matched(_mk_unet_fn, [], input_shape=(3, 64, 64), target_params=1000)


# ---------------------------------------------------------------------------
# Registry budget guard
# ---------------------------------------------------------------------------

def test_registry_budget_ceiling_raises_when_exceeded():
    with pytest.raises(ModelBudgetExceededError):
        get_model(
            name="mk_unet", channels=_PRESETS["l"], depths=[1, 1, 1, 1, 1],
            kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3,
            num_classes=1, in_channels=3, budget_ceiling=100_000,
        )


def test_registry_budget_ceiling_allow_over_budget_bypasses():
    model = get_model(
        name="mk_unet", channels=_PRESETS["l"], depths=[1, 1, 1, 1, 1],
        kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3,
        num_classes=1, in_channels=3, budget_ceiling=100_000, allow_over_budget=True,
    )
    assert count_parameters(model) > 100_000


def test_registry_budget_ceiling_not_triggered_when_under():
    model = get_model(
        name="mk_unet", channels=_PRESETS["t"], depths=[1, 1, 1, 1, 1],
        kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3,
        num_classes=1, in_channels=3, budget_ceiling=1_000_000,
    )
    assert count_parameters(model) <= 1_000_000
