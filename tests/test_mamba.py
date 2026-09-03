"""
tests/test_mamba.py — Phase 6: Mamba/VSS hybrid model family.

Covers the core selective-scan recurrence (models/auxiliary/ss2d_ref.py),
the 2D multi-directional scan's direction geometry
(models/auxiliary/ss2d.py), the VSS/PVM block, fusion, the shared decoder,
the full assembled model (including Phase 4's channel_mode integration and
Phase 5's no_decay_group/A_log/D integration), and the test the plan names
for this phase: an extension of test_determinism (see
tests/test_orchestration.py::test_determinism) to this model family, which
is the first one in this codebase to actually exercise a
determinism-guard-tripping code path (on the mamba_ssm fused-kernel path,
unreachable in this sandbox — no mamba_ssm installed here; see
CHANGELOG.md's Phase 0 entry) or, on the pure-PyTorch fallback path (what
actually runs here), to pass the ordinary equality check every other model
family's determinism test expects.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest
import torch

from models.auxiliary.ss2d import DIRECTION_SETS, SS2D, _flatten, _unflatten
from models.auxiliary.ss2d_ref import selective_scan_ref
from models.auxiliary.vss import VSSBlock, build_vss_stage
from models.build import build_width_matched
from models.decoder import MambaDecoder
from models.fusion import build_fusion
from models.registry import ModelBudgetExceededError, get_model
from orchestration.runid import experiment_paths
from orchestration.schema import validate_config
from training.determinism import (
    get_recorded_manifest_extras,
    get_recorded_nondeterminism,
    reset_recorded_nondeterminism,
)
from training.optimizers import no_decay_group
from utils.metrics import count_parameters


def _mamba_model_kwargs(**overrides):
    kwargs = dict(
        name="mamba_unet", num_classes=1, in_channels=3,
        # out_channels: MambaUNet's own constructor doesn't read this (it
        # takes num_classes) — but train.py's loss setup reads
        # model_cfg["out_channels"] directly (a pre-existing repo
        # inconsistency between the UNet-family's out_channels and the
        # MK-UNet-family's num_classes for "the same" concept). Real
        # configs get this for free via inheritance from configs/base.yaml
        # when composed normally; this standalone dict needs it explicit.
        out_channels=1,
        channels=[4, 8, 16, 24, 32], depths=[1, 1, 1, 1, 1], kernel_sizes=[1, 3, 5],
        expansion_factor=2, d_state=8, vss_expand=2, scan_directions=4,
        pvm_groups=1, drop_path_rate=0.0, fusion="cbffm", skip_policy="fused",
    )
    kwargs.update(overrides)
    if "num_classes" in overrides and "out_channels" not in overrides:
        kwargs["out_channels"] = overrides["num_classes"]
    return kwargs


# ---------------------------------------------------------------------------
# Core selective scan (ss2d_ref)
# ---------------------------------------------------------------------------

def test_selective_scan_ref_matches_hand_computation():
    """With A=0 (no decay) and delta=1, the recurrence collapses to a plain
    cumulative sum weighted by B_t*u_t — verified against that closed form
    directly, independent of the implementation's own internal logic.
    """
    torch.manual_seed(0)
    b, d, n, l = 1, 2, 3, 4
    u = torch.randn(b, d, l)
    delta = torch.ones(b, d, l)
    A = torch.zeros(d, n)
    Bt = torch.randn(b, n, l)
    Ct = torch.randn(b, n, l)

    y = selective_scan_ref(u, delta, A, Bt, Ct)

    h = torch.zeros(b, d, n)
    y_manual = []
    for t in range(l):
        h = h + Bt[:, :, t].unsqueeze(1) * u[:, :, t : t + 1]
        y_manual.append((h * Ct[:, :, t].unsqueeze(1)).sum(-1))
    y_manual = torch.stack(y_manual, dim=-1)

    assert torch.allclose(y, y_manual, atol=1e-5)


def test_selective_scan_ref_gradients_flow():
    torch.manual_seed(0)
    b, d, n, l = 1, 2, 3, 4
    u = torch.randn(b, d, l, requires_grad=True)
    delta = torch.rand(b, d, l) + 0.1
    A = -torch.rand(d, n)
    Bt = torch.randn(b, n, l)
    Ct = torch.randn(b, n, l)
    D = torch.randn(d)
    y = selective_scan_ref(u, delta, A, Bt, Ct, D=D)
    y.sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()


# ---------------------------------------------------------------------------
# 2D scan direction geometry (ss2d)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("direction", DIRECTION_SETS[8])
def test_direction_flatten_unflatten_round_trips(direction):
    x = torch.arange(2 * 3 * 4 * 5).float().reshape(2, 3, 4, 5)
    flat = _flatten(x, direction)
    assert flat.shape == (2, 3, 20)
    restored = _unflatten(flat, direction, 4, 5)
    assert torch.equal(restored, x)


@pytest.mark.parametrize("scan_directions", [1, 2, 4, 8])
@pytest.mark.parametrize("merge", ["sum", "learned"])
def test_ss2d_forward_backward(scan_directions, merge):
    torch.manual_seed(0)
    b, d, n, h, w = 2, 4, 8, 6, 6
    scan = SS2D(d_inner=d, d_state=n, scan_directions=scan_directions, merge=merge)
    x = torch.randn(b, d, h, w, requires_grad=True)
    delta = torch.rand(b, d, h, w) + 0.1
    A = -torch.rand(d, n)
    Bp = torch.randn(b, n, h, w)
    Cp = torch.randn(b, n, h, w)
    y = scan(x, delta, A, Bp, Cp, D=torch.randn(d))
    assert y.shape == (b, d, h, w)
    y.sum().backward()
    assert x.grad is not None
    assert scan.scan_impl in ("mamba_ssm", "ss2d_ref")


# ---------------------------------------------------------------------------
# VSS / PVM block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pvm_groups", [1, 2, 4])
def test_vss_block_forward_backward(pvm_groups):
    torch.manual_seed(0)
    block = VSSBlock(d_model=16, d_state=8, expand=2, pvm_groups=pvm_groups, drop_path=0.1)
    x = torch.randn(2, 6, 6, 16, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_vss_block_rejects_non_dividing_pvm_groups():
    with pytest.raises(ValueError):
        VSSBlock(d_model=15, d_state=8, expand=2, pvm_groups=4)


def test_build_vss_stage_depth():
    stage = build_vss_stage(d_model=16, depth=3, d_state=8, pvm_groups=2)
    assert len(stage) == 3
    x = torch.randn(2, 6, 6, 16)
    assert stage(x).shape == x.shape


# ---------------------------------------------------------------------------
# Fusion + decoder (already covered lightly here; heavier coverage lives in
# manual verification during development — this is regression protection).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["add", "concat", "none", "xattn", "cbffm"])
def test_fusion_modes_shape_and_grad(mode):
    channels = 8
    fusion = build_fusion(mode, channels)
    fp = torch.randn(2, channels, 10, 10, requires_grad=True)
    fa = torch.randn(2, channels, 10, 10)
    out = fusion(fp, fa)
    assert out.shape == fp.shape
    out.sum().backward()
    assert fp.grad is not None


@pytest.mark.parametrize("skip_policy", ["primary_only", "fused", "both_concat"])
def test_decoder_reaches_full_resolution(skip_policy):
    """Regression test for a real bug found during development: the first
    decoder draft stopped one stage short of the input resolution (it
    omitted the equivalent of MK-UNet's decoder5 refinement step) and
    silently returned a half-resolution output."""
    stage_channels = [4, 8, 16, 24]
    dec = MambaDecoder(stage_channels, bottleneck_channels=32, num_classes=1, skip_policy=skip_policy)
    bottleneck = torch.randn(1, 32, 4, 4)
    sizes = [32, 16, 8, 4]
    primary = [torch.randn(1, c, s, s) for c, s in zip(stage_channels, sizes)]
    auxiliary = [torch.randn(1, c, s, s) for c, s in zip(stage_channels, sizes)]
    fused = [torch.randn(1, c, s, s) for c, s in zip(stage_channels, sizes)]
    out = dec(bottleneck, primary, auxiliary, fused)
    # primary[0] (stage_channels[0], the shallowest skip) is at "H/2" in
    # MK-UNet's convention (see models/decoder.py's docstring) — the
    # decoder's final upsample doubles that once more to reach full
    # resolution, so the output is 2x primary[0]'s spatial size.
    assert out.shape == (1, 1, 64, 64)


# ---------------------------------------------------------------------------
# Full assembled model + registry/optimizer integration
# ---------------------------------------------------------------------------

def test_mamba_unet_forward_backward():
    torch.manual_seed(0)
    model = get_model(**_mamba_model_kwargs())
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)
    y.sum().backward()
    assert x.grad is not None


def test_mamba_unet_consumes_channel_mode_input():
    """Phase 4 integration: the auxiliary encoder must accept the same
    multi-channel input as the primary encoder (not hardcoded to 3) — see
    models/proposed/mamba_unet.py's "Channel input decision" docstring."""
    model = get_model(**_mamba_model_kwargs(in_channels=11))  # m5 mode's channel count
    x = torch.randn(1, 11, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)


def test_mamba_unet_multiclass_output():
    model = get_model(**_mamba_model_kwargs(num_classes=4))
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 4, 64, 64)


def test_mamba_unet_scan_impl_reported():
    model = get_model(**_mamba_model_kwargs())
    assert model.scan_impl in ("mamba_ssm", "ss2d_ref")


def test_mamba_unet_capacity_control():
    def base_fn(channels):
        return get_model(**_mamba_model_kwargs(channels=channels))

    presets = {"t": [4, 8, 16, 24, 32], "base": [16, 32, 64, 96, 160]}
    target = count_parameters(base_fn(presets["base"]))
    result = build_width_matched(
        base_fn, [presets["t"], presets["base"]], input_shape=(3, 64, 64), target_params=target, tol=0.1
    )
    assert result["within_tolerance"] is True
    assert result["channels"] == presets["base"]


def test_mamba_unet_budget_guard():
    with pytest.raises(ModelBudgetExceededError):
        get_model(**_mamba_model_kwargs(channels=[16, 32, 64, 96, 160], budget_ceiling=1000))


def test_mamba_unet_A_log_and_D_excluded_from_weight_decay():
    """Phase 5's no_decay_group() was built forward-looking for exactly
    this model family — the first one in this codebase to actually define
    A_log/D parameters. Confirms the wiring works for real, not just in
    the synthetic _TinyNet fixture tests/test_optim.py uses."""
    model = get_model(**_mamba_model_kwargs(pvm_groups=2))
    decay, no_decay = no_decay_group(model)

    a_log_names = [n for n, _ in model.named_parameters() if n.endswith("A_log")]
    d_names = [n for n, _ in model.named_parameters() if n.rsplit(".", 1)[-1] == "D"]
    assert len(a_log_names) > 0 and len(d_names) > 0

    no_decay_ids = {id(p) for p in no_decay}
    a_log_ids = {id(p) for n, p in model.named_parameters() if n.endswith("A_log")}
    d_ids = {id(p) for n, p in model.named_parameters() if n.rsplit(".", 1)[-1] == "D"}
    assert a_log_ids <= no_decay_ids
    assert d_ids <= no_decay_ids


# ---------------------------------------------------------------------------
# test_determinism (the test the plan names for this phase)
# ---------------------------------------------------------------------------

def _tiny_mamba_config(tmp_path, tiny_dataset_dir, seed: int, name_suffix: str) -> dict:
    raw = {
        "model": _mamba_model_kwargs(pvm_groups=1, scan_directions=2, fusion="add"),
        "dataset": {
            "name": "synthetic_test_dataset",
            "root": str(tiny_dataset_dir),
            "img_height": 32,
            "img_width": 32,
            "batch_size": 2,
            "num_workers": 0,
            "cache": False,
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        },
        "training": {
            "epochs": 1, "lr": 0.01, "optimizer": "adamw", "loss_type": "dice",
            "device": "cpu", "seed": seed, "amp": False, "grad_clip_mode": "none",
        },
        "k_fold": {"enabled": False},
        "checkpoint": {
            "resume": False,
            "monitor_metric": "val_dice", "mode": "max",
        },
        "early_stopping": {"enabled": False},
        "stages": [],
        "logging": {
            "experiment_name": "mamba_determinism_test", "save_overlays": False,
        },
        "output_dir": str(tmp_path / f"outputs_{name_suffix}"),
    }
    return validate_config(raw)


def test_mamba_determinism(tmp_path, tiny_dataset_dir):
    """See module docstring. Branches on which scan implementation is
    actually active — this sandbox has no mamba_ssm (see
    CHANGELOG.md's Phase 0 entry), so the ss2d_ref branch is what runs
    here; the mamba_ssm branch's assertions describe what must hold on the
    real GPU training machine instead.
    """
    from train import run_training

    scan_impl = SS2D(d_inner=4, d_state=4).scan_impl

    results = {}
    for tag in ("a", "b"):
        cfg = _tiny_mamba_config(tmp_path, tiny_dataset_dir, seed=42, name_suffix=tag)
        reset_recorded_nondeterminism()
        best_metric = run_training(cfg, fold=None)

        ckpt_path = os.path.join(
            experiment_paths(
                cfg["output_dir"], cfg["logging"]["experiment_name"], cfg["training"]["seed"]
            )["checkpoints"],
            "last.pth",
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        results[tag] = {
            "best_metric": best_metric,
            "state_dict": ckpt["model_state_dict"],
            "nondeterminism": get_recorded_nondeterminism(),
            "manifest_extras": get_recorded_manifest_extras(),
        }

    # Both runs agree on which implementation was active (a process-wide
    # fact, not something that should vary run to run).
    assert results["a"]["manifest_extras"].get("scan_impl") == scan_impl
    assert results["b"]["manifest_extras"].get("scan_impl") == scan_impl

    if scan_impl == "ss2d_ref":
        # Pure-PyTorch path: expected to be exactly as reproducible as
        # every other (non-SSM) model family's determinism test.
        assert results["a"]["nondeterminism"] == []
        assert results["b"]["nondeterminism"] == []
        assert results["a"]["best_metric"] == results["b"]["best_metric"]
        sd_a, sd_b = results["a"]["state_dict"], results["b"]["state_dict"]
        assert sd_a.keys() == sd_b.keys()
        for key in sd_a:
            assert torch.equal(sd_a[key], sd_b[key]), f"weights diverged at {key}"
    else:
        # mamba_ssm fused-kernel path (not reachable in this sandbox):
        # the spec's explicit pitfall guard is that non-determinism here
        # must be *recorded*, not silently treated as reproducible — so
        # the assertion is "it was recorded", not "the weights matched".
        assert results["a"]["nondeterminism"] or results["b"]["nondeterminism"], (
            "mamba_ssm fused-kernel path ran but no non-determinism was recorded — "
            "either torch's determinism guard didn't fire (unexpected for this "
            "kernel) or the recording hook (training/determinism.py) isn't "
            "actually catching its warning anymore."
        )
