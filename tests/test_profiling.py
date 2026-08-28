"""
tests/test_profiling.py — Phase 10: profiling/deployment module
(profiling/flops.py, latency.py, memory.py, export.py).

The test the plan names for this phase: test_flops_agreement.
"""
from __future__ import annotations

import os
import time

import pytest
import torch
import torch.nn as nn

from profiling.export import (
    ExportTimeout,
    export_all,
    try_export_onnx,
    try_export_tensorrt,
    try_export_torchscript,
)
from profiling.flops import (
    FlopsAgreementError,
    _selective_scan_flops,
    analytic_flops,
    check_flops_agreement,
    fvcore_flops,
)
from profiling.memory import checkpoint_size_mb, measure_peak_memory, measure_peak_memory_table
from profiling.latency import measure_latency, measure_latency_table
from models.registry import get_model


def _tiny_conv_model():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 1, 3, padding=1),
    )


def _mk_unet_tiny():
    return get_model(
        name="mk_unet", channels=[4, 8, 16, 24, 32], depths=[1, 1, 1, 1, 1],
        kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3,
        num_classes=1, in_channels=3,
    )


# ---------------------------------------------------------------------------
# flops.py
# ---------------------------------------------------------------------------

def test_selective_scan_flops_matches_hand_derivation():
    # Independent per-term derivation, not the formula under test:
    # deltaA (mul+exp), deltaB_u (2 muls), recurrence (mul+add), y_t
    # (N muls + N-1 adds ~= 2N), each over B*D*L*N elements/positions.
    B, D, N, L = 2, 3, 4, 5
    manual = (2 + 2 + 2) * B * D * L * N + (B * D * L) * (2 * N)
    assert _selective_scan_flops(D, N, L, batch=B) == manual


def test_analytic_flops_conv_matches_hand_formula():
    m = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    result = analytic_flops(m, (3, 16, 16))
    out = m(torch.zeros(1, 3, 16, 16))
    expected = 2 * out.numel() * 3 * 3 * 3  # 2 * out_elems * in_ch * kh * kw
    assert result["conv_linear_total"] == expected
    assert result["scan_total"] == 0


def test_analytic_flops_linear_includes_out_features():
    m = nn.Linear(64, 512)
    x = torch.zeros(1, 10, 64)
    result = analytic_flops(m, (10, 64))
    # regression check for the out_features-dropped bug this module had:
    # true FLOPs = 2 * n_output_elements * in_features, output has
    # out_features in it (10*512), not just the input's leading dims (10).
    assert result["conv_linear_total"] == 2 * 1 * 10 * 512 * 64


def test_fvcore_flops_conv_matches_2x_macs():
    m = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    fv = fvcore_flops(m, (3, 16, 16))
    out = m(torch.zeros(1, 3, 16, 16))
    expected_macs = out.numel() * 3 * 3 * 3
    assert fv["mac_style_true_flops"] == pytest.approx(2 * expected_macs)


@pytest.mark.parametrize("name,kwargs", [
    ("unet", {}),
    ("attention_unet", {}),
])
def test_check_flops_agreement_plain_convnets(name, kwargs):
    m = get_model(name=name, **kwargs)
    result = check_flops_agreement(m, (3, 32, 32), tolerance=0.05)
    assert result["agree"] is True
    assert result["relative_error"] < 0.01
    assert result["analytic_scan_total"] == 0


def test_check_flops_agreement_mk_unet():
    result = check_flops_agreement(_mk_unet_tiny(), (3, 64, 64), tolerance=0.05)
    assert result["agree"] is True
    assert result["reported_total"] > 0


def test_check_flops_agreement_raises_below_actual_error():
    # mk_unet has a small (~1%) but non-zero analytic/fvcore discrepancy —
    # a zero-tolerance check must legitimately raise on it.
    with pytest.raises(FlopsAgreementError):
        check_flops_agreement(_mk_unet_tiny(), (3, 64, 64), tolerance=0.0)


def test_check_flops_agreement_mamba_unet_scan_nonzero():
    m = get_model(
        name="mamba_unet", num_classes=1, in_channels=3, out_channels=1,
        channels=[4, 8, 16, 24, 32], depths=[1, 1, 1, 1, 1], kernel_sizes=[1, 3, 5],
        expansion_factor=2, d_state=8, vss_expand=2, scan_directions=4,
        pvm_groups=1, drop_path_rate=0.0, fusion="cbffm", skip_policy="fused",
    )
    result = check_flops_agreement(m, (3, 64, 64), tolerance=0.05)
    assert result["agree"] is True
    assert result["analytic_scan_total"] > 0
    # scan flops must be part of the reported total, not just computed and
    # discarded.
    assert result["reported_total"] > result["analytic_conv_linear_total"]


def test_flops_agreement():
    """spec §19's named test: "Analytic and tool FLOP counts within 5% for
    every registered model" — every architecturally-distinct family in
    models.registry.MODEL_REGISTRY (mk_unet_t/s/mk_unet share one class,
    covered once via mk_unet)."""
    configs = {
        "unet": dict(),
        "attention_unet": dict(),
        "mk_unet": dict(
            channels=[4, 8, 16, 24, 32], depths=[1, 1, 1, 1, 1], kernel_sizes=[1, 3, 5],
            expansion_factor=2, gag_kernel=3, num_classes=1, in_channels=3,
        ),
        "gmk_unet": dict(
            channels=[4, 8, 16, 24, 32], depths=[1, 1, 1, 1, 1], kernel_sizes=[1, 3, 5],
            expansion_factor=2, num_classes=1, in_channels=3,
        ),
        "emcad": dict(pretrain=False),
        "mamba_unet": dict(
            num_classes=1, in_channels=3, out_channels=1, channels=[4, 8, 16, 24, 32],
            depths=[1, 1, 1, 1, 1], kernel_sizes=[1, 3, 5], expansion_factor=2, d_state=8,
            vss_expand=2, scan_directions=4, pvm_groups=1, drop_path_rate=0.0,
            fusion="cbffm", skip_policy="fused",
        ),
    }
    for name, kwargs in configs.items():
        m = get_model(name=name, **kwargs)
        result = check_flops_agreement(m, (3, 64, 64), tolerance=0.05)
        assert result["agree"], f"{name}: {result}"


# ---------------------------------------------------------------------------
# latency.py
# ---------------------------------------------------------------------------

def test_measure_latency_rejects_below_spec_minimums():
    m = _tiny_conv_model()
    device = torch.device("cpu")
    with pytest.raises(ValueError):
        measure_latency(m, (3, 16, 16), device, num_warmup=5, num_runs=200)
    with pytest.raises(ValueError):
        measure_latency(m, (3, 16, 16), device, num_warmup=50, num_runs=10)


def test_measure_latency_reports_expected_fields():
    m = _tiny_conv_model()
    device = torch.device("cpu")
    result = measure_latency(m, (3, 16, 16), device, batch_size=1, num_warmup=50, num_runs=200, num_threads=1)
    assert result["batch_size"] == 1
    assert result["num_threads"] == 1
    assert result["median_ms"] > 0
    assert result["throughput_ips"] == pytest.approx(1000.0 / result["median_ms"])
    assert result["p95_ms"] >= result["median_ms"] - 1e-9


def test_measure_latency_table_keyed_by_batch_size():
    m = _tiny_conv_model()
    device = torch.device("cpu")
    table = measure_latency_table(m, (3, 16, 16), device, batch_sizes=(1, 4), num_warmup=50, num_runs=200)
    assert set(table.keys()) == {1, 4}
    assert table[1]["batch_size"] == 1
    assert table[4]["batch_size"] == 4


# ---------------------------------------------------------------------------
# memory.py
# ---------------------------------------------------------------------------

def test_measure_peak_memory_cpu_returns_none_not_zero():
    m = _tiny_conv_model()
    device = torch.device("cpu")
    result = measure_peak_memory(m, (3, 16, 16), device, batch_size=1)
    assert result["peak_allocated_mb"] is None
    assert result["peak_reserved_mb"] is None


def test_measure_peak_memory_table_keyed_by_batch_size():
    m = _tiny_conv_model()
    device = torch.device("cpu")
    table = measure_peak_memory_table(m, (3, 16, 16), device, batch_sizes=(1, 16))
    assert set(table.keys()) == {1, 16}


def test_checkpoint_size_mb(tmp_path):
    p = tmp_path / "ckpt.pth"
    torch.save({"state_dict": {"w": torch.zeros(1000)}}, p)
    size = checkpoint_size_mb(str(p))
    assert size == pytest.approx(os.path.getsize(p) / (1024 ** 2))


# ---------------------------------------------------------------------------
# export.py
# ---------------------------------------------------------------------------

def test_export_torchscript_and_onnx_succeed_for_a_real_model(tmp_path):
    m = _tiny_conv_model()
    ts = try_export_torchscript(m, (3, 16, 16), str(tmp_path / "model.pt"))
    assert ts["status"] == "ok"
    assert os.path.exists(ts["output_path"])

    onnx_res = try_export_onnx(m, (3, 16, 16), str(tmp_path / "model.onnx"))
    assert onnx_res["status"] == "ok"
    assert os.path.exists(onnx_res["output_path"])


def test_export_tensorrt_fails_cleanly_without_onnx_input(tmp_path):
    result = try_export_tensorrt(str(tmp_path / "nonexistent.onnx"), str(tmp_path / "model.trt"))
    assert result["status"] == "fail"
    assert result["error_class"] == "MissingONNXInput"


def test_export_all_returns_all_three_formats(tmp_path):
    m = _tiny_conv_model()
    results = export_all(m, (3, 16, 16), str(tmp_path))
    assert set(results.keys()) == {"torchscript", "onnx", "tensorrt"}
    assert results["torchscript"]["status"] == "ok"
    assert results["onnx"]["status"] == "ok"
    # tensorrt: ok if the package happens to be installed, fail otherwise —
    # either way it must be a real result, not an unhandled exception.
    assert results["tensorrt"]["status"] in ("ok", "fail")


class _SlowModule(nn.Module):
    def forward(self, x):
        time.sleep(3)
        return x


def test_export_timeout_resolves_to_a_clean_fail(tmp_path):
    result = try_export_torchscript(_SlowModule(), (3, 4, 4), str(tmp_path / "model.pt"), timeout_s=1)
    assert result["status"] == "fail"
    assert result["error_class"] == "ExportTimeout"
