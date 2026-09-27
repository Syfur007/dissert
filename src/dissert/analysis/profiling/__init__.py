"""
profiling/ — Phase 10 of IMPLEMENTATION_PLAN.md, spec §14's PROFILING AND
DEPLOYMENT measurement protocol: params, FLOPs (analytic + fvcore
agreement-checked), GPU/CPU latency and throughput, peak memory, checkpoint
size, and export outcomes (ONNX/TorchScript/TensorRT).

Supersedes ``utils/metrics.py``'s ``log_model_summary``/``measure_throughput``
(named for deletion by IMPLEMENTATION_PLAN.md's Phase 10 section) — that
module keeps only ``count_parameters``, reused here directly.
"""
from __future__ import annotations

from .export import export_all, try_export_onnx, try_export_tensorrt, try_export_torchscript
from .flops import FlopsAgreementError, analytic_flops, check_flops_agreement, fvcore_flops
from .latency import measure_latency, measure_latency_table
from .memory import checkpoint_size_mb, measure_peak_memory, measure_peak_memory_table

__all__ = [
    "analytic_flops",
    "fvcore_flops",
    "check_flops_agreement",
    "FlopsAgreementError",
    "measure_latency",
    "measure_latency_table",
    "measure_peak_memory",
    "measure_peak_memory_table",
    "checkpoint_size_mb",
    "export_all",
    "try_export_torchscript",
    "try_export_onnx",
    "try_export_tensorrt",
]
