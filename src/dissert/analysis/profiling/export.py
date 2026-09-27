"""
profiling/export.py — spec §14's "Export | ONNX, TorchScript, TensorRT
attempted for every model. Outcome recorded as ok | fail + error class.
Failure is a reportable result, not a blocker." row.

Every ``try_export_*`` function catches its own exceptions and returns a
result dict rather than raising — a failed export (missing optional
dependency, an op with no ONNX/TensorRT translation, a dynamic-control-flow
model TorchScript's tracer can't follow) is data for the export-outcome
matrix, not a reason to abort profiling the rest of the model.
"""
from __future__ import annotations

import os
import signal
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

_DEFAULT_TIMEOUT_S = 120


class ExportTimeout(Exception):
    """Raised when a single export attempt exceeds its time budget."""


@contextmanager
def _time_limit(seconds: int):
    """SIGALRM-based hard timeout (Unix-only — fine here: this project
    targets Linux dev boxes and Linux CI runners throughout, e.g.
    .github/workflows/ci.yml). Needed because a slow trace (e.g.
    TorchScript/ONNX export tracing through a Mamba model's sequential
    selective-scan Python loop — confirmed directly: an un-timed attempt
    at a modest 32x32 resolution was still running after 90 seconds) would
    otherwise hang the whole profiling run rather than resolving to the
    ``fail`` outcome spec explicitly allows for.
    """
    def _on_alarm(signum, frame):
        raise ExportTimeout(f"export exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _result(fmt: str, status: str, error_class: str = "", error_message: str = "", output_path: str = "", size_mb: Optional[float] = None) -> Dict[str, object]:
    return {
        "format": fmt,
        "status": status,
        "error_class": error_class,
        "error_message": error_message,
        "output_path": output_path,
        "size_mb": size_mb,
    }


def try_export_torchscript(
    model: nn.Module, input_shape: Tuple[int, int, int], out_path: str, timeout_s: int = _DEFAULT_TIMEOUT_S
) -> Dict[str, object]:
    """``torch.jit.trace`` — the same tracing mechanism fvcore's
    FlopCountAnalysis uses, so this shares that mechanism's blind spot for
    genuinely data-dependent control flow, but that is a legitimate
    ``fail`` outcome to record, not something to work around here.
    """
    was_training = model.training
    model.eval()
    try:
        dummy = torch.zeros(1, *input_shape)
        with torch.no_grad(), _time_limit(timeout_s):
            traced = torch.jit.trace(model, dummy)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        traced.save(out_path)
        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        return _result("torchscript", "ok", output_path=out_path, size_mb=size_mb)
    except Exception as exc:
        return _result("torchscript", "fail", error_class=type(exc).__name__, error_message=str(exc))
    finally:
        model.train(was_training)


def try_export_onnx(
    model: nn.Module, input_shape: Tuple[int, int, int], out_path: str, opset: int = 13,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> Dict[str, object]:
    """``torch.onnx.export`` (requires the optional ``onnx`` package to
    fully serialise; its absence is itself a legitimate ``fail:
    ModuleNotFoundError`` outcome, not special-cased away)."""
    was_training = model.training
    model.eval()
    try:
        dummy = torch.zeros(1, *input_shape)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with torch.no_grad(), _time_limit(timeout_s):
            torch.onnx.export(
                model, dummy, out_path, opset_version=opset,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        return _result("onnx", "ok", output_path=out_path, size_mb=size_mb)
    except Exception as exc:
        return _result("onnx", "fail", error_class=type(exc).__name__, error_message=str(exc))
    finally:
        model.train(was_training)


def try_export_tensorrt(onnx_path: str, out_path: str, timeout_s: int = _DEFAULT_TIMEOUT_S) -> Dict[str, object]:
    """Attempted only from an already-exported ONNX file (TensorRT's own
    standard ingestion path) via the optional ``tensorrt`` package. A
    missing ONNX input (upstream export failed) or a missing ``tensorrt``
    package both resolve to ``fail`` with the corresponding error class —
    per spec, "attempted for every model" means attempted, not guaranteed
    available in every environment (a CPU-only dev box has no GPU driver
    for TensorRT to target at all).
    """
    if not os.path.exists(onnx_path):
        return _result("tensorrt", "fail", error_class="MissingONNXInput", error_message=f"no ONNX file at {onnx_path} (ONNX export must succeed first)")
    try:
        import tensorrt as trt  # noqa: F401
    except Exception as exc:
        return _result("tensorrt", "fail", error_class=type(exc).__name__, error_message=str(exc))

    try:
        with _time_limit(timeout_s):
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                    return _result("tensorrt", "fail", error_class="OnnxParseError", error_message=errors)
            config = builder.create_builder_config()
            engine = builder.build_serialized_network(network, config)
        if engine is None:
            return _result("tensorrt", "fail", error_class="EngineBuildFailed", error_message="build_serialized_network returned None")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(engine)
        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        return _result("tensorrt", "ok", output_path=out_path, size_mb=size_mb)
    except Exception as exc:
        return _result("tensorrt", "fail", error_class=type(exc).__name__, error_message=str(exc))


def export_all(
    model: nn.Module, input_shape: Tuple[int, int, int], out_dir: str, timeout_s: int = _DEFAULT_TIMEOUT_S
) -> Dict[str, Dict[str, object]]:
    """Attempts all three formats for *model*, in the natural dependency
    order (TensorRT consumes the ONNX export's output). Returns
    ``{"torchscript": ..., "onnx": ..., "tensorrt": ...}``, each value one
    of the ``_result(...)`` dicts above — this is the export-outcome-matrix
    row for one model.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts_result = try_export_torchscript(model, input_shape, os.path.join(out_dir, "model.torchscript.pt"), timeout_s=timeout_s)
    onnx_path = os.path.join(out_dir, "model.onnx")
    onnx_result = try_export_onnx(model, input_shape, onnx_path, timeout_s=timeout_s)
    trt_result = try_export_tensorrt(onnx_path, os.path.join(out_dir, "model.trt"), timeout_s=timeout_s)
    return {"torchscript": ts_result, "onnx": onnx_result, "tensorrt": trt_result}
