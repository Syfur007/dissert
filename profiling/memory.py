"""
profiling/memory.py — spec §14's "Peak memory | torch.cuda.max_memory_allocated
after reset, at both batch sizes." row.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def measure_peak_memory(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    device: torch.device,
    batch_size: int = 1,
) -> Dict[str, Optional[float]]:
    """Peak CUDA memory for one forward pass at *batch_size*, per spec:
    ``reset_peak_memory_stats`` immediately before the pass, read
    ``max_memory_allocated`` immediately after — so the number reflects
    this pass alone, not whatever accumulated from prior measurements or
    model construction.

    Returns ``{"batch_size", "peak_allocated_mb", "peak_reserved_mb"}``,
    both None on a CPU device (peak-allocation tracking is CUDA-only; a
    CPU run reports ``None`` rather than a misleading 0.0, so a caller
    can't mistake "not measured" for "measured zero").
    """
    if device.type != "cuda":
        return {"batch_size": batch_size, "peak_allocated_mb": None, "peak_reserved_mb": None}

    was_training = model.training
    model.eval()
    try:
        torch.cuda.reset_peak_memory_stats(device)
        dummy = torch.zeros(batch_size, *input_shape, device=device)
        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    finally:
        model.train(was_training)

    return {
        "batch_size": batch_size,
        "peak_allocated_mb": float(peak_allocated),
        "peak_reserved_mb": float(peak_reserved),
    }


def measure_peak_memory_table(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    device: torch.device,
    batch_sizes: Tuple[int, ...] = (1, 16),
) -> Dict[int, Dict[str, Optional[float]]]:
    """spec's "at both batch sizes" — one measure_peak_memory call per
    batch size, keyed by batch size."""
    return {bs: measure_peak_memory(model, input_shape, device, batch_size=bs) for bs in batch_sizes}


def checkpoint_size_mb(checkpoint_path: str) -> float:
    """spec's "Checkpoint size | Weights-only FP32 file size." row — the
    file this repo's utils.checkpoint.CheckpointManager already writes
    (weights + a small amount of run metadata, all FP32; no separate
    weights-only export exists, so the checkpoint file itself is the
    measurement) sized directly, no loading required.
    """
    import os

    return os.path.getsize(checkpoint_path) / (1024 ** 2)
