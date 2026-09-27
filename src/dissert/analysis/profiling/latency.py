"""
profiling/latency.py — spec §14's latency/throughput rows:

  GPU latency  | Batch 1 and 16, >=50 warm-up iterations, explicit
               | cuda.synchronize, median of >=200 timed runs, stated precision.
  Throughput   | Derived from latency; batch size stated in the column header.
  CPU latency  | Batch 1, threads pinned and stated. Mandatory for any
               | lightweight claim.

Supersedes utils/metrics.py's old measure_throughput (5-iteration warmup,
a single un-stated batch size, no percentile reporting) and
utils/report.py's get_latency_stats (10-iteration warmup, 50 runs,
batch=1 only) — neither meets the >=50-warmup/>=200-run/stated-batch-size
protocol above.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def measure_latency(
    model: nn.Module,
    input_shape: "tuple[int, int, int]",
    device: torch.device,
    batch_size: int = 1,
    num_warmup: int = 50,
    num_runs: int = 200,
    precision: str = "fp32",
    num_threads: Optional[int] = None,
) -> Dict[str, object]:
    """One (model, batch_size, device) latency measurement, spec-compliant:
    >=50 warm-up iterations, explicit ``cuda.synchronize`` bracketing each
    timed call on a CUDA device, median of >=200 timed runs.

    Args:
        precision: label only ("fp32"/"fp16"/...) — this function does not
            itself cast the model; pass a model already in the precision
            being measured and label it to match, so the report is
            self-describing rather than silently ambiguous.
        num_threads: CPU device only — pins ``torch.set_num_threads`` for
            the duration of the measurement (spec: "threads pinned and
            stated"); restored afterwards. Ignored on CUDA.

    Returns per-image latency in ms (dividing by batch_size, so batch-1
    and batch-16 numbers are directly comparable) plus throughput in
    images/sec derived from the median: ``{"batch_size", "precision",
    "num_threads", "mean_ms", "median_ms", "std_ms", "p95_ms",
    "throughput_ips", "num_warmup", "num_runs"}``.
    """
    if num_warmup < 50:
        raise ValueError(f"measure_latency: spec requires >=50 warm-up iterations, got {num_warmup}")
    if num_runs < 200:
        raise ValueError(f"measure_latency: spec requires >=200 timed runs, got {num_runs}")

    was_training = model.training
    model.eval()

    prior_threads = None
    if device.type == "cpu" and num_threads is not None:
        prior_threads = torch.get_num_threads()
        torch.set_num_threads(num_threads)
    stated_threads = num_threads if device.type == "cpu" else None
    if device.type == "cpu" and stated_threads is None:
        stated_threads = torch.get_num_threads()

    dummy = torch.zeros(batch_size, *input_shape, device=device)
    try:
        with torch.no_grad():
            for _ in range(num_warmup):
                model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

            timings_ms = []
            for _ in range(num_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings_ms.append((time.perf_counter() - t0) * 1000.0)
    finally:
        model.train(was_training)
        if prior_threads is not None:
            torch.set_num_threads(prior_threads)

    timings_ms = np.asarray(timings_ms)
    per_image_ms = timings_ms / batch_size
    median_ms = float(np.median(per_image_ms))

    return {
        "batch_size": batch_size,
        "device": device.type,
        "precision": precision,
        "num_threads": stated_threads,
        "mean_ms": float(np.mean(per_image_ms)),
        "median_ms": median_ms,
        "std_ms": float(np.std(per_image_ms)),
        "p95_ms": float(np.percentile(per_image_ms, 95)),
        "throughput_ips": float(1000.0 / median_ms) if median_ms > 0 else 0.0,
        "num_warmup": num_warmup,
        "num_runs": num_runs,
    }


def measure_latency_table(
    model: nn.Module,
    input_shape: "tuple[int, int, int]",
    device: torch.device,
    batch_sizes: "tuple[int, ...]" = (1, 16),
    **kwargs,
) -> Dict[int, Dict[str, object]]:
    """spec's "batch 1 and 16" GPU-latency row (and reused for the "batch
    1" CPU-latency row by passing ``batch_sizes=(1,)``) — one
    ``measure_latency`` call per batch size, keyed by batch size.
    """
    return {bs: measure_latency(model, input_shape, device, batch_size=bs, **kwargs) for bs in batch_sizes}
