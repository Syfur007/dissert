"""
models/build.py — capacity control: search a model family's width-scaling
knob for a configuration matching a target parameter/FLOPs budget.

Used for the "single-encoder control" comparisons Phase 6 needs (a plain
baseline width-matched to the Mamba-hybrid model's parameter count, so a
performance difference isn't just "more parameters") — but not specific to
Mamba: any model family that already exposes a width knob (MK-UNet/GMK-UNet's
channels list, one per T/S/base/M/L preset) can be searched today.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from utils.metrics import count_parameters


def _profile(model: nn.Module, input_shape: Tuple[int, int, int]) -> Tuple[int, int]:
    """(params, flops) for *model* at *input_shape* = (C, H, W), no batch dim."""
    params = count_parameters(model)
    flops = 0
    try:
        from thop import profile
        dummy = torch.randn(1, *input_shape)
        flops_raw, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = int(flops_raw)
    except Exception:
        pass
    return params, flops


def build_width_matched(
    base_model_fn: Callable[[List[int]], nn.Module],
    width_candidates: List[List[int]],
    input_shape: Tuple[int, int, int],
    target_params: Optional[int] = None,
    target_flops: Optional[int] = None,
    tol: float = 0.1,
) -> Dict[str, Any]:
    """Search *width_candidates* for the configuration best matching a
    target parameter and/or FLOPs budget.

    Args:
        base_model_fn: channels-list -> nn.Module (e.g.
            ``functools.partial(get_model, name="mk_unet", num_classes=1,
            in_channels=3, ...)`` with every other kwarg already bound,
            called as ``base_model_fn(channels)``).
        width_candidates: candidate channel-width lists to try, in the
            order to try them (e.g. a model family's T/S/base/M/L presets,
            or a finer sweep between them).
        input_shape: (C, H, W) — no batch dim; used for FLOPs profiling.
        target_params / target_flops: at least one is required. When both
            are given, a candidate must satisfy *both* within `tol` to
            count as within-tolerance (matching is on the max of the two
            relative errors, not their average, so a good params match
            can't paper over a bad FLOPs match or vice versa).
        tol: relative-error tolerance. The first candidate (in
            *width_candidates* order) within tolerance is returned
            immediately — this is a search over a short, ordered list of
            meaningful presets, not an exhaustive optimisation, so "first
            good enough" is deliberate, not a shortcut.

    Returns:
        ``{"channels", "params", "flops", "within_tolerance",
        "relative_error"}`` for the best (or first within-tolerance)
        candidate.
    """
    if target_params is None and target_flops is None:
        raise ValueError(
            "build_width_matched: at least one of target_params/target_flops is required."
        )
    if not width_candidates:
        raise ValueError("build_width_matched: width_candidates is empty.")

    results = []
    for channels in width_candidates:
        model = base_model_fn(channels)
        params, flops = _profile(model, input_shape)

        errors = []
        if target_params is not None and target_params > 0:
            errors.append(abs(params - target_params) / target_params)
        if target_flops is not None and target_flops > 0:
            if flops > 0:
                errors.append(abs(flops - target_flops) / target_flops)
            else:
                # FLOPs profiling unavailable (thop failed/missing) but a
                # FLOPs target was requested — can't claim a match either
                # way, so this candidate can never be "within tolerance"
                # on FLOPs; contributes an infinite error rather than
                # silently being scored on params alone.
                errors.append(float("inf"))

        rel_error = max(errors) if errors else float("inf")
        result = {"channels": channels, "params": params, "flops": flops, "relative_error": rel_error}
        results.append(result)
        if rel_error <= tol:
            return {**result, "within_tolerance": True}

    best = min(results, key=lambda r: r["relative_error"])
    return {**best, "within_tolerance": False}
