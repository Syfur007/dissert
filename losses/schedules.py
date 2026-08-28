"""
losses/schedules.py — weight ramp α(t) for compound-loss terms whose
contribution should change over training, per spec §7: "Weight ramp α(t)
declared explicitly; default linear from 0" (the boundary loss's own
weight in particular — training on the raw boundary-distance functional
from epoch 0, before the model has learned anything about where the lesion
even roughly is, is known to destabilise training; ramping it in avoids
that).
"""
from __future__ import annotations

from typing import Callable, Dict


def linear_ramp(epoch: int, max_epoch: int, start: float = 0.0, end: float = 1.0) -> float:
    """Linear ramp from *start* at epoch 0 to *end* at epoch >= max_epoch,
    clamped outside that range."""
    if max_epoch <= 0:
        return end
    frac = max(0.0, min(1.0, epoch / max_epoch))
    return start + frac * (end - start)


def constant(epoch: int, max_epoch: int, value: float = 1.0) -> float:
    return value


SCHEDULES: Dict[str, Callable[..., float]] = {
    "linear": linear_ramp,
    "constant": constant,
}


def apply_schedule(spec: dict, epoch: int, max_epoch: int) -> float:
    """spec: {"type": "linear" | "constant", **kwargs for that schedule
    function}. Missing/None spec is treated as constant(1.0) — a term with
    no declared schedule always contributes at its full configured weight."""
    if not spec:
        return 1.0
    schedule_type = spec.get("type", "constant")
    if schedule_type not in SCHEDULES:
        raise ValueError(f"Unknown loss schedule type '{schedule_type}'. Known: {sorted(SCHEDULES)}")
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return SCHEDULES[schedule_type](epoch, max_epoch, **kwargs)
