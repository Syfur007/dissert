"""
analysis/erf.py — spec §12's mechanism-analysis module, ERF row: gradient-
based Effective Receptive Field (Luo et al., 2016) and its summary radius.

The gradient of a target layer's centre-unit (summed over channels)
activation w.r.t. the input image is, by the chain rule, exactly the set
of per-pixel weights that unit's value is linearly sensitive to — the ERF.
Real receptive fields grow with depth but the *effective* one (where
gradient magnitude concentrates) is almost always far smaller than the
theoretical maximum, which is the whole reason this measurement exists
rather than just reading kernel/stride arithmetic off the architecture.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_erf(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    target_layer: nn.Module,
    device: torch.device,
    n_samples: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Gradient-based ERF map: |d(target_layer's centre-spatial-unit,
    summed over channels) / d(input image)|, averaged over *n_samples*
    random inputs (a single random input already gives a stable ERF for
    an untrained or trained network alike — Luo et al.'s own experiments
    average over a handful; more samples only reduces sampling noise from
    that one draw).

    Returns an ``(H, W)`` numpy array, same spatial size as *input_shape*.
    """
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    accum: Optional[torch.Tensor] = None

    for i in range(n_samples):
        x = torch.randn(1, *input_shape, generator=gen).to(device)
        x.requires_grad_(True)

        activations = {}

        def _hook(module, inp, out):
            activations["value"] = out

        handle = target_layer.register_forward_hook(_hook)
        model(x)
        handle.remove()

        act = activations["value"]  # (1, C, h, w)
        if act.dim() != 4:
            raise ValueError(f"compute_erf: target_layer output must be (1, C, h, w), got shape {tuple(act.shape)}")
        _, _, h, w = act.shape
        centre_unit = act[:, :, h // 2, w // 2].sum()

        grad = torch.autograd.grad(centre_unit, x, retain_graph=False)[0]
        erf_map = grad.abs().sum(dim=1).squeeze(0)  # (H, W)
        accum = erf_map if accum is None else accum + erf_map

    return (accum / n_samples).detach().cpu().numpy()


def erf_radius(erf_map: np.ndarray) -> float:
    """Weighted RMS radius of *erf_map* around its own geometric centre —
    ``sqrt(sum(w * r^2) / sum(w))``, ``r`` = distance from centre, ``w`` =
    ``|erf_map|``. A uniform-weight square of side *k* has a known closed
    form (``k / sqrt(6)``) used to verify this function directly, rather
    than trusting the formula on faith.

    Returns 0.0 for an all-zero map (no gradient signal at all — a
    degenerate but well-defined case, not an error).
    """
    h, w = erf_map.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
    weights = np.abs(erf_map)
    total = weights.sum()
    if total == 0:
        return 0.0
    return float(np.sqrt((weights * dist2).sum() / total))
