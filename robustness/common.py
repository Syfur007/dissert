"""
robustness/common.py — shared inference-time corruption driver: apply a
robustness/corruptions.py function to a batch's content channels, run the
model, recompute per-image Dice, average over a test loader — the
"Metric vs severity" evaluation every degradation curve in spec §13 is
built from.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from attribution.common import predict_hard, resolve_group_slices
from metrics.region import dice as _dice

# Content-bearing channel groups a photometric/acquisition corruption
# applies to — geometry groups (xy, rtheta) are a pure function of output
# grid shape (see datasets/channels.py), not of pixel content, and every
# corruption here preserves (H, W), so they are deliberately left
# untouched (matches datasets/augment.py's own "transform content, then
# rebuild geometry" ordering rather than spatially distorting the
# coordinate channels' encoded values).
_CONTENT_GROUPS = ("rgb", "ycbcr")


def apply_corruption_to_batch(
    images: torch.Tensor,
    corruption_fn: Callable[[np.ndarray], np.ndarray],
    group_slices: Optional[Dict[str, "slice"]] = None,
) -> torch.Tensor:
    """images: (B, C, H, W) float in [0, 1]. *corruption_fn* operates on
    one (H, W, 3) uint8 RGB array (robustness.corruptions' convention) —
    applied to the ``rgb`` channel group only (the visual content a real
    sensor/compression artefact would actually touch); every other
    channel group passes through unchanged. Without *group_slices* (a
    plain 3-channel RGB-only model), the whole tensor is treated as the
    rgb group.
    """
    out = images.clone()
    rgb_slice = group_slices["rgb"] if group_slices else slice(0, images.shape[1])
    rgb = images[:, rgb_slice]
    corrupted = []
    for i in range(rgb.shape[0]):
        img_np = (rgb[i].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        corrupted_np = corruption_fn(img_np)
        corrupted_t = torch.from_numpy(corrupted_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        corrupted.append(corrupted_t)
    out[:, rgb_slice] = torch.stack(corrupted, dim=0).to(images.device)
    return out


@torch.no_grad()
def evaluate_under_corruption(
    model: nn.Module,
    test_loader,
    corruption_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    device: torch.device,
    is_multiclass: bool = False,
    group_slices: Optional[Dict[str, "slice"]] = None,
) -> Dict[str, float]:
    """Mean per-image Dice over *test_loader* with *corruption_fn* applied
    to every batch's rgb channels (``corruption_fn=None`` -> uncorrupted
    baseline, so a caller can get the clean and corrupted numbers from the
    same call shape for a degradation curve's 0-severity point).
    """
    model.eval()
    dices: List[float] = []
    for images, masks, _meta in test_loader:
        images = images.to(device)
        batch = apply_corruption_to_batch(images, corruption_fn, group_slices) if corruption_fn is not None else images
        preds, _ = predict_hard(model, batch, is_multiclass)
        preds_np = preds.cpu().numpy()
        for p, g in zip(preds_np, masks.numpy()):
            dices.append(_dice(p, g.squeeze()))
    return {"mean_dice": float(np.mean(dices)), "n": len(dices)}


def degradation_curve(
    model: nn.Module,
    test_loader,
    corruption_name: str,
    device: torch.device,
    is_multiclass: bool = False,
    group_slices: Optional[Dict[str, "slice"]] = None,
) -> List[Dict[str, float]]:
    """Metric-vs-severity degradation curve for one named corruption
    (robustness.corruptions.CORRUPTIONS key), severities 1-5 plus the
    clean (severity 0) baseline — spec's "Degradation curve: metric vs
    severity, per model" row.
    """
    from .corruptions import CORRUPTIONS, SEVERITY_LEVELS

    if corruption_name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption '{corruption_name}'. Known: {sorted(CORRUPTIONS)}")
    fn = CORRUPTIONS[corruption_name]

    curve = [{"severity": 0, **evaluate_under_corruption(model, test_loader, None, device, is_multiclass, group_slices)}]
    for severity in SEVERITY_LEVELS:
        result = evaluate_under_corruption(
            model, test_loader, lambda img, s=severity: fn(img, s), device, is_multiclass, group_slices
        )
        curve.append({"severity": severity, **result})
    return curve


def mean_corruption_error(curve: List[Dict[str, float]]) -> float:
    """spec's "Summary | Mean corruption error relative to clean, per
    model" row: mean fractional Dice drop across severities 1-5 relative
    to the severity-0 (clean) baseline in the same curve.
    """
    by_severity = {row["severity"]: row["mean_dice"] for row in curve}
    if 0 not in by_severity:
        raise ValueError("mean_corruption_error: curve has no severity=0 (clean) baseline row")
    clean = by_severity[0]
    corrupted_severities = [s for s in by_severity if s != 0]
    if not corrupted_severities:
        raise ValueError("mean_corruption_error: curve has no corrupted (severity>0) rows")
    if clean == 0:
        return float("nan")
    drops = [(clean - by_severity[s]) / clean for s in corrupted_severities]
    return float(np.mean(drops))
