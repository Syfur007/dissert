"""
robustness/geometric.py — spec §13's GEOMETRIC corruption family
(translation, off-centre crop, scale, rotation — "the key test for
geometry channels") plus spec §11's shortcut audit (coordonly Dice,
translation-shift degradation, frame jitter), reusing Phase 4's
``datasets.channels.coordonly_channels()``/channel-group machinery
directly rather than a parallel implementation.

Every transform here is applied identically to the content channels
(rgb/ycbcr) *and* the mask via one shared affine grid, via
``torch.nn.functional.grid_sample`` (bilinear for content, nearest for
the mask, so the mask stays strictly binary — never blurred into
fractional values by the resample). Geometry channels (xy, rtheta) are
left untouched: they are a pure function of the output grid's shape (see
datasets/channels.py), not of pixel content, and every transform here
preserves (H, W) — matching datasets/augment.py's own "transform content,
then rebuild geometry" ordering rather than spatially distorting the
coordinate channels' encoded values.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attribution.common import predict_hard, resolve_group_slices
from metrics.region import dice as _dice
from .corruptions import SEVERITY_LEVELS


def _affine_transform(
    content: torch.Tensor, mask: torch.Tensor, theta: torch.Tensor
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Applies the same (B, 2, 3) affine grid to *content* (bilinear) and
    *mask* (nearest, then re-thresholded at 0.5 so it stays exactly
    binary) — shared primitive every named transform below builds *theta*
    for.
    """
    grid = F.affine_grid(theta, content.shape, align_corners=False)
    content_out = F.grid_sample(content, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask_out = F.grid_sample(mask, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return content_out, (mask_out > 0.5).float()


def _identity_theta(batch_size: int, device: torch.device) -> torch.Tensor:
    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    return theta


def translate(content: torch.Tensor, mask: torch.Tensor, dx_frac: float, dy_frac: float):
    """Shifts content+mask by (*dx_frac*, *dy_frac*) fraction of half-
    width/half-height (normalised affine-grid units — e.g. dx_frac=0.1
    shifts by 10% of the image's half-width), zero-filled border."""
    b = content.shape[0]
    theta = _identity_theta(b, content.device)
    theta[:, 0, 2] = dx_frac
    theta[:, 1, 2] = dy_frac
    return _affine_transform(content, mask, theta)


def rotate(content: torch.Tensor, mask: torch.Tensor, degrees: float):
    """Rotates content+mask about the centre by *degrees* (counter-
    clockwise for a positive angle), zero-filled border."""
    b = content.shape[0]
    rad = math.radians(degrees)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    theta = torch.zeros(b, 2, 3, device=content.device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    return _affine_transform(content, mask, theta)


def scale(content: torch.Tensor, mask: torch.Tensor, factor: float):
    """Zooms content+mask by *factor* about the centre (factor > 1 zooms
    in / crops-and-enlarges; factor < 1 zooms out, shrinking the content
    with a zero-filled border around it)."""
    if factor <= 0:
        raise ValueError(f"scale factor must be positive, got {factor}")
    b = content.shape[0]
    theta = torch.zeros(b, 2, 3, device=content.device)
    theta[:, 0, 0] = 1.0 / factor
    theta[:, 1, 1] = 1.0 / factor
    return _affine_transform(content, mask, theta)


def off_centre_crop(content: torch.Tensor, mask: torch.Tensor, crop_frac: float, offset_x_frac: float, offset_y_frac: float):
    """Crops a *crop_frac*-sized window (fraction of each dimension, e.g.
    0.7 crops to 70% width/height) offset from centre by
    (*offset_x_frac*, *offset_y_frac*) (fraction of half-width/
    half-height), then resamples that window back up to the original
    resolution — composes ``scale`` (zoom to the crop size) with
    ``translate`` (recentre on the offset window) via one affine matrix.
    """
    if not 0 < crop_frac <= 1.0:
        raise ValueError(f"crop_frac must be in (0, 1], got {crop_frac}")
    b = content.shape[0]
    theta = torch.zeros(b, 2, 3, device=content.device)
    theta[:, 0, 0] = crop_frac
    theta[:, 1, 1] = crop_frac
    theta[:, 0, 2] = offset_x_frac
    theta[:, 1, 2] = offset_y_frac
    return _affine_transform(content, mask, theta)


# ---------------------------------------------------------------------------
# Severity-parameterised degradation-curve transforms (spec §13)
# ---------------------------------------------------------------------------

_TRANSLATE_SEVERITY = {1: 0.02, 2: 0.05, 3: 0.08, 4: 0.12, 5: 0.18}
_ROTATE_SEVERITY = {1: 3, 2: 7, 3: 12, 4: 18, 5: 25}
_SCALE_SEVERITY = {1: 0.95, 2: 0.88, 3: 0.80, 4: 0.70, 5: 0.60}
_CROP_SEVERITY = {1: 0.95, 2: 0.88, 3: 0.80, 4: 0.70, 5: 0.60}

GEOMETRIC_TRANSFORMS = ("translate", "rotate", "scale", "off_centre_crop")


def _apply_named_geometric(name: str, content: torch.Tensor, mask: torch.Tensor, severity: int):
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"severity must be one of {SEVERITY_LEVELS}, got {severity}")
    if name == "translate":
        d = _TRANSLATE_SEVERITY[severity]
        return translate(content, mask, d, d)
    if name == "rotate":
        return rotate(content, mask, _ROTATE_SEVERITY[severity])
    if name == "scale":
        return scale(content, mask, _SCALE_SEVERITY[severity])
    if name == "off_centre_crop":
        c = _CROP_SEVERITY[severity]
        return off_centre_crop(content, mask, c, offset_x_frac=(1 - c), offset_y_frac=0.0)
    raise ValueError(f"Unknown geometric transform '{name}'. Known: {GEOMETRIC_TRANSFORMS}")


@torch.no_grad()
def geometric_degradation_curve(
    model: nn.Module,
    test_loader,
    transform_name: str,
    ds_cfg: dict,
    modality: str,
    device: torch.device,
    is_multiclass: bool = False,
) -> List[Dict[str, float]]:
    """Metric-vs-severity curve for one named geometric transform,
    applying it to the content channel groups (+ mask) and leaving
    geometry channel groups untouched — spec's "Translation... Degradation
    curve; the key test for geometry channels" row.
    """
    model.eval()
    group_slices = resolve_group_slices(ds_cfg, modality)
    content_slices = [group_slices[g] for g in ("rgb", "ycbcr") if g in group_slices]

    def _mean_dice(transform_fn) -> Dict[str, float]:
        dices: List[float] = []
        for images, masks, _meta in test_loader:
            images = images.to(device)
            masks_dev = masks.to(device)
            out = images.clone()
            for sl in content_slices:
                content_out, mask_out = transform_fn(images[:, sl], masks_dev)
                out[:, sl] = content_out
            preds, _ = predict_hard(model, out, is_multiclass)
            preds_np = preds.cpu().numpy()
            gts_np = mask_out.cpu().numpy() if content_slices else masks.numpy()
            for p, g in zip(preds_np, gts_np):
                dices.append(_dice(p, g.squeeze()))
        return {"mean_dice": float(np.mean(dices)), "n": len(dices)}

    curve = [{"severity": 0, **_mean_dice(lambda c, m: (c, m))}]
    for severity in SEVERITY_LEVELS:
        curve.append({
            "severity": severity,
            **_mean_dice(lambda c, m, s=severity: _apply_named_geometric(transform_name, c, m, s)),
        })
    return curve


# ---------------------------------------------------------------------------
# Shortcut audit (spec §11's guarantee row)
# ---------------------------------------------------------------------------

@torch.no_grad()
def shortcut_audit(
    coordonly_model: nn.Module, test_loader, device: torch.device, threshold: float, is_multiclass: bool = False
) -> Dict[str, object]:
    """Evaluates an already-trained coord-only model (trained on
    ``datasets.channels.coordonly_channels()``'s 5-channel xy+rtheta
    input, no RGB at all) and compares its Dice against the pre-registered
    *threshold* — spec's "coordonly Dice above the pre-registered
    threshold flips a project-level flag" row: a coord-only model that
    performs suspiciously well means geometry channels alone (no visual
    appearance information) are enough to "solve" the task, which would
    undermine any claim that a full model's geometry channels are
    contributing genuine appearance-independent structure rather than
    functioning as a positional shortcut.

    Returns ``{"coordonly_dice", "threshold", "shortcut_flag"}``.
    """
    coordonly_model.eval()
    dices: List[float] = []
    for images, masks, _meta in test_loader:
        images = images.to(device)
        preds, _ = predict_hard(coordonly_model, images, is_multiclass)
        preds_np = preds.cpu().numpy()
        for p, g in zip(preds_np, masks.numpy()):
            dices.append(_dice(p, g.squeeze()))
    coordonly_dice = float(np.mean(dices))
    return {"coordonly_dice": coordonly_dice, "threshold": threshold, "shortcut_flag": coordonly_dice > threshold}


@torch.no_grad()
def frame_jitter_sensitivity(
    model: nn.Module,
    test_loader,
    ds_cfg: dict,
    modality: str,
    device: torch.device,
    jitter_frac: float = 0.02,
    n_trials: int = 5,
    seed: int = 0,
    is_multiclass: bool = False,
) -> Dict[str, object]:
    """Repeatedly applies a *small*, randomly-signed translation
    (magnitude *jitter_frac*, the geometry channels left untouched per
    this module's docstring) and reports the mean/std Dice across trials
    — spec's "frame jitter" shortcut-audit control: a model that has
    overfit to absolute frame position (e.g. via the xy channel) should
    show *more* jitter sensitivity (larger spread/drop) than one relying
    on genuine object appearance, which a small jitter barely perturbs.
    """
    rng = np.random.default_rng(seed)
    group_slices = resolve_group_slices(ds_cfg, modality)
    content_slices = [group_slices[g] for g in ("rgb", "ycbcr") if g in group_slices]
    model.eval()

    trial_dices: List[float] = []
    for _ in range(n_trials):
        dx = float(rng.uniform(-jitter_frac, jitter_frac))
        dy = float(rng.uniform(-jitter_frac, jitter_frac))
        dices: List[float] = []
        for images, masks, _meta in test_loader:
            images = images.to(device)
            masks_dev = masks.to(device)
            out = images.clone()
            mask_out = masks_dev
            for sl in content_slices:
                content_out, mask_out = translate(images[:, sl], masks_dev, dx, dy)
                out[:, sl] = content_out
            preds, _ = predict_hard(model, out, is_multiclass)
            preds_np = preds.cpu().numpy()
            gts_np = mask_out.cpu().numpy() if content_slices else masks.numpy()
            for p, g in zip(preds_np, gts_np):
                dices.append(_dice(p, g.squeeze()))
        trial_dices.append(float(np.mean(dices)))

    return {
        "mean_dice": float(np.mean(trial_dices)),
        "std_dice": float(np.std(trial_dices)),
        "n_trials": n_trials,
        "jitter_frac": jitter_frac,
        "per_trial_dice": trial_dices,
    }
