"""
attribution/integrated_grads.py — spec §11's integrated_grads.py:
"Integrated Gradients w.r.t. input, aggregated per channel (not per
pixel). Baseline = training mean image. Steps declared." Answers RQ2
cross-check ("agreement score vs occlusion").

Segmentation models have no single scalar output for captum's IG to
attribute against directly, so the model is wrapped to explain
``sigmoid(logits).sum()`` (binary) / ``softmax(logits).sum()``
(multiclass) — the total predicted-foreground mass over the whole output,
a standard choice for pixel-dense IG (e.g. Seg-Grad-CAM's own "explain the
sum of the target class map" convention, reused here rather than invented
fresh) and consistent with spec's "aggregated per channel (not per
pixel)": no per-pixel attribution map is read out of this at all, only the
per-channel total.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from .common import compute_training_mean_image, resolve_group_slices


def _foreground_mass_forward(model: nn.Module, is_multiclass: bool):
    def _forward(x: torch.Tensor) -> torch.Tensor:
        logits = model(x)
        probs = torch.softmax(logits, dim=1) if is_multiclass else torch.sigmoid(logits)
        return probs.flatten(1).sum(dim=1)
    return _forward


def run_integrated_gradients(
    model: nn.Module,
    test_loader,
    ds_cfg: dict,
    modality: str,
    device: torch.device,
    is_multiclass: bool = False,
    n_steps: int = 50,
    mean_image: Optional[torch.Tensor] = None,
    train_loader=None,
) -> Dict[str, object]:
    """Per-channel (summed to per-group) absolute attribution mass over
    *test_loader*, normalised to sum to 1 across groups so it's directly
    comparable to occlusion.py's Dice-drop proportions.

    Returns ``{"per_channel_mass": [...], "per_group_mass": {group: mass},
    "n_steps": n_steps, "n_images": int}``.
    """
    if mean_image is None:
        if train_loader is None:
            raise ValueError("run_integrated_gradients: need either mean_image or train_loader")
        mean_image = compute_training_mean_image(train_loader, device)

    model.eval()
    group_slices = resolve_group_slices(ds_cfg, modality)
    n_channels = mean_image.shape[0]

    ig = IntegratedGradients(_foreground_mass_forward(model, is_multiclass))

    per_channel_abs_sum = torch.zeros(n_channels, device=device)
    n_images = 0
    for images, _masks, _meta in test_loader:
        images = images.to(device).requires_grad_(True)
        baselines = mean_image.unsqueeze(0).expand(images.shape[0], -1, -1, -1)
        attributions = ig.attribute(images, baselines=baselines, n_steps=n_steps)
        # sum |attribution| over (H, W) and batch, per channel
        per_channel_abs_sum += attributions.abs().sum(dim=(0, 2, 3)).detach()
        n_images += images.shape[0]

    per_channel_mass = (per_channel_abs_sum / max(per_channel_abs_sum.sum(), torch.finfo(per_channel_abs_sum.dtype).eps)).cpu().numpy()

    per_group_mass: Dict[str, float] = {}
    for group, sl in group_slices.items():
        per_group_mass[group] = float(per_channel_mass[sl].sum())

    return {
        "per_channel_mass": per_channel_mass.tolist(),
        "per_group_mass": per_group_mass,
        "n_steps": n_steps,
        "n_images": n_images,
    }


def agreement_score(occlusion_result: Dict[str, Dict[str, float]], ig_result: Dict[str, object]) -> float:
    """Spearman rank correlation between occlusion's per-group Dice-drop
    ranking and integrated_grads's per-group attribution-mass ranking —
    spec's "agreement score vs occlusion" (§11's "two attribution families
    must agree before a claim" guarantee). In [-1, 1]; 1.0 = identical
    group ranking by both methods.
    """
    from scipy.stats import spearmanr

    groups = sorted(occlusion_result.keys())
    if set(groups) != set(ig_result["per_group_mass"].keys()):
        raise ValueError("agreement_score: occlusion and integrated_grads results cover different groups")
    dice_drops = [occlusion_result[g]["dice_drop"] for g in groups]
    ig_masses = [ig_result["per_group_mass"][g] for g in groups]
    if len(groups) < 2:
        return 1.0
    corr, _ = spearmanr(dice_drops, ig_masses)
    return float(corr) if not np.isnan(corr) else 0.0
