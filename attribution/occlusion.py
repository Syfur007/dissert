"""
attribution/occlusion.py — spec §11's occlusion.py: "Channel-group
occlusion at inference: replace group with its training-set mean,
recompute all metrics. Groups: RGB, YCbCr, XY, Rθ." Answers RQ2.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from metrics.region import dice as _dice

from .common import compute_training_mean_image, occlude_groups, predict_hard, resolve_group_slices


@torch.no_grad()
def run_channel_group_occlusion(
    model: nn.Module,
    test_loader,
    ds_cfg: dict,
    modality: str,
    device: torch.device,
    is_multiclass: bool = False,
    mean_image: Optional[torch.Tensor] = None,
    train_loader=None,
) -> Dict[str, Dict[str, float]]:
    """For each channel group present in the model's input (per
    ``ds_cfg["channel_mode"]``/*modality*), occlude it across the whole
    *test_loader* and report the per-image-averaged Dice drop from the
    (no-occlusion) baseline — spec's "Per-group Dice drop, per dataset,
    per seed" (this call covers one dataset/seed; the caller loops for
    the full table).

    Either pass a precomputed *mean_image* (e.g. shared across an
    occlusion+integrated_grads pair, so the training-set pass only runs
    once) or a *train_loader* to compute it from.

    Returns ``{group: {"baseline_dice", "occluded_dice", "dice_drop", "n"}}``.
    """
    if mean_image is None:
        if train_loader is None:
            raise ValueError("run_channel_group_occlusion: need either mean_image or train_loader")
        mean_image = compute_training_mean_image(train_loader, device)

    model.eval()
    group_slices = resolve_group_slices(ds_cfg, modality)

    baseline_dices: List[float] = []
    occluded_dices: Dict[str, List[float]] = {g: [] for g in group_slices}

    for images, masks, _meta in test_loader:
        images = images.to(device)
        gts = masks.numpy()

        preds, _ = predict_hard(model, images, is_multiclass)
        preds_np = preds.cpu().numpy()
        for p, g in zip(preds_np, gts):
            baseline_dices.append(_dice(p, g.squeeze()))

        for group in group_slices:
            occ_images = occlude_groups(images, mean_image, group_slices, [group])
            occ_preds, _ = predict_hard(model, occ_images, is_multiclass)
            occ_preds_np = occ_preds.cpu().numpy()
            for p, g in zip(occ_preds_np, gts):
                occluded_dices[group].append(_dice(p, g.squeeze()))

    baseline_mean = float(np.mean(baseline_dices))
    result: Dict[str, Dict[str, float]] = {}
    for group, dices in occluded_dices.items():
        occ_mean = float(np.mean(dices))
        result[group] = {
            "baseline_dice": baseline_mean,
            "occluded_dice": occ_mean,
            "dice_drop": baseline_mean - occ_mean,
            "n": len(dices),
        }
    return result
