"""
attribution/common.py — shared helpers for Phase 11's channel-group
attribution family (occlusion.py, shapley.py, integrated_grads.py): the
group→channel-index mapping every one of them slices by, the "training-set
mean image" baseline spec §11 names for both occlusion ("replace group
with its training-set mean") and integrated_grads ("Baseline = training
mean image"), and the occlusion primitive itself.

Attribution is inference-only (spec §11's guarantee) — nothing here builds
an optimizer or updates any parameter.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from datasets.augment import AugmentationPolicy
from datasets.channels import CHANNEL_GROUP_SIZES


def resolve_group_slices(ds_cfg: dict, modality: str) -> Dict[str, slice]:
    """The exact channel-group boundaries datasets.augment.AugmentationPolicy
    actually built the model's input from, for this dataset config — a
    ``{group_name: slice(start, stop)}`` mapping in channel-index order,
    e.g. ``channel_mode="m4"`` (rgb+xy+rtheta) on a colour dataset gives
    ``{"rgb": slice(0,3), "xy": slice(3,5), "rtheta": slice(5,8)}``.
    """
    groups = AugmentationPolicy(modality, ds_cfg).effective_groups
    slices: Dict[str, slice] = {}
    offset = 0
    for g in groups:
        size = CHANNEL_GROUP_SIZES[g]
        slices[g] = slice(offset, offset + size)
        offset += size
    return slices


@torch.no_grad()
def compute_training_mean_image(train_loader, device: torch.device, max_batches: Optional[int] = None) -> torch.Tensor:
    """Pixel-wise mean image over the training set, in the model's actual
    input channel layout (whatever ``channel_mode`` already built into the
    tensors ``train_loader`` yields — no separate channel construction
    here). Returns a ``(C, H, W)`` tensor on *device*.

    Args:
        max_batches: cap for a quick approximate mean (e.g. in tests) — the
            full-dataset mean (default, None) is what spec's "training-set
            mean" means for a real run.
    """
    total: Optional[torch.Tensor] = None
    n = 0
    for i, (images, _masks, _meta) in enumerate(train_loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device)
        batch_sum = images.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        n += images.shape[0]
    if total is None or n == 0:
        raise ValueError("compute_training_mean_image: train_loader yielded no batches")
    return total / n


def occlude_groups(images: torch.Tensor, mean_image: torch.Tensor, group_slices: Dict[str, slice], groups_to_occlude: List[str]) -> torch.Tensor:
    """Replace *groups_to_occlude*'s channels with ``mean_image``'s
    corresponding channels, broadcast over the batch — every other channel
    passes through unchanged. Does not mutate *images* in place.
    """
    out = images.clone()
    for g in groups_to_occlude:
        sl = group_slices[g]
        out[:, sl, :, :] = mean_image[sl, :, :].unsqueeze(0)
    return out


@torch.no_grad()
def predict_hard(model: nn.Module, images: torch.Tensor, is_multiclass: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same binary/multiclass hard-prediction convention as eval.py's
    evaluate(): sigmoid+threshold for binary, softmax+argmax for
    multiclass. Returns ``(preds, probs)``, both on the same device as
    *images* — preds is ``(B, H, W)`` uint8-valued (still float dtype;
    caller casts if needed).
    """
    outputs = model(images)
    if not is_multiclass:
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float().squeeze(1)
    else:
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).float()
    return preds, probs
