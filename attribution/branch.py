"""
attribution/branch.py — spec §11's branch.py: "Per-stage branch ablation:
zero the auxiliary contribution at stage k at inference, recompute
metrics." Answers RQ1, RQ3. Needs Phase 6's Mamba model
(models.proposed.mamba_unet.MambaUNet) specifically — the only registered
model family with a distinct auxiliary branch to ablate.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from metrics.region import dice as _dice

from .common import predict_hard


@contextmanager
def ablate_auxiliary_stage(model: nn.Module, stage_idx: int):
    """Zeroes ``model.auxiliary_encoder``'s stage *stage_idx* output for
    the duration of the context, via a forward hook that replaces the
    auxiliary encoder's returned list element with zeros — every
    downstream fusion/decoder consumer sees "no auxiliary contribution at
    this stage" with no change to MambaUNet.forward()'s own code. A
    ``register_forward_hook`` callback that returns a non-None value
    replaces the wrapped module's actual output with it (documented
    PyTorch hook behaviour), which is what makes this possible without
    monkey-patching forward itself.
    """
    def _hook(module, inputs, output):
        output = list(output)
        output[stage_idx] = torch.zeros_like(output[stage_idx])
        return output

    handle = model.auxiliary_encoder.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def run_branch_ablation(
    model: nn.Module, test_loader, device: torch.device, is_multiclass: bool = False, n_stages: int = 4
) -> Dict[int, Dict[str, float]]:
    """For each of the auxiliary branch's *n_stages* (4, matching
    MambaUNet's fixed 4-stage design), zero its contribution and recompute
    Dice over *test_loader*, reporting the drop from the (no-ablation)
    baseline — spec's "Dice drop per stage per dataset".

    Requires *model* to expose an ``auxiliary_encoder`` submodule — raises
    ``AttributeError`` early for a model that has none, rather than
    silently reporting all-zero drops for a family this analysis doesn't
    apply to.
    """
    if not hasattr(model, "auxiliary_encoder"):
        raise AttributeError(
            f"run_branch_ablation: {type(model).__name__} has no auxiliary_encoder "
            "submodule to ablate — this analysis only applies to MambaUNet-family models."
        )
    model.eval()

    def _mean_dice(loader) -> float:
        dices: List[float] = []
        for images, masks, _meta in loader:
            images = images.to(device)
            preds, _ = predict_hard(model, images, is_multiclass)
            preds_np = preds.cpu().numpy()
            for p, g in zip(preds_np, masks.numpy()):
                dices.append(_dice(p, g.squeeze()))
        return float(np.mean(dices)), len(dices)

    baseline_mean, baseline_n = _mean_dice(test_loader)

    result: Dict[int, Dict[str, float]] = {}
    for stage_idx in range(n_stages):
        with ablate_auxiliary_stage(model, stage_idx):
            stage_mean, stage_n = _mean_dice(test_loader)
        result[stage_idx] = {
            "baseline_dice": baseline_mean,
            "ablated_dice": stage_mean,
            "dice_drop": baseline_mean - stage_mean,
            "n": stage_n,
        }
    return result
