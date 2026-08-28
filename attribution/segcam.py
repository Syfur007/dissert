"""
attribution/segcam.py — spec §11's segcam.py: "Seg-Grad-CAM / Seg-XRes-CAM
for qualitative panels only. Vanilla Grad-CAM is not implemented."

Per §11's own banner: "Saliency heatmaps are supported but are qualitative
illustration only and may never be cited as evidence for a claim." Nothing
here computes a dataset-level aggregate or a p-value — every function
returns one image's heatmap, for a figure panel, not a table row.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _grad_cam_core(
    model: nn.Module,
    target_layer: nn.Module,
    image: torch.Tensor,
    target_mask: Optional[torch.Tensor],
    is_multiclass: bool,
    pixelwise_weighting: bool,
) -> torch.Tensor:
    """Shared gradient/activation capture for seg_grad_cam (GAP'd channel
    weights) and seg_xres_cam (pixel-wise weights, no GAP) — the two
    differ only in how the captured gradient is combined with the
    captured activation, everything else (hook setup, target-score
    definition, upsample+normalise) is identical.
    """
    model.eval()
    activations = {}
    gradients = {}

    def _fwd_hook(module, inp, out):
        activations["value"] = out

    def _bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(_fwd_hook)
    h2 = target_layer.register_full_backward_hook(_bwd_hook)
    try:
        image = image.clone().requires_grad_(True)
        logits = model(image)
        if target_mask is None:
            probs = torch.softmax(logits, dim=1) if is_multiclass else torch.sigmoid(logits)
            target_mask = (probs > 0.5).float()
        target_score = (logits * target_mask).sum()
        model.zero_grad(set_to_none=True)
        target_score.backward()

        acts = activations["value"]
        grads = gradients["value"]
        if pixelwise_weighting:
            cam = F.relu((grads * acts).sum(dim=1, keepdim=True))
        else:
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))

        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.detach()
    finally:
        h1.remove()
        h2.remove()


def seg_grad_cam(
    model: nn.Module, target_layer: nn.Module, image: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None, is_multiclass: bool = False,
) -> torch.Tensor:
    """Seg-Grad-CAM (Vinogradova et al., 2020): gradients of the summed
    target-region output logit w.r.t. *target_layer*'s activations,
    global-average-pooled to one weight per channel, ReLU'd weighted sum,
    upsampled to input resolution and normalised to [0, 1].

    Args:
        target_layer: the conv layer whose activations are explained —
            typically the last feature-producing conv before the
            classifier head (e.g. a MambaUNet's ``decoder.final_conv``,
            not ``decoder.out_conv`` itself).
        image: ``(1, C, H, W)`` — one image (CAM is defined per-image).
        target_mask: which output pixels to explain; defaults to the
            model's own thresholded/argmaxed prediction (explain "why did
            the model predict foreground here"), not the ground truth.

    Returns a ``(H, W)`` tensor on *image*'s device, values in [0, 1].
    """
    return _grad_cam_core(model, target_layer, image, target_mask, is_multiclass, pixelwise_weighting=False)


def seg_xres_cam(
    model: nn.Module, target_layer: nn.Module, image: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None, is_multiclass: bool = False,
) -> torch.Tensor:
    """Seg-XRes-CAM: same target/gradient setup as seg_grad_cam, but skips
    global-average-pooling the gradient into one scalar per channel before
    the ReLU'd channel sum — weights each spatial position by its own
    local gradient (HiResCAM/XGrad-CAM's fix for Grad-CAM's implicit
    "a channel's importance is spatially uniform" assumption), which
    matters more for a dense per-pixel task than for classification.
    """
    return _grad_cam_core(model, target_layer, image, target_mask, is_multiclass, pixelwise_weighting=True)
