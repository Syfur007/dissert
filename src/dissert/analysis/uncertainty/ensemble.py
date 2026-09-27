"""
uncertainty/ensemble.py — spec §12's uncertainty module, ensemble and maps
rows: "Deep ensemble over the existing seeds. Zero extra training cost."
Reuses eval.py's ``EnsembleModel`` (a handful of already-trained
fold/seed checkpoints averaged at inference) rather than introducing a
second, parallel ensembling mechanism — this module adds what
``EnsembleModel.forward()`` alone can't give: the *individual* per-member
probability maps needed for inter-seed variance, not just their mean.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


@torch.no_grad()
def predict_ensemble_members(models: List[nn.Module], images: torch.Tensor, is_multiclass: bool) -> torch.Tensor:
    """Per-member probability maps, stacked on a new leading dimension:
    ``(K, B, C, H, W)`` for multiclass (softmax over the class dim) or
    ``(K, B, 1, H, W)`` for binary (sigmoid) — K = ensemble size (the
    existing seeds/folds), every downstream uncertainty measure below is
    computed from this.
    """
    probs = []
    for m in models:
        m.eval()
        logits = m(images)
        p = torch.softmax(logits, dim=1) if is_multiclass else torch.sigmoid(logits)
        probs.append(p)
    return torch.stack(probs, dim=0)


def predictive_entropy(mean_probs: torch.Tensor, is_multiclass: bool, eps: float = 1e-8) -> torch.Tensor:
    """Per-pixel Shannon entropy of the ensemble-*mean* probability — spec's
    "Per-pixel predictive entropy" row (total uncertainty: aleatoric +
    epistemic combined, as opposed to inter_seed_variance's epistemic-only
    proxy below).

    Args:
        mean_probs: ``(B, C, H, W)`` (multiclass) or ``(B, 1, H, W)``
            (binary) — typically ``predict_ensemble_members(...).mean(0)``,
            but any mean-probability tensor of the right shape works
            (e.g. a single model's own softmax/sigmoid output, for a
            non-ensemble entropy map).

    Returns ``(B, H, W)``.
    """
    if is_multiclass:
        p = mean_probs.clamp_min(eps)
        return -(p * p.log()).sum(dim=1)
    p = mean_probs.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log()).squeeze(1)


def inter_seed_variance(member_probs: torch.Tensor, is_multiclass: bool) -> torch.Tensor:
    """Per-pixel variance across ensemble members — spec's "inter-seed
    variance" row, an epistemic-uncertainty proxy (how much do the seeds
    *disagree*, as distinct from predictive_entropy's "how uncertain is
    the averaged prediction").

    Args:
        member_probs: ``(K, B, C, H, W)`` — predict_ensemble_members()'s
            output.

    Returns ``(B, H, W)`` — multiclass variance is summed over the class
    dimension into one scalar per pixel (population variance,
    ``unbiased=False``, matching K being the *entire* ensemble, not a
    sample of a larger population).
    """
    var = member_probs.var(dim=0, unbiased=False)  # (B, C, H, W)
    if is_multiclass:
        return var.sum(dim=1)
    return var.squeeze(1)
