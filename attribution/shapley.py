"""
attribution/shapley.py — spec §11's shapley.py: "Exact Shapley over the
four channel groups (16 coalitions), per image, averaged." Answers RQ2.

The characteristic function v(S) for one image: Dice with every group NOT
in S occluded to the training-set mean (same occlusion primitive
occlusion.py uses) — S is "which groups get to see their real pixels."
v(all groups) is the ordinary un-occluded prediction; v(∅) is fully
occluded (every group replaced by the training mean).
"""
from __future__ import annotations

import itertools
from math import factorial
from typing import Dict, FrozenSet, List, Optional

import numpy as np
import torch
import torch.nn as nn

from metrics.region import dice as _dice

from .common import compute_training_mean_image, occlude_groups, predict_hard, resolve_group_slices


def _all_coalitions(groups: List[str]) -> List[FrozenSet[str]]:
    coalitions = []
    for r in range(len(groups) + 1):
        for combo in itertools.combinations(groups, r):
            coalitions.append(frozenset(combo))
    return coalitions


def shapley_values_from_characteristic_function(v: Dict[FrozenSet[str], float], groups: List[str]) -> Dict[str, float]:
    """Exact Shapley value per player from a fully-enumerated
    characteristic function ``v`` (every subset of *groups* must be a key)
    — pure combinatorics, no model/data involved, so this is independently
    testable against a hand-computed toy game.

    φ_i = Σ_{S ⊆ N\\{i}} [|S|!·(k-|S|-1)!/k!] · [v(S∪{i}) − v(S)]
    """
    k = len(groups)
    result: Dict[str, float] = {}
    for player in groups:
        others = [g for g in groups if g != player]
        total = 0.0
        for r in range(len(others) + 1):
            weight = factorial(r) * factorial(k - r - 1) / factorial(k)
            for combo in itertools.combinations(others, r):
                s = frozenset(combo)
                s_plus = s | {player}
                total += weight * (v[s_plus] - v[s])
        result[player] = total
    return result


@torch.no_grad()
def run_exact_shapley(
    model: nn.Module,
    test_loader,
    ds_cfg: dict,
    modality: str,
    device: torch.device,
    is_multiclass: bool = False,
    mean_image: Optional[torch.Tensor] = None,
    train_loader=None,
) -> Dict[str, float]:
    """Per-image exact Shapley over the model's active channel groups
    (2^k coalitions, k=4 -> 16 for an m5-mode model, spec's literal
    number), averaged over *test_loader*.

    Returns ``{group: shapley_value}`` (normalised is the caller's choice
    — raw Dice-scale Shapley values sum to ``v(all) - v(none)`` per image,
    the efficiency property of exact Shapley).
    """
    if mean_image is None:
        if train_loader is None:
            raise ValueError("run_exact_shapley: need either mean_image or train_loader")
        mean_image = compute_training_mean_image(train_loader, device)

    model.eval()
    group_slices = resolve_group_slices(ds_cfg, modality)
    groups = list(group_slices.keys())
    coalitions = _all_coalitions(groups)

    # v_per_coalition[S] = list of per-image dice, index-aligned across
    # every coalition (same iteration order over test_loader each pass).
    v_per_coalition: Dict[FrozenSet[str], List[float]] = {}
    for coalition in coalitions:
        to_occlude = [g for g in groups if g not in coalition]
        dices: List[float] = []
        for images, masks, _meta in test_loader:
            images = images.to(device)
            gts = masks.numpy()
            occ_images = occlude_groups(images, mean_image, group_slices, to_occlude) if to_occlude else images
            preds, _ = predict_hard(model, occ_images, is_multiclass)
            preds_np = preds.cpu().numpy()
            for p, g in zip(preds_np, gts):
                dices.append(_dice(p, g.squeeze()))
        v_per_coalition[coalition] = dices

    n_images = len(v_per_coalition[frozenset()])
    per_group_totals = {g: 0.0 for g in groups}
    for img_idx in range(n_images):
        v_image = {s: v_per_coalition[s][img_idx] for s in coalitions}
        phi = shapley_values_from_characteristic_function(v_image, groups)
        for g, val in phi.items():
            per_group_totals[g] += val

    return {g: total / n_images for g, total in per_group_totals.items()}
