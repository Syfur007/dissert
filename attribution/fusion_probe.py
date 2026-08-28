"""
attribution/fusion_probe.py — spec §11's fusion_probe.py: "CBFFM
gate/weight statistics per stage; correlation with GT structure size."
Answers RQ3. Needs a CBFFM-fused model (models.fusion.CBFFM — only
fusion="cbffm" has a gate to probe; other fusion modes have nothing
analogous to capture).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from models.fusion import CBFFM


@torch.no_grad()
def run_fusion_probe(model: nn.Module, test_loader, device: torch.device) -> Dict[str, object]:
    """Captures each CBFFM fusion stage's spatial gate map — a (B, 1, H, W)
    map in [0, 1] where 1 means "trust the primary branch fully here",
    read directly off ``CBFFM.gate``'s own output via a forward hook, no
    change to CBFFM's code — averaged to one scalar per image per stage,
    alongside that image's ground-truth foreground fraction ("GT structure
    size"), across *test_loader*.

    Requires ``model.fusion_stages`` (a ModuleList/list) with at least one
    CBFFM entry — raises ``AttributeError``/``ValueError`` early for a
    model/config this analysis doesn't apply to, rather than silently
    returning an empty result.

    Returns ``{"per_stage_gate_mean": {stage_idx: [mean_gate_per_image]},
    "gt_foreground_fraction": [fraction_per_image], "per_stage_correlation":
    {stage_idx: pearson_r}}`` — correlation is Pearson r between a stage's
    per-image mean gate and the per-image GT foreground fraction.
    """
    if not hasattr(model, "fusion_stages"):
        raise AttributeError(f"run_fusion_probe: {type(model).__name__} has no fusion_stages to probe.")

    cbffm_stage_idxs = [i for i, stage in enumerate(model.fusion_stages) if isinstance(stage, CBFFM)]
    if not cbffm_stage_idxs:
        raise ValueError("run_fusion_probe: no CBFFM fusion stage found on this model (fusion mode isn't 'cbffm').")

    model.eval()

    captured_gates: Dict[int, List[torch.Tensor]] = {i: [] for i in cbffm_stage_idxs}
    handles = []

    def _make_hook(stage_idx):
        def _hook(module, inputs, output):
            captured_gates[stage_idx].append(output.mean(dim=(1, 2, 3)).detach().cpu())
        return _hook

    for i in cbffm_stage_idxs:
        handles.append(model.fusion_stages[i].gate.register_forward_hook(_make_hook(i)))

    gt_fractions: List[float] = []
    try:
        for images, masks, _meta in test_loader:
            images = images.to(device)
            model(images)  # forward pass only, to trigger the gate hooks
            gt_fractions.extend(masks.flatten(1).mean(dim=1).numpy().tolist())
    finally:
        for h in handles:
            h.remove()

    per_stage_gate_mean = {i: torch.cat(vals).numpy().tolist() for i, vals in captured_gates.items()}

    per_stage_correlation: Dict[int, float] = {}
    for i, gate_means in per_stage_gate_mean.items():
        if len(gate_means) < 2 or np.std(gate_means) == 0 or np.std(gt_fractions) == 0:
            per_stage_correlation[i] = 0.0
            continue
        corr = np.corrcoef(gate_means, gt_fractions)[0, 1]
        per_stage_correlation[i] = float(corr) if not np.isnan(corr) else 0.0

    return {
        "per_stage_gate_mean": per_stage_gate_mean,
        "gt_foreground_fraction": gt_fractions,
        "per_stage_correlation": per_stage_correlation,
    }
