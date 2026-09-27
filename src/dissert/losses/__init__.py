"""
losses/ — loss module (Phase 7 of IMPLEMENTATION_PLAN.md), replacing
training/losses.py.

get_loss(name, num_classes, **kwargs) keeps the exact call signature and
loss_type strings ("bce", "dice", "structure", "combo",
"adaptive_guide_fusion") every existing config already uses — see
losses/compound.py's preset docstrings for what each maps to internally
and losses/terms.py for the standalone term implementations.
"ce"/"tversky"/"focal"/"boundary"/"compound" are new loss_type values this
package adds.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from .compound import (
    CompoundLoss,
    REDUNDANT_TERM_FAMILIES,
    StructureLoss,
    adaptive_guide_fusion_preset,
    combo_preset,
    single_term_preset,
)
from .schedules import apply_schedule, linear_ramp
from .terms import (
    bce,
    boundary,
    ce,
    compute_or_load_distance_map,
    compute_signed_distance_map,
    dice,
    focal,
    tversky,
)

_PRESET_NAMES = {"bce", "ce", "dice", "tversky", "focal", "structure", "combo", "adaptive_guide_fusion"}


def get_loss(name: str, num_classes: int = 1, **kwargs: Any) -> nn.Module:
    """Instantiate a loss by name.

    "compound" is the escape hatch for a fully custom declared term list —
    pass ``term_list=[(name, weight, schedule), ...]`` (and optionally
    ``term_kwargs={...}``) as kwargs.
    """
    name = name.lower()

    if name == "structure" and num_classes != 1:
        raise ValueError(
            "loss_type 'structure' (boundary-weighted structure loss) only "
            "supports binary segmentation (out_channels == 1). "
            "Use 'combo' or 'dice' for multi-class setups."
        )

    # A binary-only term requested for a multiclass problem auto-switches
    # to its multiclass counterpart — matches training/losses.py's old
    # get_loss("bce", num_classes>1) -> CrossEntropyLossWrapper() behaviour
    # exactly, since a bare BCE-on-logits call would otherwise hit a shape
    # mismatch against a (N, H, W) class-index target instead of failing
    # with a clear "wrong loss for this problem" message.
    if name == "bce" and num_classes > 1:
        name = "ce"

    if name == "combo":
        return combo_preset(num_classes=num_classes, **kwargs)
    if name == "adaptive_guide_fusion":
        return adaptive_guide_fusion_preset(**kwargs)
    if name == "compound":
        term_list = kwargs.pop("term_list")
        return CompoundLoss(term_list, num_classes=num_classes, **kwargs)
    if name in _PRESET_NAMES:
        return single_term_preset(name, num_classes=num_classes, **kwargs)

    raise ValueError(f"Unknown loss '{name}'. Available: {sorted(_PRESET_NAMES | {'compound'})}")


__all__ = [
    "get_loss",
    "CompoundLoss",
    "StructureLoss",
    "REDUNDANT_TERM_FAMILIES",
    "bce",
    "ce",
    "dice",
    "tversky",
    "focal",
    "boundary",
    "compute_signed_distance_map",
    "compute_or_load_distance_map",
    "apply_schedule",
    "linear_ramp",
]
