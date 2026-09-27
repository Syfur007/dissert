"""
losses/compound.py — CompoundLoss: a declared list of (term, weight,
schedule), replacing training/losses.py's ComboLoss/AdaptiveGuideFusionLoss
with thin presets expressed as fixed compound lists, per spec §7:
"Declared as a list of (term, weight, schedule). No named monolithic loss
class."

Every loss this project builds — including a "single-term" one like plain
Dice — is a CompoundLoss with a one-entry term_list, not a special case;
see the presets at the bottom of this module. That keeps exactly one
forward()/set_epoch() code path for every loss_type value instead of a
CompoundLoss path for "combo"-like configs and a separate bare-function
path for everything else.

StructureLoss is kept as its own class, used as a named CompoundLoss term
("structure") — it's a *boundary-weighted* BCE+IoU (local-average pixel
weighting near mask edges), not the spec's distance-transform boundary
loss (losses/terms.py's `boundary()`, a different thing). Naming both
"boundary" would be actively misleading about which one a config is
requesting.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import terms as T
from .schedules import apply_schedule

LossTermSpec = Tuple[str, float, Optional[Dict[str, Any]]]

# Terms taking (logits, targets, **kwargs) directly, from losses/terms.py.
# "boundary" and "structure" are handled separately in CompoundLoss.forward
# — boundary needs distance_maps, structure is its own nn.Module below.
_LOGIT_TERMS = {"bce", "ce", "dice", "tversky", "focal"}

# "Monotonically related" term families the redundancy guard
# (orchestration/schema.py) checks — region-overlap losses that measure
# essentially the same thing (Tversky at alpha=beta=0.5 *is* Dice; see
# losses/terms.py's dice()/tversky() docstrings), so stacking two from the
# same family is very likely a config mistake, not a deliberate ensemble.
REDUNDANT_TERM_FAMILIES: List[set] = [{"dice", "tversky"}]


class StructureLoss(nn.Module):
    """Boundary-*weighted* structure loss (weighted BCE + weighted IoU) —
    matches the official MK-UNet training script (PraNet/Polyp-PVT
    lineage). A local-average term up-weights pixels near mask boundaries
    so boundary precision directly drives the loss signal.

    **Binary segmentation only** (out_channels == 1).
    """

    def __init__(self, boundary_weight: float = 5.0, pool_kernel_size: int = 31):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.pool_kernel_size = pool_kernel_size
        self.pool_padding = pool_kernel_size // 2

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        local_avg = F.avg_pool2d(
            targets, kernel_size=self.pool_kernel_size, stride=1, padding=self.pool_padding
        )
        weit = 1 + self.boundary_weight * torch.abs(local_avg - targets)

        wbce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        probs = torch.sigmoid(logits)
        inter = ((probs * targets) * weit).sum(dim=(2, 3))
        union = ((probs + targets) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


class CompoundLoss(nn.Module):
    """
    Args:
        term_list: [(name, weight, schedule_spec), ...]. name is one of
            "bce"/"ce"/"dice"/"tversky"/"focal"/"boundary" (losses/terms.py)
            or "structure" (StructureLoss above). schedule_spec is
            losses.schedules.apply_schedule's ``spec`` dict, or None for a
            constant weight.
        term_kwargs: {name: {...}} — each term's own kwargs (e.g.
            ``{"tversky": {"alpha": 0.3, "beta": 0.7}}``), kept separate
            from the (name, weight, schedule) tuple so that tuple's shape
            stays uniform regardless of which term it names.
    """

    def __init__(
        self,
        term_list: List[LossTermSpec],
        num_classes: int = 1,
        term_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        super().__init__()
        if not term_list:
            raise ValueError("CompoundLoss: term_list must have at least one entry")
        self.term_list = term_list
        self.num_classes = num_classes
        self.term_kwargs = term_kwargs or {}
        self._epoch = 0
        self._max_epoch = 1

        self._structure = (
            StructureLoss(**self.term_kwargs.get("structure", {}))
            if any(n == "structure" for n, _, _ in term_list)
            else None
        )

    def set_epoch(self, epoch: int, max_epoch: int) -> None:
        """Call once per epoch so schedule-driven term weights (e.g. the
        boundary loss's linear ramp) know where they are in training. A
        CompoundLoss with no scheduled terms simply ignores this — every
        schedule defaults to constant(1.0)."""
        self._epoch = epoch
        self._max_epoch = max(max_epoch, 1)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        distance_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total = None
        for name, weight, schedule in self.term_list:
            kwargs = self.term_kwargs.get(name, {})
            if name == "structure":
                value = self._structure(logits, targets)
            elif name == "boundary":
                if distance_maps is None:
                    raise ValueError(
                        "CompoundLoss: a 'boundary' term requires distance_maps to be "
                        "passed to forward() — see losses/terms.py's compute_or_load_distance_map()."
                    )
                value = T.boundary(torch.sigmoid(logits), distance_maps, **kwargs)
            elif name in _LOGIT_TERMS:
                value = getattr(T, name)(logits, targets, **kwargs)
            else:
                raise ValueError(f"CompoundLoss: unknown term '{name}'")

            w = weight * apply_schedule(schedule, self._epoch, self._max_epoch)
            total = w * value if total is None else total + w * value
        return total


# ---------------------------------------------------------------------------
# Presets — every training/losses.py loss_type value maps to one of these,
# preserving its exact prior numerical behaviour.
# ---------------------------------------------------------------------------

def single_term_preset(name: str, num_classes: int = 1, **term_kwargs) -> CompoundLoss:
    return CompoundLoss([(name, 1.0, None)], num_classes=num_classes, term_kwargs={name: term_kwargs})


def combo_preset(num_classes: int = 1, bce_weight: float = 0.5, dice_weight: float = 0.5) -> CompoundLoss:
    """Replaces training/losses.py's ComboLoss. num_classes==1 uses bce;
    num_classes>1 uses ce — same branch the old class made."""
    ce_term = "bce" if num_classes == 1 else "ce"
    return CompoundLoss(
        [(ce_term, bce_weight, None), ("dice", dice_weight, None)], num_classes=num_classes,
    )


def adaptive_guide_fusion_preset(alpha: float = 0.5) -> CompoundLoss:
    """Replaces training/losses.py's AdaptiveGuideFusionLoss — the
    fixed-alpha behaviour only. That class's learnable-alpha option is
    dropped, not preserved as a compound-list case it can't cleanly
    express (a CompoundLoss term weight is a plain float, not an
    nn.Parameter): it was already effectively dead code — its own
    docstring notes it "requires adding a separate parameter group with an
    appropriate learning rate", which train.py's optimizer setup never
    did, so a learnable=True config would build a learnable alpha the
    optimizer never actually updated. No existing config sets
    loss_type: "adaptive_guide_fusion" at all (confirmed by inventory).
    """
    return CompoundLoss([("structure", alpha, None), ("dice", 1.0 - alpha, None)])
