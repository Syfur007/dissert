"""
training/losses.py — Loss function registry & factory.

Usage::

    from training.losses import get_loss

    criterion = get_loss('combo', num_classes=1, bce_weight=0.5, dice_weight=0.5)

Mirrors the model-registry pattern in models/registry.py so adding a new loss
is a single-line registration — no train.py edit required.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Primitive loss implementations
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Dice loss for binary or multi-class image segmentation.

    Args:
        smooth: Laplace smoothing term to avoid division by zero.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            # Binary segmentation
            probs   = torch.sigmoid(logits).view(-1)
            targets = targets.view(-1).float()
            inter   = (probs * targets).sum()
            dice    = (2.0 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
            return 1.0 - dice

        # Multi-class segmentation
        num_classes = logits.shape[1]
        probs       = torch.softmax(logits, dim=1)

        # Normalise targets to one-hot (N, C, H, W)
        if targets.ndim == 4 and targets.shape[1] == num_classes:
            one_hot = targets.float()
        else:
            if targets.ndim == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            one_hot = nn.functional.one_hot(
                targets.long(), num_classes=num_classes
            ).permute(0, 3, 1, 2).float()

        dice_loss = 0.0
        for c in range(num_classes):
            p_c   = probs[:, c].reshape(-1)
            t_c   = one_hot[:, c].reshape(-1)
            inter = (p_c * t_c).sum()
            dice_loss += 1.0 - (2.0 * inter + self.smooth) / (p_c.sum() + t_c.sum() + self.smooth)
        return dice_loss / num_classes


class CrossEntropyLossWrapper(nn.Module):
    """Thin wrapper around nn.CrossEntropyLoss that normalises target shape/dtype.

    Masks loaded from disk aren't guaranteed to be squeezed to (N, H, W) and
    cast to int64, so calling nn.CrossEntropyLoss directly on raw masks can
    throw or silently misbehave.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        return self.ce(logits, targets.long())


class ComboLoss(nn.Module):
    """Combines BCE (or CE) and Dice loss to leverage both objectives.

    Args:
        num_classes:  1 → BCEWithLogitsLoss + Dice; >1 → CrossEntropy + Dice.
        bce_weight:   Weight for the BCE/CE term.
        dice_weight:  Weight for the Dice term.
    """

    def __init__(self, num_classes: int = 1, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.dice        = DiceLoss()
        self.ce          = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            ce_loss   = self.ce(logits, targets.float())
            dice_loss = self.dice(logits, targets)
        else:
            if targets.ndim == 4 and targets.shape[1] == 1:
                ce_targets = targets.squeeze(1).long()
            elif targets.ndim == 4 and targets.shape[1] > 1:
                ce_targets = targets.argmax(dim=1).long()
            else:
                ce_targets = targets.long()
            ce_loss   = self.ce(logits, ce_targets)
            dice_loss = self.dice(logits, ce_targets)

        return self.bce_weight * ce_loss + self.dice_weight * dice_loss


class StructureLoss(nn.Module):
    """Boundary-weighted structure loss (weighted BCE + weighted IoU).

    Matches the official MK-UNet training script (PraNet/Polyp-PVT lineage).
    A local-average term up-weights pixels near mask boundaries so boundary
    precision directly drives the loss signal.

    **Binary segmentation only** (out_channels == 1). Use 'combo' or 'dice'
    for multi-class setups.

    Args:
        boundary_weight:   Multiplier for the boundary emphasis term.
        pool_kernel_size:  Size of the average-pooling kernel used to compute
                           the local neighbourhood average.
    """

    def __init__(self, boundary_weight: float = 5.0, pool_kernel_size: int = 31):
        super().__init__()
        self.boundary_weight  = boundary_weight
        self.pool_kernel_size = pool_kernel_size
        self.pool_padding     = pool_kernel_size // 2

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets   = targets.float()
        local_avg = nn.functional.avg_pool2d(
            targets, kernel_size=self.pool_kernel_size,
            stride=1, padding=self.pool_padding
        )
        weit  = 1 + self.boundary_weight * torch.abs(local_avg - targets)

        wbce  = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        wbce  = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        probs = torch.sigmoid(logits)
        inter = ((probs * targets) * weit).sum(dim=(2, 3))
        union = ((probs + targets) * weit).sum(dim=(2, 3))
        wiou  = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


class AdaptiveGuideFusionLoss(nn.Module):
    """Structure loss + Dice loss fused with a configurable mixing ratio.

    This is the planned AdaptiveGuideFusionLoss.  The mixing ratio α is a
    fixed scalar (config-driven) by default; set ``learnable=True`` to make it
    a ``nn.Parameter`` the optimizer can tune (requires adding a separate
    parameter group with an appropriate learning rate).

    Args:
        alpha:     Weight for StructureLoss (1-alpha goes to DiceLoss).
        learnable: If True, α is a bounded learnable parameter.
    """

    def __init__(self, alpha: float = 0.5, learnable: bool = False):
        super().__init__()
        self.structure = StructureLoss()
        self.dice      = DiceLoss()
        if learnable:
            # Store in logit space so sigmoid keeps it in (0, 1) without clamping
            self._alpha_logit = nn.Parameter(
                torch.tensor(float(torch.log(torch.tensor(alpha / (1.0 - alpha + 1e-6)))))
            )
            self._learnable = True
        else:
            self.register_buffer('_alpha_fixed', torch.tensor(alpha))
            self._learnable = False

    @property
    def alpha(self) -> torch.Tensor:
        if self._learnable:
            return torch.sigmoid(self._alpha_logit)
        return self._alpha_fixed

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        return a * self.structure(logits, targets) + (1.0 - a) * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

LOSS_REGISTRY = {
    "bce":                    None,   # resolved dynamically (depends on num_classes)
    "dice":                   DiceLoss,
    "structure":              StructureLoss,
    "combo":                  ComboLoss,
    "adaptive_guide_fusion":  AdaptiveGuideFusionLoss,
}


def get_loss(name: str, num_classes: int = 1, **kwargs) -> nn.Module:
    """Instantiate a loss function by name.

    Args:
        name:        Loss name (see LOSS_REGISTRY keys).
        num_classes: Number of segmentation classes (1 = binary).
        **kwargs:    Forwarded to the loss constructor.  For 'combo', useful
                     keys are ``bce_weight`` and ``dice_weight``.

    Returns:
        Instantiated ``nn.Module`` loss.

    Raises:
        ValueError: For unknown names or incompatible name/num_classes combos.
    """
    name = name.lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )

    if name == "bce":
        if num_classes == 1:
            return nn.BCEWithLogitsLoss()
        return CrossEntropyLossWrapper()

    if name == "structure" and num_classes != 1:
        raise ValueError(
            "loss_type 'structure' (boundary-weighted structure loss) only "
            "supports binary segmentation (out_channels == 1). "
            "Use 'combo' or 'dice' for multi-class setups."
        )

    cls = LOSS_REGISTRY[name]

    # Inject num_classes for losses that care about it
    if name == "combo":
        kwargs.setdefault("num_classes", num_classes)

    return cls(**kwargs)
