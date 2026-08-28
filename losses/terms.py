"""
losses/terms.py — standalone loss terms, per spec §7: "Each a standalone
callable with documented reduction and epsilon."

`bce`/`ce`/`dice` are ported from training/losses.py's DiceLoss/
CrossEntropyLossWrapper/ComboLoss's BCE branch — same numerics, just pulled
out as standalone functions instead of living inside a monolithic loss
class. `tversky`, `boundary`, `focal` are new (zero prior implementation in
this repo).

Every term here takes raw logits (not probabilities) except `boundary`,
which is defined on probabilities in the boundary-loss literature (Kervadec
et al., 2019) — documented on that function specifically.
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def bce(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Binary cross-entropy on logits. targets: same shape as logits, in
    {0, 1} (float or int, cast to float here)."""
    return F.binary_cross_entropy_with_logits(logits, targets.float(), reduction=reduction)


def ce(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Multi-class cross-entropy. logits: (N, C, H, W); targets: (N, H, W)
    or (N, 1, H, W) class-index map (not one-hot) — squeezed/cast here so a
    raw mask loaded straight from disk (float, possibly channel-dim-1)
    works without the caller normalising it first."""
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    return F.cross_entropy(logits, targets.long(), reduction=reduction)


def dice(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Dice loss (1 - soft Dice coefficient). Binary (logits.shape[1]==1)
    or multi-class (macro-averaged over classes, targets one-hot or a
    class-index map — both accepted, same as ce()). `smooth`: Laplace term
    added to numerator and denominator, avoiding a 0/0 on an
    empty-prediction/empty-target pair.
    """
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits).view(-1)
        t = targets.view(-1).float()
        inter = (probs * t).sum()
        return 1.0 - (2.0 * inter + smooth) / (probs.sum() + t.sum() + smooth)

    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    if targets.ndim == 4 and targets.shape[1] == num_classes:
        one_hot = targets.float()
    else:
        t = targets.squeeze(1) if (targets.ndim == 4 and targets.shape[1] == 1) else targets
        one_hot = F.one_hot(t.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    total = 0.0
    for c in range(num_classes):
        p_c, t_c = probs[:, c].reshape(-1), one_hot[:, c].reshape(-1)
        inter = (p_c * t_c).sum()
        total = total + (1.0 - (2.0 * inter + smooth) / (p_c.sum() + t_c.sum() + smooth))
    return total / num_classes


def tversky(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.5, beta: float = 0.5,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Tversky loss (1 - Tversky index), binary only.

    TI = TP / (TP + alpha*FN + beta*FP). alpha=beta=0.5 makes this
    numerically identical to dice() (a useful sanity-checkable property —
    see tests/test_losses.py::test_tversky_reduces_to_dice_at_half_half).
    alpha>beta penalises false negatives more (higher recall emphasis);
    beta>alpha penalises false positives more.
    """
    if logits.shape[1] != 1:
        raise ValueError("tversky() is binary-only (logits.shape[1] must be 1)")
    probs = torch.sigmoid(logits).view(-1)
    t = targets.view(-1).float()
    tp = (probs * t).sum()
    fn = ((1 - probs) * t).sum()
    fp = (probs * (1 - t)).sum()
    ti = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1.0 - ti


def focal(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss (Lin et al., 2017): -alpha*(1-p_t)^gamma * log(p_t),
    p_t = p if target==1 else 1-p. Down-weights already-easy (high p_t)
    pixels so the loss concentrates on hard/misclassified ones — the
    standard remedy for the extreme foreground/background pixel imbalance
    segmentation masks have.
    """
    t = targets.float()
    bce_raw = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * t + (1 - p) * (1 - t)
    loss = alpha * (1 - p_t).pow(gamma) * bce_raw
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# ---------------------------------------------------------------------------
# Boundary loss (Kervadec et al., 2019) — real distance-transform, cached
# ---------------------------------------------------------------------------

def _distance_map_cache_path(mask_path: str, cache_dir: str) -> str:
    key = hashlib.sha1(os.path.abspath(mask_path).encode("utf-8")).hexdigest()[:20]
    return os.path.join(cache_dir, f"{key}.npy")


def compute_signed_distance_map(mask: np.ndarray) -> np.ndarray:
    """mask: (H, W) binary (or {0, 255}) array. Returns a signed Euclidean
    distance transform: positive outside the foreground, negative inside,
    ~0 at the boundary — the standard boundary-loss target field (Kervadec
    et al., 2019, eq. 1-2)."""
    from scipy.ndimage import distance_transform_edt

    binary = mask > 127 if mask.max() > 1 else mask.astype(bool)
    if binary.all() or not binary.any():
        # No boundary exists (all-foreground or all-background) — the
        # distance transform is undefined in the usual sense; return an
        # all-zero field so this pair contributes exactly 0 to the
        # boundary loss rather than an arbitrary large constant.
        return np.zeros(mask.shape, dtype=np.float32)
    dist_out = distance_transform_edt(~binary)
    dist_in = distance_transform_edt(binary)
    return (dist_out - dist_in).astype(np.float32)


def compute_or_load_distance_map(
    mask_path: str, cache_dir: str = "artifacts/boundary_cache"
) -> np.ndarray:
    """Disk-cached compute_signed_distance_map(), keyed by *mask_path*'s
    absolute path — per spec §7: "Distance transforms precomputed per
    dataset and cached to disk." Meant for a one-time preprocessing pass
    over a dataset's mask files (mirrors datasets/preprocess.py's
    build_manifest()'s "computed once" ethos), not a per-training-step call.
    """
    import cv2

    path = _distance_map_cache_path(mask_path, cache_dir)
    if os.path.exists(path):
        return np.load(path)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found while computing boundary distance map: {mask_path}")
    dmap = compute_signed_distance_map(mask)

    os.makedirs(cache_dir, exist_ok=True)
    # np.save() appends ".npy" if the filename doesn't already end with it
    # — naming the temp file with the extension up front keeps the actual
    # write target unambiguous instead of reasoning about that behaviour.
    tmp_path = f"{path}.tmp.npy"
    np.save(tmp_path, dmap)
    os.replace(tmp_path, path)
    return dmap


def boundary(probs: torch.Tensor, distance_maps: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Boundary loss (Kervadec et al., 2019): mean(probs * signed_distance_map).

    Unlike every other term in this module, this one takes **probabilities**
    (already sigmoid'd), not logits — the boundary loss is defined that way
    in the source paper (a linear functional of the softmax/sigmoid output,
    not a log-likelihood), and applying it to raw logits would give it an
    entirely different (and much larger, unbounded) scale.

    Args:
        probs: (N, 1, H, W) sigmoid probabilities.
        distance_maps: (N, 1, H, W) precomputed signed distance transform
            of the ground truth (see compute_or_load_distance_map) — passed
            in rather than computed here, since per-training-step
            computation is exactly what the spec's caching requirement
            exists to avoid.
    """
    per_image = (probs * distance_maps).mean(dim=tuple(range(1, probs.ndim)))
    if reduction == "mean":
        return per_image.mean()
    if reduction == "sum":
        return per_image.sum()
    return per_image
