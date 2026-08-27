"""
datasets/channels.py — shared multi-channel input construction (Phase 4 of
IMPLEMENTATION_PLAN.md).

Consolidates two previously-private, drifting implementations:
  - models/proposed/gmk_unet.py's private `_ycbcr()` — the exact BT.601
    coefficients now live in `_YCBCR_COEFFS` here; `ycbcr_from_rgb_tensor()`
    is what gmk_unet.py imports and calls instead of defining its own.
  - models/baseline/emcad.py's ad hoc "if grayscale, repeat to 3ch"
    forward()-time branch — `modality_effective_channels()` closes this at
    the data level instead: a grayscale dataset's channel_mode never
    constructs YCbCr channels (degenerate/constant for a repeated-gray
    source) in the first place, so no model needs its own inline patch.

Five channel modes (m1..m5), built from named channel GROUPS so a
consistent, predictable channel ordering is available for Phase 11's
per-group Shapley attribution:

    m1              rgb
    m2              rgb + xy
    m3              rgb + ycbcr
    m4              rgb + xy + rtheta
    m5 (all)        rgb + xy + ycbcr + rtheta

Every group-producing function here works on a numpy (H, W, C) frame — the
same representation Albumentations produces before ToTensorV2() converts to
a tensor — because build_channels() is meant to run in that exact slot in
the pipeline: see datasets/augment.py's AugmentationPolicy, which enforces
geometric augmentation *then* build_channels() (ORDER=post), so XY/Rθ
coordinate channels are regenerated from the actual augmented crop rather
than being augmented themselves like the RGB channels are.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# YCbCr (BT.601) — single source of truth for the coefficients
# ---------------------------------------------------------------------------

# (y_r, y_g, y_b), (cb_r, cb_g, cb_b, cb_offset), (cr_r, cr_g, cr_b, cr_offset)
_YCBCR_COEFFS = {
    "y": (0.299000, 0.587000, 0.114000),
    "cb": (-0.168736, -0.331264, 0.500000, 0.5),
    "cr": (0.500000, -0.418688, -0.081312, 0.5),
}


def ycbcr_from_rgb(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """RGB (H, W, 3) float in [0, 1] -> (luma (H, W, 1), chroma (H, W, 2)).

    Numpy counterpart of ycbcr_from_rgb_tensor() — used by build_channels()
    at data-loading time. Same _YCBCR_COEFFS as the tensor version GMK-UNet
    calls at forward-time, so the two paths can never numerically drift
    apart even though they operate on different array types (a live
    differentiable GPU tensor inside a model's forward() cannot share a
    literal function body with a numpy op run once per __getitem__).
    """
    r, g, b = image[..., 0:1], image[..., 1:2], image[..., 2:3]
    yr, yg, yb = _YCBCR_COEFFS["y"]
    cbr, cbg, cbb, cb_off = _YCBCR_COEFFS["cb"]
    crr, crg, crb, cr_off = _YCBCR_COEFFS["cr"]

    y = yr * r + yg * g + yb * b
    cb = cbr * r + cbg * g + cbb * b + cb_off
    cr = crr * r + crg * g + crb * b + cr_off
    return y, np.concatenate([cb, cr], axis=-1)


def ycbcr_from_rgb_tensor(x):
    """torch (B, 3, H, W) -> (luma (B, 1, H, W), chroma (B, 2, H, W)).

    Drop-in replacement for gmk_unet.py's private _ycbcr() — identical
    formula (same _YCBCR_COEFFS), same call signature, same return shape.
    Import is local so importing datasets.channels never requires torch to
    be installed for pure data-prep use (e.g. an offline stats script).
    """
    import torch

    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    yr, yg, yb = _YCBCR_COEFFS["y"]
    cbr, cbg, cbb, cb_off = _YCBCR_COEFFS["cb"]
    crr, crg, crb, cr_off = _YCBCR_COEFFS["cr"]

    y = yr * r + yg * g + yb * b
    cb = cbr * r + cbg * g + cbb * b + cb_off
    cr = crr * r + crg * g + crb * b + cr_off
    return y, torch.cat([cb, cr], dim=1)


# ---------------------------------------------------------------------------
# Geometry channels
# ---------------------------------------------------------------------------

def xy_channels(h: int, w: int) -> np.ndarray:
    """(H, W, 2) absolute-position channels, each in [-1, 1] — x varies
    along columns, y along rows (image convention: origin top-left)."""
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)  # each (H, W)
    return np.stack([xx, yy], axis=-1)


def r_theta_channels(h: int, w: int) -> np.ndarray:
    """(H, W, 3) polar-position channels relative to the frame centre: r
    (Euclidean radius, normalised to [0, ~1.41] by the same [-1, 1] extent
    xy_channels uses), sin(theta), cos(theta).

    theta is encoded as (sin, cos), not a raw angle, specifically to avoid
    the -pi/pi branch cut a raw angle channel would have directly through
    the middle of the frame — see tests/test_channels.py::test_theta_continuity.
    """
    xy = xy_channels(h, w)
    x, y = xy[..., 0], xy[..., 1]
    r = np.sqrt(x**2 + y**2).astype(np.float32)
    theta = np.arctan2(y, x)
    return np.stack([r, np.sin(theta).astype(np.float32), np.cos(theta).astype(np.float32)], axis=-1)


def randproj_channels(h: int, w: int, n: int = 2, seed: int = 0) -> np.ndarray:
    """(H, W, n) random-projection control channels: xy_channels() passed
    through a fixed (seeded) random linear map. An ablation control — if a
    model's accuracy benefits from *these* just as much as from the real
    geometry channels, the real channels aren't earning their place on
    geometric information, just on "the model likes having more numbers to
    work with." Fixed per seed, not per-call, so the same control channels
    are reproducible across a run.
    """
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((2, n)).astype(np.float32)
    xy = xy_channels(h, w)  # (H, W, 2)
    return xy @ proj  # (H, W, n)


def coordonly_channels(h: int, w: int) -> np.ndarray:
    """(H, W, 5) xy_channels + r_theta_channels with **no RGB** — an
    ablation control for the shortcut audit (Phase 13/S11): if a model
    trained on position alone scores non-trivially, that's a red flag that
    RGB-mode results might partly be exploiting frame position rather than
    lesion appearance.
    """
    return np.concatenate([xy_channels(h, w), r_theta_channels(h, w)], axis=-1)


# ---------------------------------------------------------------------------
# Channel-group registry and mode assembly
# ---------------------------------------------------------------------------

CHANNEL_GROUP_SIZES: Dict[str, int] = {"rgb": 3, "xy": 2, "ycbcr": 3, "rtheta": 3}

MODE_GROUPS: Dict[str, List[str]] = {
    "m1": ["rgb"],
    "m2": ["rgb", "xy"],
    "m3": ["rgb", "ycbcr"],
    "m4": ["rgb", "xy", "rtheta"],
    "m5": ["rgb", "xy", "ycbcr", "rtheta"],
}


def _group_channels(name: str, image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if name == "rgb":
        return image
    if name == "xy":
        return xy_channels(h, w)
    if name == "ycbcr":
        luma, chroma = ycbcr_from_rgb(image)
        return np.concatenate([luma, chroma], axis=-1)
    if name == "rtheta":
        return r_theta_channels(h, w)
    raise ValueError(f"Unknown channel group '{name}'. Known: {sorted(CHANNEL_GROUP_SIZES)}")


def build_channels_from_groups(image: np.ndarray, groups: List[str]) -> np.ndarray:
    """Build a multi-channel input from an explicit, already-resolved group
    list — the primitive build_channels() (mode-name based) and
    datasets.augment.AugmentationPolicy (modality-filtered-group-list
    based) both reduce to. No mode-name lookup involved, so a caller
    holding an arbitrary/filtered group list (e.g. modality_effective_channels()'s
    output) never needs to find a mode name that happens to match it.
    """
    unknown = set(groups) - set(CHANNEL_GROUP_SIZES)
    if unknown:
        raise ValueError(f"Unknown channel group(s) {sorted(unknown)}. Known: {sorted(CHANNEL_GROUP_SIZES)}")
    return np.concatenate([_group_channels(g, image) for g in groups], axis=-1)


def build_channels(image: np.ndarray, mode: str, order: Optional[List[str]] = None) -> np.ndarray:
    """Build the full multi-channel input for *mode* from an (H, W, 3)
    float RGB frame in [0, 1].

    Args:
        image: (H, W, 3) float array in [0, 1] (post geometric-augmentation
            — see module docstring's ORDER=post note).
        mode: one of MODE_GROUPS's keys ("m1".."m5").
        order: optional explicit group ordering (must be a permutation of
            MODE_GROUPS[mode]) — the channel-group boundaries this produces
            are what Phase 11's per-group Shapley attribution slices by, so
            a caller that needs a specific, predictable layout can pin it
            here instead of relying on MODE_GROUPS' default order.

    Returns:
        (H, W, C) float array, channel groups concatenated in *order* (or
        MODE_GROUPS[mode]'s default order).
    """
    if mode not in MODE_GROUPS:
        raise ValueError(f"Unknown channel mode '{mode}'. Known: {sorted(MODE_GROUPS)}")
    groups = order if order is not None else MODE_GROUPS[mode]
    if set(groups) != set(MODE_GROUPS[mode]):
        raise ValueError(
            f"order {groups} is not a permutation of mode '{mode}''s groups {MODE_GROUPS[mode]}"
        )
    return build_channels_from_groups(image, groups)


def modality_effective_channels(mode: str, modality: str) -> List[str]:
    """The channel groups *mode* actually resolves to once *modality* is
    taken into account.

    For modality="grayscale", drops "ycbcr" — Cb/Cr computed from a
    repeated-gray "RGB" frame are constant (0.5, 0.5) everywhere, pure
    dead weight the model would otherwise have to learn is uninformative.
    This is the fix for models/baseline/emcad.py's old inline
    "if grayscale, repeat to 3ch" gap: the *data* a grayscale-modality
    dataset produces never contains those degenerate channels in the first
    place, so no model needs its own ad hoc patch for it.

    modality="colour" is a no-op (every requested group is meaningful).
    """
    if modality not in ("colour", "grayscale"):
        raise ValueError(f"modality must be 'colour' or 'grayscale', got '{modality}'")
    groups = MODE_GROUPS.get(mode)
    if groups is None:
        raise ValueError(f"Unknown channel mode '{mode}'. Known: {sorted(MODE_GROUPS)}")
    if modality == "grayscale":
        return [g for g in groups if g != "ycbcr"]
    return list(groups)


def effective_channel_count(mode: str, modality: str) -> int:
    return sum(CHANNEL_GROUP_SIZES[g] for g in modality_effective_channels(mode, modality))


# ---------------------------------------------------------------------------
# Per-channel normalisation stats — cached per dataset+mode, hashed
# ---------------------------------------------------------------------------

def _stats_cache_key(dataset_name: str, mode: str, modality: str) -> str:
    payload = json.dumps(
        {"dataset": dataset_name, "mode": mode, "modality": modality}, sort_keys=True
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def channel_norm_stats(
    dataset_name: str,
    mode: str,
    modality: str,
    compute_fn,
    cache_dir: str = "artifacts/channel_stats",
) -> Dict[str, list]:
    """Return {"mean": [...], "std": [...]} for (dataset_name, mode,
    modality)'s effective channel layout, from an on-disk cache keyed by a
    hash of that triple — computing per-channel stats over an entire
    dataset is not something to redo on every run.

    Args:
        compute_fn: zero-arg callable invoked only on a cache miss; must
            return {"mean": [...], "std": [...]} matching
            effective_channel_count(mode, modality). Kept as an injected
            callback (rather than this function knowing how to iterate a
            dataset itself) so it stays decoupled from any specific
            Dataset/DataModule implementation.
    """
    key = _stats_cache_key(dataset_name, mode, modality)
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    stats = compute_fn()
    n_expected = effective_channel_count(mode, modality)
    if len(stats["mean"]) != n_expected or len(stats["std"]) != n_expected:
        raise ValueError(
            f"compute_fn returned {len(stats['mean'])} channels, expected "
            f"{n_expected} for mode='{mode}' modality='{modality}'"
        )

    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(stats, f, indent=2)
    os.replace(tmp_path, path)
    return stats
