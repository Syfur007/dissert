"""
tests/test_channels.py — Phase 4: channel construction + AugmentationPolicy.

The five tests IMPLEMENTATION_PLAN.md names for this phase:
test_val_test_no_augment, test_mask_interpolation, test_channel_order,
test_theta_continuity, test_grayscale_drops_colour.
"""
from __future__ import annotations

import numpy as np
import pytest

from datasets.augment import AugmentationPolicy
from datasets.channels import (
    CHANNEL_GROUP_SIZES,
    MODE_GROUPS,
    build_channels,
    modality_effective_channels,
    r_theta_channels,
    ycbcr_from_rgb,
)


def _synthetic_rgb(size=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((size, size, 3), dtype=np.float32)


def _sharp_mask(size=64):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    return mask


def _base_ds_cfg(size=64, **overrides):
    cfg = {
        "img_height": size,
        "img_width": size,
        "name": "test_dataset",
        "augmentation": {
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.5,
            "random_rotate90_p": 0.5,
            "shift_scale_rotate": {"shift_limit": 0.1, "scale_limit": 0.1, "rotate_limit": 30, "p": 1.0},
            "brightness_contrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 1.0},
        },
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# test_val_test_no_augment
# ---------------------------------------------------------------------------

def test_val_test_no_augment():
    policy = AugmentationPolicy(modality="colour", ds_cfg=_base_ds_cfg())
    image = (_synthetic_rgb() * 255).astype(np.uint8)
    mask = _sharp_mask()

    val_tf = policy.transform(train=False)
    out1 = val_tf(image=image, mask=mask)
    out2 = val_tf(image=image, mask=mask)

    # Deterministic: same input, same output, every call — no randomness
    # anywhere in the val/test pipeline.
    assert np.array_equal(np.asarray(out1["image"]), np.asarray(out2["image"]))
    assert np.array_equal(np.asarray(out1["mask"]), np.asarray(out2["mask"]))

    # The train pipeline, by contrast, genuinely varies run to run (with
    # shift_scale_rotate/brightness_contrast p=1.0 in this fixture) —
    # confirms the *train* transform isn't secretly non-random too, which
    # would make the val/test determinism check above trivially true for
    # the wrong reason.
    train_tf = policy.transform(train=True)
    train_outs = [np.asarray(train_tf(image=image, mask=mask)["image"]) for _ in range(8)]
    assert any(not np.array_equal(train_outs[0], o) for o in train_outs[1:])

    # Literal spec §19 wording: "Val/test transform pipeline contains only
    # resize + normalise" — not just "happens to be deterministic" (a
    # pipeline with, say, a fixed-seed augmentation op would also pass the
    # determinism check above without satisfying this).
    val_op_names = [type(t).__name__ for t in val_tf.transforms]
    assert val_op_names == ["Resize", "Normalize", "ToTensorV2"], val_op_names


# ---------------------------------------------------------------------------
# test_mask_interpolation
# ---------------------------------------------------------------------------

def test_mask_interpolation():
    """A sharp binary mask, put through geometric augmentation (rotation in
    particular — the op most likely to introduce interpolation artefacts),
    must come out with no intermediate grey values: Albumentations uses
    nearest-neighbour for the mask target regardless of the image
    interpolation mode, and this confirms that holds for this project's
    actual configured pipeline, not just Albumentations' documented
    default.
    """
    cfg = _base_ds_cfg()
    cfg["augmentation"]["shift_scale_rotate"]["rotate_limit"] = 45
    cfg["augmentation"]["shift_scale_rotate"]["p"] = 1.0
    policy = AugmentationPolicy(modality="colour", ds_cfg=cfg)

    image = (_synthetic_rgb() * 255).astype(np.uint8)
    mask = _sharp_mask()

    for seed in range(10):
        np.random.seed(seed)
        out = policy.transform(train=True)(image=image, mask=mask)
        out_mask = np.asarray(out["mask"])
        uniques = np.unique(out_mask)
        assert set(uniques.tolist()) <= {0, 255}, (
            f"mask has intermediate values after augmentation: {uniques} "
            "(nearest-neighbour interpolation was not used)"
        )


# ---------------------------------------------------------------------------
# test_channel_order
# ---------------------------------------------------------------------------

def test_channel_order():
    image = _synthetic_rgb()
    default = build_channels(image, "m4")  # rgb, xy, rtheta
    reordered = build_channels(image, "m4", order=["rtheta", "rgb", "xy"])

    assert default.shape == reordered.shape
    n_rgb, n_xy, n_rt = CHANNEL_GROUP_SIZES["rgb"], CHANNEL_GROUP_SIZES["xy"], CHANNEL_GROUP_SIZES["rtheta"]

    # Default order: rgb | xy | rtheta
    assert np.array_equal(default[..., :n_rgb], image)
    # Reordered: rtheta | rgb | xy
    assert np.array_equal(reordered[..., :n_rt], default[..., n_rgb + n_xy :])
    assert np.array_equal(reordered[..., n_rt : n_rt + n_rgb], image)


def test_channel_order_rejects_non_matching_groups():
    image = _synthetic_rgb()
    with pytest.raises(ValueError):
        build_channels(image, "m1", order=["rgb", "xy"])  # xy not in m1


# ---------------------------------------------------------------------------
# test_theta_continuity
# ---------------------------------------------------------------------------

def test_theta_continuity():
    """sin(theta)/cos(theta) must vary smoothly along the negative x-axis,
    where a *raw* atan2 angle channel wraps from +pi to -pi.

    The branch cut lives at x<0, y=0: fix a column at negative x (avoiding
    x=0, where the origin itself is a genuine r=0 singularity no encoding
    can smooth over — that's a different, unavoidable discontinuity, not
    what sin/cos-encoding fixes) and sweep down through y=0. A raw
    atan2(y, x) channel jumps by ~2*pi there; sin/cos must not jump at all.
    """
    size = 65
    rt = r_theta_channels(size, size)
    sin_theta = rt[..., 1]
    cos_theta = rt[..., 2]

    col = size // 4  # x < 0 here (linspace(-1, 1, size)'s first quarter)
    x_at_col = np.linspace(-1.0, 1.0, size)[col]
    assert x_at_col < 0

    raw_theta_col = np.arctan2(np.linspace(-1.0, 1.0, size), x_at_col)
    max_step_raw = np.abs(np.diff(raw_theta_col)).max()
    assert max_step_raw > 3.0  # confirms this column really does cross the branch cut

    max_step_sin = np.abs(np.diff(sin_theta[:, col])).max()
    max_step_cos = np.abs(np.diff(cos_theta[:, col])).max()
    assert max_step_sin < 0.5
    assert max_step_cos < 0.5


def test_ycbcr_matches_bt601_reference():
    image = np.zeros((4, 4, 3), dtype=np.float32)
    image[..., 0] = 1.0  # pure red
    luma, chroma = ycbcr_from_rgb(image)
    # BT.601 pure red: Y=0.299, Cb=0.5-0.168736=0.331264, Cr=0.5+0.5=1.0
    assert np.allclose(luma, 0.299, atol=1e-5)
    assert np.allclose(chroma[..., 0], 0.5 - 0.168736, atol=1e-5)
    assert np.allclose(chroma[..., 1], 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# test_grayscale_drops_colour
# ---------------------------------------------------------------------------

def test_grayscale_drops_colour():
    for mode in MODE_GROUPS:
        groups = modality_effective_channels(mode, "grayscale")
        assert "ycbcr" not in groups
        # Every non-ycbcr group in the mode is still present.
        assert groups == [g for g in MODE_GROUPS[mode] if g != "ycbcr"]

    # colour is a no-op.
    for mode in MODE_GROUPS:
        assert modality_effective_channels(mode, "colour") == MODE_GROUPS[mode]


def test_grayscale_drops_colour_via_augmentation_policy():
    cfg = _base_ds_cfg(channel_mode="m5")
    policy = AugmentationPolicy(modality="grayscale", ds_cfg=cfg)
    assert "ycbcr" not in policy.effective_groups
    assert policy.effective_channel_count() == sum(
        CHANNEL_GROUP_SIZES[g] for g in ("rgb", "xy", "rtheta")
    )

    image = _synthetic_rgb()
    out = policy.build_input_channels(image)
    assert out.shape[-1] == policy.effective_channel_count()


def test_augmentation_policy_dataset_specific_intensity_scale():
    # BUSI (registered grayscale-ultrasound subtype) gets scaled-down
    # brightness/contrast limits relative to a colour dataset with the
    # same requested limits.
    cfg = _base_ds_cfg(name="busi")
    busi_policy = AugmentationPolicy(modality="grayscale", ds_cfg=cfg)
    colour_policy = AugmentationPolicy(modality="colour", ds_cfg=_base_ds_cfg(name="clinicdb"))

    busi_bc = busi_policy.transform(train=True).transforms[-3]  # RandomBrightnessContrast
    colour_bc = colour_policy.transform(train=True).transforms[-3]
    assert busi_bc.brightness_limit[1] < colour_bc.brightness_limit[1]
