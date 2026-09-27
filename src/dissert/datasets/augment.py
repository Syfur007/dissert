"""
datasets/augment.py — AugmentationPolicy: built once per dataset from
(modality, ds_cfg), wrapping datasets/transforms.py's build_transforms().

Enforces ORDER=post: geometric augmentation runs on (image, mask) first via
Albumentations, then build_channels() regenerates geometry (XY/Rθ) and
colour-derived (YCbCr) channels from *that already-augmented* frame — never
the other way around. A pre-augmentation XY channel would encode stale
coordinates once the frame is flipped/rotated/cropped; regenerating it
post-augmentation is the only way an XY/Rθ channel means what it claims to
mean.

Modality-conditioned op tables (spec §5.4): colour vs. grayscale-ultrasound
vs. grayscale-microscopy get different augmentation intensity defaults
(aggressive brightness/contrast jitter that's reasonable for natural colour
images can push ultrasound speckle or microscopy staining intensity outside
any range a real scanner/slide would ever produce). The *config* schema
(orchestration/schema.py) only exposes the coarse `dataset.modality:
colour|grayscale` distinction the spec fixes at dataset level — the finer
ultrasound-vs-microscopy split used for augmentation intensity is resolved
here from `dataset.name` via `_GRAYSCALE_SUBTYPE`, since it's a fixed fact
about a specific named dataset, not something a two-value config flag can
express. Unrecognised grayscale datasets default to the same conservative
scale as ultrasound rather than the full colour-intensity augmentation set.
"""
from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np

from .channels import MODE_GROUPS, build_channels_from_groups, modality_effective_channels
from .transforms import build_transforms

MODALITIES = ("colour", "grayscale")

# dataset.name (lowercased) -> grayscale subtype, for augmentation-intensity
# purposes only (channel construction only ever needs the coarse
# colour/grayscale distinction — see modality_effective_channels()).
# No grayscale-microscopy dataset is registered in this repo yet; the
# category exists here so one can be added without widening the config
# schema's modality field.
_GRAYSCALE_SUBTYPE = {
    "busi": "ultrasound",
}

# subtype -> augmentation-intensity scale layered on top of ds_cfg's own
# augmentation.brightness_contrast values. 1.0 = no adjustment.
_BRIGHTNESS_CONTRAST_SCALE = {
    "colour": 1.0,
    "ultrasound": 0.5,
    "microscopy": 0.5,
    "generic-grayscale": 0.5,  # unrecognised grayscale dataset: be conservative
}


class AugmentationPolicy:
    """
    Usage::

        policy = AugmentationPolicy(modality="colour", ds_cfg=config["dataset"])
        image, mask = policy.transform(train=True)(image=image, mask=mask).values()
        model_input = policy.build_input_channels(image)  # ORDER=post
    """

    def __init__(self, modality: str, ds_cfg: dict):
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got '{modality}'")
        self.modality = modality

        self.channel_mode = ds_cfg.get("channel_mode", "m1")
        if self.channel_mode not in MODE_GROUPS:
            raise ValueError(f"Unknown channel_mode '{self.channel_mode}'. Known: {sorted(MODE_GROUPS)}")

        requested_order = ds_cfg.get("channel_order")
        self._effective_groups = modality_effective_channels(self.channel_mode, modality)
        if requested_order is not None:
            filtered = [g for g in requested_order if g in self._effective_groups]
            if set(filtered) != set(self._effective_groups):
                raise ValueError(
                    f"channel_order {requested_order} does not cover the effective "
                    f"channel groups {self._effective_groups} for channel_mode="
                    f"'{self.channel_mode}' modality='{modality}'"
                )
            self._effective_groups = filtered

        self._subtype = self._resolve_subtype(modality, ds_cfg.get("name", ""))
        adjusted_cfg = self._apply_modality_overrides(ds_cfg)
        h, w = ds_cfg["img_height"], ds_cfg["img_width"]
        self._train_tf, self._val_tf = build_transforms(h, w, adjusted_cfg)

    @staticmethod
    def _resolve_subtype(modality: str, dataset_name: str) -> str:
        if modality == "colour":
            return "colour"
        return _GRAYSCALE_SUBTYPE.get(dataset_name.lower(), "generic-grayscale")

    def _apply_modality_overrides(self, ds_cfg: dict) -> dict:
        scale = _BRIGHTNESS_CONTRAST_SCALE[self._subtype]
        if scale == 1.0:
            return ds_cfg
        adjusted = copy.deepcopy(ds_cfg)
        aug = adjusted.setdefault("augmentation", {})
        bc = aug.setdefault("brightness_contrast", {})
        bc["brightness_limit"] = bc.get("brightness_limit", 0.2) * scale
        bc["contrast_limit"] = bc.get("contrast_limit", 0.2) * scale
        return adjusted

    def transform(self, train: bool):
        """The underlying Albumentations Compose — val/test always get the
        no-augmentation pipeline (build_transforms already enforces this;
        see tests/test_channels.py::test_val_test_no_augment)."""
        return self._train_tf if train else self._val_tf

    @property
    def effective_groups(self) -> List[str]:
        return list(self._effective_groups)

    def effective_channel_count(self) -> int:
        from .channels import CHANNEL_GROUP_SIZES
        return sum(CHANNEL_GROUP_SIZES[g] for g in self._effective_groups)

    def build_input_channels(self, image: np.ndarray) -> np.ndarray:
        """Call this on the frame *after* self.transform(train)(...) has
        already run — see module docstring's ORDER=post note. image is
        (H, W, 3) float in [0, 1]."""
        return build_channels_from_groups(image, self._effective_groups)
