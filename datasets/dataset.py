"""
Generic dataset for image-mask segmentation pairs.

Features
--------
- Accepts either ``(image_dir, mask_dir, filenames)`` or a pre-built
  list of ``(img_path, mask_path)`` pairs.
- Optional integrity validation at construction (``validate=True``):
  confirms every image/mask pair exists, is readable, and has matching
  spatial dimensions.  All broken pairs are reported in a single error
  rather than crashing mid-epoch.
- Optional in-RAM caching (``cache=True``): loads all raw images and
  masks on construction if the estimated footprint is under
  ``cache_size_limit_gb``.  Skipped with a warning when the dataset is
  too large.
"""
import os
import cv2
import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset


class DataIntegrityError(Exception):
    """Raised when one or more image/mask pairs fail the integrity check."""


class MedicalSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir=None,
        mask_dir=None,
        filenames=None,
        pairs=None,
        transform=None,
        validate: bool = False,
        cache: bool = False,
        cache_size_limit_gb: float = 4.0,
    ):
        self.transform = transform
        # None  → no caching; dict → caching enabled (filled in _prefetch)
        self._cache: dict | None = {} if cache else None

        if pairs is not None:
            self.pairs = list(pairs)
        else:
            if filenames is None:
                filenames = (
                    sorted(f for f in os.listdir(image_dir)
                           if os.path.isfile(os.path.join(image_dir, f)))
                    if os.path.exists(image_dir) else []
                )
            mask_names_map = {}
            if mask_dir and os.path.exists(mask_dir):
                mask_names_map = {
                    os.path.splitext(f)[0].lower(): f
                    for f in os.listdir(mask_dir)
                }
            self.pairs = []
            for fname in filenames:
                stem = os.path.splitext(fname)[0].lower()
                mask_file = (
                    mask_names_map.get(stem)
                    or mask_names_map.get(f"{stem}_mask")
                    or mask_names_map.get(f"{stem}_gt")
                    or fname
                )
                self.pairs.append((
                    os.path.join(image_dir, fname),
                    os.path.join(mask_dir, mask_file),
                ))

        if validate:
            self._validate()

        if cache and self._cache is not None:
            self._prefetch(cache_size_limit_gb)

    # ------------------------------------------------------------------
    # Integrity validation
    # ------------------------------------------------------------------

    def _validate(self):
        """Check every pair for existence, readability, and spatial alignment."""
        errors = []
        for img_path, mask_path in self.pairs:
            pair_errors = []
            img, mask = None, None

            if not os.path.isfile(img_path):
                pair_errors.append("image missing")
            else:
                img = cv2.imread(img_path)
                if img is None:
                    pair_errors.append("image unreadable")

            if not os.path.isfile(mask_path):
                pair_errors.append("mask missing")
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    pair_errors.append("mask unreadable")

            if img is not None and mask is not None:
                if img.shape[:2] != mask.shape[:2]:
                    pair_errors.append(
                        f"dimension mismatch: image {img.shape[:2]} vs mask {mask.shape[:2]}"
                    )

            if pair_errors:
                errors.append(f"  {img_path}: {', '.join(pair_errors)}")

        if errors:
            raise DataIntegrityError(
                f"Dataset integrity check failed "
                f"({len(errors)}/{len(self.pairs)} broken pairs):\n"
                + "\n".join(errors)
            )
        logger.info(f"Dataset integrity check passed ({len(self.pairs)} pairs).")

    # ------------------------------------------------------------------
    # In-RAM caching
    # ------------------------------------------------------------------

    def _prefetch(self, limit_gb: float):
        """Load all raw images/masks into self._cache if within size limit."""
        if not self.pairs:
            return

        probe_img  = cv2.imread(self.pairs[0][0])
        probe_mask = cv2.imread(self.pairs[0][1], cv2.IMREAD_GRAYSCALE)
        if probe_img is None:
            logger.warning("Cache: could not read probe image; caching disabled.")
            self._cache = None
            return

        bytes_per_pair = probe_img.nbytes + (
            probe_mask.nbytes if probe_mask is not None else 0
        )
        estimated_gb = bytes_per_pair * len(self.pairs) / (1024 ** 3)

        if estimated_gb > limit_gb:
            logger.warning(
                f"Cache: estimated footprint {estimated_gb:.2f} GB "
                f"exceeds limit {limit_gb:.2f} GB; caching disabled."
            )
            self._cache = None
            return

        logger.info(
            f"Caching {len(self.pairs)} pairs into RAM "
            f"(~{estimated_gb:.2f} GB)..."
        )
        broken = []
        for idx, (img_path, mask_path) in enumerate(self.pairs):
            img  = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                # Don't cache a broken pair — leave it out of self._cache so
                # __getitem__ falls through to its non-cached path, which
                # raises a clear FileNotFoundError naming the missing file
                # instead of silently handing None to the transform
                # pipeline (an opaque crash deep inside albumentations,
                # far from the actual cause).
                broken.append(img_path if img is None else mask_path)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._cache[idx] = (img, mask)
        if broken:
            logger.warning(
                f"Cache: {len(broken)} pair(s) could not be read and were "
                f"left uncached (will raise on access): {broken}"
            )
        logger.info("Caching complete.")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self._cache is not None and idx in self._cache:
            image, mask = self._cache[idx]
        else:
            img_path, mask_path = self.pairs[idx]
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if mask.max() > 1.0:
            # Binarize rather than dividing unthresholded: some source masks
            # (e.g. ClinicDB) aren't strictly {0, 255} — antialiased/
            # compressed edges leave intermediate grey values, which would
            # otherwise become fractional soft labels concentrated exactly
            # at object boundaries, where HD95/ASD are most sensitive.
            mask = (mask > 127).float()

        return image, mask.float()
