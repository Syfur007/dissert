"""
datasets/busi.py — BUSI (Breast Ultrasound Images) dataset handler.

**Structurally complete but UNVERIFIED against real data** — no BUSI files
exist in this environment (or anywhere reachable from it) to test against.
Built against the dataset's publicly documented layout: a
``Dataset_BUSI_with_GT`` root containing ``benign/``, ``malignant/``,
``normal/`` subfolders, each holding ``<case>.png`` images paired with
``<case>_mask.png`` masks (some cases have extra ``<case>_mask_1.png``,
``<case>_mask_2.png`` for multiple lesions in one image — only the primary
``_mask.png`` is used here, matching what nearly every public BUSI loader
implementation does). Sanity-check this handler's directory/filename
assumptions against your actual download before first real use — see
IMPLEMENTATION_PLAN.md's Phase 3 section and CHANGELOG.md's Phase 3 entry.

BUSI has no official train/val/test split (papers create their own), so —
unlike ClinicDB/ColonDB — this handler does its own seeded random split via
``dataset.split`` ratios, the same convention ``_GenericHandler`` uses for
any other unregistered flat-directory dataset.

Deduplication (``preprocess.dedup()``) is **mandatory, not optional**: BUSI
is known to contain near-duplicate images across (and within) its
benign/malignant/normal folders, which a naive split could scatter across
train and test — see IMPLEMENTATION_PLAN.md's Phase 3 section.
"""
import math
import os
from typing import List, Tuple

import numpy as np
from loguru import logger

from .dataset import MedicalSegmentationDataset

_CLASS_DIRS = ("benign", "malignant", "normal")


class BUSI:
    """
    Config keys (under ``dataset``):
        root  : str  — path to the dataset root (default: data/busi/Dataset_BUSI_with_GT)
        split : dict — {train: 0.8, val: 0.1, test: 0.1} (required — no official split)
        dedup : bool — run preprocess.dedup() before splitting (default: True; set False
                only to skip the (slow, one-time) dedup pass on a run you know is already
                deduplicated — never to skip it outright on first use)
    """

    NAME = "busi"
    # One case == one image here (unlike ClinicDB/ColonDB's video-frame
    # extracts), so frame-level subject_id is not a known limitation for
    # this dataset the way it is for those two — no artefact flag needed
    # for that reason. It IS flagged for the class label living only in the
    # directory name, not in `meta`, in case a future consumer expects it.
    ARTEFACT_FLAGS = {"class_label_in_directory_name_not_in_meta": True}

    def __init__(self, cfg: dict, seed: int = 42):
        root = cfg.get("root", "data/busi/Dataset_BUSI_with_GT")
        self._root = root
        self._seed = seed
        self._dedup = cfg.get("dedup", True)
        self._split_ratios = cfg.get("split")
        if not self._split_ratios:
            raise ValueError(
                "BUSI has no official split — 'dataset.split: {train: 0.8, val: 0.1, "
                "test: 0.1}' is required in config."
            )
        self._splits = None  # lazily computed on first get_dataset()/get_kfold_pairs() call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_all_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for cls in _CLASS_DIRS:
            cls_dir = os.path.join(self._root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if "_mask" in fname or not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                stem, ext = os.path.splitext(fname)
                mask_name = f"{stem}_mask{ext}"
                mask_path = os.path.join(cls_dir, mask_name)
                if os.path.exists(mask_path):
                    pairs.append((os.path.join(cls_dir, fname), mask_path))
                else:
                    logger.warning(f"BUSI: no mask found for {fname} in {cls_dir}; skipping.")
        return pairs

    def _build_splits(self, seed: int) -> None:
        if self._splits is not None:
            return

        pairs = self._scan_all_pairs()
        if not pairs:
            raise FileNotFoundError(
                f"No BUSI pairs found under {self._root}. Expected "
                f"{self._root}/{{benign,malignant,normal}}/<case>.png + <case>_mask.png."
            )

        if self._dedup:
            from .preprocess import dedup
            excluded = dedup(pairs)
            if excluded:
                logger.warning(
                    f"BUSI: preprocess.dedup() flagged {len(excluded)}/{len(pairs)} "
                    "near-duplicate images for exclusion (mandatory per spec)."
                )
            pairs = [p for i, p in enumerate(pairs) if i not in set(excluded)]

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(pairs))
        n = len(pairs)
        n_train = math.floor(self._split_ratios.get("train", 0.8) * n)
        n_val = math.floor(self._split_ratios.get("val", 0.1) * n)

        self._splits = {
            "train": [pairs[i] for i in indices[:n_train]],
            "val": [pairs[i] for i in indices[n_train : n_train + n_val]],
            "test": [pairs[i] for i in indices[n_train + n_val :]],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataset(self, split: str, transform=None, **kwargs) -> MedicalSegmentationDataset:
        self._build_splits(self._seed)
        kwargs.setdefault("source_dataset", self.NAME)
        kwargs.setdefault("artefact_flags", self.ARTEFACT_FLAGS)
        return MedicalSegmentationDataset(
            pairs=self._splits[split], transform=transform, **kwargs
        )

    def get_kfold_pairs(self) -> list:
        """train+val pairs only — test stays held out, same contract as
        datasets/polyp/{clinicdb,colondb}.py."""
        self._build_splits(self._seed)
        return list(self._splits["train"]) + list(self._splits["val"])
