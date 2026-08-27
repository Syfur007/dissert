"""
Data module hierarchy for medical image segmentation.

Classes
-------
BaseDataModule
    Shared init: stride-snapping, transform wiring (via build_transforms),
    seeded DataLoader factory.

StandardSplitDataModule(BaseDataModule)
    Uses pre-defined train/val/test splits from a registered dataset handler.
    Falls back to auto-partitioning a flat directory when ``dataset.split``
    ratios are provided and no registered handler matches.

KFoldDataModule(BaseDataModule)
    K-fold cross-validation with stable fold persistence (fold assignments
    are serialised to JSON on first run and reloaded on resume).

Dataset handler registry
------------------------
Add new handlers to ``DATASETS`` dict; each must implement:
    get_dataset(split, transform, **kwargs) → MedicalSegmentationDataset
    get_kfold_pairs()                       → list of [img_path, mask_path]
"""
import math
import os
import json
import random

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from .dataset import MedicalSegmentationDataset
from .transforms import build_transforms
from .polyp.clinicdb import ClinicDB
from .polyp.colondb import ColonDB

_MODEL_STRIDE = 32

DATASETS: dict = {
    ClinicDB.NAME: ClinicDB,
    ColonDB.NAME:  ColonDB,
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _snap_to_stride(value: int, stride: int = _MODEL_STRIDE) -> int:
    """Round *value* to the nearest positive multiple of *stride*."""
    return max(stride, int(round(value / stride)) * stride)


def _make_worker_init_fn(base_seed: int):
    """Return a DataLoader worker initialiser that seeds each worker reproducibly."""
    def worker_init_fn(worker_id: int):
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return worker_init_fn


# ── Generic flat-directory handler ──────────────────────────────────────────

class _GenericHandler:
    """
    Handler for datasets that ship as a flat ``images/`` + ``masks/``
    directory without pre-made train/val/test sub-trees.

    Requires ``dataset.split: {train: 0.8, val: 0.1, test: 0.1}`` in config.
    Files are shuffled with the training seed for reproducibility.
    """

    def __init__(self, cfg: dict, seed: int):
        split_ratios = cfg.get("split")
        if not split_ratios:
            raise ValueError(
                "Dataset not registered and 'dataset.split' is absent. "
                "Add 'split: {train: 0.8, val: 0.1, test: 0.1}' to your config "
                "or register a handler in datasets/datamodule.py."
            )
        root     = cfg["root"]
        img_dir  = os.path.join(root, "images")
        mask_dir = os.path.join(root, "masks")
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            raise FileNotFoundError(
                f"Expected '{img_dir}' and '{mask_dir}' to exist for generic auto-split."
            )

        all_names = sorted(
            f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))
        )
        rng     = np.random.default_rng(seed)
        indices = rng.permutation(len(all_names))

        n       = len(all_names)
        n_train = math.floor(split_ratios.get("train", 0.8) * n)
        n_val   = math.floor(split_ratios.get("val",   0.1) * n)

        self._splits = {
            "train": [all_names[i] for i in indices[:n_train]],
            "val":   [all_names[i] for i in indices[n_train : n_train + n_val]],
            "test":  [all_names[i] for i in indices[n_train + n_val :]],
        }
        self._img_dir  = img_dir
        self._mask_dir = mask_dir

    def get_dataset(self, split: str, transform=None, **kwargs) -> MedicalSegmentationDataset:
        return MedicalSegmentationDataset(
            self._img_dir, self._mask_dir,
            filenames=self._splits[split],
            transform=transform,
            **kwargs,
        )


# ── Base ─────────────────────────────────────────────────────────────────────

class BaseDataModule:
    """
    Shared initialisation for all data module subclasses.

    Responsibilities
    ----------------
    - Snap spatial dims to the model's stride (multiple-of-32 requirement).
    - Build train/val transforms from ``ds_cfg`` via ``build_transforms``.
    - Provide a seeded ``_make_loader`` factory so every DataLoader gets
      reproducible worker seeds.
    """

    def __init__(self, config: dict):
        self.config = config
        ds_cfg      = config["dataset"]
        self._seed  = config.get("training", {}).get("seed", 42)

        # ── Stride alignment ─────────────────────────────────────────────
        raw_h, raw_w = ds_cfg["img_height"], ds_cfg["img_width"]
        h = _snap_to_stride(raw_h)
        w = _snap_to_stride(raw_w)

        if h != raw_h or w != raw_w:
            logger.info(
                f"Image size snapped to nearest multiple of {_MODEL_STRIDE}: "
                f"({raw_h}×{raw_w}) → ({h}×{w})."
            )
        else:
            logger.info(f"Image size {h}×{w} is aligned to stride {_MODEL_STRIDE}.")

        # Write resolved dims back so the rest of the pipeline sees them.
        ds_cfg["img_height"] = h
        ds_cfg["img_width"]  = w

        # ── Transforms ───────────────────────────────────────────────────
        self._train_tf, self._val_tf = build_transforms(h, w, ds_cfg)

        # ── DataLoader shared kwargs ──────────────────────────────────────
        self._ldr_kw = dict(
            batch_size     = ds_cfg["batch_size"],
            num_workers    = ds_cfg["num_workers"],
            pin_memory     = True,
            worker_init_fn = _make_worker_init_fn(self._seed),
        )
        # persistent_workers keeps worker processes (and their RNG state)
        # alive across epochs. Without it, workers respawn every epoch and
        # worker_init_fn reseeds them with the *same* fixed seed each time —
        # since Albumentations decides whether to apply a transform via the
        # bare global `random`/`np.random` calls, that means every epoch
        # applies the exact same sequence of augmentations to the same
        # samples, silently collapsing augmentation diversity across a run.
        if ds_cfg["num_workers"] > 0:
            self._ldr_kw["persistent_workers"] = True

        # ── Dataset construction kwargs (validate / cache) ────────────────
        self._ds_kwargs = dict(
            validate            = ds_cfg.get("validate", False),
            cache               = ds_cfg.get("cache", False),
            cache_size_limit_gb = ds_cfg.get("cache_size_limit_gb", 4.0),
        )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            shuffle   = shuffle,
            drop_last = shuffle and len(dataset) > self._ldr_kw["batch_size"],
            **self._ldr_kw,
        )


# ── Standard split ────────────────────────────────────────────────────────────

class StandardSplitDataModule(BaseDataModule):
    """
    Data module for datasets with pre-defined train/val/test splits.

    If the dataset name is not in the registry *and* ``dataset.split``
    ratios are configured, falls back to ``_GenericHandler`` which
    auto-partitions a flat ``images/`` + ``masks/`` directory.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        ds_cfg = config["dataset"]
        name   = ds_cfg["name"].lower()

        if name in DATASETS:
            self.handler = DATASETS[name](ds_cfg)
        else:
            logger.info(
                f"Dataset '{name}' not in registry; "
                "falling back to generic auto-split handler."
            )
            self.handler = _GenericHandler(ds_cfg, self._seed)

    def get_standard_loaders(self):
        """Return ``(train_loader, val_loader)``."""
        train_ds = self.handler.get_dataset("train", self._train_tf, **self._ds_kwargs)
        val_ds   = self.handler.get_dataset("val",   self._val_tf,   **self._ds_kwargs)
        return self._make_loader(train_ds, True), self._make_loader(val_ds, False)

    def get_test_loader(self):
        """Return the test DataLoader, or ``None`` if no test samples exist."""
        test_ds = self.handler.get_dataset("test", self._val_tf, **self._ds_kwargs)
        return self._make_loader(test_ds, False) if len(test_ds) > 0 else None


# ── K-Fold ────────────────────────────────────────────────────────────────────

class KFoldDataModule(BaseDataModule):
    """
    Data module for k-fold cross-validation.

    Fold assignments are serialised to ``<checkpoint_dir>/<exp>/fold_splits.json``
    on the first call to ``get_fold_loaders`` and reloaded on subsequent calls,
    so resumed runs always use the same split.

    CAUTION — frame-level splitting, no video/sequence grouping:
    Folds are drawn from ``sklearn.model_selection.KFold`` over individual
    (image, mask) pairs. CVC-ClinicDB and CVC-ColonDB are frame extracts
    from a small number of colonoscopy video sequences, so numerically
    nearby frames are often near-duplicates of the same polyp from the same
    sequence. A plain per-frame KFold can place near-duplicate frames in
    both the train and validation side of a fold, which would inflate
    validation Dice/mIoU relative to true generalisation. Fixing this
    properly needs a frame → source-video mapping (for
    ``sklearn.model_selection.GroupKFold``) that isn't recoverable from the
    filenames shipped with this dataset (they're flat sequential integers
    with no encoded sequence id) — so it isn't implemented here. If you
    have that mapping for your data, group-aware splitting should replace
    the plain ``KFold`` call in ``_load_or_create_fold_splits`` below.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        ds_cfg = config["dataset"]
        name   = ds_cfg["name"].lower()

        if name not in DATASETS:
            raise ValueError(
                f"KFoldDataModule requires a registered dataset handler. "
                f"Got '{name}'. Registered: {list(DATASETS)}. "
                f"K-Fold is not supported for generic auto-split datasets."
            )

        self.handler = DATASETS[name](ds_cfg)
        self.kf_cfg  = config.get("k_fold", {})

        exp_name        = config.get("logging", {}).get("experiment_name", "experiment")
        save_dir        = config.get("checkpoint", {}).get("save_dir", "checkpoints")
        self._fold_file = os.path.join(save_dir, exp_name, "fold_splits.json")

    def _load_or_create_fold_splits(self) -> list:
        """
        Load fold splits from disk if available, otherwise compute and persist them.
        Returns a list of dicts: ``[{'train': [[img, mask], ...], 'val': [...]}, ...]``.
        """
        n_splits = self.kf_cfg.get("n_splits", 5)

        if os.path.exists(self._fold_file):
            with open(self._fold_file) as f:
                cached = json.load(f)
            if cached.get("n_splits") == n_splits and cached.get("seed") == self._seed:
                return cached["folds"]
            logger.warning(
                f"Cached fold splits at {self._fold_file} were built with "
                f"n_splits={cached.get('n_splits')}, seed={cached.get('seed')}, but "
                f"the current config requests n_splits={n_splits}, seed={self._seed}. "
                "Regenerating fold splits — the old file will be overwritten. "
                "Any checkpoint resumed against a specific fold index may no "
                "longer correspond to the same data as before."
            )

        all_pairs = np.array(self.handler.get_kfold_pairs())
        if len(all_pairs) == 0:
            raise RuntimeError("No samples found for k-fold splitting.")

        # Frame-level split — no video/sequence grouping. See the class
        # docstring above: near-duplicate frames from the same source video
        # can land in both sides of a fold. Swap for GroupKFold if you have
        # a frame → video mapping for your data.
        kf    = KFold(n_splits=n_splits, shuffle=True, random_state=self._seed)
        folds = [
            {"train": all_pairs[ti].tolist(), "val": all_pairs[vi].tolist()}
            for ti, vi in kf.split(all_pairs)
        ]

        os.makedirs(os.path.dirname(self._fold_file), exist_ok=True)
        with open(self._fold_file, "w") as f:
            json.dump({"n_splits": n_splits, "seed": self._seed, "folds": folds}, f)

        return folds

    def get_fold_loaders(self, fold_idx: int):
        """Return ``(train_loader, val_loader)`` for *fold_idx*."""
        folds = self._load_or_create_fold_splits()
        if not (0 <= fold_idx < len(folds)):
            raise ValueError(
                f"fold_idx {fold_idx} out of range [0, {len(folds)})."
            )
        fold     = folds[fold_idx]
        train_ds = MedicalSegmentationDataset(
            pairs=fold["train"], transform=self._train_tf, **self._ds_kwargs
        )
        val_ds = MedicalSegmentationDataset(
            pairs=fold["val"], transform=self._val_tf, **self._ds_kwargs
        )
        return self._make_loader(train_ds, True), self._make_loader(val_ds, False)

    def get_test_loader(self):
        """Return the test DataLoader, or ``None`` if no test samples exist."""
        test_ds = self.handler.get_dataset("test", self._val_tf, **self._ds_kwargs)
        return self._make_loader(test_ds, False) if len(test_ds) > 0 else None
