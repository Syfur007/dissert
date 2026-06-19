"""
Unified data module for medical image segmentation.

Flow
----
Standard  : handler.get_dataset('train'/'val'/'test') → pre-defined split lists
K-Fold    : handler.get_kfold_pairs() → merge train+val → KFold → save to JSON
              On resume the JSON is re-loaded, so fold assignments are stable.
"""
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from .dataset import MedicalSegmentationDataset
from .transforms import get_train_transforms, get_val_transforms
from .polyp.clinicdb import ClinicDB
from .polyp.colondb import ColonDB

# ── Registry ────────────────────────────────────────────────────────────────
DATASETS: dict = {
    ClinicDB.NAME: ClinicDB,
    ColonDB.NAME:  ColonDB,
}


class SegmentationDataModule:
    """
    Unified data module. Selects the right dataset handler by ``config.dataset.name``.

    Public API (unchanged from before):
        get_standard_loaders()       → (train_loader, val_loader)
        get_fold_loaders(fold_idx)   → (train_loader, val_loader)
        get_test_loader()            → test_loader | None
    """

    def __init__(self, config: dict):
        self.config  = config
        ds_cfg       = config["dataset"]
        self.kf_cfg  = config.get("k_fold", {})

        name = ds_cfg["name"].lower()
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Registered: {list(DATASETS)}")

        self.handler = DATASETS[name](ds_cfg)

        h, w = ds_cfg["img_height"], ds_cfg["img_width"]
        self._train_tf = get_train_transforms(h, w)
        self._val_tf   = get_val_transforms(h, w)

        self._ldr_kw = dict(
            batch_size  = ds_cfg["batch_size"],
            num_workers = ds_cfg["num_workers"],
            pin_memory  = True,
        )

        # Path for persisting fold splits (enables training resume)
        exp_name       = config.get("logging", {}).get("experiment_name", "experiment")
        save_dir       = config.get("checkpoint", {}).get("save_dir", "checkpoints")
        self._fold_file = os.path.join(save_dir, exp_name, "fold_splits.json")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            shuffle   = shuffle,
            drop_last = shuffle and len(dataset) > self._ldr_kw["batch_size"],
            **self._ldr_kw,
        )

    def _load_or_create_fold_splits(self) -> list:
        """
        Load fold splits from disk if they exist, otherwise compute and save them.
        Returns a list of dicts: [{'train': [...], 'val': [...]}, ...]
        Each inner list contains [img_path, mask_path] pairs (lists, not tuples, for JSON).
        """
        if os.path.exists(self._fold_file):
            with open(self._fold_file) as f:
                return json.load(f)["folds"]

        n_splits = self.kf_cfg.get("n_splits", 5)
        seed     = self.config["training"].get("seed", 42)
        all_pairs = np.array(self.handler.get_kfold_pairs())  # shape (N, 2)

        if len(all_pairs) == 0:
            raise RuntimeError("No samples found for k-fold splitting.")

        kf     = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds  = [
            {"train": all_pairs[ti].tolist(), "val": all_pairs[vi].tolist()}
            for ti, vi in kf.split(all_pairs)
        ]

        os.makedirs(os.path.dirname(self._fold_file), exist_ok=True)
        with open(self._fold_file, "w") as f:
            json.dump({"n_splits": n_splits, "seed": seed, "folds": folds}, f)

        return folds

    # ── Public API ────────────────────────────────────────────────────────────

    def get_standard_loaders(self):
        """Return (train_loader, val_loader) using the dataset's pre-defined splits."""
        train_ds = self.handler.get_dataset("train", self._train_tf)
        val_ds   = self.handler.get_dataset("val",   self._val_tf)
        return self._make_loader(train_ds, True), self._make_loader(val_ds, False)

    def get_fold_loaders(self, fold_idx: int):
        """Return (train_loader, val_loader) for a specific k-fold index."""
        folds    = self._load_or_create_fold_splits()
        n_splits = self.kf_cfg.get("n_splits", len(folds))

        if not (0 <= fold_idx < len(folds)):
            raise ValueError(f"fold_idx {fold_idx} out of range for {n_splits} folds.")

        fold     = folds[fold_idx]
        train_ds = MedicalSegmentationDataset(pairs=fold["train"], transform=self._train_tf)
        val_ds   = MedicalSegmentationDataset(pairs=fold["val"],   transform=self._val_tf)
        return self._make_loader(train_ds, True), self._make_loader(val_ds, False)

    def get_test_loader(self):
        """Return the test DataLoader, or None if no test samples are available."""
        test_ds = self.handler.get_dataset("test", self._val_tf)
        return self._make_loader(test_ds, False) if len(test_ds) > 0 else None
