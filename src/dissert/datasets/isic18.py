"""
datasets/isic18.py — ISIC 2018 (Task 1-2, lesion segmentation) dataset
handler.

**Structurally complete but UNVERIFIED against real data** — no ISIC18
files exist in this environment (or anywhere reachable from it) to test
against. Built against the challenge's publicly documented official
directory layout:

    <root>/ISIC2018_Task1-2_Training_Input/       ISIC_XXXXXXX.jpg
    <root>/ISIC2018_Task1_Training_GroundTruth/    ISIC_XXXXXXX_segmentation.png
    <root>/ISIC2018_Task1-2_Validation_Input/
    <root>/ISIC2018_Task1_Validation_GroundTruth/
    <root>/ISIC2018_Task1-2_Test_Input/
    <root>/ISIC2018_Task1_Test_GroundTruth/

Unlike BUSI, ISIC18 ships an **official** train/val/test split (this is
the split every published benchmark number against ISIC18 uses) — the
challenge's own directory partition IS the published split, used directly
rather than re-derived from a random seed. Sanity-check the exact directory
names and the ``_segmentation`` mask-filename suffix against your actual
download before first real use — see IMPLEMENTATION_PLAN.md's Phase 3
section and CHANGELOG.md's Phase 3 entry.

Config keys (under ``dataset``):
    root : str — path to the dataset root (default: data/isic18)
    Per-split image/mask directory overrides (train_img_dir, train_mask_dir,
    val_img_dir, val_mask_dir, test_img_dir, test_mask_dir) — for a layout
    that doesn't match the official directory names above.
"""
import os

from .dataset import MedicalSegmentationDataset

_DEFAULT_DIRS = {
    "train": (
        "ISIC2018_Task1-2_Training_Input",
        "ISIC2018_Task1_Training_GroundTruth",
    ),
    "val": (
        "ISIC2018_Task1-2_Validation_Input",
        "ISIC2018_Task1_Validation_GroundTruth",
    ),
    "test": (
        "ISIC2018_Task1-2_Test_Input",
        "ISIC2018_Task1_Test_GroundTruth",
    ),
}


class ISIC18:
    NAME = "isic18"
    # One image == one lesion case in ISIC18's design (no multi-frame/
    # multi-visit relationship the way ClinicDB/ColonDB have) — no
    # frame-level-identity artefact flag needed the way those two carry one.
    ARTEFACT_FLAGS = {}

    def __init__(self, cfg: dict, seed: int = 42):
        # seed is part of the handler interface (datasets/datamodule.py's
        # DATASETS registry) but unused here — ISIC18's split is the
        # challenge's own official directory partition, not randomly drawn.
        root = cfg.get("root", "data/isic18")
        self._splits = {}
        for split, (default_img_dirname, default_mask_dirname) in _DEFAULT_DIRS.items():
            img_dir = cfg.get(f"{split}_img_dir", os.path.join(root, default_img_dirname))
            mask_dir = cfg.get(f"{split}_mask_dir", os.path.join(root, default_mask_dirname))
            self._splits[split] = (img_dir, mask_dir)

    def _resolve_pairs(self, split: str):
        """Build explicit (img_path, mask_path) pairs rather than relying on
        MedicalSegmentationDataset's built-in filename-pairing (which only
        tolerates a `_mask`/`_gt` suffix) — ISIC18's official
        `<stem>_segmentation.png` mask-naming convention needs its own
        pairing logic, kept local to this handler rather than widening the
        shared suffix-tolerance list for every other dataset."""
        img_dir, mask_dir = self._splits[split]
        if not os.path.exists(img_dir):
            return []
        pairs = []
        for fname in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, fname)
            if not os.path.isfile(img_path):
                continue
            stem = os.path.splitext(fname)[0]
            mask_path = os.path.join(mask_dir, f"{stem}_segmentation.png")
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
        return pairs

    def get_dataset(self, split: str, transform=None, **kwargs) -> MedicalSegmentationDataset:
        pairs = self._resolve_pairs(split)
        kwargs.setdefault("source_dataset", self.NAME)
        kwargs.setdefault("artefact_flags", self.ARTEFACT_FLAGS)
        return MedicalSegmentationDataset(pairs=pairs, transform=transform, **kwargs)

    def get_kfold_pairs(self) -> list:
        """train+val pairs only — test stays held out, same contract as
        datasets/polyp/{clinicdb,colondb}.py. K-Fold CV on top of an
        officially-split benchmark dataset is an unusual thing to want (the
        official split exists specifically so published numbers are
        comparable) — supported here for consistency with the other
        handlers, not because it's the typical way to use ISIC18."""
        pairs = []
        for split in ("train", "val"):
            pairs.extend(self.get_dataset(split).pairs)
        return pairs
