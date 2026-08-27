import os
from ..dataset import MedicalSegmentationDataset


class ClinicDB:
    """
    ClinicDB polyp dataset handler.

    Expected layout:
        <root>/train/{images, masks}/
        <root>/val/{images, masks}/
        <root>/test/{images, masks}/
        <root>/train_list_clinicdb.txt   (optional — falls back to scanning the directory)
        <root>/val_list_clinicdb.txt
        <root>/test_list_clinicdb.txt

    Config keys (under ``dataset``):
        root        : str  — path to the dataset root (default: data/polyp/ClinicDB)
        train_list  : str  — override path for the train list file
        val_list    : str  — override path for the val list file
        test_list   : str  — override path for the test list file
    """

    NAME = "clinicdb"

    # See datasets/dataset.py's _default_subject_id_fn / METADATA_KEYS:
    # filenames are flat sequential integers with no recoverable
    # source-video id, so subject_id honestly falls back to frame-level
    # identity — this flag is that caveat's machine-readable home (was
    # previously only in KFoldDataModule's class docstring).
    ARTEFACT_FLAGS = {"frame_level_only_no_video_grouping": True}

    def __init__(self, cfg: dict, seed: int = 42):
        # seed is part of the handler interface (datasets/datamodule.py's
        # DATASETS registry) but unused here — ClinicDB ships pre-made
        # train/val/test lists, nothing to randomly split.
        root = cfg.get("root", "data/polyp/ClinicDB")
        # (image_dir, mask_dir, list_file) per split
        self._splits = {
            "train": (
                os.path.join(root, "train", "images"),
                os.path.join(root, "train", "masks"),
                cfg.get("train_list", os.path.join(root, "train_list_clinicdb.txt")),
            ),
            "val": (
                os.path.join(root, "val", "images"),
                os.path.join(root, "val", "masks"),
                cfg.get("val_list", os.path.join(root, "val_list_clinicdb.txt")),
            ),
            "test": (
                os.path.join(root, "test", "images"),
                os.path.join(root, "test", "masks"),
                cfg.get("test_list", os.path.join(root, "test_list_clinicdb.txt")),
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_files(self, split: str):
        """Return (img_dir, mask_dir, [filenames]) for a given split."""
        img_dir, mask_dir, list_path = self._splits[split]
        if list_path and os.path.exists(list_path):
            with open(list_path) as f:
                names = [line.strip() for line in f if line.strip()]
        elif os.path.exists(img_dir):
            names = sorted(os.listdir(img_dir))
        else:
            names = []
        return img_dir, mask_dir, names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataset(self, split: str, transform=None, **kwargs) -> MedicalSegmentationDataset:
        """Return a ``MedicalSegmentationDataset`` for *split* ('train'/'val'/'test')."""
        img_dir, mask_dir, names = self._resolve_files(split)
        kwargs.setdefault("source_dataset", self.NAME)
        kwargs.setdefault("artefact_flags", self.ARTEFACT_FLAGS)
        return MedicalSegmentationDataset(img_dir, mask_dir, filenames=names, transform=transform, **kwargs)

    def get_kfold_pairs(self) -> list:
        """
        Return all (img_path, mask_path) pairs from train+val splits.
        Used by the datamodule to build k-fold cross-validation splits.
        The test split is intentionally excluded so it remains a held-out set.

        Delegates to get_dataset() rather than pairing filenames directly,
        so k-fold splitting shares the same mask-filename resolution
        (case/suffix tolerant — handles `_mask`/`_gt` suffixes etc.) as every
        other code path instead of assuming image and mask filenames match
        byte-for-byte.
        """
        pairs = []
        for split in ("train", "val"):
            pairs.extend(self.get_dataset(split).pairs)
        return pairs
