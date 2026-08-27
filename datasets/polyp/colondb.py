import os
from ..dataset import MedicalSegmentationDataset


class ColonDB:
    """
    ColonDB polyp dataset handler.

    Expected layout:
        <root>/train/{images, masks}/
        <root>/val/{images, masks}/
        <root>/test/{images, masks}/
        <root>/train_list_colondb.txt   (optional — falls back to scanning the directory)
        <root>/val_list_colondb.txt
        <root>/test_list_colondb.txt

    Config keys (under ``dataset``):
        root        : str  — path to the dataset root (default: data/polyp/ColonDB)
        train_list  : str  — override path for the train list file
        val_list    : str  — override path for the val list file
        test_list   : str  — override path for the test list file
    """

    NAME = "colondb"

    def __init__(self, cfg: dict):
        root = cfg.get("root", "data/polyp/ColonDB")
        # (image_dir, mask_dir, list_file) per split
        self._splits = {
            "train": (
                os.path.join(root, "train", "images"),
                os.path.join(root, "train", "masks"),
                cfg.get("train_list", os.path.join(root, "train_list_colondb.txt")),
            ),
            "val": (
                os.path.join(root, "val", "images"),
                os.path.join(root, "val", "masks"),
                cfg.get("val_list", os.path.join(root, "val_list_colondb.txt")),
            ),
            "test": (
                os.path.join(root, "test", "images"),
                os.path.join(root, "test", "masks"),
                cfg.get("test_list", os.path.join(root, "test_list_colondb.txt")),
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
