"""
tests/test_data_contract.py — Phase 3: (image, mask, meta) contract, leakage
guards, external-dataset guard, and the test-loader guard.

The four tests IMPLEMENTATION_PLAN.md names for this phase:
test_no_subject_overlap, test_external_never_trained, test_test_loader_guard,
test_sweep_cannot_see_test.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from datasets.splits import (
    ExternalDatasetError,
    LeakageError,
    TestLoaderGuardError,
    assert_no_subject_overlap,
    duplicate_cross_check,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# (image, mask, meta) contract
# ---------------------------------------------------------------------------

def test_getitem_returns_image_mask_meta(tiny_dataset_dir):
    from datasets.dataset import METADATA_KEYS, MedicalSegmentationDataset

    img_dir = tiny_dataset_dir / "images"
    mask_dir = tiny_dataset_dir / "masks"
    names = sorted(p.name for p in img_dir.iterdir())

    ds = MedicalSegmentationDataset(
        str(img_dir), str(mask_dir), filenames=names,
        source_dataset="synthetic_test_dataset",
        artefact_flags={"frame_level_only_no_video_grouping": True},
    )
    item = ds[0]
    assert len(item) == 3
    image, mask, meta = item
    assert set(METADATA_KEYS) <= set(meta.keys())
    assert meta["source_dataset"] == "synthetic_test_dataset"
    assert meta["artefact_flags"] == {"frame_level_only_no_video_grouping": True}
    assert meta["subject_id"] == Path(names[0]).stem
    assert meta["spacing"] == ()  # not None — see dataset.py's collate note


def test_getitem_collates_through_a_real_dataloader(tiny_config):
    # The actual risk this contract change carries: torch's default
    # collate_fn choking on the new meta dict (e.g. a bare None inside it —
    # confirmed it does, hence dataset.py uses () for unset spacing, not
    # None). Exercise a real DataLoader, not just __getitem__ in isolation.
    from datasets import StandardSplitDataModule

    dm = StandardSplitDataModule(tiny_config)
    train_loader, _ = dm.get_standard_loaders()
    images, masks, meta = next(iter(train_loader))
    assert images.ndim == 4
    assert masks.ndim == 4
    assert "subject_id" in meta
    assert len(meta["subject_id"]) == images.shape[0]


def test_all_four_consumers_accept_the_new_contract():
    """Static check that no live code still unpacks a DataLoader batch as a
    bare 2-tuple — this is the "single coordinated change" the plan
    requires (a half-migrated state fails on the very first batch, in
    whichever consumer got missed). Parses each file's AST rather than
    grepping so this can't be fooled by a comment mentioning the old shape.
    """
    consumers = [
        "eval.py",
        "utils/metrics.py",
        "training/trainer.py",
        "training/callbacks.py",
    ]
    for rel_path in consumers:
        source = (REPO_ROOT / rel_path).read_text()
        tree = ast.parse(source, filename=rel_path)
        for node in ast.walk(tree):
            # for images, masks in X:  ->  For.target is a Tuple of 2 Names
            if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
                names = [
                    elt.id for elt in node.target.elts if isinstance(elt, ast.Name)
                ]
                if names[:2] == ["images", "masks"] or (
                    len(names) == 2 and names[0] == "images"
                ):
                    assert len(node.target.elts) == 3, (
                        f"{rel_path}: found a 2-tuple DataLoader unpack "
                        f"{names} — should be a 3-tuple (image, mask, meta)"
                    )


# ---------------------------------------------------------------------------
# test_no_subject_overlap
# ---------------------------------------------------------------------------

def test_no_subject_overlap():
    # Passes on disjoint sets.
    assert_no_subject_overlap(["a", "b"], ["c", "d"], ["e", "f"])

    # Raises LeakageError the moment any pair of splits shares a subject.
    with pytest.raises(LeakageError):
        assert_no_subject_overlap(["a", "b"], ["b", "c"], ["d"])
    with pytest.raises(LeakageError):
        assert_no_subject_overlap(["a"], ["b"], ["a"])


def test_no_subject_overlap_against_real_clinicdb_kfold_split():
    """Exercise the guard against a real dataset handler's actual fold
    assignment, not just synthetic id lists — this is the point the plan
    makes explicitly: a leakage guard is meaningless if only tested against
    made-up data. ClinicDB's subject_id is frame-level (== filename stem),
    so this is trivially satisfied (KFold partitions indices, and each
    index has a unique filename/subject_id) — the value here is regression
    protection, and proving the guard runs cleanly against the real handler
    interface end to end.
    """
    from datasets.polyp.clinicdb import ClinicDB

    root = REPO_ROOT / "data" / "polyp" / "ClinicDB"
    if not root.exists():
        pytest.skip("data/polyp/ClinicDB not present in this environment")

    handler = ClinicDB({"root": str(root)})
    pairs = handler.get_kfold_pairs()
    if not pairs:
        pytest.skip("no ClinicDB train/val pairs found")

    from datasets.dataset import _default_subject_id_fn
    from sklearn.model_selection import KFold

    ids = [_default_subject_id_fn(img_path) for img_path, _ in pairs]
    train_idx, val_idx = next(KFold(n_splits=5, shuffle=True, random_state=42).split(ids))
    train_ids = [ids[i] for i in train_idx]
    val_ids = [ids[i] for i in val_idx]

    assert_no_subject_overlap(train_ids, val_ids)  # must not raise


def test_duplicate_cross_check(tmp_path):
    # Two distinct paths pointing at byte-identical content, placed in
    # different splits -> flagged, even though the filenames differ.
    (tmp_path / "a.png").write_bytes(b"identical-bytes")
    (tmp_path / "b.png").write_bytes(b"identical-bytes")
    (tmp_path / "c.png").write_bytes(b"different-bytes")

    pairs_by_split = {
        "train": [(str(tmp_path / "a.png"), "mask_a")],
        "test": [(str(tmp_path / "b.png"), "mask_b"), (str(tmp_path / "c.png"), "mask_c")],
    }
    dupes = duplicate_cross_check(pairs_by_split)
    assert len(dupes) == 1
    (entries,) = dupes.values()
    splits_involved = {split for split, _, _ in entries}
    assert splits_involved == {"train", "test"}


# ---------------------------------------------------------------------------
# test_external_never_trained
# ---------------------------------------------------------------------------

def test_external_never_trained(tiny_dataset_dir):
    from datasets.datamodule import _GenericHandler

    cfg = {"name": "synthetic_external", "root": str(tiny_dataset_dir)}
    handler = _GenericHandler(cfg, seed=0, external=True)

    with pytest.raises(ExternalDatasetError):
        handler.get_dataset("train")
    with pytest.raises(ExternalDatasetError):
        handler.get_dataset("val")

    # "test" must still work, and must contain every sample (external ==
    # held-out-only, not held-out-and-discarded).
    test_ds = handler.get_dataset("test")
    all_images = list((tiny_dataset_dir / "images").iterdir())
    assert len(test_ds) == len(all_images)


def test_external_flows_through_standard_split_datamodule(tmp_path, tiny_dataset_dir):
    from datasets import StandardSplitDataModule
    from datasets.splits import ExternalDatasetError
    from orchestration.schema import validate_config

    raw = {
        "model": {"name": "unet", "in_channels": 3, "out_channels": 1, "features": [4, 8, 16, 32]},
        "dataset": {
            "name": "synthetic_external_ds",
            "root": str(tiny_dataset_dir),
            "img_height": 64, "img_width": 64, "batch_size": 2, "num_workers": 0,
            "external": True,
        },
        "training": {"epochs": 1, "lr": 0.01, "loss_type": "dice", "device": "cpu", "seed": 42},
        "k_fold": {"enabled": False},
        "checkpoint": {"save_dir": str(tmp_path / "ckpt"), "resume": False},
        "logging": {"log_dir": str(tmp_path / "logs"), "tb_dir": str(tmp_path / "runs"),
                    "experiment_name": "ext"},
    }
    cfg = validate_config(raw)
    dm = StandardSplitDataModule(cfg)
    with pytest.raises(ExternalDatasetError):
        dm.get_standard_loaders()


# ---------------------------------------------------------------------------
# test_test_loader_guard
# ---------------------------------------------------------------------------

def test_test_loader_guard(tiny_config):
    from datasets import StandardSplitDataModule
    from orchestration.ledger import LedgerWriter

    dm = StandardSplitDataModule(tiny_config)
    ledger_dir = tiny_config["checkpoint"]["save_dir"] + "_ledger"

    # No token / empty token -> refused.
    with pytest.raises(TestLoaderGuardError):
        dm.get_test_loader("", ledger_dir=ledger_dir)
    with pytest.raises(TestLoaderGuardError):
        dm.get_test_loader("not-a-real-token", ledger_dir=ledger_dir)

    # A token minted by issue_test_token() is accepted.
    token = LedgerWriter(ledger_dir).issue_test_token(run_id="r1", config_hash="h1")
    loader = dm.get_test_loader(token, ledger_dir=ledger_dir)
    assert loader is not None
    assert len(loader.dataset) > 0

    # The mint is recorded, not just accepted silently.
    assert LedgerWriter(ledger_dir).has_test_token(token)


# ---------------------------------------------------------------------------
# test_sweep_cannot_see_test
# ---------------------------------------------------------------------------

def test_sweep_cannot_see_test():
    """Static guarantee: nothing on the training/sweep path can reach the
    guarded test loader — train.py, training/trainer.py, search.py, and
    orchestration/runner.py must contain zero references to
    get_test_loader. (eval.py and datasets/datamodule.py are the only two
    legitimate references — the guard's definition and its one guarded call
    site.) A grep-able static fact, not a runtime behaviour, but exactly
    what Phase 14's later "re-verified static-import guarantee" builds on.
    """
    sweep_path_files = [
        "train.py",
        "training/trainer.py",
        "search.py",
        "orchestration/runner.py",
    ]
    for rel_path in sweep_path_files:
        source = (REPO_ROOT / rel_path).read_text()
        assert "get_test_loader" not in source, (
            f"{rel_path} references get_test_loader — the sweep/training "
            "path must never be able to reach the guarded test set"
        )


# ---------------------------------------------------------------------------
# BUSI / ISIC18 handlers — "the concrete proof the retrofit generalizes"
# (IMPLEMENTATION_PLAN.md). No real BUSI/ISIC18 data exists in this
# environment, so these exercise the handler *interface* (get_dataset /
# get_kfold_pairs, directory-layout parsing, mask-pairing) against synthetic
# data built to match each dataset's publicly documented layout — see
# datasets/busi.py and datasets/isic18.py's module docstrings for the
# "unverified against real data" caveat this doesn't lift.
# ---------------------------------------------------------------------------

def _write_pair(img_path, mask_path, rng, size=32):
    import cv2
    img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    mask = (rng.random((size, size)) > 0.6).astype(np.uint8) * 255
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(mask_path), mask)


def test_busi_handler_interface(tmp_path):
    import numpy as np

    from datasets.busi import BUSI

    root = tmp_path / "busi"
    rng = np.random.default_rng(0)
    counts = {"benign": 5, "malignant": 4, "normal": 3}
    for cls, n in counts.items():
        (root / cls).mkdir(parents=True)
        for i in range(n):
            _write_pair(root / cls / f"{cls} ({i}).png", root / cls / f"{cls} ({i})_mask.png", rng)

    handler = BUSI(
        {"root": str(root), "split": {"train": 0.6, "val": 0.2, "test": 0.2}, "dedup": False},
        seed=0,
    )
    train_ds = handler.get_dataset("train")
    val_ds = handler.get_dataset("val")
    test_ds = handler.get_dataset("test")
    total = sum(counts.values())
    assert len(train_ds) + len(val_ds) + len(test_ds) == total

    _, _, meta = train_ds[0]
    assert meta["source_dataset"] == "busi"
    assert len(handler.get_kfold_pairs()) == len(train_ds) + len(val_ds)


def test_busi_dedup_is_mandatory_and_excludes_duplicates(tmp_path):
    import numpy as np

    from datasets.busi import BUSI

    root = tmp_path / "busi"
    (root / "benign").mkdir(parents=True)
    rng = np.random.default_rng(1)
    import cv2
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    mask = (rng.random((64, 64)) > 0.6).astype(np.uint8) * 255
    cv2.imwrite(str(root / "benign" / "benign (0).png"), img)
    cv2.imwrite(str(root / "benign" / "benign (0)_mask.png"), mask)
    cv2.imwrite(str(root / "benign" / "benign (1).png"), img)  # exact duplicate
    cv2.imwrite(str(root / "benign" / "benign (1)_mask.png"), mask)

    handler = BUSI(
        {"root": str(root), "split": {"train": 0.5, "val": 0.0, "test": 0.5}, "dedup": True},
        seed=0,
    )
    handler._build_splits(0)
    total = sum(len(v) for v in handler._splits.values())
    assert total == 1  # one of the identical pair excluded


def test_isic18_handler_interface(tmp_path):
    import numpy as np

    from datasets.isic18 import ISIC18

    root = tmp_path / "isic18"
    dirs = {
        "train": ("ISIC2018_Task1-2_Training_Input", "ISIC2018_Task1_Training_GroundTruth"),
        "val": ("ISIC2018_Task1-2_Validation_Input", "ISIC2018_Task1_Validation_GroundTruth"),
        "test": ("ISIC2018_Task1-2_Test_Input", "ISIC2018_Task1_Test_GroundTruth"),
    }
    rng = np.random.default_rng(0)
    counts = {"train": 4, "val": 2, "test": 2}
    import cv2
    for split, (img_dirname, mask_dirname) in dirs.items():
        (root / img_dirname).mkdir(parents=True)
        (root / mask_dirname).mkdir(parents=True)
        for i in range(counts[split]):
            img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            mask = (rng.random((32, 32)) > 0.6).astype(np.uint8) * 255
            cv2.imwrite(str(root / img_dirname / f"ISIC_{i:07d}.jpg"), img)
            cv2.imwrite(str(root / mask_dirname / f"ISIC_{i:07d}_segmentation.png"), mask)

    handler = ISIC18({"root": str(root)}, seed=0)
    for split, n in counts.items():
        assert len(handler.get_dataset(split)) == n

    _, _, meta = handler.get_dataset("test")[0]
    assert meta["source_dataset"] == "isic18"
    assert meta["subject_id"] == "ISIC_0000000"
    assert len(handler.get_kfold_pairs()) == counts["train"] + counts["val"]
