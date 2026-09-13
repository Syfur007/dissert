"""
tests/test_fold_splits.py — KFoldDataModule's shared, content-addressed
fold partition.

The partition file is keyed by config_hash rather than by seed, which is
what makes every seed of a multi-seed sweep train on the same split and
what makes a resumed run reuse the split its checkpoint was trained
against. Both properties are silent when they break — the run still
completes, the numbers are just wrong — so they are asserted here directly.

KFoldDataModule requires a *registered* dataset handler (K-Fold is
deliberately unsupported for generic auto-split directories), and the
synthetic fixture dataset is not registered. These tests therefore register
a minimal handler over the fixture's files for the duration of each test,
which is enough to exercise every code path in
``_load_or_create_fold_splits``.
"""
from __future__ import annotations

import copy
import json
import os

import pytest

from datasets.datamodule import DATASETS, KFoldDataModule
from datasets.splits import FoldSplitDriftError
from orchestration.runid import config_hash, experiment_paths


class _FakeKFoldHandler:
    """Minimal registered-handler stand-in: the only method
    ``_load_or_create_fold_splits`` needs is ``get_kfold_pairs``."""

    NAME = "kfold_test_dataset"
    ARTEFACT_FLAGS: dict = {}

    def __init__(self, cfg: dict, seed: int):
        root = cfg["root"]
        img_dir = os.path.join(root, "images")
        mask_dir = os.path.join(root, "masks")
        self._pairs = [
            [os.path.join(img_dir, name), os.path.join(mask_dir, name)]
            for name in sorted(os.listdir(img_dir))
        ]

    def get_kfold_pairs(self):
        return self._pairs

    def get_dataset(self, split, transform=None, **kwargs):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def kfold_config(tiny_config, monkeypatch):
    """tiny_config rewired onto a registered K-Fold-capable handler."""
    monkeypatch.setitem(DATASETS, _FakeKFoldHandler.NAME, _FakeKFoldHandler)

    cfg = copy.deepcopy(tiny_config)
    cfg["dataset"]["name"] = _FakeKFoldHandler.NAME
    cfg["k_fold"] = {"enabled": True, "n_splits": 2}
    return cfg


def _fold_file(config: dict) -> str:
    return experiment_paths(
        config["output_dir"], config["logging"]["experiment_name"],
        config_hash(config), config["training"]["seed"],
    )["fold_splits"]


def test_fold_splits_shared_across_seeds(kfold_config):
    """Different seeds, one partition. Only model-init/training randomness
    may vary across the seeds of a sweep — never the data split itself, or
    the seeds are not comparable and averaging them is meaningless."""
    cfg_a = copy.deepcopy(kfold_config)
    cfg_a["training"]["seed"] = 7
    cfg_b = copy.deepcopy(kfold_config)
    cfg_b["training"]["seed"] = 1337

    folds_a = KFoldDataModule(cfg_a)._load_or_create_fold_splits()
    folds_b = KFoldDataModule(cfg_b)._load_or_create_fold_splits()

    assert _fold_file(cfg_a) == _fold_file(cfg_b)
    assert folds_a == folds_b

    exp_root = os.path.dirname(_fold_file(cfg_a))
    written = [p for p in os.listdir(exp_root) if p.endswith("-fold_splits.json")]
    assert len(written) == 1


def test_fold_splits_not_rewritten_on_resume(kfold_config):
    """A resumed run reuses the cached partition byte-for-byte — asserted on
    mtime, since a rewrite that happened to produce identical content would
    still mean the guarantee doesn't hold."""
    dm = KFoldDataModule(kfold_config)
    dm._load_or_create_fold_splits()

    path = _fold_file(kfold_config)
    before = os.stat(path).st_mtime_ns

    resumed = copy.deepcopy(kfold_config)
    resumed["checkpoint"]["resume"] = True
    folds = KFoldDataModule(resumed)._load_or_create_fold_splits()

    assert os.stat(path).st_mtime_ns == before
    assert folds == dm._load_or_create_fold_splits()


def test_resume_flag_does_not_change_the_partition_path(kfold_config):
    """checkpoint.resume is stripped from config_hash, which is *why* a
    resumed run finds the same file."""
    resumed = copy.deepcopy(kfold_config)
    resumed["checkpoint"]["resume"] = True
    assert _fold_file(kfold_config) == _fold_file(resumed)


def test_drift_raises_on_resume(kfold_config):
    """A cached partition that doesn't match the config is unreachable
    short of a truncated-hash collision — but if it happened under a resume
    the run would train on data its checkpoint never saw, so it must fail
    loudly rather than regenerate."""
    KFoldDataModule(kfold_config)._load_or_create_fold_splits()

    path = _fold_file(kfold_config)
    with open(path) as f:
        cached = json.load(f)
    cached["config_hash"] = "0" * 40
    with open(path, "w") as f:
        json.dump(cached, f)

    resumed = copy.deepcopy(kfold_config)
    resumed["checkpoint"]["resume"] = True
    with pytest.raises(FoldSplitDriftError):
        KFoldDataModule(resumed)._load_or_create_fold_splits()


def test_drift_still_regenerates_without_resume(kfold_config):
    """The pre-existing warn-and-regenerate escape hatch is untouched for a
    run that has no checkpoint to contradict."""
    KFoldDataModule(kfold_config)._load_or_create_fold_splits()

    path = _fold_file(kfold_config)
    with open(path) as f:
        cached = json.load(f)
    cached["config_hash"] = "0" * 40
    with open(path, "w") as f:
        json.dump(cached, f)

    fresh = copy.deepcopy(kfold_config)
    fresh["checkpoint"]["resume"] = False
    folds = KFoldDataModule(fresh)._load_or_create_fold_splits()

    assert len(folds) == 2
    with open(path) as f:
        assert json.load(f)["config_hash"] == config_hash(fresh)
