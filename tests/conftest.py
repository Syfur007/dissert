"""
tests/conftest.py — shared pytest fixtures.

The fixtures here build a tiny *synthetic* image/mask dataset on disk (a
handful of small PNGs) rather than depending on the real (gitignored,
~1 GB) data/ directory, so the test suite runs anywhere this repo is
checked out — including CI, which never has data/ populated.
"""
from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest

from orchestration.schema import validate_config

N_TRAIN, N_VAL, N_TEST = 6, 2, 2
# Smallest size that's still a clean multiple of the model stride (32) —
# datasets/datamodule.py snaps to this stride regardless, so anything
# smaller would silently be upsized.
IMG_SIZE = 64


def _make_pair(rng: np.random.Generator, size: int = IMG_SIZE):
    image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    mask = (rng.random((size, size)) > 0.7).astype(np.uint8) * 255
    return image, mask


@pytest.fixture
def tiny_dataset_dir(tmp_path):
    """Build ``<tmp_path>/data/{images,masks}/*.png`` — the flat layout
    ``datasets.datamodule._GenericHandler`` auto-splits via
    ``dataset.split`` ratios. Returns the dataset root (a ``pathlib.Path``,
    i.e. the parent of ``images/``/``masks/``)."""
    root = tmp_path / "data"
    img_dir = root / "images"
    mask_dir = root / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    n_total = N_TRAIN + N_VAL + N_TEST
    for i in range(n_total):
        image, mask = _make_pair(rng)
        cv2.imwrite(str(img_dir / f"img_{i:03d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / f"img_{i:03d}.png"), mask)

    return root


@pytest.fixture
def tiny_config(tmp_path, tiny_dataset_dir):
    """A minimal, schema-valid resolved config pointing at
    ``tiny_dataset_dir`` — small enough to build+train+eval a real model in
    well under a second on CPU. Piped through
    ``orchestration.schema.validate_config`` so this fixture can never drift
    out of sync with the schema it's meant to exercise.

    Every path (checkpoints/logs/runs) lives under this test's own
    ``tmp_path`` — nothing here ever touches the real repo's
    checkpoints/logs/runs directories. Callers that need to change a value
    should ``copy.deepcopy()`` this first (it's a plain dict, shared
    mutable state across uses in the same test otherwise).
    """
    n_total = N_TRAIN + N_VAL + N_TEST
    raw = {
        "model": {
            "name": "unet",
            "in_channels": 3,
            "out_channels": 1,
            "features": [4, 8, 16, 32],
        },
        "dataset": {
            "name": "synthetic_test_dataset",
            "root": str(tiny_dataset_dir),
            "img_height": IMG_SIZE,
            "img_width": IMG_SIZE,
            "batch_size": 2,
            "num_workers": 0,
            "validate": False,
            "cache": False,
            "split": {
                "train": N_TRAIN / n_total,
                "val": N_VAL / n_total,
                "test": N_TEST / n_total,
            },
        },
        "training": {
            "epochs": 1,
            "lr": 0.01,
            "optimizer": "adamw",
            "loss_type": "dice",
            "device": "cpu",
            "seed": 42,
            "amp": False,
            "grad_clip_mode": "none",
        },
        "k_fold": {"enabled": False},
        "checkpoint": {
            "resume": False,
            "monitor_metric": "val_dice",
            "mode": "max",
        },
        "early_stopping": {"enabled": False},
        "stages": [],
        "logging": {
            "experiment_name": "pytest_synth",
            "save_overlays": False,
        },
        "output_dir": str(tmp_path / "outputs" / "experiments"),
    }
    return validate_config(raw)


@pytest.fixture
def tiny_config_factory(tiny_config):
    """Factory returning a fresh deep copy of ``tiny_config`` each call, for
    tests that need several independently-mutable configs (e.g. comparing
    two seeds)."""
    def _make():
        return copy.deepcopy(tiny_config)
    return _make
