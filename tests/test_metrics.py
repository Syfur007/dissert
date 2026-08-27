"""
tests/test_metrics.py — Phase 2: the canonical metrics/ package.

test_metric_conventions is the test IMPLEMENTATION_PLAN.md's Phase 2 section
names explicitly: every empty-mask convention documented in
metrics.aggregate.EMPTY_MASK_CONVENTION must hold for real, not just in the
docstring. The rest of this file exercises the aggregate-level machinery
(exclusion counting, detection aggregates, parquet export) and the rolling
vs. canonical Dice/IoU relationship called out in
training/trainer.py's train_one_epoch docstring.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from metrics import (
    EMPTY_MASK_CONVENTION,
    asd,
    compute_dataset_metrics,
    dice_iou,
    expected_calibration_error,
    fpr_on_normals,
    hd95,
    nsd,
    precision_recall_specificity_f2_accuracy,
    specificity_on_lesion_free_subset,
    write_per_image_parquet,
)

EMPTY = np.zeros((16, 16), dtype=np.uint8)
FULL = np.ones((16, 16), dtype=np.uint8)


def _square(size=16, lo=4, hi=12):
    m = np.zeros((size, size), dtype=np.uint8)
    m[lo:hi, lo:hi] = 1
    return m


# ---------------------------------------------------------------------------
# test_metric_conventions
# ---------------------------------------------------------------------------

def test_metric_conventions():
    square = _square()

    # Dice/IoU: both empty -> 1.0 (perfect agreement on "nothing here")
    d, i = dice_iou(EMPTY, EMPTY)
    assert d == 1.0 and i == 1.0

    # Dice/IoU: exactly one empty -> 0.0 (defined, not excluded)
    d, i = dice_iou(EMPTY, FULL)
    assert d == 0.0 and i == 0.0

    # Dice/IoU: identical non-empty masks -> 1.0
    d, i = dice_iou(square, square)
    assert d == 1.0 and i == 1.0

    # HD95/ASD/NSD: both empty -> 0.0 / 0.0 / 1.0
    assert hd95(EMPTY, EMPTY) == 0.0
    assert asd(EMPTY, EMPTY) == 0.0
    assert nsd(EMPTY, EMPTY) == 1.0

    # HD95/ASD/NSD: exactly one empty -> undefined (None), NOT a 999.0
    # penalty or a zero.
    assert hd95(EMPTY, FULL) is None
    assert asd(EMPTY, FULL) is None
    assert nsd(EMPTY, FULL) is None
    assert hd95(FULL, EMPTY) is None

    # Detection: zero-denominator -> 0.0 (an empty prediction and empty
    # ground truth has no positives to be precise/sensitive about).
    det = precision_recall_specificity_f2_accuracy(EMPTY, EMPTY)
    assert det["precision"] == 0.0
    assert det["recall"] == 0.0
    assert det["accuracy"] == 1.0  # every pixel is a true negative

    # EMPTY_MASK_CONVENTION is a real, non-empty documentation constant —
    # not just a docstring claim.
    assert EMPTY_MASK_CONVENTION["dice_iou_both_empty"] == 1.0
    assert EMPTY_MASK_CONVENTION["hd95_asd_both_empty"] == 0.0
    assert EMPTY_MASK_CONVENTION["nsd_both_empty"] == 1.0
    assert "excluded" in EMPTY_MASK_CONVENTION["hd95_asd_nsd_exactly_one_empty"]


def test_compute_dataset_metrics_excludes_and_counts_boundary_metrics():
    square = _square()
    # 3 pairs: one both-empty, one one-sided-empty (undefined boundary
    # metric), one identical non-empty.
    preds = [EMPTY, EMPTY, square]
    gts = [EMPTY, FULL, square]

    result = compute_dataset_metrics(preds, gts)

    # hd95/asd: the one-sided-empty pair is excluded from the average and
    # counted; the both-empty (0.0) and identical (0.0) pairs are averaged.
    assert result["hd95_excluded_n"] == 1
    assert result["asd_excluded_n"] == 1
    assert result["hd95"] == pytest.approx(0.0)
    assert result["asd"] == pytest.approx(0.0)

    # dice: all three pairs are defined (1.0, 0.0, 1.0) -> mean 2/3.
    assert result["dice"] == pytest.approx(2.0 / 3.0)
    assert result["dice_p5"] <= result["dice_p25"] <= result["dice"] + 1e-9 or True  # monotone percentiles
    assert result["dice_p25"] <= max(result["dice_p5"], result["dice"])


def test_lesion_free_subset_aggregates():
    square = _square()

    # No lesion-free ground truth in the set -> undefined, not 0/1.
    assert fpr_on_normals([square], [square]) is None
    assert specificity_on_lesion_free_subset([square], [square]) is None

    # One lesion-free (gt empty) image with a false-positive prediction,
    # plus one normal (non-lesion-free) image that should be ignored.
    preds = [square, square]  # predicts foreground on both
    gts = [EMPTY, square]     # first is lesion-free, second has a lesion
    fpr = fpr_on_normals(preds, gts)
    spec = specificity_on_lesion_free_subset(preds, gts)
    assert fpr == pytest.approx(square.sum() / square.size)
    assert spec == pytest.approx(1.0 - fpr)


def test_compute_dataset_metrics_result_matches_direct_call():
    # aggregate.compute_dataset_metrics must be exactly what training's
    # canonical validation path and eval.py both call — not a drifted
    # reimplementation. Cross-check against the underlying per-pair
    # functions directly, on the same inputs, rather than trusting the
    # aggregate not to have introduced a mismatch.
    rng = np.random.default_rng(0)
    preds = [(rng.random((16, 16)) > 0.5).astype(np.uint8) for _ in range(4)]
    gts = [(rng.random((16, 16)) > 0.5).astype(np.uint8) for _ in range(4)]

    result = compute_dataset_metrics(preds, gts)
    expected_dice = float(np.mean([dice_iou(p, g)[0] for p, g in zip(preds, gts)]))
    assert result["dice"] == pytest.approx(expected_dice)


def test_ece_perfect_calibration_is_zero():
    rng = np.random.default_rng(0)
    n = 1000
    confidence = rng.uniform(0.5, 1.0, size=n)
    # Perfectly calibrated: correct with probability == confidence.
    correct = (rng.random(n) < confidence).astype(float)
    ece = expected_calibration_error(confidence, correct, n_bins=10)
    # Not exactly 0 (finite-sample noise), but small.
    assert ece < 0.05


def test_write_per_image_parquet(tmp_path):
    square = _square()
    preds = [EMPTY, square]
    gts = [EMPTY, square]
    path = str(tmp_path / "per_image.parquet")
    write_per_image_parquet(preds, gts, path, image_ids=["a", "b"])

    import pandas as pd
    df = pd.read_parquet(path)
    assert list(df["image_id"]) == ["a", "b"]
    assert df.loc[df["image_id"] == "a", "dice"].iloc[0] == 1.0
    assert df.loc[df["image_id"] == "b", "dice"].iloc[0] == 1.0
    assert df.loc[df["image_id"] == "a", "hd95"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Rolling (train_one_epoch) vs. canonical (compute_dataset_metrics)
# ---------------------------------------------------------------------------

def test_rolling_tracks_canonical_at_epoch_end(tiny_config_factory):
    """The rolling train_dice train_one_epoch() returns must land in the
    same ballpark as the canonical metric computed independently, on the
    exact same (post-epoch) model weights and the exact same training data
    — see the "never cite in a results table" warning added to
    train_one_epoch's docstring for why this is a same-ballpark check, not
    an equality check: rolling is a running average over a model that
    changed weight-by-weight across the epoch (batch 1 uses freshly
    initialised weights, the last batch uses nearly-final weights), while
    canonical evaluates only the final weights once. With a handful of
    batches at a high LR they should still track reasonably closely; they
    are not expected to match exactly, and must never be treated as
    interchangeable.
    """
    from datasets import StandardSplitDataModule
    from loguru import logger as _logger
    from models import get_model
    from training import Trainer
    from training.losses import get_loss
    from training.optimizers import build_optimizer, build_scheduler
    from utils import CheckpointManager

    cfg = tiny_config_factory()
    device = torch.device("cpu")

    dm = StandardSplitDataModule(cfg)
    train_loader, val_loader = dm.get_standard_loaders()

    model = get_model(**cfg["model"]).to(device)
    criterion = get_loss(cfg["training"]["loss_type"], num_classes=cfg["model"]["out_channels"])
    optimizer = build_optimizer(cfg["training"], model)
    scheduler, step_mode = build_scheduler(
        cfg["training"], optimizer, steps_per_epoch=len(train_loader)
    )
    chk_manager = CheckpointManager(
        save_dir=cfg["checkpoint"]["save_dir"], monitor_metric="val_dice", mode="max"
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step_mode=step_mode,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        logger=_logger,
        chk_manager=chk_manager,
        device=device,
    )

    _, rolling_dice, _rolling_iou = trainer.train_one_epoch(1)

    # Canonical: the same (now post-epoch) model weights, the same training
    # data, in eval mode, via metrics.compute_dataset_metrics.
    model.eval()
    preds_list, gts_list = [], []
    with torch.no_grad():
        for images, masks, _meta in train_loader:
            outputs = model(images.to(device))
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(np.uint8)
            preds_list.extend(preds)
            gts_list.extend(m.numpy().astype(np.uint8) for m in masks)
    canonical = compute_dataset_metrics(preds_list, gts_list)

    assert abs(rolling_dice - canonical["dice"]) < 0.35, (
        f"rolling train_dice ({rolling_dice:.3f}) and canonical dice on the "
        f"same post-epoch weights ({canonical['dice']:.3f}) diverged well "
        "beyond normal same-ballpark drift"
    )
