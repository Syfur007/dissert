"""
train.py — Training entry-point (thin script).

Responsibilities:
  1. Parse CLI arguments and load YAML config.
  2. Apply CLI overrides to config.
  3. For each fold (or once for standard training):
       a. Build DataModule → loaders
       b. Build model
       c. Build criterion (training.losses.get_loss)
       d. Build optimizer / scheduler (training.optimizers)
       e. Build CheckpointManager, EarlyStopping, TensorBoardTracker
       f. Build EMA (optional)
       g. Build callbacks list
       h. Resume from checkpoint (restore model/opt/scheduler/EMA/RNG states)
       i. Trainer(...).fit(start_epoch)

All loss classes, training loops, and the old run_training() function now
live in the training/ package.  This file has no training logic.
"""

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch

from datasets import KFoldDataModule, StandardSplitDataModule
from models import get_model
from utils.config import load_config
from training import EMA, Trainer
from training.callbacks import (
    PeriodicCheckpointCallback,
    PredictionOverlayCallback,
    TensorBoardCallback,
    TrainingCurvePlotCallback,
)
from orchestration.runid import config_hash, run_id as compute_run_id
from training.determinism import reset_recorded_nondeterminism, seed_everything
from training.losses import get_loss
from training.optimizers import build_optimizer, build_scheduler
from utils import (
    CheckpointManager,
    EarlyStopping,
    TensorBoardTracker,
    setup_logger,
    log_model_summary,
)


# ---------------------------------------------------------------------------
# Per-fold training run
# ---------------------------------------------------------------------------

def run_training(config: dict, fold=None, run_id: Optional[str] = None) -> float:
    """Build all components and run Trainer.fit() for one fold (or non-CV run).

    Args:
        run_id: identifies this run for checkpoint provenance (embedded into
            every saved checkpoint alongside the config hash and current git
            commit — see utils.checkpoint.CheckpointManager.save). Computed
            from the config + seed + fold when not supplied, so a bare
            ``python train.py`` still produces addressable checkpoints, not
            just runs launched through orchestration.runner.

    Returns:
        Best monitored metric value.
    """
    training_cfg = config["training"]
    dataset_cfg  = config["dataset"]
    kfold_cfg    = config.get("k_fold", {})
    chk_cfg      = config.get("checkpoint", {})
    log_cfg      = config.get("logging", {})
    es_cfg       = config.get("early_stopping", {})

    device = torch.device(training_cfg["device"] if torch.cuda.is_available() else "cpu")
    reset_recorded_nondeterminism()
    seed_everything(training_cfg["seed"])

    # ── Run identity ───────────────────────────────────────────────────
    resolved_config_hash = config_hash(config)
    resolved_run_id = run_id or compute_run_id(
        resolved_config_hash, seed=training_cfg["seed"], fold=fold
    )

    # ── Logging ────────────────────────────────────────────────────────
    base_exp        = log_cfg['experiment_name']          # e.g. mkunet_t_clinicdb
    fold_prefix     = f"_fold{fold}" if fold is not None else ""
    experiment_name = f"{base_exp}{fold_prefix}"          # used for TB / checkpoints

    # Log filename is e.g. "fold0", "fold1", or base name for non-CV runs.
    # The directory is always logs/{base_exp}/ so all folds share one folder.
    log_filename = f"fold{fold}" if fold is not None else base_exp
    logger, exp_log_dir = setup_logger(log_cfg["log_dir"], base_exp, log_filename=log_filename)
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment log dir: {exp_log_dir}")

    # ── Data ───────────────────────────────────────────────────────────
    if fold is not None:
        logger.info(f"Initializing fold {fold}/{kfold_cfg['n_splits']-1} loaders...")
        dm = KFoldDataModule(config)
        train_loader, val_loader = dm.get_fold_loaders(fold)
    else:
        logger.info("Initializing standard train/val loaders...")
        dm = StandardSplitDataModule(config)
        train_loader, val_loader = dm.get_standard_loaders()

    logger.info(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    # ── Model ──────────────────────────────────────────────────────────
    model_cfg = config["model"]
    model     = get_model(**model_cfg).to(device)

    input_shape = (1, model_cfg["in_channels"], dataset_cfg["img_height"], dataset_cfg["img_width"])
    # model_summary.txt lands inside the experiment subdir
    log_model_summary(model, input_shape, logger, log_dir=exp_log_dir)

    # ── Loss ───────────────────────────────────────────────────────────
    loss_kwargs = training_cfg.get("loss_kwargs", {}) or {}
    criterion   = get_loss(
        training_cfg["loss_type"],
        num_classes=model_cfg["out_channels"],
        **loss_kwargs,
    ).to(device)

    # ── Optimizer / Scheduler ──────────────────────────────────────────
    optimizer                        = build_optimizer(training_cfg, model.parameters())
    scheduler, scheduler_step_mode   = build_scheduler(
        training_cfg, optimizer, steps_per_epoch=len(train_loader)
    )

    # ── AMP ────────────────────────────────────────────────────────────
    amp_requested = training_cfg.get("amp", False)
    use_amp       = amp_requested and device.type == "cuda"
    if amp_requested and not use_amp:
        logger.warning("AMP requested but CUDA not available; training in full precision.")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("AMP training enabled.")

    # Log gradient-clipping config
    gc_mode = training_cfg.get("grad_clip_mode", "value")
    if gc_mode == "value":
        logger.info(f"Grad clip | mode=value | value={training_cfg.get('grad_clip_value', 0.5)}")
    elif gc_mode == "norm":
        logger.info(f"Grad clip | mode=norm  | max_norm={training_cfg.get('grad_clip_norm', 1.0)}")
    else:
        logger.info("Grad clip disabled.")

    # ── Checkpoint & Early Stopping ────────────────────────────────────
    checkpoint_dir = os.path.join(
        chk_cfg.get("save_dir", "checkpoints"), log_cfg["experiment_name"]
    )
    chk_manager = CheckpointManager(
        save_dir        = checkpoint_dir,
        monitor_metric  = chk_cfg.get("monitor_metric", "val_dice"),
        mode            = chk_cfg.get("mode", "max"),
        min_delta       = es_cfg.get("min_delta", 0.0),
        config_hash     = resolved_config_hash,
        run_id          = resolved_run_id,
    )

    es_mode      = chk_cfg.get("mode", "max")
    early_stopper = None
    if es_cfg.get("enabled", False):
        early_stopper = EarlyStopping(
            patience  = es_cfg.get("patience", 20),
            min_delta = es_cfg.get("min_delta", 0.0),
            mode      = es_mode,
            verbose   = True,
        )
        logger.info(
            f"EarlyStopping | patience={early_stopper.patience} | "
            f"min_delta={early_stopper.min_delta} | mode={es_mode}"
        )

    # ── EMA ────────────────────────────────────────────────────────────
    ema_cfg = training_cfg.get("ema", {}) or {}
    ema     = None
    if ema_cfg.get("enabled", False):
        ema = EMA(model, decay=ema_cfg.get("decay", 0.9999))
        logger.info(f"EMA enabled | decay={ema.decay}")

    # ── TensorBoard tracker ────────────────────────────────────────────
    tracker = TensorBoardTracker(log_cfg["tb_dir"], experiment_name)

    # ── Callbacks ──────────────────────────────────────────────────────
    callbacks = [
        TensorBoardCallback(tracker),
    ]

    # Periodic epoch snapshots
    periodic_k = chk_cfg.get("periodic_save_every", 0)
    if periodic_k > 0:
        callbacks.append(
            PeriodicCheckpointCallback(
                save_every = periodic_k,
                save_dir   = checkpoint_dir,
                fold       = fold,
            )
        )

    # Prediction overlay visualisation
    overlay_save_every = log_cfg.get("overlay_save_every", 10)
    overlay_n_samples  = log_cfg.get("overlay_n_samples", 4)
    if log_cfg.get("save_overlays", True) and overlay_save_every > 0:
        overlay_dir = os.path.join(exp_log_dir, "overlays")
        callbacks.append(
            PredictionOverlayCallback(
                val_loader  = val_loader,
                device      = device,
                save_dir    = overlay_dir,
                n_samples   = overlay_n_samples,
                save_every  = overlay_save_every,
                tb_tracker  = tracker,
            )
        )
        logger.info(
            f"PredictionOverlay | every {overlay_save_every} epochs | "
            f"{overlay_n_samples} samples | dir={overlay_dir}"
        )

    # Offline training-curve plots (runs at the very end of training)
    callbacks.append(
        TrainingCurvePlotCallback(
            tb_log_dir = tracker.log_dir,   # the TensorBoard event directory
            out_dir    = exp_log_dir,
        )
    )

    # ── Resume ─────────────────────────────────────────────────────────
    start_epoch   = 1
    fold_suffix   = f"_fold{fold}" if fold is not None else ""

    if chk_cfg.get("resume", False):
        chk_path = chk_cfg.get("checkpoint_path") or os.path.join(
            checkpoint_dir, f"last{fold_suffix}.pth"
        )
        if os.path.exists(chk_path):
            start_epoch, loaded_metric, _ = chk_manager.load(
                chk_path, model, optimizer, scheduler, scaler=scaler
            )
            start_epoch += 1
            if loaded_metric is not None:
                chk_manager.best_metric = loaded_metric

            # The historical best score lives in best{fold_suffix}.pth, not
            # necessarily in whatever checkpoint we just resumed from (that's
            # usually last.pth, whose metric_val is only the last completed
            # epoch's score). Re-seed best_metric from the actual best
            # checkpoint so a resumed run can't mistake a mediocre epoch for
            # an improvement and overwrite a genuinely better checkpoint.
            best_path = os.path.join(checkpoint_dir, f"best{fold_suffix}.pth")
            if os.path.exists(best_path):
                best_ckpt = torch.load(best_path, map_location="cpu")
                if best_ckpt.get("metric_val") is not None:
                    chk_manager.best_metric = best_ckpt["metric_val"]
                    logger.info(
                        f"Resumed best_metric from {best_path}: {chk_manager.best_metric:.4f}"
                    )

            raw_ckpt = torch.load(chk_path, map_location="cpu")

            # Restore EarlyStopper
            if early_stopper is not None:
                es_state = raw_ckpt.get("early_stopper_state")
                if es_state is not None:
                    early_stopper.load_state_dict(es_state)
                    logger.info(
                        f"EarlyStopping restored | best={early_stopper.best_metric:.4f} | "
                        f"counter={early_stopper.counter}/{early_stopper.patience}"
                    )
                else:
                    early_stopper.restore(best_metric=loaded_metric)

            # Restore EMA shadow weights
            if ema is not None and raw_ckpt.get("ema_state"):
                ema.load_state_dict(raw_ckpt["ema_state"])
                logger.info("EMA state restored from checkpoint.")

            # Restore RNG states
            if raw_ckpt.get("rng_state_python") is not None:
                random.setstate(raw_ckpt["rng_state_python"])
            if raw_ckpt.get("rng_state_numpy") is not None:
                np.random.set_state(raw_ckpt["rng_state_numpy"])
            if raw_ckpt.get("rng_state_torch") is not None:
                torch.set_rng_state(raw_ckpt["rng_state_torch"])
            if raw_ckpt.get("rng_state_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(raw_ckpt["rng_state_cuda"])
            logger.info("RNG states restored from checkpoint.")
        else:
            logger.warning(f"No checkpoint at {chk_path}. Starting from scratch.")

    # ── Train ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model                = model,
        criterion            = criterion,
        optimizer            = optimizer,
        scheduler            = scheduler,
        scheduler_step_mode  = scheduler_step_mode,
        train_loader         = train_loader,
        val_loader           = val_loader,
        config               = config,
        logger               = logger,
        chk_manager          = chk_manager,
        device               = device,
        fold                 = fold,
        early_stopper        = early_stopper,
        callbacks            = callbacks,
        ema                  = ema,
        scaler               = scaler,
    )

    logger.info(f"Starting training from epoch {start_epoch}...")
    return trainer.fit(start_epoch=start_epoch)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train PyTorch Segmentation Pipeline")
    parser.add_argument("--config",          type=str,   default="configs/experiment/mkunet/mkunet_t_clinicdb.yaml")
    parser.add_argument("--fold",            type=int,   default=None,
                        help="Specific K-Fold index to train (0-indexed)")
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--batch-size",      type=int,   default=None)
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--model",           type=str,   default=None)
    parser.add_argument("--dataset_dir",     type=str,   default=None)
    parser.add_argument("--amp",             action="store_true")
    parser.add_argument("--grad-clip-mode",  type=str,   default=None,
                        choices=["value", "norm", "none"])
    parser.add_argument("--grad-clip-value", type=float, default=None)
    parser.add_argument("--grad-clip-norm",  type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.fold is not None:
        config["k_fold"]["enabled"] = True
    if args.resume:
        config["checkpoint"]["resume"] = True
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.model is not None:
        config["model"]["name"] = args.model
    if args.dataset_dir is not None:
        config["dataset"]["root"] = args.dataset_dir
    if args.amp:
        config["training"]["amp"] = True
    if args.grad_clip_mode is not None:
        config["training"]["grad_clip_mode"] = args.grad_clip_mode
    if args.grad_clip_value is not None:
        config["training"]["grad_clip_value"] = args.grad_clip_value
    if args.grad_clip_norm is not None:
        config["training"]["grad_clip_norm"] = args.grad_clip_norm

    kfold_cfg = config.get("k_fold", {})

    if kfold_cfg.get("enabled", False) and args.fold is None:
        n_splits   = kfold_cfg.get("n_splits", 5)
        run_folds  = kfold_cfg.get("run_folds") or list(range(n_splits))

        print(f"K-Fold Cross Validation ({n_splits} splits) over folds: {run_folds}")
        fold_results = []
        for f in run_folds:
            print(f"\n{'='*20} TRAINING FOLD {f} {'='*20}")
            try:
                fold_results.append((f, run_training(config, fold=f), None))
            except Exception as e:
                # Don't let one bad fold (e.g. a CUDA OOM) take down the
                # whole multi-hour sweep and lose every already-completed
                # fold's results — log it and move on, same as search.py
                # already does per-trial.
                print(f"Fold {f} failed with error: {e}")
                fold_results.append((f, None, str(e)))

        print(f"\n{'='*20} K-FOLD SUMMARY {'='*20}")
        print(f"Metric ({config['checkpoint']['monitor_metric']}) per fold:")
        successful_scores = []
        for f, score, error in fold_results:
            if error is None:
                print(f"  Fold {f}: {score:.4f}")
                successful_scores.append(score)
            else:
                print(f"  Fold {f}: FAILED — {error}")
        if successful_scores:
            print(
                f"Mean: {np.mean(successful_scores):.4f} ± {np.std(successful_scores):.4f} "
                f"(over {len(successful_scores)}/{len(run_folds)} successful folds)"
            )
        else:
            print("No folds completed successfully.")
    else:
        run_training(config, fold=args.fold)


if __name__ == "__main__":
    main()