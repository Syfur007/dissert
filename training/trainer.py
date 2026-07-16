"""
training/trainer.py — Core Trainer class.

The Trainer owns the entire training lifecycle:

    fit()             — outer loop: stage iteration → epoch loop
    train_one_epoch() — single epoch forward/backward/optimizer pass
    validate()        — full validation pass with metric computation

Everything that was inline in ``run_training()`` inside ``train.py`` now lives
here.  train.py becomes a thin script that builds components and calls
``Trainer(...).fit(start_epoch)``.

Multi-scale training, AMP, gradient clipping, gradient accumulation, EMA,
multi-stage/freeze, callbacks, early stopping, and checkpoint orchestration
are all handled inside this class.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils import CheckpointManager, EarlyStopping, compute_dataset_metrics
from training.callbacks import Callback
from training.ema import EMA
from training.optimizers import build_optimizer, build_scheduler


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round_to_divisor(value: float, divisor: int) -> int:
    """Round *value* to the nearest positive multiple of *divisor*.

    Used for multi-scale training to keep spatial dimensions compatible with
    the model's stride-2 downsampling stages (32 for MK-UNet's 5 pooling
    layers).
    """
    rounded = int(round(value / divisor)) * divisor
    return max(divisor, rounded)


def _apply_grad_clip(
    model: nn.Module,
    mode: str,
    clip_value: float,
    clip_norm: float,
) -> None:
    """Apply the configured gradient clipping strategy."""
    if mode == "value":
        nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
    elif mode == "norm":
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    elif mode == "none":
        pass
    else:
        raise ValueError(
            f"Unknown grad_clip_mode '{mode}'. Expected 'value', 'norm', or 'none'."
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates the full training lifecycle.

    Args:
        model:         The model to train.
        criterion:     Loss function (``nn.Module``).
        optimizer:     Optimizer instance.
        scheduler:     LR scheduler instance (or None).
        scheduler_step_mode: ``'epoch'`` or ``'batch'`` (from build_scheduler).
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        config:        Full config dict (all top-level keys).
        logger:        Loguru logger or compatible.
        chk_manager:   CheckpointManager instance.
        device:        torch.device.
        fold:          Current fold index (None for non-CV runs).
        early_stopper: EarlyStopping instance (or None).
        callbacks:     List of Callback instances.
        ema:           EMA instance (or None).
        scaler:        GradScaler for AMP (or None).
    """

    def __init__(
        self,
        model:                nn.Module,
        criterion:            nn.Module,
        optimizer:            torch.optim.Optimizer,
        scheduler:            Any,
        scheduler_step_mode:  str,
        train_loader:         torch.utils.data.DataLoader,
        val_loader:           torch.utils.data.DataLoader,
        config:               Dict[str, Any],
        logger:               Any,
        chk_manager:          CheckpointManager,
        device:               torch.device,
        fold:                 Optional[int] = None,
        early_stopper:        Optional[EarlyStopping] = None,
        callbacks:            Optional[List[Callback]] = None,
        ema:                  Optional[EMA] = None,
        scaler:               Optional[torch.cuda.amp.GradScaler] = None,
    ):
        self.model               = model
        self.criterion           = criterion
        self.optimizer           = optimizer
        self.scheduler           = scheduler
        self.scheduler_step_mode = scheduler_step_mode
        self.train_loader        = train_loader
        self.val_loader          = val_loader
        self.config              = config
        self.logger              = logger
        self.chk_manager         = chk_manager
        self.device              = device
        self.fold                = fold
        self.early_stopper       = early_stopper
        self.callbacks           = callbacks or []
        self.ema                 = ema
        self.scaler              = scaler

        # Shorthand sub-configs
        self._tcfg  = config.get("training", {})
        self._chkcfg = config.get("checkpoint", {})

        # Gradient-clipping config
        self._gc_mode  = self._tcfg.get("grad_clip_mode",  "value")
        self._gc_value = self._tcfg.get("grad_clip_value", 0.5)
        self._gc_norm  = self._tcfg.get("grad_clip_norm",  1.0)

        # AMP flag
        self._use_amp = scaler is not None

        # Gradient accumulation
        self._accum_steps = max(1, int(self._tcfg.get("accumulate_grad_batches", 1)))

        # Multi-scale config
        self._ms_cfg = self._tcfg.get("multi_scale", None)

        # Monitor metric (strip leading "val_" prefix stored in checkpoint config)
        monitor_raw = self._chkcfg.get("monitor_metric", "val_dice")
        self._monitor_key = monitor_raw.replace("val_", "")

        # Fold suffix used in checkpoint filenames
        self._fold_suffix = f"_fold{fold}" if fold is not None else ""

        # Checkpoint directory (for in-loop extra-state persistence)
        log_cfg = config.get("logging", {})
        self._chk_dir = os.path.join(
            self._chkcfg.get("save_dir", "checkpoints"),
            log_cfg.get("experiment_name", "experiment"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, start_epoch: int = 1) -> float:
        """Run the full training loop, respecting multi-stage config if present.

        Args:
            start_epoch: First epoch to run (>1 when resuming).

        Returns:
            Best monitored metric value achieved during training.
        """
        stages = self.config.get("stages", [])

        if stages:
            return self._fit_staged(stages, start_epoch)
        else:
            total_epochs = self._tcfg.get("epochs", 50)
            return self._fit_range(start_epoch, total_epochs)

    # ------------------------------------------------------------------
    # Internal: epoch-range loop (shared by staged and non-staged paths)
    # ------------------------------------------------------------------

    def _fit_range(self, start_epoch: int, end_epoch: int) -> float:
        """Run epochs [start_epoch, end_epoch] inclusive."""
        self._call("on_train_start")

        for epoch in range(start_epoch, end_epoch + 1):
            self._call("on_epoch_start", epoch)

            train_loss, train_dice, train_iou = self.train_one_epoch(epoch)
            val_metrics = self._run_validate()

            # LR scheduler step
            self._step_scheduler(val_metrics)

            # Assemble full metrics dict
            current_lr = self.optimizer.param_groups[0]["lr"]
            val_metrics["lr"]         = current_lr
            val_metrics["train_loss"] = train_loss
            val_metrics["train_dice"] = train_dice
            val_metrics["train_iou"]  = train_iou

            # Log summary line
            self.logger.info(
                f"Epoch {epoch:03d} | "
                f"Loss: {train_loss:.4f} | "
                f"Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Dice: {val_metrics['dice']:.4f} | "
                f"Val mIoU: {val_metrics['miou']:.4f} | "
                f"Val HD95: {val_metrics['hd95']:.2f} | "
                f"Val ASD: {val_metrics['asd']:.2f} | "
                f"LR: {current_lr:.6f}"
            )

            # Per-class breakdown (multiclass only) at DEBUG verbosity
            pc = val_metrics.get("per_class", {})
            if pc:
                class_dice = pc.get("dice", [])
                class_iou  = pc.get("iou",  [])
                lines = [f"  Class {c}: Dice={class_dice[c]:.4f} IoU={class_iou[c]:.4f}"
                         for c in range(len(class_dice))]
                self.logger.debug("Per-class Val metrics:\n" + "\n".join(lines))

            # Callbacks (includes TensorBoard logging)
            self._call("on_epoch_end", epoch, val_metrics)

            # Checkpoint
            monitored_val = val_metrics[self._monitor_key]
            is_best       = self.chk_manager.is_better(monitored_val)
            self.chk_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metric_val=monitored_val,
                fold=self.fold,
                is_best=is_best,
                scaler=self.scaler,
            )
            # Persist extra state (RNG + EarlyStopper) into the last checkpoint
            self._persist_extra_state(epoch)

            # Early stopping
            if self.early_stopper is not None and self.early_stopper(monitored_val):
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        self._call("on_train_end")
        return self.chk_manager.best_metric

    def _fit_staged(self, stages: List[Dict[str, Any]], start_epoch: int) -> float:
        """Multi-stage training: iterate stages sequentially.

        Each stage dict supports:
            epochs  (int):         epochs to run in this stage
            lr      (float):       learning rate for this stage (rebuilds optimizer)
            freeze  (list[str]):   submodule name fragments to freeze

        Stages are run sequentially; the epoch counter is continuous across
        stages so checkpoint/ES logic is not disrupted.
        """
        epoch_cursor = 1
        for stage_idx, stage in enumerate(stages):
            stage_epochs = stage.get("epochs", self._tcfg.get("epochs", 50))
            stage_lr     = stage.get("lr", self._tcfg["lr"])
            freeze_keys  = stage.get("freeze", [])

            stage_end = epoch_cursor + stage_epochs - 1

            # Apply freezing
            self._set_frozen(freeze_keys, frozen=True)
            self.logger.info(
                f"[Stage {stage_idx}] epochs {epoch_cursor}–{stage_end} | "
                f"lr={stage_lr} | freeze={freeze_keys}"
            )

            # Rebuild optimizer with stage LR so frozen params are excluded
            stage_tcfg = {**self._tcfg, "lr": stage_lr}
            trainable  = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = build_optimizer(stage_tcfg, trainable)
            self.scheduler, self.scheduler_step_mode = build_scheduler(
                stage_tcfg, self.optimizer,
                steps_per_epoch=len(self.train_loader),
            )

            # Run this stage's epoch range (skip completed epochs on resume)
            effective_start = max(start_epoch, epoch_cursor)
            if effective_start <= stage_end:
                self._fit_range(effective_start, stage_end)

            # Unfreeze everything before the next stage
            self._set_frozen(freeze_keys, frozen=False)
            epoch_cursor = stage_end + 1

        return self.chk_manager.best_metric

    # ------------------------------------------------------------------
    # Core epoch methods
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int):
        """One full training pass over the training DataLoader.

        Supports:
        - Multi-scale training (``training.multi_scale``)
        - AMP (GradScaler)
        - Gradient accumulation (``training.accumulate_grad_batches``)
        - EMA update after each optimizer step
        - Batch-level callbacks (``on_batch_end``)
        - Rolling Dice/IoU computed from raw logits (no medpy required)

        Returns:
            Tuple of (mean_train_loss, mean_train_dice, mean_train_iou).
        """
        self.model.train()
        running_loss    = 0.0
        running_samples = 0
        running_dice    = 0.0
        running_iou     = 0.0

        ms_enabled      = self._ms_cfg.get("enabled", False) if self._ms_cfg else False
        ms_scales       = self._ms_cfg.get("scales", [0.75, 1.0, 1.25]) if self._ms_cfg else [1.0]
        ms_size_div     = self._ms_cfg.get("size_divisor", 32) if self._ms_cfg else 32
        ms_mode         = self._ms_cfg.get("mode", "all_scales") if self._ms_cfg else "all_scales"

        scales_to_run   = ms_scales if ms_enabled else [1.0]
        track_scale     = 1.0 if 1.0 in scales_to_run else scales_to_run[-1]

        # For gradient accumulation: zero once at the start
        self.optimizer.zero_grad(set_to_none=True)
        accum_counter   = 0

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, masks) in pbar:
            images_full = images.to(self.device)
            masks_full  = masks.to(self.device)
            h, w        = images_full.shape[2], images_full.shape[3]

            scales_this_batch = (
                [random.choice(ms_scales)] if (ms_enabled and ms_mode == "random")
                else scales_to_run
            )

            last_loss_value = 0.0
            for scale in scales_this_batch:
                if scale != 1.0:
                    new_h   = _round_to_divisor(h * scale, ms_size_div)
                    new_w   = _round_to_divisor(w * scale, ms_size_div)
                    imgs_s  = nn.functional.interpolate(
                        images_full, size=(new_h, new_w), mode="bilinear", align_corners=False
                    )
                    masks_s = nn.functional.interpolate(masks_full, size=(new_h, new_w), mode="nearest")
                else:
                    imgs_s, masks_s = images_full, masks_full

                # Forward pass
                if self._use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(imgs_s)
                        loss    = self.criterion(outputs, masks_s)
                else:
                    outputs = self.model(imgs_s)
                    loss    = self.criterion(outputs, masks_s)

                # Scale loss for gradient accumulation
                loss = loss / self._accum_steps

                # Backward
                if self._use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                last_loss_value = loss.item() * self._accum_steps  # unscaled for display

                # Rolling Dice/IoU — computed only on the tracking scale, no extra memory
                if scale == track_scale:
                    running_loss    += last_loss_value * imgs_s.size(0)
                    running_samples += imgs_s.size(0)
                    with torch.no_grad():
                        if outputs.shape[1] == 1:
                            p = (torch.sigmoid(outputs) > 0.5).float()
                            t = (masks_s > 0.5).float()
                        else:
                            n_cls = outputs.shape[1]
                            p = nn.functional.one_hot(
                                torch.argmax(outputs, dim=1), n_cls
                            ).permute(0, 3, 1, 2).float()
                            t = nn.functional.one_hot(
                                masks_s.long().squeeze(1), n_cls
                            ).permute(0, 3, 1, 2).float()
                        inter      = (p * t).sum(dim=(1, 2, 3))
                        union_sum  = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
                        batch_dice = (2.0 * inter / union_sum.clamp(min=1e-6)).mean().item()
                        batch_iou  = (inter / (union_sum - inter).clamp(min=1e-6)).mean().item()
                    running_dice += batch_dice * imgs_s.size(0)
                    running_iou  += batch_iou  * imgs_s.size(0)

            # Gradient accumulation: only step every N batches
            accum_counter += 1
            if accum_counter >= self._accum_steps:
                if self._use_amp:
                    self.scaler.unscale_(self.optimizer)
                    _apply_grad_clip(self.model, self._gc_mode, self._gc_value, self._gc_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    _apply_grad_clip(self.model, self._gc_mode, self._gc_value, self._gc_norm)
                    self.optimizer.step()

                if self.ema is not None:
                    self.ema.update()

                self.optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

                # Batch-level scheduler step (e.g. OneCycleLR)
                if self.scheduler is not None and self.scheduler_step_mode == "batch":
                    self.scheduler.step()

            mean_dice = running_dice / max(running_samples, 1)
            mean_iou  = running_iou  / max(running_samples, 1)
            pbar.set_postfix(
                loss=f"{last_loss_value:.4f}",
                dice=f"{mean_dice:.4f}",
                iou=f"{mean_iou:.4f}",
            )
            self._call("on_batch_end", batch_idx, last_loss_value)

        mean_loss = running_loss / max(running_samples, 1)
        mean_dice = running_dice / max(running_samples, 1)
        mean_iou  = running_iou  / max(running_samples, 1)
        return mean_loss, mean_dice, mean_iou

    def validate(self) -> Dict[str, Any]:
        """Public validate (uses live model weights — no EMA swap)."""
        return self._validate_model(self.model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_validate(self) -> Dict[str, Any]:
        """Validate, swapping to EMA weights if EMA is enabled."""
        if self.ema is not None:
            with self.ema.average_parameters():
                return self._validate_model(self.model)
        return self._validate_model(self.model)

    def _validate_model(self, model: nn.Module) -> Dict[str, Any]:
        """Core validation loop — model-agnostic (works with EMA swap)."""
        model.eval()
        running_loss = 0.0
        preds_list: list = []
        gts_list:   list = []

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                masks  = masks.to(self.device)

                if self._use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss    = self.criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss    = self.criterion(outputs, masks)

                running_loss += loss.item() * images.size(0)

                if outputs.shape[1] == 1:
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).cpu().numpy().astype(np.uint8)
                else:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)

                preds_list.extend(preds)
                gts_list.extend(m.cpu().numpy().astype(np.uint8) for m in masks)

        val_loss          = running_loss / len(self.val_loader.dataset)
        metrics           = compute_dataset_metrics(preds_list, gts_list)
        metrics["loss"]   = val_loss
        return metrics

    def _step_scheduler(self, val_metrics: Dict[str, Any]) -> None:
        """Advance the LR scheduler by one epoch.

        Handles ReduceLROnPlateau (needs monitored metric value) transparently.
        """
        if self.scheduler is None or self.scheduler_step_mode != "epoch":
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_metrics.get(self._monitor_key, 0.0))
        else:
            self.scheduler.step()

    def _call(self, hook: str, *args, **kwargs) -> None:
        """Fire a named callback hook on all registered callbacks."""
        for cb in self.callbacks:
            getattr(cb, hook)(self, *args, **kwargs)

    def _set_frozen(self, name_fragments: List[str], frozen: bool) -> None:
        """Freeze or unfreeze named submodules.

        A submodule is matched if any fragment in *name_fragments* is a
        substring of its fully-qualified name.  An empty list is a no-op.
        """
        if not name_fragments:
            return
        for name, module in self.model.named_modules():
            if any(frag in name for frag in name_fragments):
                for param in module.parameters():
                    param.requires_grad_(not frozen)
        verb = "Froze" if frozen else "Unfroze"
        self.logger.info(f"{verb} modules matching: {name_fragments}")

    def _persist_extra_state(self, epoch: int) -> None:
        """Append EarlyStopping and RNG states to the last checkpoint in-place.

        This keeps the checkpoint self-contained so a resumed run can fully
        reconstruct training state without extra side-files.
        """
        last_path = os.path.join(self._chk_dir, f"last{self._fold_suffix}.pth")
        if not os.path.exists(last_path):
            return

        extra = torch.load(last_path, map_location="cpu")

        # RNG states
        extra["rng_state_python"] = random.getstate()
        extra["rng_state_numpy"]  = np.random.get_state()
        extra["rng_state_torch"]  = torch.get_rng_state()
        if torch.cuda.is_available():
            extra["rng_state_cuda"] = torch.cuda.get_rng_state_all()

        # EarlyStopper
        if self.early_stopper is not None:
            extra["early_stopper_state"] = self.early_stopper.state_dict()

        # EMA shadow weights
        if self.ema is not None:
            extra["ema_state"] = self.ema.state_dict()

        torch.save(extra, last_path)
