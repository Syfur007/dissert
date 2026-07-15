"""
training/callbacks.py — Lightweight callback hook system.

No external framework required.  The Trainer maintains a ``list[Callback]``
and calls each hook at the appropriate moment in the training loop.

Adding a new behaviour (e.g. prediction-overlay logging, LR-range test) means
writing a new Callback subclass — the core loop never needs to change again.

Built-in callbacks
------------------
PeriodicCheckpointCallback  — saves epoch_NNNN.pth every K epochs
TensorBoardCallback         — wraps TensorBoardTracker log_dict calls
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

if TYPE_CHECKING:
    # Avoid circular import at runtime; only for type hints.
    from training.trainer import Trainer


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Callback:
    """Base callback class.  Override any hooks you need; all are no-ops by default.

    All hooks receive the ``Trainer`` instance as their first argument so any
    trainer attribute (model, optimizer, config, logger …) is accessible.
    """

    def on_train_start(self, trainer: "Trainer") -> None:
        """Called once before the first epoch."""

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called once after the last epoch (or after early stopping)."""

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the start of every epoch, before train_one_epoch."""

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        """Called at the end of every epoch, after validate().

        Args:
            epoch:   Current epoch number (1-indexed).
            metrics: Dict of all metrics computed this epoch (includes
                     'train_loss', 'loss', 'dice', 'miou', 'lr', …).
        """

    def on_batch_end(
        self,
        trainer: "Trainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        """Called after every optimizer step inside train_one_epoch.

        Args:
            batch_idx: 0-indexed batch number within the current epoch.
            loss:      Scalar loss value for this batch.
        """


# ---------------------------------------------------------------------------
# Built-in: PeriodicCheckpointCallback
# ---------------------------------------------------------------------------

class PeriodicCheckpointCallback(Callback):
    """Save a snapshot ``epoch_{N:04d}{fold_suffix}.pth`` every *save_every* epochs.

    This is separate from the best/last checkpoints managed by CheckpointManager
    — it gives you a full timeline of snapshots for post-hoc analysis or
    rolling-back to any intermediate epoch.

    Args:
        save_every: Checkpoint interval in epochs.  0 or negative → disabled.
        save_dir:   Directory where snapshots are written (same as CheckpointManager
                    save_dir by default; overridable for isolation).
        fold:       Optional fold index, appended as ``_fold{fold}`` in the filename.
    """

    def __init__(self, save_every: int, save_dir: str, fold: Optional[int] = None):
        self.save_every = save_every
        self.save_dir   = save_dir
        self.fold       = fold
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        if self.save_every <= 0 or epoch % self.save_every != 0:
            return

        fold_suffix = f"_fold{self.fold}" if self.fold is not None else ""
        path = os.path.join(self.save_dir, f"epoch_{epoch:04d}{fold_suffix}.pth")

        state: Dict[str, Any] = {
            "epoch":              epoch,
            "model_state_dict":   trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "metrics":            metrics,
            "fold":               self.fold,
        }
        if trainer.scheduler is not None:
            state["scheduler_state_dict"] = trainer.scheduler.state_dict()
        if trainer.scaler is not None:
            state["scaler_state"] = trainer.scaler.state_dict()

        torch.save(state, path)
        trainer.logger.info(f"[PeriodicCheckpoint] Saved snapshot → {path}")


# ---------------------------------------------------------------------------
# Built-in: TensorBoardCallback
# ---------------------------------------------------------------------------

class TensorBoardCallback(Callback):
    """Log epoch metrics to TensorBoard via TensorBoardTracker.

    The tracker is passed in at construction time (already initialised in
    train.py) so this callback is pure decoration around the existing tracker.

    Args:
        tracker: An initialised ``TensorBoardTracker`` instance.
        prefix:  Prefix applied to all metric keys (default: ``'epoch'``).
    """

    def __init__(self, tracker: Any, prefix: str = "epoch"):
        self.tracker = tracker
        self.prefix  = prefix

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        self.tracker.log_dict(metrics, step=epoch, prefix=self.prefix)

    def on_train_end(self, trainer: "Trainer") -> None:
        self.tracker.close()
