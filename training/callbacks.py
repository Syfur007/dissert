"""
training/callbacks.py — Lightweight callback hook system.

No external framework required.  The Trainer maintains a ``list[Callback]``
and calls each hook at the appropriate moment in the training loop.

Adding a new behaviour (e.g. prediction-overlay logging, LR-range test) means
writing a new Callback subclass — the core loop never needs to change again.

Built-in callbacks
------------------
PeriodicCheckpointCallback   — saves epoch_NNNN.pth every K epochs
TensorBoardCallback          — wraps TensorBoardTracker log_dict calls
PredictionOverlayCallback    — saves side-by-side pred/GT overlays every N epochs
TrainingCurvePlotCallback    — dumps offline matplotlib PNGs at training end
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch

from utils import atomic_torch_save

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
                     'train_loss', 'train_dice', 'train_iou', 'loss',
                     'dice', 'miou', 'lr', …).
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
                    save_dir by default — already fold-scoped by the caller).
        fold:       Optional fold index, recorded into the snapshot's state dict.
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

        path = os.path.join(self.save_dir, f"epoch_{epoch:04d}.pth")

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

        atomic_torch_save(state, path)
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
        # Log all scalar metrics; skip the nested per_class dict
        flat = {k: v for k, v in metrics.items() if k != "per_class" and isinstance(v, (int, float))}
        self.tracker.log_dict(flat, step=epoch, prefix=self.prefix)

    def on_train_end(self, trainer: "Trainer") -> None:
        self.tracker.close()


# ---------------------------------------------------------------------------
# Built-in: PredictionOverlayCallback
# ---------------------------------------------------------------------------

class PredictionOverlayCallback(Callback):
    """Save side-by-side prediction vs. ground-truth overlay grids every N epochs.

    Grabs the first ``n_samples`` images from the first batch of
    ``val_loader`` and produces a ``[Input Image | Ground Truth Mask |
    Prediction Mask | Ground Truth Overlay on Image | Prediction Overlay on
    Image]`` grid saved as ``overlay_epoch_{epoch:04d}.png`` in ``save_dir``.

    Optionally logs the grid image to TensorBoard if ``tb_tracker`` is provided.

    Args:
        val_loader:  Validation DataLoader used during training.
        device:      Torch device to run inference on.
        save_dir:    Directory where overlay PNGs are written.
        n_samples:   Number of images per overlay grid (default: 4).
        save_every:  Epoch interval (default: 10).  0 → disabled.
        tb_tracker:  Optional ``TensorBoardTracker`` for image logging.
    """

    def __init__(
        self,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        save_dir: str,
        n_samples: int = 4,
        save_every: int = 10,
        tb_tracker: Optional[Any] = None,
    ):
        self.val_loader  = val_loader
        self.device      = device
        self.save_dir    = save_dir
        self.n_samples   = n_samples
        self.save_every  = save_every
        self.tb_tracker  = tb_tracker
        os.makedirs(save_dir, exist_ok=True)

        # Pre-fetch a fixed sample batch so overlays are reproducible across epochs
        self._sample_images: Optional[torch.Tensor] = None
        self._sample_masks:  Optional[torch.Tensor] = None
        self._prefetched = False

    def _prefetch(self) -> None:
        """Grab the first batch from val_loader (called lazily on first use)."""
        if self._prefetched:
            return
        try:
            images, masks, _meta = next(iter(self.val_loader))
            n = min(self.n_samples, images.shape[0])
            self._sample_images = images[:n]
            self._sample_masks  = masks[:n]
        except Exception:
            pass
        self._prefetched = True

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        if self.save_every <= 0 or epoch % self.save_every != 0:
            return

        self._prefetch()
        if self._sample_images is None:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            trainer.logger.warning("[PredictionOverlay] matplotlib not installed; skipping.")
            return

        imgs   = self._sample_images.to(self.device)
        masks  = self._sample_masks.to(self.device)
        n      = imgs.shape[0]
        is_bin = trainer.model.training  # save state

        trainer.model.eval()
        # If EMA is enabled, the epoch's logged metrics came from validating
        # under the EMA shadow weights (Trainer._run_validate) — run the
        # overlay forward pass under those same weights, otherwise the saved
        # image doesn't correspond to the score reported alongside it.
        ema_ctx = trainer.ema.average_parameters() if trainer.ema is not None else nullcontext()
        with ema_ctx:
            with torch.no_grad():
                if trainer._use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = trainer.model(imgs)
                else:
                    outputs = trainer.model(imgs)

        if is_bin:
            trainer.model.train()

        # Convert outputs → hard predictions
        if outputs.shape[1] == 1:
            preds = (torch.sigmoid(outputs) > 0.5).squeeze(1).cpu().numpy().astype(np.float32)
            gts   = masks.squeeze(1).cpu().numpy().astype(np.float32)
        else:
            preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.float32)
            gts   = masks.squeeze(1).cpu().numpy().astype(np.float32) \
                    if masks.shape[1] == 1 else \
                    torch.argmax(masks, dim=1).cpu().numpy().astype(np.float32)

        # Normalise input images for display (first channel shown for grayscale; RGB for 3-ch)
        imgs_np = imgs.cpu().numpy()

        def make_overlay(base_img: np.ndarray, mask_img: np.ndarray) -> np.ndarray:
            """Blend a mask onto the image for a quick overlay preview."""
            if base_img.shape[0] == 3:
                base = np.transpose(base_img, (1, 2, 0))
            else:
                base = np.repeat(base_img[:1].transpose(1, 2, 0), 3, axis=2)

            base = (base - base.min()) / (base.max() - base.min() + 1e-8)
            mask = mask_img.astype(np.float32)

            if mask.max() > 1:
                cmap = plt.get_cmap("tab20")
                colors = cmap((mask.astype(np.int32) % 20) / 19.0)[..., :3]
                alpha = (mask > 0)[..., None].astype(np.float32) * 0.45
            else:
                colors = np.zeros((*mask.shape, 3), dtype=np.float32)
                colors[..., 1] = mask
                alpha = mask[..., None] * 0.45

            return np.clip(base * (1.0 - alpha) + colors * alpha, 0.0, 1.0)

        # Build grid: n rows × 5 columns [Input | GT Mask | Pred Mask | GT Overlay | Pred Overlay]
        n_cols = 5
        fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 3, n * 3))
        if n == 1:
            axes = axes[np.newaxis, :]

        col_titles = [
            "Image",
            "GT Mask",
            "Prediction Mask",
            "GT Overlay",
            "Prediction Overlay",
        ]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=11, fontweight="bold")

        for row in range(n):
            img = imgs_np[row]
            # Normalise image to [0, 1] for display
            if img.shape[0] == 3:
                # RGB: CHW → HWC
                img_disp = np.transpose(img, (1, 2, 0))
                img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
            else:
                img_disp = img[0]
                img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)

            axes[row, 0].imshow(img_disp, cmap=None if img.shape[0] == 3 else "gray")
            axes[row, 1].imshow(gts[row],   cmap="tab20" if gts[row].max() > 1 else "gray")
            axes[row, 2].imshow(preds[row], cmap="tab20" if preds[row].max() > 1 else "gray")
            axes[row, 3].imshow(make_overlay(img, gts[row]))
            axes[row, 4].imshow(make_overlay(img, preds[row]))

            for col in range(n_cols):
                axes[row, col].axis("off")

        fig.suptitle(f"Epoch {epoch}", fontsize=13)
        fig.tight_layout()

        save_path = os.path.join(self.save_dir, f"overlay_epoch_{epoch:04d}.png")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        trainer.logger.info(f"[PredictionOverlay] Saved → {save_path}")

        # Log to TensorBoard as image
        if self.tb_tracker is not None:
            try:
                import torchvision
                # Re-read the saved PNG as a tensor (avoids keeping the large fig in memory)
                from torchvision.io import read_image
                grid_tensor = read_image(save_path).float() / 255.0
                self.tb_tracker.log_image(f"overlays/epoch_{epoch:04d}", grid_tensor, step=epoch)
            except Exception:
                pass  # TensorBoard image logging is best-effort


# ---------------------------------------------------------------------------
# Built-in: TrainingCurvePlotCallback
# ---------------------------------------------------------------------------

class TrainingCurvePlotCallback(Callback):
    """Dump offline matplotlib PNG training curves at the end of training.

    Reads the TensorBoard event file that ``TensorBoardCallback`` has been
    writing throughout training and saves one PNG per scalar tag to ``out_dir``.

    Args:
        tb_log_dir: TensorBoard run directory (the fold-scoped ``tensorboard``
                    dir from ``experiment_paths()``).
        out_dir:    The fold-scoped ``plots`` dir from ``experiment_paths()``
                    — PNGs land in its ``curves/`` subfolder.
    """

    def __init__(self, tb_log_dir: str, out_dir: str):
        self.tb_log_dir = tb_log_dir
        self.out_dir    = out_dir

    def on_train_end(self, trainer: "Trainer") -> None:
        try:
            from utils.plot_training import plot_training_curves
        except ImportError:
            trainer.logger.warning(
                "[TrainingCurvePlot] utils.plot_training unavailable; skipping."
            )
            return

        plots_dir = os.path.join(self.out_dir, "curves")
        try:
            saved = plot_training_curves(
                tb_log_dir=self.tb_log_dir,
                out_dir=plots_dir,
            )
            trainer.logger.info(
                f"[TrainingCurvePlot] Saved {len(saved)} curve plot(s) → {plots_dir}"
            )
        except Exception as exc:
            trainer.logger.warning(f"[TrainingCurvePlot] Could not generate plots: {exc}")
