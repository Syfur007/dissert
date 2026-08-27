import os
import torch
from loguru import logger


def atomic_torch_save(state: dict, path: str) -> None:
    """Write via a temp file + os.replace so a crash mid-write can't leave a
    truncated/corrupted checkpoint at *path*. This matters most for
    last.pth: it's the sole resume target, so a torn write there is
    unrecoverable rather than just losing one snapshot."""
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.
    Supports tracking best performance metric and resuming training configurations.

    Note: periodic epoch snapshots (epoch_NNNN.pth) are handled entirely by
    ``training.callbacks.PeriodicCheckpointCallback``, not here — the two
    used to both write the same file every interval, with this class's plain
    write always winning and silently discarding the callback's richer
    per-epoch metrics payload.
    """
    def __init__(self, save_dir, monitor_metric="val_dice", mode="max", min_delta: float = 0.0):
        self.save_dir       = save_dir
        self.monitor_metric = monitor_metric
        self.mode           = mode
        self.min_delta      = min_delta
        self.best_metric    = float('-inf') if mode == "max" else float('inf')
        os.makedirs(save_dir, exist_ok=True)

    def is_better(self, current_val):
        # Mirrors EarlyStopping's _is_improvement exactly (same min_delta),
        # so "best.pth" and EarlyStopping's notion of "best" never diverge.
        if self.mode == "max":
            return current_val > self.best_metric + self.min_delta
        else:
            return current_val < self.best_metric - self.min_delta

    def save(self, model, optimizer, scheduler, epoch, metric_val, fold=None, is_best=False, scaler=None):
        """Save training states including weights, optimizer status, scheduler, and epoch."""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metric_val': metric_val,
            'monitor_metric': self.monitor_metric,
            'fold': fold
        }

        # persist GradScaler state so a resumed AMP run
        # continues with the calibrated loss-scale factor rather than
        # restarting from the default, which risks overflow/underflow.
        if scaler is not None:
            state['scaler_state'] = scaler.state_dict()

        fold_suffix = f"_fold{fold}" if fold is not None else ""

        # Save latest epoch checkpoint
        last_path = os.path.join(self.save_dir, f"last{fold_suffix}.pth")
        atomic_torch_save(state, last_path)

        if is_best:
            self.best_metric = metric_val
            best_path = os.path.join(self.save_dir, f"best{fold_suffix}.pth")
            atomic_torch_save(state, best_path)
            logger.info(f"Saved new best model checkpoint to {best_path} with {self.monitor_metric}: {metric_val:.4f}")
        else:
            logger.debug(f"Saved last checkpoint to {last_path}")


    def load(self, checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
        """Load checkpoint weights and optionally restore optimizer/scheduler/scaler status."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        state_dict = checkpoint['model_state_dict']
        model_keys = set(model.state_dict().keys())

        # Reconcile a 'module.' prefix mismatch (checkpoint saved from a
        # DataParallel/DistributedDataParallel-wrapped model, loaded into a
        # bare model, or vice versa). Without this, every key would fail to
        # match and strict=False below would silently load nothing at all.
        ckpt_has_prefix  = any(k.startswith("module.") for k in state_dict)
        model_has_prefix = any(k.startswith("module.") for k in model_keys)
        if ckpt_has_prefix and not model_has_prefix:
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        elif model_has_prefix and not ckpt_has_prefix:
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}

        missing    = model_keys - set(state_dict.keys())
        unexpected = set(state_dict.keys()) - model_keys
        if missing:
            logger.warning(f"Missing keys while loading {checkpoint_path}: {sorted(missing)}")
        if unexpected:
            logger.warning(f"Unexpected keys while loading {checkpoint_path}: {sorted(unexpected)}")

        model.load_state_dict(state_dict, strict=False)
        
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # restore GradScaler state when available so AMP resumes
        # with the calibrated loss scale rather than the default initial value.
        if scaler is not None and checkpoint.get('scaler_state') is not None:
            scaler.load_state_dict(checkpoint['scaler_state'])
            logger.info("GradScaler state restored from checkpoint.")
            
        epoch = checkpoint.get('epoch', 0)
        metric_val = checkpoint.get('metric_val', None)
        fold = checkpoint.get('fold', None)
        
        # Update our tracker's best metric with loaded value if this was a best checkpoint
        if metric_val is not None:
            self.best_metric = metric_val
            
        logger.info(f"Resumed model at epoch {epoch} (fold {fold}) with monitored metric value: {metric_val}")
        return epoch, metric_val, fold
