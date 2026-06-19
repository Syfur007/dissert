import os
import torch
from loguru import logger

class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.
    Supports tracking best performance metric and resuming training configurations.
    """
    def __init__(self, save_dir, monitor_metric="val_dice", mode="max"):
        self.save_dir = save_dir
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.best_metric = float('-inf') if mode == "max" else float('inf')
        os.makedirs(save_dir, exist_ok=True)

    def is_better(self, current_val):
        if self.mode == "max":
            return current_val > self.best_metric
        else:
            return current_val < self.best_metric

    def save(self, model, optimizer, scheduler, epoch, metric_val, fold=None, is_best=False):
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
        
        fold_suffix = f"_fold{fold}" if fold is not None else ""
        
        # Save latest epoch checkpoint
        last_path = os.path.join(self.save_dir, f"last{fold_suffix}.pth")
        torch.save(state, last_path)
        
        if is_best:
            self.best_metric = metric_val
            best_path = os.path.join(self.save_dir, f"best{fold_suffix}.pth")
            torch.save(state, best_path)
            logger.info(f"Saved new best model checkpoint to {best_path} with {self.monitor_metric}: {metric_val:.4f}")
        else:
            logger.debug(f"Saved last checkpoint to {last_path}")

    def load(self, checkpoint_path, model, optimizer=None, scheduler=None):
        """Load checkpoint weights and optionally restore optimizer/scheduler status."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle cases where model is wrapped in DataParallel / DistributedDataParallel
        state_dict = checkpoint['model_state_dict']
        if not next(model.parameters()).device == 'cpu' and hasattr(model, 'module'):
            # Model has 'module.' prefix but state_dict might not, or vice versa
            pass
            
        model.load_state_dict(state_dict, strict=False)
        
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        epoch = checkpoint.get('epoch', 0)
        metric_val = checkpoint.get('metric_val', None)
        fold = checkpoint.get('fold', None)
        
        # Update our tracker's best metric with loaded value if this was a best checkpoint
        if metric_val is not None:
            self.best_metric = metric_val
            
        logger.info(f"Resumed model at epoch {epoch} (fold {fold}) with monitored metric value: {metric_val}")
        return epoch, metric_val, fold
