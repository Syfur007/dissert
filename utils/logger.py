import os
import sys
from loguru import logger

def setup_logger(log_dir, experiment_name):
    """Set up Loguru logger to log to console and log files."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    # Configure loguru: remove default handler first
    logger.remove()
    
    # Add clean console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB"
    )
    return logger

class TensorBoardTracker:
    """Experiment tracker using PyTorch built-in TensorBoard or TensorBoardX."""
    def __init__(self, tb_dir, experiment_name):
        self.log_dir = os.path.join(tb_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            try:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir)
            except ImportError:
                self.writer = None
                logger.warning("Tensorboard is not installed. Experiment metrics will not be logged visually.")

    def log_scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_dict(self, metrics_dict, step, prefix=""):
        if self.writer:
            for k, v in metrics_dict.items():
                tag = f"{prefix}/{k}" if prefix else k
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(tag, v, step)

    def close(self):
        if self.writer:
            self.writer.close()
