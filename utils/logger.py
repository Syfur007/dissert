import os
import sys
from loguru import logger


def setup_logger(log_dir: str, log_filename: str):
    """Set up Loguru logger to log to console and a file inside *log_dir*.

    *log_dir* is the already-fully-resolved directory to log into (the
    ``logs`` entry of ``orchestration.runid.experiment_paths()``) — this
    function does no further path construction beyond the filename itself.
    All folds and eval runs share one *log_dir* with distinct filenames:

        outputs/experiments/my_experiment-s42/logs/
            fold0.log
            fold1.log
            eval.log

    Returns:
        (logger, log_dir) — the configured Loguru logger and the directory
        path (returned for convenience, same value as the *log_dir* input).
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_filename}.log")

    # Configure loguru: remove default handler first
    logger.remove()

    # Add clean console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
    )
    return logger, log_dir


class TensorBoardTracker:
    """Experiment tracker using PyTorch built-in TensorBoard or TensorBoardX."""

    def __init__(self, log_dir: str):
        """*log_dir* is the already-fully-resolved, fold-scoped tensorboard
        directory (the ``tensorboard`` entry of ``experiment_paths()``)."""
        self.log_dir = log_dir
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
                logger.warning(
                    "Tensorboard is not installed. Experiment metrics will not be logged visually."
                )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_dict(self, metrics_dict: dict, step: int, prefix: str = "") -> None:
        if self.writer:
            for k, v in metrics_dict.items():
                tag = f"{prefix}/{k}" if prefix else k
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(tag, v, step)

    def log_image(self, tag: str, img_tensor, step: int) -> None:
        """Log a CHW or HW image tensor to TensorBoard."""
        if self.writer:
            self.writer.add_image(tag, img_tensor, step)

    def close(self) -> None:
        if self.writer:
            self.writer.close()
