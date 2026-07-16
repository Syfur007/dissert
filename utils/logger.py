import os
import sys
from loguru import logger


def setup_logger(log_dir: str, experiment_name: str, log_filename: str = ""):
    """Set up Loguru logger to log to console and a per-experiment log file.

    The experiment directory is always ``{log_dir}/{experiment_name}/``.
    The log file inside is ``{log_filename}.log`` when *log_filename* is
    provided, falling back to ``{experiment_name}.log``.

    This separation lets all folds, eval runs, and summaries share the same
    parent directory while keeping their log files distinct:

        logs/my_experiment/
            fold0.log
            fold1.log
            eval.log
            model_summary.txt
            overlays/
            plots/

    Returns:
        (logger, exp_log_dir) — the configured Loguru logger and the
        experiment-specific directory path.
    """
    exp_log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(exp_log_dir, exist_ok=True)

    fname    = log_filename if log_filename else experiment_name
    log_file = os.path.join(exp_log_dir, f"{fname}.log")

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
    return logger, exp_log_dir


class TensorBoardTracker:
    """Experiment tracker using PyTorch built-in TensorBoard or TensorBoardX."""

    def __init__(self, tb_dir: str, experiment_name: str):
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
