from .logger import setup_logger, TensorBoardTracker
from .early_stopping import EarlyStopping
from .checkpoint import CheckpointManager, atomic_torch_save
from .metrics import count_parameters
from .report import (
    EvaluationReporter,
    get_model_memory_size,
    get_latency_stats,
    get_gpu_memory_usage,
    get_environment_info,
)
from .visualize import (
    save_confusion_matrix,
    save_roc_curve,
    save_pr_curve,
)
from .plot_training import plot_training_curves

# Segmentation-quality metrics (Dice/IoU/HD95/ASD/...) live in the top-level
# metrics/ package (see metrics/aggregate.py's compute_dataset_metrics) —
# import from there directly, not from utils.

__all__ = [
    # logger
    "setup_logger",
    "TensorBoardTracker",
    # training utilities
    "CheckpointManager",
    "atomic_torch_save",
    "EarlyStopping",
    # profiling — see profiling/ (Phase 10) for FLOPs/latency/memory/export
    "count_parameters",
    # report
    "EvaluationReporter",
    "get_model_memory_size",
    "get_latency_stats",
    "get_gpu_memory_usage",
    "get_environment_info",
    # visualize
    "save_confusion_matrix",
    "save_roc_curve",
    "save_pr_curve",
    # offline plots
    "plot_training_curves",
]
