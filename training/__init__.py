"""
training/ — modular training infrastructure.

Public surface:
    Trainer              — orchestrates fit / train_one_epoch / validate
    get_loss             — loss factory
    build_optimizer      — optimizer factory
    build_scheduler      — scheduler factory
    Callback             — base hook class
    PeriodicCheckpointCallback
    TensorBoardCallback
    EMA                  — exponential moving average weight wrapper
"""

from .trainer import Trainer
from .losses import get_loss
from .optimizers import build_optimizer, build_scheduler
from .callbacks import Callback, PeriodicCheckpointCallback, TensorBoardCallback
from .ema import EMA

__all__ = [
    "Trainer",
    "get_loss",
    "build_optimizer",
    "build_scheduler",
    "Callback",
    "PeriodicCheckpointCallback",
    "TensorBoardCallback",
    "EMA",
]
