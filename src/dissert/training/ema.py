"""
training/ema.py — Exponential Moving Average (EMA) weight wrapper.

Usage::

    ema = EMA(model, decay=0.9999)

    # After each optimizer step inside the training loop:
    ema.update()

    # At validation time:
    with ema.average_parameters():
        val_metrics = trainer.validate()
    # Model weights are automatically restored after the context exits.

EMA is a well-established trick that often yields 0.5–1 point of free Dice/IoU
at essentially zero training-time cost — the shadow weights do not participate
in backprop, only in evaluation.

No changes to the model class are required.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class EMA:
    """Maintains a set of exponentially-averaged shadow parameters.

    After each optimizer step call ``ema.update()``.  Use the
    ``ema.average_parameters()`` context manager to temporarily replace the
    model weights with the shadow weights for validation.

    Args:
        model: The model whose parameters are shadowed.
        decay: EMA decay factor (default 0.9999).  Higher values → slower but
               smoother averaging.  Typical range: 0.99–0.9999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model  = model
        self.decay  = decay
        # Deep-copy creates the shadow parameters on the same device as model
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        # Freeze shadow so it never accumulates gradient
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow weights: shadow_p = decay * shadow_p + (1-decay) * model_p."""
        for s_param, m_param in zip(self.shadow.parameters(), self.model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
        # Also copy buffers (BatchNorm running stats, etc.)
        for s_buf, m_buf in zip(self.shadow.buffers(), self.model.buffers()):
            s_buf.copy_(m_buf)

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Context manager that temporarily swaps model weights → shadow weights.

        On exit the original model weights are restored so training can continue
        seamlessly.

        Example::

            with ema.average_parameters():
                metrics = validate(ema.shadow, val_loader, criterion, device)
        """
        # Stash live weights
        original_state = copy.deepcopy(self.model.state_dict())
        # Install shadow weights into the live model
        self.model.load_state_dict(self.shadow.state_dict())
        try:
            yield
        finally:
            # Restore live weights
            self.model.load_state_dict(original_state)

    def state_dict(self) -> dict:
        """Serialise shadow weights for checkpoint persistence."""
        return {
            "shadow_state_dict": self.shadow.state_dict(),
            "decay":             self.decay,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore shadow weights from a checkpoint state dict."""
        self.decay = state.get("decay", self.decay)
        self.shadow.load_state_dict(state["shadow_state_dict"])
