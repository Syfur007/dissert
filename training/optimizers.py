"""
training/optimizers.py — Optimizer and LR-scheduler factories.

Usage::

    from training.optimizers import build_optimizer, build_scheduler

    optimizer = build_optimizer(training_cfg, model)
    scheduler, step_mode = build_scheduler(training_cfg, optimizer, steps_per_epoch=len(train_loader))

``step_mode`` is either ``'epoch'`` (call scheduler.step() once per epoch) or
``'batch'`` (call scheduler.step() once per batch).  The Trainer uses this
value to know when to advance the schedule without inspecting the scheduler
type itself.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    OneCycleLR,
)

_NORM_MODULE_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm, nn.LayerNorm,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)

# Bare parameter names (not dotted paths) excluded from weight decay
# regardless of which module owns them. `.bias` is handled separately
# (it's a near-universal convention across module types); A_log/D are
# Mamba/SSM-specific — no model in this registry defines them yet, but
# every published Mamba implementation excludes them from decay, and
# Phase 6 needs this in place before that model exists, not patched in
# after the fact.
_NO_DECAY_PARAM_NAMES = {"A_log", "D"}


def no_decay_group(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Split model.named_parameters() into (decay, no_decay) by *name*,
    not by tensor shape: a bias, a normalisation layer's affine scale/shift
    (BatchNorm/GroupNorm/LayerNorm/InstanceNorm's weight+bias), or a
    parameter literally named A_log/D goes in no_decay; everything else
    (conv/linear weight matrices, embeddings) goes in decay.
    """
    norm_param_ids = {
        id(p)
        for module in model.modules()
        if isinstance(module, _NORM_MODULE_TYPES)
        for p in module.parameters(recurse=False)
    }

    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "bias" or leaf in _NO_DECAY_PARAM_NAMES or id(param) in norm_param_ids:
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_optimizer(cfg: Dict[str, Any], model: nn.Module) -> optim.Optimizer:
    """Instantiate an optimizer from config, with weight decay split into
    two parameter groups via no_decay_group() — biases, normalisation-layer
    affine parameters, and (forward-looking, for Phase 6) A_log/D never get
    weight-decayed, regardless of optimizer choice.

    Config keys read (all under ``training:``):
        optimizer      (str)   — 'adam' | 'adamw' | 'sgd'
        lr             (float) — base learning rate
        weight_decay   (float) — L2 penalty (default 1e-4)
        adam_betas     (list)  — [β₁, β₂] for Adam/AdamW (default [0.9, 0.999])
        adam_eps       (float) — ε for Adam/AdamW (default 1e-8)
        momentum       (float) — momentum for SGD (default 0.9)
        nesterov       (bool)  — Nesterov SGD (default False)

    Args:
        cfg:   The ``training`` sub-dict from the full config.
        model: The model to optimise — **not** ``model.parameters()``;
            no_decay_group() needs each parameter's dotted name and owning
            module, not just the bare tensors.

    Returns:
        Configured optimizer instance.
    """
    name         = cfg.get("optimizer", "adamw").lower()
    lr           = cfg["lr"]
    weight_decay = cfg.get("weight_decay", 1e-4)
    betas        = tuple(cfg.get("adam_betas", [0.9, 0.999]))
    eps          = cfg.get("adam_eps", 1e-8)
    momentum     = cfg.get("momentum", 0.9)
    nesterov     = cfg.get("nesterov", False)

    decay_params, no_decay_params = no_decay_group(model)
    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    logger.info(
        f"Optimizer param groups | decay: {len(decay_params)} tensors | "
        f"no_decay: {len(no_decay_params)} tensors"
    )

    if name == "adam":
        return optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)
    elif name == "adamw":
        return optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    elif name == "sgd":
        return optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=nesterov)
    else:
        raise ValueError(
            f"Unknown optimizer '{name}'. Supported: 'adam', 'adamw', 'sgd'."
        )


def build_scheduler(
    cfg: Dict[str, Any],
    optimizer: optim.Optimizer,
    steps_per_epoch: int = 0,
) -> Tuple[Any, str]:
    """Instantiate an LR scheduler from config.

    Config keys read (all under ``training:``):
        scheduler              (str)   — 'cosine' | 'step' | 'plateau' | 'onecycle' | 'none'
        epochs                 (int)   — total training epochs (used by cosine/onecycle)
        lr_step_size           (int)   — epoch step size for StepLR (default 10)
        lr_gamma               (float) — decay factor for StepLR (default 0.5)
        reduce_lr_patience     (int)   — ReduceLROnPlateau patience (default 5)
        reduce_lr_factor       (float) — ReduceLROnPlateau factor (default 0.5)
        warmup_epochs          (int)   — pct_start for OneCycleLR (default 5)

    Args:
        cfg:             The ``training`` sub-dict from the full config.
        optimizer:       The optimizer whose LR will be scheduled.
        steps_per_epoch: Number of optimizer steps per epoch.  Required when
                         ``scheduler='onecycle'``; ignored otherwise.

    Returns:
        Tuple of (scheduler | None, step_mode) where step_mode is
        'epoch' or 'batch'.  None is returned (with step_mode='epoch')
        when scheduler is 'none'.
    """
    name = cfg.get("scheduler", "none").lower()

    if name == "none":
        return None, "epoch"

    epochs = cfg.get("epochs", 50)

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs), "epoch"

    elif name == "step":
        return StepLR(
            optimizer,
            step_size=cfg.get("lr_step_size", 10),
            gamma=cfg.get("lr_gamma", 0.5),
        ), "epoch"

    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=cfg.get("mode", "max"),          # inherit checkpoint mode
            patience=cfg.get("reduce_lr_patience", 5),
            factor=cfg.get("reduce_lr_factor", 0.5),
        ), "epoch"   # Trainer handles .step(metric) for ReduceLROnPlateau

    elif name == "onecycle":
        if steps_per_epoch <= 0:
            raise ValueError(
                "build_scheduler: steps_per_epoch must be > 0 when scheduler='onecycle'."
            )
        warmup_epochs = cfg.get("warmup_epochs", 5)
        return OneCycleLR(
            optimizer,
            max_lr=cfg["lr"],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warmup_epochs / max(epochs, 1),
        ), "batch"

    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. Supported: 'cosine', 'step', 'plateau', 'onecycle', 'none'."
        )
