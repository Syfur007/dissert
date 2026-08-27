"""
tests/test_optim.py — Phase 5: parameter-group optimizer.

The test the plan names for this phase: test_param_groups.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.optimizers import build_optimizer, no_decay_group


class _TinyNet(nn.Module):
    """A small net exercising every no_decay_group() case: a conv weight+
    bias, a BatchNorm's weight+bias, and a bare parameter literally named
    A_log (Mamba/SSM-style, forward-looking for Phase 6)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, bias=True)
        self.bn = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8, 2, bias=True)
        self.A_log = nn.Parameter(torch.zeros(4, 4))
        self.D = nn.Parameter(torch.zeros(4))


def test_param_groups():
    model = _TinyNet()
    decay, no_decay = no_decay_group(model)

    decay_ids = {id(p) for p in decay}
    no_decay_ids = {id(p) for p in no_decay}

    assert id(model.conv.weight) in decay_ids
    assert id(model.fc.weight) in decay_ids

    assert id(model.conv.bias) in no_decay_ids
    assert id(model.fc.bias) in no_decay_ids
    assert id(model.bn.weight) in no_decay_ids
    assert id(model.bn.bias) in no_decay_ids
    assert id(model.A_log) in no_decay_ids
    assert id(model.D) in no_decay_ids

    # Every trainable parameter accounted for exactly once.
    n_params = sum(1 for _ in model.parameters())
    assert len(decay) + len(no_decay) == n_params
    assert decay_ids.isdisjoint(no_decay_ids)


def test_param_groups_excludes_frozen_params():
    model = _TinyNet()
    for p in model.fc.parameters():
        p.requires_grad_(False)

    decay, no_decay = no_decay_group(model)
    all_ids = {id(p) for p in decay} | {id(p) for p in no_decay}
    assert id(model.fc.weight) not in all_ids
    assert id(model.fc.bias) not in all_ids
    assert id(model.conv.weight) in all_ids  # still trainable, still included


def test_build_optimizer_applies_weight_decay_only_to_decay_group():
    model = _TinyNet()
    cfg = {"optimizer": "adamw", "lr": 0.001, "weight_decay": 0.05}
    optimizer = build_optimizer(cfg, model)

    assert len(optimizer.param_groups) == 2
    wds = {round(g["weight_decay"], 6) for g in optimizer.param_groups}
    assert wds == {0.0, 0.05}

    no_decay_group_ = next(g for g in optimizer.param_groups if g["weight_decay"] == 0.0)
    no_decay_ids = {id(p) for p in no_decay_group_["params"]}
    assert id(model.A_log) in no_decay_ids
    assert id(model.bn.weight) in no_decay_ids


def test_build_optimizer_sgd_also_gets_param_groups():
    model = _TinyNet()
    cfg = {"optimizer": "sgd", "lr": 0.01, "weight_decay": 0.01, "momentum": 0.9}
    optimizer = build_optimizer(cfg, model)
    assert len(optimizer.param_groups) == 2
