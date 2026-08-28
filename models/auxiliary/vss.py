"""
models/auxiliary/vss.py — VSS (Visual State Space) block with PVM
(Parallel Vision Mamba grouping), per spec §6: "VSS blocks with PVM. Config
exposes: scan_directions ∈ {1,2,4,8}, d_state, expand, PVM group count,
per-stage depth, DropPath rate."

Standard Mamba/VMamba block structure, adapted for 2D vision input:

    x (B,H,W,C)
      │
    LayerNorm
      │
    in_proj (Linear: C -> 2*d_inner)  ──split──►  x_in (B,H,W,d_inner)   z (B,H,W,d_inner) [gate]
      │                                              │
      │                                        depthwise Conv2d(3x3) + SiLU
      │                                              │
      │                                    ┌─────────┴─────────┐
      │                                    │   PVM: split into  │
      │                                    │   `pvm_groups`      │
      │                                    │   channel groups,   │
      │                                    │   each an           │
      │                                    │   independent,      │
      │                                    │   narrower SS2D     │
      │                                    │   scan (its own     │
      │                                    │   dt/B/C/A/D) —     │
      │                                    │   this is the       │
      │                                    │   "parallel" in     │
      │                                    │   Parallel Vision   │
      │                                    │   Mamba: G smaller  │
      │                                    │   scans instead of  │
      │                                    │   one full-width    │
      │                                    │   one, each gated   │
      │                                    │   by its own slice  │
      │                                    │   of z              │
      │                                    └─────────┬─────────┘
      │                                        concat groups
      │                                              │
      │                                        out_norm (LayerNorm)
      │                                              │
      │                                        out_proj (Linear: d_inner -> C)
      │                                              │
      └──────────────────────── + (residual, DropPath) ◄┘

PVM group count is this module's own reasonable, clearly-documented
interpretation of the spec's "PVM group count" knob — the spec (§6) names
the config surface but not PVM's exact internals; per-group independent
scans is the design choice made here, chosen because it's what actually
makes "PVM group count" a meaningful capacity/efficiency knob (more groups
= narrower, cheaper per-group scans) rather than a no-op.
"""
from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from .ss2d import SS2D


class _PVMGroup(nn.Module):
    """One PVM group: its own dt/B/C projection, A_log/D parameters, and
    SS2D scan, operating on `group_dim` channels."""

    def __init__(self, group_dim: int, d_state: int, dt_rank: int, scan_directions: int, merge: str):
        super().__init__()
        self.group_dim = group_dim
        self.d_state = d_state

        self.x_proj = nn.Linear(group_dim, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, group_dim, bias=True)
        self.dt_rank = dt_rank

        # A_log parametrisation: A = -exp(A_log), initialised so each
        # state index n has decay rate -(n+1) at delta=1 — the standard
        # Mamba S6 init (HiPPO-inspired, not a random init), giving each
        # of the d_state channels of state a distinct initial timescale.
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(group_dim, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(group_dim))

        self.scan = SS2D(d_inner=group_dim, d_state=d_state, scan_directions=scan_directions, merge=merge)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """x, z: (B, group_dim, H, W). Returns (B, group_dim, H, W)."""
        b, c, h, w = x.shape
        x_chlast = x.permute(0, 2, 3, 1)  # (B, H, W, group_dim)
        proj = self.x_proj(x_chlast)  # (B, H, W, dt_rank + 2*d_state)
        dt_raw, B_raw, C_raw = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        delta = self.dt_proj(dt_raw).permute(0, 3, 1, 2)  # (B, group_dim, H, W) -- bias applied here
        Bp = B_raw.permute(0, 3, 1, 2)  # (B, d_state, H, W)
        Cp = C_raw.permute(0, 3, 1, 2)
        A = -torch.exp(self.A_log)  # (group_dim, d_state), always negative

        return self.scan(x, delta, A, Bp, Cp, D=self.D, z=z, delta_softplus=True)

    @property
    def scan_impl(self) -> str:
        return self.scan.scan_impl


class VSSBlock(nn.Module):
    """One VSS block. Operates on channels-last (B, H, W, C) tensors,
    matching the VMamba/Swin convention (as opposed to this repo's other
    encoders, which are channels-first (B, C, H, W) — models/decoder.py and
    models/proposed/mamba_unet.py handle the permute at the branch
    boundary, not this module).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        scan_directions: Literal[1, 2, 4, 8] = 4,
        merge: Literal["sum", "learned"] = "sum",
        pvm_groups: int = 1,
        drop_path: float = 0.0,
        conv_kernel: int = 3,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        if self.d_inner % pvm_groups != 0:
            raise ValueError(
                f"pvm_groups={pvm_groups} must evenly divide d_inner={self.d_inner} "
                f"(d_model={d_model} * expand={expand})"
            )
        self.pvm_groups = pvm_groups
        self.group_dim = self.d_inner // pvm_groups
        dt_rank = dt_rank or max(d_model // 16, 1)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=conv_kernel,
            padding=conv_kernel // 2, groups=self.d_inner,
        )
        self.act = nn.SiLU()

        self.pvm = nn.ModuleList([
            _PVMGroup(self.group_dim, d_state, dt_rank, scan_directions, merge)
            for _ in range(pvm_groups)
        ])

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C). Returns (B, H, W, C)."""
        shortcut = x
        x = self.norm(x)
        xz = self.in_proj(x)  # (B, H, W, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = x_in.permute(0, 3, 1, 2)  # (B, d_inner, H, W)
        x_in = self.act(self.conv2d(x_in))
        z = z.permute(0, 3, 1, 2)  # (B, d_inner, H, W) -- gate, per-group sliced below

        x_groups = x_in.chunk(self.pvm_groups, dim=1)
        z_groups = z.chunk(self.pvm_groups, dim=1)
        y = torch.cat(
            [group(xg, zg) for group, xg, zg in zip(self.pvm, x_groups, z_groups)],
            dim=1,
        )  # (B, d_inner, H, W)

        y = y.permute(0, 2, 3, 1)  # (B, H, W, d_inner)
        y = self.out_norm(y)
        out = self.out_proj(y)  # (B, H, W, d_model)
        return shortcut + self.drop_path(out)

    @property
    def scan_impl(self) -> str:
        """Which scan implementation (mamba_ssm fused kernel or ss2d_ref
        pure-PyTorch) this block's PVM groups actually ran on — every
        group uses the same implementation (it's a process-wide
        availability check, not per-instance), so the first group's value
        represents the whole block."""
        return self.pvm[0].scan_impl


def build_vss_stage(
    d_model: int,
    depth: int,
    d_state: int = 16,
    expand: int = 2,
    scan_directions: Literal[1, 2, 4, 8] = 4,
    merge: Literal["sum", "learned"] = "sum",
    pvm_groups: int = 1,
    drop_path_rate: float = 0.0,
) -> nn.Sequential:
    """`depth` VSSBlocks in sequence — one encoder stage. drop_path_rate is
    linearly ramped from 0 to drop_path_rate across the stage's blocks
    (the standard stochastic-depth schedule), not applied uniformly.
    """
    if depth <= 0:
        raise ValueError(f"depth must be >= 1, got {depth}")
    drop_rates = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
    return nn.Sequential(*[
        VSSBlock(
            d_model=d_model, d_state=d_state, expand=expand,
            scan_directions=scan_directions, merge=merge,
            pvm_groups=pvm_groups, drop_path=drop_rates[i],
        )
        for i in range(depth)
    ])
