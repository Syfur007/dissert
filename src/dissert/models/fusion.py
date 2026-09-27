"""
models/fusion.py — swappable primary/auxiliary branch fusion, per spec §6:
"fusion ∈ {cbffm, concat, add, xattn, none}, plus fuse_stages list.
Swappable without touching the encoder."

Contrasts deliberately with GMK-UNet's `ExponentialDecayGating`
(models/proposed/gmk_unet.py) — a fusion mechanism hard-baked into that
model's own forward(), impossible to swap without editing the model. Every
fusion mode here shares one interface (`forward(f_primary, f_auxiliary) ->
Tensor`, same shape as the inputs) so `models/proposed/mamba_unet.py`
selects one via config and never branches on which mode is active.

`fuse_stages` (a list of stage indices, 1-based, matching the primary
encoder's stage numbering) controls *where* fusion happens — a stage not
in `fuse_stages` uses the primary branch's own feature untouched
(equivalent to `none` at that stage specifically), regardless of the
model-wide `fusion` setting. build_fusion_stages() builds one fusion module
per stage accordingly.
"""
from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn

FusionMode = Literal["cbffm", "concat", "add", "xattn", "none"]


class AddFusion(nn.Module):
    """f_primary + f_auxiliary. Requires matching channel counts; no
    learnable parameters — the simplest possible baseline fusion."""

    def forward(self, f_primary: torch.Tensor, f_auxiliary: torch.Tensor) -> torch.Tensor:
        return f_primary + f_auxiliary


class ConcatFusion(nn.Module):
    """concat(f_primary, f_auxiliary) along channels, then a 1x1 conv back
    down to the primary branch's channel count — so the decoder (built for
    the primary branch's widths) never needs to know fusion happened."""

    def __init__(self, channels: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_primary: torch.Tensor, f_auxiliary: torch.Tensor) -> torch.Tensor:
        return self.reduce(torch.cat([f_primary, f_auxiliary], dim=1))


class NoneFusion(nn.Module):
    """Primary branch only — the auxiliary branch's output at this stage is
    computed (it still contributes to the loss via whatever stages/paths
    *do* fuse it) but discarded here. The ablation control for "does fusing
    the auxiliary branch at this stage help at all.\""""

    def forward(self, f_primary: torch.Tensor, f_auxiliary: torch.Tensor) -> torch.Tensor:
        return f_primary


class CrossAttentionFusion(nn.Module):
    """Lightweight spatial cross-attention: primary queries, auxiliary
    keys/values (auxiliary features inform where in the primary map to
    attend, not the reverse) — computed at reduced spatial resolution via
    a pooled token set to keep the attention matrix a manageable size for
    a full-resolution feature map."""

    def __init__(self, channels: int, num_heads: int = 4, pool_size: int = 8):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.pool_size = pool_size
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, f_primary: torch.Tensor, f_auxiliary: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_primary.shape
        q = f_primary.flatten(2).transpose(1, 2)  # (B, H*W, C) -- full resolution queries
        kv = self.pool(f_auxiliary).flatten(2).transpose(1, 2)  # (B, pool*pool, C)
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        return f_primary + self.proj(attended)


class CBFFM(nn.Module):
    """Cross-Branch Feature Fusion Module: a per-pixel, per-branch gate
    (not a single global scalar the way GMK-UNet's ExponentialDecayGating
    is) — a small conv over concat(f_primary, f_auxiliary) produces a
    1-channel spatial gate map g in [0, 1]; output is
    ``g * proj_primary(f_primary) + (1-g) * proj_auxiliary(f_auxiliary)``.
    Each branch is projected before blending so the gate can also learn to
    reweight *which channels* matter from each branch, not just a spatial
    on/off mask.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj_primary = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )
        self.proj_auxiliary = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels // 2 or 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2 or 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2 or 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, f_primary: torch.Tensor, f_auxiliary: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([f_primary, f_auxiliary], dim=1))  # (B, 1, H, W)
        return g * self.proj_primary(f_primary) + (1 - g) * self.proj_auxiliary(f_auxiliary)


_FUSION_CLASSES = {
    "add": AddFusion,
    "concat": ConcatFusion,
    "none": NoneFusion,
    "xattn": CrossAttentionFusion,
    "cbffm": CBFFM,
}


def build_fusion(mode: FusionMode, channels: int) -> nn.Module:
    if mode not in _FUSION_CLASSES:
        raise ValueError(f"Unknown fusion mode '{mode}'. Known: {sorted(_FUSION_CLASSES)}")
    if mode in ("add", "none"):
        return _FUSION_CLASSES[mode]()
    return _FUSION_CLASSES[mode](channels)


def build_fusion_stages(
    mode: FusionMode,
    stage_channels: List[int],
    fuse_stages: Optional[List[int]] = None,
) -> nn.ModuleList:
    """One fusion module per encoder stage (1-based indices matching
    `stage_channels`' order). Stages not listed in `fuse_stages` (default:
    every stage) get a NoneFusion regardless of `mode`.
    """
    n_stages = len(stage_channels)
    active_stages = set(fuse_stages) if fuse_stages is not None else set(range(1, n_stages + 1))
    unknown = active_stages - set(range(1, n_stages + 1))
    if unknown:
        raise ValueError(f"fuse_stages {sorted(unknown)} out of range for {n_stages} stages")

    return nn.ModuleList([
        build_fusion(mode, stage_channels[i]) if (i + 1) in active_stages else NoneFusion()
        for i in range(n_stages)
    ])
