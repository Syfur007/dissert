"""
models/decoder.py — single decoder shared by every internal MambaUNet
variant, per spec §6: "Single decoder shared by all internal variants.
Skip policy ∈ {primary_only, fused, both_concat} declared in config."

Reuses models/blocks.py's existing DecoderBlock (bilinear-upsample +
concat-skip + DoubleConv) and CBAM primitives (ChannelAttention/
SpatialAttention) rather than reimplementing upsampling/skip-fusion logic
a second time — and applies them the same way MK-UNet/GMK-UNet's own
decoders do (CBAM gate on every stage's output), so a comparison between
this model and those baselines isn't confounded by an unrelated decoder
change.

skip_policy controls what feeds each stage's skip connection:
    primary_only  — the primary (MK-UNet) encoder's own feature only;
                    the auxiliary branch's skip is ignored entirely (it
                    may still influence the bottleneck via fusion there).
    fused         — models/fusion.py's fused primary+auxiliary feature
                    for that stage (whichever fusion mode the model was
                    built with).
    both_concat   — primary and auxiliary skips concatenated directly
                    (double the channel width DecoderBlock's DoubleConv
                    must reduce), bypassing models/fusion.py — lets the
                    decoder's own conv learn the primary/auxiliary mixing
                    instead of a dedicated fusion module.
"""
from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn

from .blocks import ChannelAttention, DecoderBlock, DoubleConv, SpatialAttention

SkipPolicy = Literal["primary_only", "fused", "both_concat"]
_VALID_SKIP_POLICIES = ("primary_only", "fused", "both_concat")


class MambaDecoder(nn.Module):
    def __init__(
        self,
        stage_channels: List[int],  # primary encoder's width per stage, shallow -> deep
        bottleneck_channels: int,
        num_classes: int,
        skip_policy: SkipPolicy = "fused",
        bilinear: bool = True,
    ):
        super().__init__()
        if skip_policy not in _VALID_SKIP_POLICIES:
            raise ValueError(f"skip_policy must be one of {_VALID_SKIP_POLICIES}, got '{skip_policy}'")
        self.skip_policy = skip_policy
        n_stages = len(stage_channels)
        skip_multiplier = 2 if skip_policy == "both_concat" else 1

        prev_channels = bottleneck_channels
        self.blocks = nn.ModuleList()
        self.channel_attns = nn.ModuleList()
        self.spatial_attn = SpatialAttention()

        # Build deepest-stage-first (decoding order); stage_channels itself
        # stays shallow->deep so callers describe the encoder naturally.
        for stage_idx in reversed(range(n_stages)):
            skip_channels = stage_channels[stage_idx] * skip_multiplier
            out_channels = stage_channels[stage_idx]
            self.blocks.append(
                DecoderBlock(prev_channels + skip_channels, out_channels, bilinear=bilinear)
            )
            self.channel_attns.append(ChannelAttention(out_channels))
            prev_channels = out_channels

        # stage_channels' shallowest entry (t1, in MK-UNet's naming) is
        # already at H/2, not full input resolution (MK-UNet's own
        # maxpool2d-after-every-stage convention — see
        # models/proposed/mamba_unet.py's docstring). One more upsample
        # (no skip connection available at this resolution) + refinement
        # conv is needed to reach H — mirrors MK-UNet's own decoder5 step
        # in models/baseline/mk_unet.py's MK_UNet_Base.forward().
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = DoubleConv(prev_channels, prev_channels)

        self.out_conv = nn.Conv2d(prev_channels, num_classes, kernel_size=1)

    def _make_skip(
        self,
        primary_skip: torch.Tensor,
        auxiliary_skip: Optional[torch.Tensor],
        fused_skip: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.skip_policy == "primary_only":
            return primary_skip
        if self.skip_policy == "fused":
            return fused_skip
        return torch.cat([primary_skip, auxiliary_skip], dim=1)  # both_concat

    def forward(
        self,
        bottleneck: torch.Tensor,
        primary_skips: List[torch.Tensor],
        auxiliary_skips: List[Optional[torch.Tensor]],
        fused_skips: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """All three skip lists are shallow -> deep (index 0 = highest
        resolution, index -1 = deepest, immediately above the bottleneck)
        — the same order stage_channels was given in __init__.
        auxiliary_skips/fused_skips entries may be None for whichever
        entries skip_policy doesn't need (the caller,
        models/proposed/mamba_unet.py, still computes the auxiliary branch
        regardless — this decoder just doesn't have to consume it under
        "primary_only").
        """
        x = bottleneck
        n_stages = len(primary_skips)
        for i, stage_idx in enumerate(reversed(range(n_stages))):
            skip = self._make_skip(
                primary_skips[stage_idx], auxiliary_skips[stage_idx], fused_skips[stage_idx]
            )
            x = self.blocks[i](x, skip)
            x = self.channel_attns[i](x) * x
            x = self.spatial_attn(x) * x

        x = self.final_conv(self.final_up(x))
        return self.out_conv(x)
