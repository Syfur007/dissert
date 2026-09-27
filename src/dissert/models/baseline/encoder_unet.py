"""
models/baseline/encoder_unet.py — Phase 2's one new model family (plan
§6.4): any models.encoders.build_encoder output plus a plain U-Net decoder.
A fixed-decoder test bed for comparing pretraining sources on equal footing
(the "controlled comparison" §2.1.1 calls for) — the decoder never changes,
only the encoder + its weights do, so a performance difference is
attributable to the pretraining choice, not a confounded architecture
change. No other new architecture is added this phase.

The decoder reuses models/blocks.py's DecoderBlock/DoubleConv exactly as
models/baseline/unet.py's own UNet does (concat-skip + bilinear upsample) —
generalised from UNet's fixed 4-level, from-scratch encoder to any number of
stages build_encoder returns, since a timm encoder's stage count/strides
vary by architecture (§6.1). Unlike models/decoder.py's MambaDecoder, there
is no primary/auxiliary/fused skip distinction to make — a plain encoder has
exactly one skip per stage.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import DecoderBlock, DoubleConv
from ..encoders import build_encoder
from ..registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("encoder_unet")
class EncoderUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder: Optional[Union[Dict[str, Any], str]] = None,
        bilinear: bool = True,
        **kwargs,
    ):
        """
        Args:
            encoder: dict (or models.encoders.EncoderSpec) — same Phase 2
                weight spec EMCADNet takes, e.g. ``{"name": "resnet50",
                "weights": "timm:resnet50.a1_in1k"}``. Defaults to an
                untrained pvt_v2_b2 (weights: none) so ``get_model(name=
                "encoder_unet")`` works with no config, matching every other
                registered family's zero-config-buildable convention.
        """
        super().__init__()
        encoder_spec = encoder if encoder is not None else {"name": "pvt_v2_b2", "weights": "none"}
        built = build_encoder(encoder_spec, in_chans=in_channels)
        self.encoder = built.module
        self.pretrained_cfg = built.pretrained_cfg
        self.weight_record = built.weight_record

        channels = built.channels  # deepest stage first
        n_stages = len(channels)

        self.blocks = nn.ModuleList()
        prev_channels = channels[0]
        for stage_idx in range(1, n_stages):
            skip_channels = channels[stage_idx]
            self.blocks.append(DecoderBlock(prev_channels + skip_channels, skip_channels, bilinear=bilinear))
            prev_channels = skip_channels

        self.refine = DoubleConv(prev_channels, prev_channels)
        self.out_conv = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)          # shallow -> deep (timm's features_only convention)
        feats = list(reversed(feats))    # deepest -> shallow, matching self.encoder's channel order

        d = feats[0]
        for i, block in enumerate(self.blocks):
            d = block(d, feats[i + 1])

        d = self.refine(d)
        out = self.out_conv(d)
        if out.shape[-2:] != x.shape[-2:]:
            # The shallowest extracted stage is rarely at full input
            # resolution (e.g. stride 4 for PVTv2/ResNet) — interpolate
            # directly to the input size rather than computing an exact
            # upsample factor per architecture.
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out
