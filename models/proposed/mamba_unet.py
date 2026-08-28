"""
models/proposed/mamba_unet.py — Mamba/VSS hybrid model family (spec §6,
IMPLEMENTATION_PLAN.md Phase 6).

Primary encoder is explicitly MK-UNet's existing encoder — this module
imports and reuses `mk_irb_bottleneck` / `MultiKernelInvertedResidualBlock`
from models/baseline/mk_unet.py rather than reimplementing it, so the two
models share the exact same primary-branch numerics; any performance
difference is attributable to the auxiliary VSS branch and fusion, not to
a second, drifted copy of the encoder.

Auxiliary encoder: 4 stages of models/auxiliary/vss.py's VSS/PVM blocks,
each stage downsampling by 2x via a strided-conv "patch merging" step —
producing feature maps at the *same* 4 resolutions
(H/2, H/4, H/8, H/16) MK-UNet's t1..t4 skip connections use, so
per-stage fusion (models/fusion.py) is shape-compatible without any
resizing. VSSBlock operates channels-last (B,H,W,C); this module permutes
at the primary/auxiliary boundary, since every other branch in this
codebase (including the shared decoder) is channels-first (B,C,H,W).

**Channel input decision** (per IMPLEMENTATION_PLAN.md's Phase 6 section):
the auxiliary VSS encoder consumes the *same* multi-channel input as the
primary encoder (whatever dataset.channel_mode produces via Phase 4) — both
branches' stems take `in_channels`, not a hardcoded 3. This is deliberate:
if the auxiliary branch got raw RGB independently while the primary got
the full channel_mode stack, Phase 11's per-channel-group Shapley
attribution couldn't tell whether a channel group's contribution came from
the primary or auxiliary branch.

No pretrained weights are used by this model (MK-UNet's encoder is trained
from scratch here, same as the baseline MK-UNet configs), so the spec's
"stem inflation" knob (mean-copy | zero-init | random, for adapting
pretrained RGB-conv weights to a wider channel_mode input) doesn't apply
yet — it becomes relevant once/if a pretrained-backbone variant of this
model is added.
"""
from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baseline.mk_unet import mk_irb_bottleneck
from ..decoder import MambaDecoder, SkipPolicy
from ..fusion import FusionMode, build_fusion_stages
from ..registry import MODEL_REGISTRY
from ..auxiliary.vss import build_vss_stage


class _PatchMerge(nn.Module):
    """2x spatial downsample + channel-width change, via a strided conv —
    the auxiliary branch's between-stage downsampling (the VSS-literature
    analogue of MK-UNet's maxpool2d between encoder stages)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _AuxiliaryVSSEncoder(nn.Module):
    """4 VSS stages at H/2, H/4, H/8, H/16 — matching MK-UNet's t1..t4
    skip resolutions exactly. Returns the list of 4 stage outputs
    (channels-first), shallow -> deep.
    """

    def __init__(
        self,
        in_channels: int,
        stage_channels: List[int],  # length 4, shallow -> deep
        depths: List[int],          # length 4, per-stage VSS block count
        d_state: int,
        expand: int,
        scan_directions: int,
        merge: str,
        pvm_groups: int,
        drop_path_rate: float,
    ):
        super().__init__()
        if len(stage_channels) != 4 or len(depths) != 4:
            raise ValueError(
                f"_AuxiliaryVSSEncoder needs exactly 4 stage widths and depths "
                f"(one per skip resolution), got {len(stage_channels)} widths, {len(depths)} depths"
            )
        # Stem: full-res input -> H/2 at stage_channels[0], matching t1's
        # resolution (MK-UNet's t1 is already post-maxpool, i.e. H/2 — see
        # module docstring).
        self.stem = _PatchMerge(in_channels, stage_channels[0])

        self.stages = nn.ModuleList([
            build_vss_stage(
                d_model=stage_channels[i], depth=depths[i], d_state=d_state,
                expand=expand, scan_directions=scan_directions, merge=merge,
                pvm_groups=pvm_groups, drop_path_rate=drop_path_rate,
            )
            for i in range(4)
        ])
        self.downsamples = nn.ModuleList([
            _PatchMerge(stage_channels[i], stage_channels[i + 1]) for i in range(3)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        x = self.stem(x)  # (B, C0, H/2, W/2)
        for i, stage in enumerate(self.stages):
            x_chlast = x.permute(0, 2, 3, 1)  # VSSBlock is channels-last
            x_chlast = stage(x_chlast)
            x = x_chlast.permute(0, 3, 1, 2)  # back to channels-first
            outs.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        return outs

    @property
    def scan_impl(self) -> str:
        """Which selective-scan implementation actually ran — see
        models/auxiliary/ss2d.py's docstring. Every VSSBlock in this
        encoder resolves to the same value (a process-wide availability
        check), so the first stage's first block represents the whole
        encoder."""
        return self.stages[0][0].scan_impl


@MODEL_REGISTRY.register("mamba_unet")
class MambaUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        channels: List[int] = [16, 32, 64, 96, 160],
        depths: List[int] = [1, 1, 1, 1, 1],
        kernel_sizes: List[int] = [1, 3, 5],
        expansion_factor: int = 2,
        # Auxiliary VSS encoder (spec §6):
        vss_depths: Optional[List[int]] = None,  # default: depths[:4]
        d_state: int = 16,
        vss_expand: int = 2,
        scan_directions: Literal[1, 2, 4, 8] = 4,
        merge: Literal["sum", "learned"] = "sum",
        pvm_groups: int = 1,
        drop_path_rate: float = 0.1,
        # Fusion (spec §6):
        fusion: FusionMode = "cbffm",
        fuse_stages: Optional[List[int]] = None,  # default: all 4 stages
        # Decoder (spec §6):
        skip_policy: SkipPolicy = "fused",
        bilinear: bool = True,
        **kwargs,
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError(f"channels must have 5 entries (4 skip stages + bottleneck), got {len(channels)}")
        vss_depths = vss_depths if vss_depths is not None else list(depths[:4])

        # ── Primary encoder: MK-UNet's own blocks, reused not reimplemented ──
        self.encoder1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], 1, expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1, expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1, expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1, expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], 1, expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        # ── Auxiliary encoder: same multi-channel input as the primary ──
        self.auxiliary_encoder = _AuxiliaryVSSEncoder(
            in_channels=in_channels, stage_channels=list(channels[:4]), depths=vss_depths,
            d_state=d_state, expand=vss_expand, scan_directions=scan_directions,
            merge=merge, pvm_groups=pvm_groups, drop_path_rate=drop_path_rate,
        )

        # ── Fusion (per stage) ──
        self.fusion_stages = build_fusion_stages(fusion, list(channels[:4]), fuse_stages=fuse_stages)

        # ── Shared decoder ──
        self.decoder = MambaDecoder(
            stage_channels=list(channels[:4]), bottleneck_channels=channels[4],
            num_classes=num_classes, skip_policy=skip_policy, bilinear=bilinear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            # Same defensive fallback GMK-UNet/EMCAD use for an
            # unexpectedly single-channel input; the *intended* path for a
            # genuinely grayscale dataset is Phase 4's
            # modality_effective_channels(), producing a properly-sized
            # multi-channel frame before the model ever sees it.
            x = x.repeat(1, 3, 1, 1)

        # Primary branch (MK-UNet encoder)
        t1 = F.max_pool2d(self.encoder1(x), 2, 2)
        t2 = F.max_pool2d(self.encoder2(t1), 2, 2)
        t3 = F.max_pool2d(self.encoder3(t2), 2, 2)
        t4 = F.max_pool2d(self.encoder4(t3), 2, 2)
        bottleneck = F.max_pool2d(self.encoder5(t4), 2, 2)

        # Auxiliary branch (VSS), same input x — see module docstring's
        # "Channel input decision".
        aux1, aux2, aux3, aux4 = self.auxiliary_encoder(x)

        primary_skips = [t1, t2, t3, t4]
        auxiliary_skips = [aux1, aux2, aux3, aux4]
        fused_skips = [
            self.fusion_stages[i](primary_skips[i], auxiliary_skips[i]) for i in range(4)
        ]

        return self.decoder(bottleneck, primary_skips, auxiliary_skips, fused_skips)

    @property
    def scan_impl(self) -> str:
        """Which selective-scan implementation the auxiliary branch ran on
        this process (mamba_ssm fused kernel vs. ss2d_ref pure-PyTorch
        fallback) — read by orchestration/manifest.py's caller so it's
        recorded in the run manifest alongside the determinism flag (see
        models/auxiliary/ss2d.py's docstring for why)."""
        return self.auxiliary_encoder.scan_impl
