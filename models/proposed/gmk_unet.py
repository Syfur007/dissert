import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


__all__ = ['GMK_UNet']


from ..blocks import _init_weights, act_layer as _act, channel_shuffle as _channel_shuffle, ChannelAttention, SpatialAttention
from ..registry import MODEL_REGISTRY

class _MultiKernelDWConv(nn.Module):
    def __init__(self, channels, kernel_sizes, stride, activation='relu6', parallel=True):
        super().__init__()
        self.parallel = parallel
        self.dwconvs  = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, k, stride, k // 2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                _act(activation, inplace=True),
            ) for k in kernel_sizes
        ])
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, x):
        out = []
        for dw in self.dwconvs:
            y = dw(x)
            out.append(y)
            if not self.parallel:
                x = x + y
        return out


class _MKIR(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion=2, parallel=True,
                 add=True, kernel_sizes=[1, 3, 5], activation='relu6'):
        super().__init__()
        assert stride in (1, 2)
        self.add        = add
        self.use_skip   = stride == 1
        ex_c            = in_c * expansion
        n_scales        = len(kernel_sizes)
        combined        = ex_c if add else ex_c * n_scales

        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, ex_c, 1, bias=False),
            nn.BatchNorm2d(ex_c),
            _act(activation, inplace=True),
        )
        self.mkdwconv = _MultiKernelDWConv(ex_c, kernel_sizes, stride, activation, parallel)
        self.pconv2   = nn.Sequential(
            nn.Conv2d(combined, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = (
            nn.Conv2d(in_c, out_c, 1, bias=False) if (self.use_skip and in_c != out_c) else None
        )
        self._gcd     = math.gcd(combined, out_c)
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, x):
        dw_outs = self.mkdwconv(self.pconv1(x))
        feat    = sum(dw_outs) if self.add else torch.cat(dw_outs, dim=1)
        feat    = _channel_shuffle(feat, self._gcd)
        out     = self.pconv2(feat)
        if self.use_skip:
            res = self.shortcut(x) if self.shortcut is not None else x
            return res + out
        return out


def _mk_stage(in_c, out_c, n, expansion=2, parallel=True,
              add=True, kernel_sizes=[1, 3, 5], activation='relu6'):
    kw = dict(expansion=expansion, parallel=parallel, add=add,
              kernel_sizes=kernel_sizes, activation=activation)
    return nn.Sequential(
        _MKIR(in_c, out_c, 1, **kw),
        *[_MKIR(out_c, out_c, 1, **kw) for _ in range(n - 1)],
    )


# ---------------------------------------------------------------------------
# GMK-specific blocks
# ---------------------------------------------------------------------------

def _ycbcr(x):
    """Differentiable RGB → YCbCr (BT.601). Returns luma [B,1,H,W] and chroma [B,2,H,W]."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  =  0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b + 0.5
    cr =  0.500000 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y, torch.cat([cb, cr], dim=1)


class ExponentialDecayGating(nn.Module):
    """
    Fuses RGB, luma, and chroma encoder features with learnable softmax
    weights. log_w starts at zeros → uniform (1/3, 1/3, 1/3) initialisation.
    """
    def __init__(self, channels):
        super().__init__()
        def _proj():
            return nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        self.proj_rgb    = _proj()
        self.proj_luma   = _proj()
        self.proj_chroma = _proj()
        self.log_w = nn.Parameter(torch.zeros(3))

    def forward(self, f_rgb, f_luma, f_chroma):
        w = torch.softmax(self.log_w, dim=0)
        return (w[0] * self.proj_rgb(f_rgb) +
                w[1] * self.proj_luma(f_luma) +
                w[2] * self.proj_chroma(f_chroma))


class GroupedAttentionGate(nn.Module):
    """
    GAG with an optional dual-skip RGB branch (W_r).

    Construction-time switch: pass F_rgb to build W_r; pass None to disable.
    The W_r branch is never built when use_rgb_skip=False in GMK_UNet, so no
    dead parameters are allocated when the RGB skip is turned off.
    """
    def __init__(self, F_g, F_l, F_int, F_rgb=None, kernel_size=1, groups=1, activation='relu'):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        if F_g % groups != 0 or F_l % groups != 0 or F_int % groups != 0 or (F_rgb is not None and F_rgb % groups != 0):
            raise ValueError(
                f"GroupedAttentionGate: groups={groups} must evenly divide "
                f"F_g={F_g}, F_l={F_l}, F_int={F_int}"
                + (f", and F_rgb={F_rgb}" if F_rgb is not None else "")
                + ". This is normally called with groups=channels[i]//2, so "
                "an odd channel count at that stage is what triggers this — "
                "use an even 'channels' list, or pass a compatible 'groups' explicitly."
            )
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,   F_int, kernel_size, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,   F_int, kernel_size, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_r = None
        if F_rgb is not None:
            self.W_r = nn.Sequential(
                nn.Conv2d(F_rgb, F_int, kernel_size, padding=kernel_size // 2, groups=groups, bias=True),
                nn.BatchNorm2d(F_int),
            )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.act = _act(activation, inplace=True)
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, g, x, x_rgb=None):
        if (x_rgb is not None) != (self.W_r is not None):
            raise ValueError("x_rgb must be supplied iff F_rgb was provided at construction.")
        combined = self.W_g(g) + self.W_x(x)
        if x_rgb is not None:
            combined = combined + self.W_r(x_rgb)
        psi  = self.psi(self.act(combined))
        skip = x + x_rgb if x_rgb is not None else x
        return skip * psi


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@MODEL_REGISTRY.register("gmk_unet")
class GMK_UNet(nn.Module):
    """
    Guided Multi-color-space K-UNet.

    Three parallel MKIR encoder paths (RGB, luma Y, chroma CbCr) encode the
    input independently. At each of the four skip levels an EDG block fuses
    all three encoder outputs into a single gated skip. Depending on
    use_rgb_skip, the GAG at each decoder stage receives either the EDG-fused
    skip alone or both the EDG-fused skip and the raw RGB encoder output as a
    dual-skip signal. Luma/chroma encoders run for stages 1–4 only.

    Config knobs
    ------------
    use_rgb_skip : bool  (model.use_rgb_skip in YAML)
        True  — GAG receives EDG skip + raw RGB skip (dual-skip, W_r built).
        False — GAG receives only the EDG-fused skip (no W_r, no extra params).
    """
    def __init__(self, num_classes=1, in_channels=3,
                 channels=[16, 32, 64, 96, 160],
                 depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=2,
                 gag_kernel=3,
                 use_rgb_skip=True,
                 **kwargs):
        super().__init__()
        self.use_rgb_skip = use_rgb_skip
        kw = dict(expansion=expansion_factor, parallel=True, add=True, kernel_sizes=kernel_sizes)

        # RGB encoder — full 5-stage path to bottleneck
        self.rgb_enc1 = _mk_stage(in_channels, channels[0], depths[0], **kw)
        self.rgb_enc2 = _mk_stage(channels[0], channels[1], depths[1], **kw)
        self.rgb_enc3 = _mk_stage(channels[1], channels[2], depths[2], **kw)
        self.rgb_enc4 = _mk_stage(channels[2], channels[3], depths[3], **kw)
        self.rgb_enc5 = _mk_stage(channels[3], channels[4], depths[4], **kw)

        # Luma encoder — stages 1–4 (Y: 1 channel in, channels[i] out)
        self.luma_enc1 = _mk_stage(1,           channels[0], depths[0], **kw)
        self.luma_enc2 = _mk_stage(channels[0], channels[1], depths[1], **kw)
        self.luma_enc3 = _mk_stage(channels[1], channels[2], depths[2], **kw)
        self.luma_enc4 = _mk_stage(channels[2], channels[3], depths[3], **kw)

        # Chroma encoder — stages 1–4 (CbCr: 2 channels in, channels[i] out)
        self.chroma_enc1 = _mk_stage(2,           channels[0], depths[0], **kw)
        self.chroma_enc2 = _mk_stage(channels[0], channels[1], depths[1], **kw)
        self.chroma_enc3 = _mk_stage(channels[1], channels[2], depths[2], **kw)
        self.chroma_enc4 = _mk_stage(channels[2], channels[3], depths[3], **kw)

        # EDG fusion at each skip level
        self.edg1 = ExponentialDecayGating(channels[0])
        self.edg2 = ExponentialDecayGating(channels[1])
        self.edg3 = ExponentialDecayGating(channels[2])
        self.edg4 = ExponentialDecayGating(channels[3])

        # GAG — W_r branch is only built when use_rgb_skip=True.
        # When False: F_rgb=None → no W_r parameters, no dual-skip at runtime.
        f_rgb = channels  # shorthand; indexed per stage below
        def _gag(i):
            c    = channels[i]
            frgb = c if use_rgb_skip else None
            return GroupedAttentionGate(c, c, c // 2, F_rgb=frgb,
                                        kernel_size=gag_kernel, groups=c // 2)
        self.AG1 = _gag(3)
        self.AG2 = _gag(2)
        self.AG3 = _gag(1)
        self.AG4 = _gag(0)

        # Decoder
        self.decoder1 = _mk_stage(channels[4], channels[3], 1, **kw)
        self.decoder2 = _mk_stage(channels[3], channels[2], 1, **kw)
        self.decoder3 = _mk_stage(channels[2], channels[1], 1, **kw)
        self.decoder4 = _mk_stage(channels[1], channels[0], 1, **kw)
        self.decoder5 = _mk_stage(channels[0], channels[0], 1, **kw)

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)
        self.SA  = SpatialAttention()

        self.out = nn.Conv2d(channels[0], num_classes, 1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        luma, chroma = _ycbcr(x)

        r1 = F.max_pool2d(self.rgb_enc1(x),        2, 2)
        l1 = F.max_pool2d(self.luma_enc1(luma),     2, 2)
        c1 = F.max_pool2d(self.chroma_enc1(chroma), 2, 2)
        s1 = self.edg1(r1, l1, c1)

        r2 = F.max_pool2d(self.rgb_enc2(r1), 2, 2)
        l2 = F.max_pool2d(self.luma_enc2(l1), 2, 2)
        c2 = F.max_pool2d(self.chroma_enc2(c1), 2, 2)
        s2 = self.edg2(r2, l2, c2)

        r3 = F.max_pool2d(self.rgb_enc3(r2), 2, 2)
        l3 = F.max_pool2d(self.luma_enc3(l2), 2, 2)
        c3 = F.max_pool2d(self.chroma_enc3(c2), 2, 2)
        s3 = self.edg3(r3, l3, c3)

        r4 = F.max_pool2d(self.rgb_enc4(r3), 2, 2)
        l4 = F.max_pool2d(self.luma_enc4(l3), 2, 2)
        c4 = F.max_pool2d(self.chroma_enc4(c3), 2, 2)
        s4 = self.edg4(r4, l4, c4)

        out = F.max_pool2d(self.rgb_enc5(r4), 2, 2)

        # use_rgb_skip controls whether the raw RGB tensor is passed as the
        # second skip argument to each GAG. When False, x_rgb=None and GAG
        # gates only on the EDG-fused skip (W_r was not built, so no params
        # are wasted).
        rgb_skips = (r4, r3, r2, r1) if self.use_rgb_skip else (None, None, None, None)

        # size= (not scale_factor=) so each upsample lands exactly on its
        # skip tensor's spatial dims even when H/W aren't multiples of 32 —
        # floor-division in the encoder's max_pool2d can make the
        # exact-scale_factor-2 upsample a pixel or two off from the skip
        # tensor it's about to be gated/added against.
        out = self.CA1(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder1(out), size=s4.shape[2:], mode='bilinear', align_corners=False))
        out = out + self.AG1(g=out, x=s4, x_rgb=rgb_skips[0])

        out = self.CA2(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder2(out), size=s3.shape[2:], mode='bilinear', align_corners=False))
        out = out + self.AG2(g=out, x=s3, x_rgb=rgb_skips[1])

        out = self.CA3(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder3(out), size=s2.shape[2:], mode='bilinear', align_corners=False))
        out = out + self.AG3(g=out, x=s2, x_rgb=rgb_skips[2])

        out = self.CA4(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder4(out), size=s1.shape[2:], mode='bilinear', align_corners=False))
        out = out + self.AG4(g=out, x=s1, x_rgb=rgb_skips[3])

        out = self.CA5(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder5(out), size=x.shape[2:], mode='bilinear', align_corners=False))

        return self.out(out)