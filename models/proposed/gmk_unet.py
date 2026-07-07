import math
import torch
from torch import nn
import torch.nn.functional as F

from functools import partial
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


__all__ = ['MK_UNet', 'GMK_UNet']


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


# ---------------------------------------------------------------------------
# Building blocks (unchanged from MK-UNet)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        if self.in_planes < ratio:
            ratio = self.in_planes
        self.reduced_channels = self.in_planes // ratio
        if self.out_planes is None:
            self.out_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(in_planes, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7, 11), 'kernel size must be 3, 7, or 11'
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class GroupedAttentionGate(nn.Module):
    """
    Attention gate with an optional third skip branch (W_r) for the dual-skip
    mode used in GMK_UNet, where both the EDG-fused skip and the raw RGB skip
    are passed simultaneously. When F_rgb is None the gate behaves identically
    to the original MK-UNet attention gate.

    Guard: W_r is only built when F_rgb is explicitly provided. The forward
    requires x_rgb only when W_r was built — passing x_rgb without W_r (or
    vice-versa) raises an error rather than silently gating with a mismatched
    signal.
    """
    def __init__(self, F_g, F_l, F_int, F_rgb=None, kernel_size=1, groups=1, activation='relu'):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_r = None
        if F_rgb is not None:
            self.W_r = nn.Sequential(
                nn.Conv2d(F_rgb, F_int, kernel_size, stride=1, padding=kernel_size // 2, bias=True),
                nn.BatchNorm2d(F_int)
            )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, g, x, x_rgb=None):
        if (x_rgb is not None) != (self.W_r is not None):
            raise ValueError(
                "x_rgb must be supplied iff F_rgb was provided at construction."
            )
        combined = self.W_g(g) + self.W_x(x)
        if x_rgb is not None:
            combined = combined + self.W_r(x_rgb)
        psi = self.psi(self.activation(combined))
        # Gate is applied to the full skip signal (EDG + RGB when dual-skip).
        skip = x + x_rgb if x_rgb is not None else x
        return skip * psi


class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super().__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, stride, k // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            )
            for k in kernel_sizes
        ])
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MultiKernelInvertedResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True,
                 add=True, kernel_sizes=[1, 3, 5], activation='relu6'):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip_connection = stride == 1

        ex_c = int(in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, ex_c, 1, bias=False),
            nn.BatchNorm2d(ex_c),
            act_layer(activation, inplace=True)
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(ex_c, kernel_sizes, stride, activation, dw_parallel)
        self.combined_channels = ex_c if add else ex_c * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.conv1x1 = None
        if self.use_skip_connection and in_c != out_c:
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add:
            dout = sum(dwconv_outs)
        else:
            dout = torch.cat(dwconv_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.conv1x1 is not None:
                x = self.conv1x1(x)
            return x + out
        return out


def mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2, dw_parallel=True,
                      add=True, kernel_sizes=[1, 3, 5], activation='relu6'):
    blocks = [MultiKernelInvertedResidualBlock(
        in_c, out_c, s, expansion_factor, dw_parallel, add, kernel_sizes, activation
    )]
    for _ in range(1, n):
        blocks.append(MultiKernelInvertedResidualBlock(
            out_c, out_c, 1, expansion_factor, dw_parallel, add, kernel_sizes, activation
        ))
    return nn.Sequential(*blocks)


# ---------------------------------------------------------------------------
# GMK-UNet additions
# ---------------------------------------------------------------------------

def rgb_to_ycbcr(x):
    """
    Differentiable RGB → YCbCr conversion (BT.601, values in [0, 1]).
    Returns luma [B,1,H,W] and chroma [B,2,H,W] (CbCr) separately.
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  =  0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b + 0.5
    cr =  0.500000 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y, torch.cat([cb, cr], dim=1)


class ExponentialDecayGating(nn.Module):
    """
    EDG block. Merges RGB, luma, and chroma encoder features at one scale
    using learnable softmax-weighted projections.

    All three encoder streams output the same number of channels at each stage
    (they share the same `channels` progression), so a single `channels` arg
    covers both input and output. The log_w parameter is initialised to zeros
    so softmax starts at uniform (1/3, 1/3, 1/3) weighting.
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


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# channels presets:
#   MK_UNet-T : [4,  8,  16,  24,  32]
#   MK_UNet-S : [8,  16, 32,  48,  80]
#   MK_UNet   : [16, 32, 64,  96,  160]
#   MK_UNet-M : [32, 64, 128, 192, 320]
#   MK_UNet-L : [64, 128,256, 384, 512]

class MK_UNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, channels=[16, 32, 64, 96, 160],
                 depths=[1, 1, 1, 1, 1], kernel_sizes=[1, 3, 5], expansion_factor=2,
                 gag_kernel=3, **kwargs):
        super().__init__()
        kw = dict(expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        self.encoder1 = mk_irb_bottleneck(in_channels,   channels[0], depths[0], 1, **kw)
        self.encoder2 = mk_irb_bottleneck(channels[0],   channels[1], depths[1], 1, **kw)
        self.encoder3 = mk_irb_bottleneck(channels[1],   channels[2], depths[2], 1, **kw)
        self.encoder4 = mk_irb_bottleneck(channels[2],   channels[3], depths[3], 1, **kw)
        self.encoder5 = mk_irb_bottleneck(channels[3],   channels[4], depths[4], 1, **kw)

        self.AG1 = GroupedAttentionGate(channels[3], channels[3], channels[3] // 2, kernel_size=gag_kernel, groups=channels[3] // 2)
        self.AG2 = GroupedAttentionGate(channels[2], channels[2], channels[2] // 2, kernel_size=gag_kernel, groups=channels[2] // 2)
        self.AG3 = GroupedAttentionGate(channels[1], channels[1], channels[1] // 2, kernel_size=gag_kernel, groups=channels[1] // 2)
        self.AG4 = GroupedAttentionGate(channels[0], channels[0], channels[0] // 2, kernel_size=gag_kernel, groups=channels[0] // 2)

        self.decoder1 = mk_irb_bottleneck(channels[4], channels[3], 1, 1, **kw)
        self.decoder2 = mk_irb_bottleneck(channels[3], channels[2], 1, 1, **kw)
        self.decoder3 = mk_irb_bottleneck(channels[2], channels[1], 1, 1, **kw)
        self.decoder4 = mk_irb_bottleneck(channels[1], channels[0], 1, 1, **kw)
        self.decoder5 = mk_irb_bottleneck(channels[0], channels[0], 1, 1, **kw)

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)
        self.SA  = SpatialAttention()

        self.out1 = nn.Conv2d(channels[2], num_classes, 1)
        self.out2 = nn.Conv2d(channels[1], num_classes, 1)
        self.out3 = nn.Conv2d(channels[0], num_classes, 1)
        self.out4 = nn.Conv2d(channels[0], num_classes, 1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        out = F.max_pool2d(self.encoder1(x),   2, 2); t1 = out
        out = F.max_pool2d(self.encoder2(out),  2, 2); t2 = out
        out = F.max_pool2d(self.encoder3(out),  2, 2); t3 = out
        out = F.max_pool2d(self.encoder4(out),  2, 2); t4 = out
        out = F.max_pool2d(self.encoder5(out),  2, 2)

        out = self.CA1(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False))
        out = torch.add(out, self.AG1(g=out, x=t4))

        out = self.CA2(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False))
        p1  = F.interpolate(self.out1(out), scale_factor=8, mode='bilinear', align_corners=False)
        out = torch.add(out, self.AG2(g=out, x=t3))

        out = self.CA3(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False))
        p2  = F.interpolate(self.out2(out), scale_factor=4, mode='bilinear', align_corners=False)
        out = torch.add(out, self.AG3(g=out, x=t2))

        out = self.CA4(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode='bilinear', align_corners=False))
        p3  = F.interpolate(self.out3(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.add(out, self.AG4(g=out, x=t1))

        out = self.CA5(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear', align_corners=False))
        p4  = self.out4(out)

        return p4  # deep-supervision variant: return [p4, p3, p2, p1]


class GMK_UNet(nn.Module):
    """
    Guided Multi-color-space K-UNet.

    Three parallel MKIR encoder paths (RGB, luma Y, chroma CbCr) operate on
    the input simultaneously. At each of the four skip levels an EDG block
    fuses the three encoder outputs into a single gated skip tensor. The GAG
    at each decoder stage receives both the EDG-fused skip and the raw RGB
    encoder output as a dual-skip signal. The decoder and CA/SA attention are
    identical to MK_UNet; the output is a single segmentation head.

    Luma/chroma encoders run for stages 1–4 only: their stage-5 bottleneck
    outputs are not consumed by any EDG or decoder, so stage-5 luma/chroma
    encoder blocks would be allocated and forward-passed but never contribute
    to gradients — unnecessary compute. RGB runs all 5 stages as usual.
    """
    def __init__(self, num_classes=1, in_channels=3, channels=[16, 32, 64, 96, 160],
                 depths=[1, 1, 1, 1, 1], kernel_sizes=[1, 3, 5], expansion_factor=2,
                 gag_kernel=3, **kwargs):
        super().__init__()
        kw = dict(expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        # RGB encoder (5 stages — full path to bottleneck)
        self.rgb_enc1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], 1, **kw)
        self.rgb_enc2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1, **kw)
        self.rgb_enc3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1, **kw)
        self.rgb_enc4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1, **kw)
        self.rgb_enc5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], 1, **kw)

        # Luma encoder (stages 1–4; Y is 1 channel)
        self.luma_enc1 = mk_irb_bottleneck(1,           channels[0], depths[0], 1, **kw)
        self.luma_enc2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1, **kw)
        self.luma_enc3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1, **kw)
        self.luma_enc4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1, **kw)

        # Chroma encoder (stages 1–4; CbCr is 2 channels)
        self.chroma_enc1 = mk_irb_bottleneck(2,           channels[0], depths[0], 1, **kw)
        self.chroma_enc2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1, **kw)
        self.chroma_enc3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1, **kw)
        self.chroma_enc4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1, **kw)

        # EDG fusion at each skip level
        self.edg1 = ExponentialDecayGating(channels[0])
        self.edg2 = ExponentialDecayGating(channels[1])
        self.edg3 = ExponentialDecayGating(channels[2])
        self.edg4 = ExponentialDecayGating(channels[3])

        # GAG with dual-skip (EDG-fused + raw RGB); F_rgb == F_l since both
        # are channels[i] — separate W_r branch lets the gate weight the two
        # sources differently rather than treating them identically.
        self.AG1 = GroupedAttentionGate(channels[3], channels[3], channels[3] // 2, F_rgb=channels[3], kernel_size=gag_kernel, groups=channels[3] // 2)
        self.AG2 = GroupedAttentionGate(channels[2], channels[2], channels[2] // 2, F_rgb=channels[2], kernel_size=gag_kernel, groups=channels[2] // 2)
        self.AG3 = GroupedAttentionGate(channels[1], channels[1], channels[1] // 2, F_rgb=channels[1], kernel_size=gag_kernel, groups=channels[1] // 2)
        self.AG4 = GroupedAttentionGate(channels[0], channels[0], channels[0] // 2, F_rgb=channels[0], kernel_size=gag_kernel, groups=channels[0] // 2)

        # Decoder (identical structure to MK_UNet)
        self.decoder1 = mk_irb_bottleneck(channels[4], channels[3], 1, 1, **kw)
        self.decoder2 = mk_irb_bottleneck(channels[3], channels[2], 1, 1, **kw)
        self.decoder3 = mk_irb_bottleneck(channels[2], channels[1], 1, 1, **kw)
        self.decoder4 = mk_irb_bottleneck(channels[1], channels[0], 1, 1, **kw)
        self.decoder5 = mk_irb_bottleneck(channels[0], channels[0], 1, 1, **kw)

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

        luma, chroma = rgb_to_ycbcr(x)

        # Parallel encoding — each stream is max-pooled at every stage.
        # EDG fuses all three at each skip level; raw RGB skip is kept
        # separately for the dual-skip GAG.
        r1 = F.max_pool2d(self.rgb_enc1(x),      2, 2)
        l1 = F.max_pool2d(self.luma_enc1(luma),   2, 2)
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

        # Bottleneck — RGB stream only
        out = F.max_pool2d(self.rgb_enc5(r4), 2, 2)

        # Decoder — CA/SA recalibrate bottleneck before each upsample;
        # GAG uses EDG skip (x=sN) + raw RGB skip (x_rgb=rN) as dual signal.
        out = self.CA1(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False))
        out = out + self.AG1(g=out, x=s4, x_rgb=r4)

        out = self.CA2(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False))
        out = out + self.AG2(g=out, x=s3, x_rgb=r3)

        out = self.CA3(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False))
        out = out + self.AG3(g=out, x=s2, x_rgb=r2)

        out = self.CA4(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode='bilinear', align_corners=False))
        out = out + self.AG4(g=out, x=s1, x_rgb=r1)

        out = self.CA5(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear', align_corners=False))

        return self.out(out)
