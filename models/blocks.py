import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """A standard Conv -> BatchNorm -> ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    """A residual convolution block (Conv -> BN -> ReLU -> Conv -> BN + shortcut -> ReLU)."""
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class DoubleConv(nn.Module):
    """Two successive Convolution blocks, standard in U-Net architectures."""
    def __init__(self, in_channels, out_channels, mid_channels=None, bias=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ConvBlock(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            ConvBlock(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        return self.double_conv(x)

class EncoderBlock(nn.Module):
    """An encoder/down-sampling block: MaxPool followed by DoubleConv."""
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, bias=bias)

    def forward(self, x):
        return self.conv(self.maxpool(x))

class DecoderBlock(nn.Module):
    """A decoder/up-sampling block. Support Bilinear upsampling or Transposed Convolution."""
    def __init__(self, in_channels, out_channels, bilinear=True, bias=False):
        super().__init__()
        # If bilinear, use normal convolution to reduce channels after upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, bias=bias)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, bias=bias)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # In case the input size is odd, pad the upsampled tensor
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionBlock(nn.Module):
    """
    Additive Attention Gate (from Attention U-Net).
    Filters the skip connection features `x` (from encoder) using gating signal `g` (from decoder).
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Downsample/align gating signal W_g(g) and spatial feature map W_x(x)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Align spatial dimensions of gating signal and skip connection if they differ
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi
