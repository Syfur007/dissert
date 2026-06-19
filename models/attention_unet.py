import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv, EncoderBlock, AttentionBlock

class AttentionDecoderBlock(nn.Module):
    """
    Decoder block with an Attention Gate.
    Applies Attention to the encoder skip connection before concatenation.
    """
    def __init__(self, in_channels, out_channels, skip_channels, bilinear=True, bias=False):
        super().__init__()
        # in_channels is the channel count of the lower layer decoder input
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling, channels are in_channels // 2 (if we use standard reduction)
            # But Upsample itself doesn't reduce channels, so we need to account for it
            up_channels = in_channels // 2
            self.conv = DoubleConv(up_channels + skip_channels, out_channels, (up_channels + skip_channels) // 2, bias=bias)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_channels = in_channels // 2
            self.conv = DoubleConv(up_channels + skip_channels, out_channels, bias=bias)
            
        # Attention Gate: g is the gating signal (lower decoder, up_channels), x is the skip connection (skip_channels)
        self.att = AttentionBlock(F_g=up_channels, F_l=skip_channels, F_int=up_channels // 2)

    def forward(self, x_dec, x_skip):
        x_dec_up = self.up(x_dec)
        
        # Align spatial dimensions in case they differ
        diffY = x_skip.size()[2] - x_dec_up.size()[2]
        diffX = x_skip.size()[3] - x_dec_up.size()[3]
        x_dec_up = F.pad(x_dec_up, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        
        # Apply attention gate
        x_skip_att = self.att(g=x_dec_up, x=x_skip)
        
        # Concatenate and convolve
        x = torch.cat([x_skip_att, x_dec_up], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture.
    Uses attention gates to filter skip connections.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Initial double convolution
        self.inc = DoubleConv(in_channels, features[0])
        
        # Encoder (Downsamplers)
        self.down1 = EncoderBlock(features[0], features[1])
        self.down2 = EncoderBlock(features[1], features[2])
        self.down3 = EncoderBlock(features[2], features[3])
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = EncoderBlock(features[3], features[3] * 2 // factor)
        
        # Decoder (Upsamplers with Attention Gates)
        # Note: in_channels is features[i]*2 // factor, skip_channels is features[i-1]
        self.up1 = AttentionDecoderBlock(features[3] * 2, features[3] // factor, features[3], bilinear)
        self.up2 = AttentionDecoderBlock(features[3], features[2] // factor, features[2], bilinear)
        self.up3 = AttentionDecoderBlock(features[2], features[1] // factor, features[1], bilinear)
        self.up4 = AttentionDecoderBlock(features[1], features[0], features[0], bilinear)
        
        # Output classification projection
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
