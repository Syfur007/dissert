import torch
import torch.nn as nn
from ..blocks import DoubleConv, EncoderBlock, DecoderBlock

class UNet(nn.Module):
    """
    Standard modular U-Net implementation.
    Allows customization of input channels, output classes/channels, and features list.
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
        
        # Decoder (Upsamplers)
        self.up1 = DecoderBlock(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = DecoderBlock(features[3], features[2] // factor, bilinear)
        self.up3 = DecoderBlock(features[2], features[1] // factor, bilinear)
        self.up4 = DecoderBlock(features[1], features[0], bilinear)
        
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
