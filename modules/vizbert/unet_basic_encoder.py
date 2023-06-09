""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        batch, c, h, w = x.size()
        x = x.view(batch, self.channels, h * w).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        # attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1).view(batch, c, h, w)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """


class UNetBasicEncoder(nn.Module):
    def __init__(self, in_channels, bilinear=False):
        super(UNetBasicEncoder, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.sa1 = (SelfAttention(128))
        self.down2 = (Down(128, 256))
        self.sa2 = SelfAttention(256)
        self.down3 = (Down(256, 512))
        self.sa3 = (SelfAttention(512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.sa4 = (SelfAttention(1024 // factor))

    def forward(self, x):
        x1 = self.inc(x)
        print("inc: ", x1.shape)
        x2 = self.down1(x1)
        print("down1: ", x2.shape)
        # x2 = self.sa1(x2)
        # print("sa1: ", x2.shape)
        x3 = self.down2(x2)
        print("down2: ", x3.shape)
        x3 = self.sa2(x3)
        print("sa2: ", x3.shape)
        x4 = self.down3(x3)
        print("down3: ", x4.shape)
        x4 = self.sa3(x4)
        print("sa3: ", x4.shape)
        x5 = self.down4(x4)
        print("down4: ", x5.shape)
        # x5 = self.sa4(x5) 

        return x5

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
