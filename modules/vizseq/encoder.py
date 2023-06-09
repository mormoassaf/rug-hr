""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5


class VizSeqEncoder(torch.nn.Module):

    def __init__(self,
                 embedding_size=768,
                 max_sequence_length=256,
                 in_channels=1,
                 spatial_size=(128, 2048),
                 ):
        super().__init__()
        self.spatial_size = spatial_size
        self.encoder = UNetBasicEncoder(in_channels=in_channels)
        # output size is (batch, last_channel=1024, spatial_size[0]//2**4, spatial_size[1]//2**4)
        #  output of encoding2sequence goal (batch, sequence_length, embedding_size)
        out_h = spatial_size[0] // (2 ** 4)
        out_w = spatial_size[1] // (2 ** 4)
        self.encoding2embedding = torch.nn.Sequential(
            torch.nn.Conv2d(1024, max_sequence_length, kernel_size=(1, 1)),
            # outputs (batch, embedding_size, out_h, out_w)
            torch.nn.ReLU(),  # outputs (batch, max_sequence_length, out_h, out_w)
            torch.nn.Flatten(start_dim=2),  # outputs (batch, max_sequence_length, out_h*out_w)
            torch.nn.Linear(out_h * out_w, embedding_size),
            # outputs (batch, sequence_length, embedding_size)
        )
        # resnet = models.resnet50(pretrained=True)
        # layers = list(resnet.children())  # remove the last layer
        # layers[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # layers = layers[:-3]
        # self.encoder = torch.nn.Sequential(*layers)
        # self.encoding2embedding = torch.nn.Sequential(
        #     torch.nn.Conv2d(1024, max_sequence_length, kernel_size=(1, 1)), # outputs (batch, max_sequence_length, 8, 128)
        #     torch.nn.SiLU(), # outputs (batch, embedding_size, 8, 128)
        #     torch.nn.Flatten(start_dim=2), # outputs (batch, max_sequence_length, 8*128)
        #     torch.nn.Linear(8*128, embedding_size), # outputs (batch, max_sequence_length, vocab_size)
        # )

    """
    image: (batch_size, in_channels, *spatial_size)
    timestep: (batch_size, 1)

    """

    def forward(self, image):
        encoding = self.encoder(image)
        encoding = self.encoding2embedding(encoding)

        return encoding
