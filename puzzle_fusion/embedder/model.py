from torch import nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def get_model(path=None, use_gpu=True):
    device = 'cuda' if use_gpu else 'cpu'
    model = UNet(3, 3)
    if path:
        state_dict = torch.load(path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] #remove 'module'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    if use_gpu:
        # model = nn.DataParallel(model)
        model.to(device)
    return model

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
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        self.down4 = (Down(128, 256))
        self.down5 = (Down(256, 256))
        self.down6 = (Down(256, 256))
        self.down7 = (Down(256, 32))
        # self.down8 = (Down(512, 1024))
        # self.up1 = (Up(1024, 512))
        self.up2 = (Up(32, 256))
        self.up3 = (Up(256, 256))
        self.up4 = (Up(256, 256))
        self.up5 = (Up(256, 128))
        self.up6 = (Up(128, 64))
        self.up7 = (Up(64, 32))
        self.up8 = (Up(32, 16))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x, pred_image=True):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)
        x = self.down7(x)
        x_embed = x
        if pred_image:
        # x = self.down8(x)
        # x = self.up1(x)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
            x = self.up5(x)
            x = self.up6(x)
            x = self.up7(x)
            x = self.up8(x)
            logits = self.outc(x)
            return logits
        else:
            return x
