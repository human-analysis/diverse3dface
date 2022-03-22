import torch
import torch.nn as nn
import torch.nn.functional as F

# 2D-Unet Model taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch))
        self.act = nn.ReLU(inplace=True)

        self.res_conv = None
        self.res_norm = None
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1, padding=0)
            self.res_norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        residual = x

        x = self.conv(x)

        if self.res_conv is not None:
            residual = self.res_norm(self.res_conv(residual))

        x += residual
        x = self.act(x)

        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UnetSeg(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, final_act='Softmax2d', intermediate=False, dropout_p=0):
        super(UnetSeg, self).__init__()
        self.n_channels = in_channels
        self.n_classes =  out_channels
        self.intermediate = intermediate
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)

        self.inc = InConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.down5 = Down(256, 256)
        self.up1 = Up(512, 256, bilinear=bilinear)
        self.up2 = Up(512, 128, bilinear=bilinear)
        self.up3 = Up(256, 64, bilinear=bilinear)
        self.up4 = Up(128, 32, bilinear=bilinear)
        self.up5 = Up(64, 32, bilinear=bilinear)
        self.outc = OutConv(32, out_channels)
        self.final_act = getattr(nn, final_act)()

    def forward(self, x):
        x1 = self.inc(x)
        if hasattr(self, 'dropout'):
            x1 = self.dropout(x1)
        x2 = self.down1(x1)
        if hasattr(self, 'dropout'):
            x2 = self.dropout(x2)
        x3 = self.down2(x2)
        if hasattr(self, 'dropout'):
            x3 = self.dropout(x3)
        x4 = self.down3(x3)
        if hasattr(self, 'dropout'):
            x4 = self.dropout(x4)
        x5 = self.down4(x4)
        if hasattr(self, 'dropout'):
            x5 = self.dropout(x5)
        x6 = self.down5(x5)
        if hasattr(self, 'dropout'):
            x6 = self.dropout(x6)
        x = self.up1(x6, x5)
        if self.intermediate:
            interm = [self.interm1(x)]
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.up2(x, x4)
        if self.intermediate:
            interm.append(self.interm2(x))
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.up3(x, x3)
        if self.intermediate:
            interm.append(self.interm3(x))
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.up4(x, x2)
        if self.intermediate:
            interm.append(self.interm4(x))
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.up5(x, x1)
        x = self.outc(x)
        x = self.final_act(x)

        if self.intermediate:
            return x, interm
        else:
            return x
