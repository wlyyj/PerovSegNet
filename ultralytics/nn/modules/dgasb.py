from timm.models.layers import DropPath
import torch.nn as nn
from torch import Tensor
import torch
from .conv import Conv

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

class ConvFFN(nn.Module):
    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pw1 = Conv(in_channels, internal_channels, k=1, s=1, g=1)
        self.se = SELayer(internal_channels, 16)
        self.pw2 = Conv(internal_channels, out_channels, k=1, s=1, g=1)
        self.nonlinear = nn.GELU()
        self.add = in_channels == out_channels

    def forward(self, x):
        out = self.pw1(x)
        out = self.se(out)
        out = self.pw2(out)
        return x + self.drop_path(out) if self.add else self.drop_path(out)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(DilatedConv, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DGSTkg(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(DGSTkg, self).__init__()
        nn.SiLU()  # default activation
        branch_features = output_c // 4
        self.conv = Conv(input_c, output_c, k=1, s=1, g=1)
        self.branch = nn.Sequential(
            self.depthwise_conv(branch_features, branch_features, kernel_s=7, stride=1, padding=3),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )
        self.branch1 = SELayer(branch_features, 4)
        self.branch2 = DilatedConv(branch_features, branch_features, kernel_size=3, dilation=2)  # 使用空洞卷积
        self.branch3 = Conv(branch_features, branch_features, 3, 1, g=1, p=1)

        self.convffn = ConvFFN(output_c, output_c * 4, output_c, drop_path=0.2)

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        out = torch.cat((x1, self.branch3(x2), self.branch2(x3), self.branch(x4)), dim=1)
        out = channel_shuffle(out, 2)
        out = self.convffn(out)
        return out


