import torch
from torch import nn


def residual_repeat(Block, n, in_channel, out_channel, first_block):
    blocks = []
    for i in range(n):
        if i == 0 and not first_block:
            blocks.append(Block(
                in_channel,
                out_channel,
                conv1x1=True,
                stride=2
            ))
        else:
            blocks.append(Block(
                out_channel,
                out_channel,
                conv1x1=False,
                stride=1
            ))
    return blocks


class ResidualBasicBlock(nn.Module):
    def __init__(s, in_channel, out_channel, stride=1):
        super().__init__()
        s.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channel)
        )
        if stride != 1 or in_channel != out_channel:
            s.short = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel)
            )
        else:
            s.short = lambda x: x
        s.final = nn.ReLU(inplace=True)

    def forward(s, x):
        y1 = s.layers(x)
        y2 = s.short(x)
        y = y1 + y2
        y = s.final(y)
        return y


class ResidualBottleneckBlock(nn.Module):

    def __init__(s, in_channel, out_channel, stride=1):
        super().__init__()
        s.expansion = 4
        s.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel * s.expansion, 1, bias=False),
            nn.BatchNorm2d(out_channel * s.expansion)
        )
        if stride != 1 or in_channel != out_channel * s.expansion:
            s.short = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel)
            )
        else:
            s.short = lambda x: x

        s.final = nn.ReLU(inplace=True)

    def forward(s, x):
        y1 = s.layers(x)
        y2 = s.short(x)
        return s.final(y1 + y2)


class SpatialSqueezeBlock(nn.Module):
    def __init__(s, in_channel, N=2):
        super().__init__()
        s.pool = nn.AvgPool2d(N)
        s.dwconv = nn.Conv2d(in_channel, in_channel, 3,
                             padding=1, bias=False, groups=in_channel)
        s.prelu = nn.PReLU()
        s.pwconv = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        s.upsample = nn.ConvTranspose2d(
            in_channel, in_channel, 2, stride=2, bias=False)

    def forward(s, x):
        y = s.pool(x)
        y = s.dwconv(y)
        y = s.prelu(y)
        y = s.pwconv(y)
        y = s.upsample(y)
        return y


class SpatialSqueezeModule(nn.Module):
    def __init__(s, in_channel):
        super().__init__()
        s.group = in_channel
        s.conv1 = nn.Conv2d(in_channel, in_channel //
                            2, 1, bias=False, groups=1)
        s.squeeze1 = SpatialSqueezeBlock(in_channel // 2)
        s.squeeze2 = SpatialSqueezeBlock(in_channel // 2)
        s.prelu = nn.PReLU()

    def forward(s, y):
        y = channel_shuffle(y, s.group)
        y0 = y
        y = s.conv1(y)
        y1 = s.squeeze1(y)
        y2 = s.squeeze2(y)
        y = torch.cat([y1, y2], dim=1)
        y = y + y0
        y = s.prelu(y)
        return y
