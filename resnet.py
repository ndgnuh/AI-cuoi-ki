import torch
import torch.nn as nn


def residual_repeat(Block, n, in_channel, out_channel, first_block):
    blocks = []
    for i in range(n):
        if i == 0 and not first_block:
            blocks.append(Block(
                in_channel,
                out_channel,
                stride=2
            ))
        else:
            blocks.append(Block(
                out_channel,
                out_channel,
                stride=1
            ))
    return blocks


class ResidualBasicBlock(nn.Module):
    expansion = 1

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
    expansion = 4

    def __init__(s, in_channel, out_channel, stride=1):
        super().__init__()
        expansion = ResidualBasicBlock.expansion

        s.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel * expansion, 1, bias=False),
            nn.BatchNorm2d(out_channel * expansion)
        )
        if stride != 1 or in_channel != out_channel * expansion:
            s.short = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * expansion, 1),
                nn.BatchNorm2d(out_channel * expansion)
            )
        else:
            s.short = lambda x: x

        s.final = nn.ReLU(inplace=True)

    def forward(s, x):
        y1 = s.layers(x)
        y2 = s.short(x)
        return s.final(y1 + y2)


class ResNet(nn.Module):

    def __init__(s, Block, repeats, classes=100, in_channel=3, basesize=64):
        super().__init__()
        s.basesize = basesize
        s.expansion = 4 if Block == ResidualBottleneckBlock else 1

        s.entry = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, padding=1)
        )

        strides = [1, 2, 2, 2]
        in_channels = [64, 64, 128, 256]
        out_channels = [64, 128, 256, 512]
        s.blocks = nn.Sequential(
            *residual_repeat(Block, repeats[0],
                             64, 64, first_block=True),
            *residual_repeat(Block, repeats[1], 64,
                             128, first_block=False),
            *residual_repeat(Block, repeats[2], 128,
                             256, first_block=False),
            *residual_repeat(Block, repeats[3], 256,
                             512, first_block=False),
        )

        s.pool = nn.AvgPool2d(2)
        s.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, classes),
            nn.LeakyReLU()
        )

    def featuremap(s, x):
        y = s.entry(x)
        y = s.blocks(y)
        y = s.pool(y)
        return y

    def forward(s, x):
        y = s.featuremap(x)
        y = s.out(y)
        return y


config = {
    '18': [ResidualBasicBlock, [2, 2, 2, 2]],
    '34': [ResidualBasicBlock, [3, 4, 6, 3]],
    '50': [ResidualBottleneckBlock, [3, 4, 6, 3]],
    '101': [ResidualBottleneckBlock, [3, 4, 23, 3]],
    '152': [ResidualBottleneckBlock, [3, 8, 36, 3]]
}


def ResNet18(*args, **kwargs):
    Block, repeats = config['18']
    return ResNet(Block, repeats, *args, **kwargs)


def ResNet34(*args, **kwargs):
    Block, repeats = config['34']
    return ResNet(Block, repeats, *args, **kwargs)


def ResNet50(*args, **kwargs):
    Block, repeats = config['50']
    return ResNet(Block, repeats, *args, **kwargs)


def ResNet101(*args, **kwargs):
    Block, repeats = config['101']
    return ResNet(Block, repeats, *args, **kwargs)


def ResNet152(*args, **kwargs):
    Block, repeats = config['152']
    return ResNet(Block, repeats, *args, **kwargs)


models = {
    'ResNet': ResNet,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
}
