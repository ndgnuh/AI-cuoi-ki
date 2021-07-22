import torch
import torch.nn as nn

import blocks


class ResNet(nn.Module):

    def __init__(s, Block, repeats, classes=100, in_channel=3, basesize=64):
        super().__init__()
        s.basesize = basesize
        s.expansion = 4 if Block == blocks.ResidualBottleneckBlock else 1

        s.entry = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, padding=1)
        )

        strides = [1, 2, 2, 2]
        in_channels = [64, 64, 128, 256]
        out_channels = [64, 128, 256, 512]
        s.blocks = nn.Sequential(
            *blocks.residual_repeat(Block, repeats[0],
                                    64, 64, first_block=True),
            *blocks.residual_repeat(Block, repeats[1], 64,
                                    128, first_block=False),
            *blocks.residual_repeat(Block, repeats[2], 128,
                                    256, first_block=False),
            *blocks.residual_repeat(Block, repeats[3], 256,
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
    '18': [blocks.ResidualBasicBlock, [2, 2, 2, 2]],
    '34': [blocks.ResidualBasicBlock, [3, 4, 6, 3]],
    '50': [blocks.ResidualBottleneckBlock, [3, 4, 6, 3]],
    '101': [blocks.ResidualBottleneckBlock, [3, 4, 23, 3]],
    '152': [blocks.ResidualBottleneckBlock, [3, 8, 36, 3]]
}


def ResNetModel(size):
    Block, repeats = config[size]

    def _(*args, **kwargs):
        return ResNet(Block, repeats, *args, **kwargs)
    return _


models = {'ResNet': ResNet}
for size in config.keys():
    models["ResNet" + size] = ResNetModel(size)
