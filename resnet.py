import torch
import torch.nn as nn

__all__ = ["ResNet"]


class SkipConnection(nn.Module):
    def __init__(self, layers, connection):
        super().__init__()
        self.layers = layers
        self.connection = connection

    def forward(self, x):
        if isinstance(self.connection, nn.Module):
            return self.connection(self.layers(x))
        else:
            return self.connection(x, self.layers(x))


def BasicBlock(c1, c2, connection, stride=1):
    return SkipConnection(
        nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=stride, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c2)
        ), connection)


def BottleNeck(c1, c2, connection, stride=1, expand=4):
    return SkipConnection(nn.Sequential(
        nn.Conv2d(c1, c2, 1, bias=False),
        nn.BatchNorm2d(c2),
        nn.ReLU(),
        nn.Conv2d(c2, c2, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c2),
        nn.ReLU(),
        nn.Conv2d(c2, c2 * expand, 1, bias=False),
        nn.BatchNorm2d(c2 * expand)
    ), connection)


def make_res_layer(Block, c1, c2, repeat, expand, stride=1):
    ce = c2 * expand
    if stride == 1 and c2 == c1:
        connection = sum
    else:
        connection = SkipConnection(nn.Sequential(
            nn.Conv2d(c2, c2, 1, stride=stride, bias=False),
            nn.BatchNorm2d(c2)
        ), sum)
    layers = [Block(c1, c2, connection, stride=stride)]
    for _ in range(repeat - 1):
        layers += [Block(ce, c2, sum)]

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    config = {
        18: ((2, 2, 2, 2), BasicBlock, 1),
        34: ((3, 4, 6, 3), BasicBlock, 1),
        50: ((3, 4, 6, 3), BottleNeck, 4),
        101: ((3, 4, 23, 3), BottleNeck, 4),
        152: ((3, 8, 36, 3), BottleNeck, 4),
    }

    def __init__(self, size, c1, c2):
        super().__init__()
        self.size = size

        channels = [64, 128, 256, 512]
        inchannel = channels[0]
        strides = [1, 1, 1, 2]
        repeats, Block, expand = ResNet.config[size]

        self.entry = nn.Sequential(
            nn.Conv2d(c1, inchannel, 7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(inchannel)
        )

        self.pooling = nn.MaxPool2d(3, padding=1, stride=2)

        self.layers = []
        for (outchannel, stride, repeat) in zip(channels, strides, repeats):
            self.layers += [make_res_layer(Block, inchannel, outchannel,
                                           repeat=repeat, expand=expand, stride=stride)]
            inchannel = outchannel * expand
        self.layers = nn.Sequential(*self.layers)
        if c2 is not None:
            self.head = nn.Sequential(
                nn.AvgPool2d(7, padding=3),
                nn.Flatten(),
                nn.Linear(channels[-1] * expand, c2)
            )
        else:
            self.head = None

    def forward(self, x):
        y = x
        y = self.entry(y)
        y = self.pooling(y)
        y = self.layers(y)
        if self.head:
            y = self.head(y)
        return y
