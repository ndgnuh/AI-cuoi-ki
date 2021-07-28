import torch.nn as nn
import torch
import torch.functional as F
from functools import reduce, singledispatch
import resnet


class FCN(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(FCN, self).__init__()
        self.entry = nn.Conv2d(in_channel, 64, 1, bias=False)

        def makelayer(cin, cout, repeat):
            layer = []
            for i in range(repeat):
                out = cout if i == repeat - 1 else cin
                layer.append(nn.Conv2d(cin, out, 3, padding=1, bias=False))
                layer.append(nn.ReLU())
            layer.append(nn.MaxPool2d(2, stride=2))
            layer.append(nn.BatchNorm2d(cout))
            return nn.Sequential(*layer)

        def upscale():
            return nn.ConvTranspose2d(out_channel, out_channel, 2, stride=2, bias=False)

        self.layer128 = makelayer(64, 128, 2)
        self.layer256 = makelayer(128, 256, 3)
        self.layer512 = makelayer(256, 512, 3)
        self.layer4096 = makelayer(512, 4096, 3)
        self.middle = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.middle512 = nn.Conv2d(512, out_channel, 1, bias=False)
        self.middle256 = nn.Conv2d(256, out_channel, 1, bias=False)
        self.middle128 = nn.Conv2d(128, out_channel, 1, bias=False)
        self.middle4096 = nn.Sequential(
            nn.Conv2d(4096, 4096, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(4096, 4096, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(4096, 4096, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(4096, out_channel, 1, bias=False)
        )
        self.decode1 = upscale()
        self.decode2 = upscale()
        self.decode3 = upscale()
        self.decode4 = upscale()
        self.output = nn.Sigmoid()
        # self.output = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(imsize * imsize, out_channel)
        # )

    # Size: size of image n * n
    def forward(self, x):  # size-n
        y = self.entry(x)  # size-n
        y128 = self.layer128(y)  # size n/2
        y256 = self.layer256(y128)  # size n/4
        y512 = self.layer512(y256)  # size n/8
        y4096 = self.layer4096(y512)  # size n/16
        y4096 = self.middle4096(y4096)
        y = self.decode1(y4096) + self.middle512(y512)  # size n/8
        y = self.decode2(y) + self.middle256(y256)  # size n/4
        y = self.decode3(y) + self.middle128(y128)  # size n / 2
        y = self.decode4(y) + self.middle(x)  # size n
        return self.output(torch.sum(y, dim=1))


class SkipConnection(nn.Module):
    def __init__(s, layers, con):
        super().__init__()
        s.layers = layers
        s.con = con
        for i, l in enumerate(layers):
            setattr(s, f'layer_{i}', l)

    def forward(s, x):
        if isinstance(s.layers, list):
            return s.forward_multiple(x)
        else:
            return s.forward_single(x)

    def forward_single(s, x):
        mx = s.layers(x)
        return s.con(x, mx)

    def forward_multiple(s, x):
        mxs = [m(x) for m in s.layers]
        return s.con([x]+mxs)


class Untitled(nn.Module):
    def __init__(s, in_channel, classes, imgsize):
        super().__init__()
        def samepad(n): return n//2
        s.entry = nn.Conv2d(in_channel, 64, 1, bias=False)
        s.conv = SkipConnection(
            [
                nn.Sequential(
                    nn.Conv2d(64, 64, i, padding=samepad(i), bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                )
                for i in [1, 3, 5, 7, 9]
            ],
            sum
        )
        s.out = nn.Sequential(
            nn.Conv2d(64, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.Flatten(),
            nn.Linear(imgsize * imgsize * in_channel, classes)
        )

    def forward(s, y):
        y = s.entry(y)
        y = s.conv(y)
        y = s.out(y)
        return y


class SegModel1(nn.Module):
    def __init__(s):
        super().__init__()
        s.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.AvgPool2d(2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
            nn.Sigmoid())

    def forward(s, y):
        return s.layers(y)


class SegModel2(nn.Module):
    def __init__(s):
        super().__init__()
        s.entry = nn.Conv2d(3, 32, 1)
        s.blocks = [s.entry]
        nblocks = 2
        for i in range(nblocks):
            block = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.AvgPool2d(2),
                nn.ReLU(inplace=True)
            )
            setattr(s, f"block{i}", block)
            s.blocks.append(block)
        for i in range(nblocks):
            block = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 2, stride=2),
                nn.ReLU(inplace=True)
            )
            s.blocks.append(block)
            setattr(s, f"up_block{i}", block)
        s.out = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        s.blocks.append(s.out)

    def forward(s, y):
        for block in s.blocks:
            y = block(y)
        return y


def all_models():
    models = {}
    models.update(resnet.models)

    G = globals()
    for k in G:
        if type(G[k]) == type and issubclass(G[k], nn.Module):
            models[k] = G[k]
    return models


all_models = all_models()
