import torch.nn as nn
import torch
import torch.functional as F
from functools import reduce, singledispatch

class InvertedResidual(nn.Module):
    def __init__(s, i, o, st):
        super(InvertedResidual, s).__init__()
        assert st in [1, 2]

        s.res = i == o and st == 1
        s.feed = nn.Sequential(
                nn.Conv2d(i, o, 1),
                nn.BatchNorm2d(o),
                nn.ReLU6(inplace=True),
                nn.Conv2d(o, o, 3, groups=o, padding=1, stride=st),
                nn.BatchNorm2d(o),
                nn.ReLU6(inplace=True),
                nn.Conv2d(o, o, 1),
                )

    def forward(s, x):
        if s.res:
            return x + s.feed(x)
        else:
            return s.feed(x)

class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        settings = [
            [3, 16, 1],
            [16, 24, 1],
            [24, 24, 1],
            [24, 32, 1],
            [32, 32, 1],
            ]
        layers = []
        for (i, o, st) in settings:
            layers.append(InvertedResidual(i, o, st))
        layers.extend([
            nn.Conv2d(32, 1, 7, padding=3),
            nn.Sigmoid()
        ])
        self.feed = nn.Sequential(*layers)
            
    def forward(s, x):
        return s.feed(x)

class FCN8(nn.Module):

    def __init__(self, numclass = 1):
        self.__cepoch = 0
        super(FCN8, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv53 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, padding=3)
        self.dropout1 = nn.Dropout(0.85)

        self.conv7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)
        self.dropout2 = nn.Dropout(0.85)

        self.conv8 = nn.Conv2d(in_channels=4096, out_channels=numclass, kernel_size=1)

        self.tranconv1 = nn.ConvTranspose2d(in_channels=numclass, out_channels=512, kernel_size=4, stride=2)

        self.tranconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=2, output_padding=1)

        self.tranconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=numclass, kernel_size=16, stride=8, padding=4)

    def set_cepoch(self, ce):
        self.__cepoch = ce

    def forward(self, x):
        x = self.conv12(self.conv11(x))
        x = self.pool1(x)
        x = self.pool2(self.conv22(self.conv21(x)))
        x1 = self.pool3(self.conv33(self.conv32(self.conv31(x))))
        x2 = self.pool4(self.conv43(self.conv42(self.conv41(x1))))
        x = self.pool5(self.conv53(self.conv52(self.conv51(x2))))
        x = self.dropout1(self.conv6(x))
        x = self.dropout2(self.conv7(x))
        x = self.conv8(x)
        x = self.tranconv1(x)
        print(x.shape, x2.shape)
        x = x2 + x
        x = self.tranconv2(x)
        x = x1 + x
        x = self.tranconv3(x)
        return x

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
        y = self.output(y)
        return y


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


class SkipConnection(nn.Module):
    def __init__(s, skips, con):
        super().__init__()
        s.skips = skips
        s.con = con
        # Just for pretty printing
        # forward doesn't need this
        for (i, skip) in enumerate(skips):
            setattr(s, f"skip_{i}", skip)

    def forward(s, x):
        ys = [skip(x) for skip in s.skips]
        return s.con(x, *ys)


class SegModel3(nn.Module):
    def __init__(s, nblocks=4):
        super().__init__()
        # Build from the middle
        layer = SkipConnection([
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.MaxPool2d(2),  # downscale 2x
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 2, stride=2)  # upscale 2x
            )
        ], sum)
        layer = SkipConnection([
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                layer,
                nn.ConvTranspose2d(64, 32, 2, stride=2),
            )
        ], sum)
        layer = SkipConnection([
            nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=true),
                layer,
                nn.ConvTranspose2d(32, 3, 2, stride=2),
            )
        ], sum)
        s.layer = layer
        s.out = nn.Sequential(
            nn.Conv2d(3, 1, 1),
            nn.Sigmoid()
        )

    def forward(s, x):
        return s.out(s.layer(x))


def sumoutput(x, mx1, mx2):
    return mx1 + mx2


class SegModel4(nn.Module):
    def __init__(s, channels=None):
        super(SegModel4, s).__init__()
        if channels is None:
            channels = ([64, 128, 256, 512, 1024])
        channels.reverse()
        s.entry = nn.Conv2d(3, channels[-1], 7, padding=3)
        s.blocks = [s.entry]
        block = None
        for i, (c2, c1) in enumerate(zip(channels, channels[1:])):
            feed = nn.Sequential(
                *[b for b in [
                        nn.Conv2d(c1, c2, 3, padding=1),
                        block,
                        nn.AvgPool2d(2),
                        nn.BatchNorm2d(c2),
                        nn.LeakyReLU(inplace=True),
                        nn.ConvTranspose2d(c2, c1, 2, stride=2)
                ] if b is not None]
            )
            block = SkipConnection(
                [feed],
                sum
            )

        s.feed = block
        s.blocks.append(block)

        channels.reverse()
        s.output = nn.Sequential(
            nn.Conv2d(channels[0], 1, 7, padding=3),
            nn.Sigmoid()
        )
        s.blocks.append(s.output)

    def forward(s, x):
        y = x
        for b in s.blocks:
            y = b(y)
        return y

# Based on alex net
class SegModel5(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        s.pool1 = nn.MaxPool2d(3, stride=2)
        s.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        s.pool2 = nn.MaxPool2d(3, stride=2)
        s.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        s.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        s.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        s.pool5 = nn.MaxPool2d(3, stride=2)

        s.tconv1 = nn.ConvTranspose2d(256, 1, 3, stride=2)
        s.tconv2 = nn.ConvTranspose2d(1, 1, 3, stride=2)
        s.tconv3 = nn.ConvTranspose2d(1, 1, 3, stride=2)
        s.tconv4 = nn.ConvTranspose2d(1, 1, 11, stride=4)
        s.out = nn.Sigmoid()

    def forward(s, x):
        y = s.conv3(s.pool2(s.conv2(s.pool1(s.conv1(x)))))
        y = s.conv4(y) + y
        y = s.pool5(s.conv5(y))
        y = s.tconv1(y)
        y = s.tconv2(y)
        y = s.tconv3(y)
        y = s.tconv4(y)
        y = nn.functional.interpolate(y, (x.shape[2], x.shape[3]))
        y = s.out(y)
        return y

# Model 5 but add ReLu and BatchNorm
# and the tconv are 3 channels
class SegModel6(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        s.pool1 = nn.MaxPool2d(3, stride=2)
        s.relu1 = nn.ReLU6(inplace=True)
        s.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        s.pool2 = nn.MaxPool2d(3, stride=2)
        s.relu2 = nn.ReLU6(inplace=True)
        s.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        s.btnm3 = nn.BatchNorm2d(384)
        s.relu3 = nn.ReLU6(inplace=True)
        s.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        s.relu4 = nn.ReLU6(inplace=True)
        s.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        s.pool5 = nn.MaxPool2d(3, stride=2)
        s.btnm5 = nn.BatchNorm2d(256)
        s.relu5 = nn.ReLU6(inplace=True)

        s.tconv1 = nn.ConvTranspose2d(256, 1, 3, stride=2)
        s.tconv2 = nn.ConvTranspose2d(1, 1, 3, stride=2)
        s.tconv3 = nn.ConvTranspose2d(1, 1, 3, stride=2)
        s.tconv4 = nn.ConvTranspose2d(1, 1, 11, stride=4)
        s.out = nn.Sigmoid()

    def forward(s, x):
        y = s.conv1(x)
        y = s.pool1(y)
        y = s.relu1(y)
        y = s.conv2(y)
        y = s.pool2(y)
        y = s.relu2(y)
        y = s.conv3(y)
        y = s.btnm3(y)
        y = s.relu3(y)
        y = s.conv4(y) + y
        y = s.relu4(y)
        y = s.conv5(y)
        y = s.pool5(y)
        y = s.btnm5(y)
        y = s.relu5(y)
        y = s.tconv1(y)
        y = s.tconv2(y)
        y = s.tconv3(y)
        y = s.tconv4(y)
        y = nn.functional.interpolate(y, (x.shape[2], x.shape[3]))
        y = s.out(y)
        return y

def sobel_kernel():
    s = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    s = [s.rot90(i).unsqueeze(0) for i in range(4)]
    s = torch.cat(s, 0).unsqueeze(1)
    return torch.cat([s, s, s], 1).float()
    

# Model 5, but with sobel preprocess
class SegModel7(nn.Module):
    def __init__(s):
        super().__init__()
        s.sobel = sobel_kernel()
        s.conv1 = nn.Conv2d(7, 96, 11, stride=4)
        s.pool1 = nn.MaxPool2d(3, stride=2)
        s.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        s.pool2 = nn.MaxPool2d(3, stride=2)
        s.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        s.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        s.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        s.pool5 = nn.MaxPool2d(3, stride=2)
        s.tconv1 = nn.ConvTranspose2d(256, 1, 3, stride=2)
        s.tconv2 = nn.ConvTranspose2d(1, 1, 3, stride=2)
        s.tconv3 = nn.ConvTranspose2d(1, 1, 3, stride=2)
        s.tconv4 = nn.ConvTranspose2d(1, 1, 11, stride=4)
        s.out = nn.Sigmoid()
    def forward(s, x):
        s.sobel = type(x)(s.sobel)
        y = torch.conv2d(x, s.sobel, padding=1)
        y = torch.cat([x, y], 1)
        y = s.conv1(y)
        y = s.pool1(y)
        y = s.conv2(y)
        y = s.pool2(y)
        y = s.conv3(y)
        y = s.conv4(y) + y
        y = s.conv5(y)
        y = s.pool5(y)
        y = s.tconv1(y)
        y = s.tconv2(y)
        y = s.tconv3(y)
        y = s.tconv4(y)
        y = nn.functional.interpolate(y, (x.shape[2], x.shape[3]))
        y = s.out(y)
        return y

    def to(s, d):
        if isinstance(d, torch.device):
            s.sobel = s.sobel.to(d)
        return super(SegModel7, s).to(d)
