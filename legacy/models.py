import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


# Depthwise separatable convolution
class DSConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DSConv2d, self).__init__()
        in_channel = args[0] if len(args) > 0 else kwargs["in_channel"]
        out_channel = args[1] if len(args) > 1 else kwargs["out_channel"]
        self.dw = nn.Conv2d(*args, **kwargs, groups=in_channel)
        self.pw = nn.Conv2d(out_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.ac = HSwish(inplace=True)

    def forward(self, y):
        y = self.dw(y)
        y = self.bn(y)
        y = self.ac(y)
        y = self.pw(y)
        y = self.bn(y)
        y = self.ac(y)
        return y


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.out(y)
        return y


def sobel_kernel():
    s = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    s = [s.rot90(i).unsqueeze(0) for i in range(4)]
    s = torch.cat(s, 0).unsqueeze(1)
    s = torch.cat([s, s, s], 1).float()
    return s


def edge_detection_kernel():
    k1 = torch.tensor([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]).unsqueeze(0)
    k2 = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).unsqueeze(0)
    k3 = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).unsqueeze(0)
    ks = torch.cat([k1, k2, k3], 0).unsqueeze(1)
    ks = torch.cat([ks, ks, ks], 1).float()
    return ks


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
            nn.AvgPool1d(2),
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
                nn.MaxPool2d(2),
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
                nn.ReLU(inplace=True),
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
                        nn.MaxPool2d(2),
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


class FixedConv2d(nn.Module):
    # Fixed kernel convolution layer
    def __init__(s, k, *args, **kwargs):
        super(FixedConv2d, s).__init__()
        s.kernel = k
        s.args = args
        s.kwargs = kwargs

    def forward(s, x):
        # use type field to compare string instead of two object
        if s.kernel.device.type != x.device.type:
            s.kernel = s.kernel.to(x.device)
        return torch.conv2d(x, s.kernel, *s.args, **s.kwargs)

# Model 5, but with sobel preprocess


class SegModel7(nn.Module):
    def __init__(s):
        super().__init__()
        s.sobel = FixedConv2d(sobel_kernel(), padding=1)
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
        y = s.sobel(x)
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

#     def to(s, d):
#         if isinstance(d, torch.device):
#             s.sobel = s.sobel.to(d)
#         return super(SegModel7, s).to(d)


class ResidualBlock(nn.Module):
    def __init__(s, i, n):
        super().__init__()
        s.conv1 = nn.Conv2d(i, n, 3, padding=1)
        s.bn1 = nn.BatchNorm2d(n)
        s.relu1 = nn.ReLU(inplace=True)
        s.conv2 = nn.Conv2d(n, n, 3, padding=1, groups=n)
        s.bn2 = nn.BatchNorm2d(n)
        if i != n:
            s.conv_skip = nn.Conv2d(i, n, 1)
            s.has_conv1x1 = True
        else:
            s.has_conv1x1 = False

    def forward(s, x):
        y = s.conv1(x)
        y = s.bn1(y)
        y = s.relu1(y)
        y = s.conv2(y)
        y = s.bn2(y)
        if s.has_conv1x1:
            y = y + s.conv_skip(x)
        else:
            y = y + x
        return y

# Use more residual,
# leverage stride for lighter computation


class SegModel8(nn.Module):
    def __init__(s):
        super(SegModel8, s).__init__()
        s.conv1 = nn.Conv2d(3, 64, 7, stride=2)
        s.pool1 = nn.MaxPool2d(2)
        s.resi1 = ResidualBlock(64, 128)

        s.conv2 = nn.Conv2d(128, 128, 3, dilation=2)
        s.pool2 = nn.MaxPool2d(2)
        s.resi2 = ResidualBlock(128, 256)

        s.conv3 = nn.Conv2d(256, 256, 3, dilation=4)
        s.pool3 = nn.MaxPool2d(2)
        s.resi3 = ResidualBlock(256, 64)

        s.tconv1 = nn.ConvTranspose2d(64, 1, 4, stride=4)
        s.tconv2 = nn.ConvTranspose2d(1, 1, 8, stride=2)
        s.tconv3 = nn.ConvTranspose2d(1, 1, 7, stride=2)

        s.out = nn.Sigmoid()

    def forward(s, x):
        y = s.conv1(x)
        y = s.pool1(y)
        y = s.resi1(y)
        y = s.conv2(y)
        y = s.pool2(y)
        y = s.resi2(y)
        y = s.conv3(y)
        y = s.pool3(y)
        y = s.resi3(y)
        y = s.tconv1(y)
        y = s.tconv2(y)
        y = s.tconv3(y)
        y = F.interpolate(y, (x.shape[2], x.shape[3]), mode='bilinear')
        y = s.out(y)
        return y


# Like model 8, adding the sobel kernel
class SegModel9(nn.Module):
    def __init__(s):
        super(SegModel9, s).__init__()
        s.sobel = FixedConv2d(sobel_kernel(), padding=1)

        s.conv1 = nn.Conv2d(7, 64, 7, stride=2)
        s.pool1 = nn.MaxPool2d(2)
        s.resi1 = ResidualBlock(64, 128)

        s.conv2 = nn.Conv2d(128, 128, 3, dilation=2)
        s.pool2 = nn.MaxPool2d(2)
        s.resi2 = ResidualBlock(128, 256)

        s.conv3 = nn.Conv2d(256, 256, 3, dilation=4)
        s.pool3 = nn.MaxPool2d(2)
        s.resi3 = ResidualBlock(256, 64)

        s.tconv1 = nn.ConvTranspose2d(64, 1, 4, stride=4)
        s.tconv2 = nn.ConvTranspose2d(1, 1, 8, stride=2)
        s.tconv3 = nn.ConvTranspose2d(1, 1, 7, stride=2)

        s.out = nn.Sigmoid()

    def forward(s, x):
        y = s.sobel(x)
        y = torch.cat([x, y], 1)
        y = s.conv1(y)
        y = s.pool1(y)
        y = s.resi1(y)
        y = s.conv2(y)
        y = s.pool2(y)
        y = s.resi2(y)
        y = s.conv3(y)
        y = s.pool3(y)
        y = s.resi3(y)
        y = s.tconv1(y)
        y = s.tconv2(y)
        y = s.tconv3(y)
        y = F.interpolate(y, (x.shape[2], x.shape[3]), mode='bilinear')
        y = s.out(y)
        return y

# Just mix a bunch of stuffs
# After experimenting, this is a failed one


class SegModel10(nn.Module):
    def __init__(self):
        super(SegModel10, self).__init__()
        # Reduce the size
        self.conv1 = nn.Conv2d(3, 64, 7, dilation=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 5, dilation=4, stride=2)
        self.conv3 = nn.Conv2d(128, 128, 3, dilation=3, stride=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)

        # Squeeze
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.pool2x = nn.MaxPool2d((2, 1))
        self.pool2y = nn.MaxPool2d((1, 2))
        self.tconv1 = nn.ConvTranspose2d(64, 64, 4, stride=2)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)

        self.lin1 = nn.Conv2d(64, 64, 1)
        self.lin2 = nn.Conv2d(64, 64, 1)
        self.lin3 = nn.Conv2d(64, 64, 1)

        self.bn3 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 1, 1)
        self.sigmoid = nn.Sigmoid()

        self.pool_x4 = nn.MaxPool2d((4, 1), stride=2)
        self.pool_y4 = nn.MaxPool2d((1, 4), stride=2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.pool1(y)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv5(y)
        y_orig = y
        y = F.adaptive_max_pool2d(y, (1, 1))
        y = self.lin3(self.lin2(self.lin1(y)))
        y = torch.multiply(y_orig, y)
        y = self.tconv1(y)
        y = torch.cat([F.interpolate(self.conv6(y_orig), y.shape[2:]), y], 1)

        # Rescale
        y = self.bn3(y)
        y = self.conv7(y)
        y = F.interpolate(y, (x.shape[2], x.shape[3]))
        y = self.sigmoid(y)
        return y


# Channel wise attention
class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4.0):
        super(SqueezeBlock, self).__init__()
        if divide > 1:
            self.dense = nn.Sequential(
                nn.Linear(exp_size, int(exp_size / divide)),
                nn.PReLU(int(exp_size / divide)),
                nn.Linear(int(exp_size / divide), exp_size),
                nn.PReLU(exp_size),
            )
        else:
            self.dense = nn.Sequential(
                nn.Linear(exp_size, exp_size),
                nn.PReLU(exp_size)
            )

    def forward(self, x):
        batch, channels, height, width = x.size()
        y = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        y = self.dense(y)
        y = y.view(batch, channels, 1, 1)
        return y


class SegModel11(nn.Module):
    def __init__(s):
        super(SegModel11, s).__init__()
        s.conv1 = nn.Conv2d(3, 64, 3, stride=2)
        s.pool1 = nn.MaxPool2d(2)

        s.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        s.pool2 = nn.MaxPool2d(2)

        s.bnrelu = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        s.conv3 = nn.Conv2d(128, 256, 5, padding=1)
        s.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        s.conv5 = nn.Conv2d(512, 1024, 3, padding=1)

        s.tconv1 = nn.ConvTranspose2d(1024, 256, 4, stride=2)
        s.tconv2 = nn.ConvTranspose2d(256, 64, 4, stride=2)
        s.tconv3 = nn.ConvTranspose2d(64, 64, 4, stride=2)

        s.sqz = SqueezeBlock(64, divide=1)
        s.sqz_tconv1 = nn.ConvTranspose2d(64, 64, 11, stride=4)
        s.sqz_tconv2 = nn.ConvTranspose2d(64, 64, 7, stride=3)
        s.sqz_bn = nn.BatchNorm2d(64)
        s.out = nn.Hardsigmoid()

    def forward(s, x):
        y = s.conv1(x)
        y = s.pool1(y)
        y1 = y

        y_sq = s.sqz(y)
        y_sq = s.sqz_tconv1(y_sq)
        y_sq = s.sqz_tconv2(y_sq)
        y_sq = s.sqz_bn(y_sq)

        y = s.conv2(y)
        y = s.pool2(y)
        y = s.bnrelu(y)

        y = s.conv3(y)
        y = s.conv4(y)
        y = s.conv5(y)

        y = s.tconv1(y)
        y = s.tconv2(y)
        y = y1 + F.interpolate(y, y1.shape[2:])
        y = y + torch.multiply(y, F.interpolate(y_sq, y.shape[2:]))
        y = s.tconv3(y)
        y = torch.sum(y, 1).unsqueeze(1)
        y = F.interpolate(y, x.shape[2:])
        y = s.out(y)
        return y


class SegModel11Fixed(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(size)
        self.layers = SegModel11()

    def forward(self, x):
        y = self.pool(x)
        y = self.layers(y)
        y = F.interpolate(y, x.shape[2:])
        return y


class SegModel12(nn.Module):
    def __init__(s):
        super(SegModel12, s).__init__()
        s.sobel = FixedConv2d(sobel_kernel(), padding=1)
        s.edges = FixedConv2d(edge_detection_kernel(), padding=1)

        s.conv1 = nn.Conv2d(10, 64, 3, stride=2)
        s.pool1 = nn.MaxPool2d(2)

        s.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        s.pool2 = nn.MaxPool2d(2)

        s.bnrelu = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        s.conv3 = nn.Conv2d(128, 256, 5, padding=1)
        s.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        s.conv5 = nn.Conv2d(512, 1024, 3, padding=1)

        s.tconv1 = nn.ConvTranspose2d(1024, 256, 4, stride=2)
        s.tconv2 = nn.ConvTranspose2d(256, 64, 4, stride=2)
        s.tconv3 = nn.ConvTranspose2d(64, 64, 4, stride=2)

        s.sqz = SqueezeBlock(64, divide=1)
        s.sqz_tconv1 = nn.ConvTranspose2d(64, 64, 11, stride=4)
        s.sqz_tconv2 = nn.ConvTranspose2d(64, 64, 7, stride=3)
        s.sqz_bn = nn.BatchNorm2d(64)
        s.out = nn.Sigmoid()

    def forward(s, x):
        sobel = s.sobel(x)
        edges = s.edges(x)
        y = torch.cat([x, sobel, edges], 1)

        y = s.conv1(y)
        y = s.pool1(y)
        y1 = y

        y_sq = s.sqz(y)
        y_sq = s.sqz_tconv1(y_sq)
        y_sq = s.sqz_tconv2(y_sq)
        y_sq = s.sqz_bn(y_sq)

        y = s.conv2(y)
        y = s.pool2(y)
        y = s.bnrelu(y)

        y = s.conv3(y)
        y = s.conv4(y)
        y = s.conv5(y)

        y = s.tconv1(y)
        y = s.tconv2(y)
        y = y1 + F.interpolate(y, y1.shape[2:])
        y = y + torch.multiply(y, F.interpolate(y_sq, y.shape[2:]))
        y = s.tconv3(y)
        y = F.interpolate(y, x.shape[2:])
        y = torch.sum(y, 1).unsqueeze(1)
        y = s.out(y)
        return y


# inspired by
# https://ai.googleblog.com/2020/10/background-features-in-google-meet.html?m=1
# Did not add skip connection
class SegModel13(nn.Module):
    def __init__(self, in_size=128):
        super(SegModel13, self).__init__()
        self.pool_entry = nn.AdaptiveMaxPool2d(in_size)

        sizes = [in_size//(2**i) for i in [1, 2, 3, 4]]
        channels = [64, 128, 256, 512]
        for i, size in enumerate(sizes):
            if i == 0:
                conv = nn.Conv2d(3, channels[i], 7, padding=3)
            else:
                conv = nn.Conv2d(channels[i - 1], channels[i], 3, padding=1)
            pool = nn.AdaptiveMaxPool2d(size)
            sq = SqueezeBlock(channels[i], 1)
            ex = nn.Conv2d(channels[i], channels[i], 1)
            bn = nn.BatchNorm2d(channels[i])
            setattr(self, f"conv{i}", conv)
            setattr(self, f"pool{i}", pool)
            setattr(self, f"sq{i}", sq)
            setattr(self, f"ex{i}", ex)
            setattr(self, f"bn{i}", bn)

        channels.reverse()
        for i, (c1, c2) in enumerate(zip(channels, channels[1:])):
            tconv = nn.ConvTranspose2d(c1, c2, 2, stride=2)
            setattr(self, f"tconv{i}", tconv)
        self.relu = nn.ReLU(inplace=True)
        self.tconv3 = nn.ConvTranspose2d(
                channels[-1], channels[-1], 2, stride=2)

        self.out = nn.Sequential(
            nn.Conv2d(channels[-1], 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # To 256x256
        y = self.pool_entry(x)

        for i in range(4):
            conv = getattr(self, f"conv{i}")
            pool = getattr(self, f"pool{i}")
            sq = getattr(self, f"sq{i}")
            ex = getattr(self, f"ex{i}")
            bn = getattr(self, f"bn{i}")
            y = conv(y)
            y = pool(y)
            y1 = y2 = y
            y1 = bn(y)
            y1 = self.relu(y1)
            y2 = sq(y2)
            y2 = ex(y2)
            y = torch.multiply(y1, y2)

        for i in range(4):
            y = getattr(self, f"tconv{i}")(y)

        y = F.interpolate(y, x.shape[2:])
        y = self.out(y)
        return y


class SegModel14(nn.Module):
    def __init__(self, in_size):
        super(SegModel14, self).__init__()
        self.entry_pool = nn.AdaptiveMaxPool2d(in_size)
        self.entry_conv = nn.Conv2d(3, 64, 1)

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.relu = nn.ReLU6(inplace=True)

        self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.se1 = SqueezeBlock(128)
        self.se2 = SqueezeBlock(256)
        self.se3 = SqueezeBlock(512)
        self.dwconv1 = nn.Conv2d(512, 256, 3, padding=1, groups=256)
        self.dwconv2 = nn.Conv2d(256, 128, 3, padding=1, groups=128)
        self.dwconv3 = nn.Conv2d(128, 64, 3, padding=1, groups=64)

        self.out = nn.Sequential(
                nn.Conv2d(64, 1, 3, padding=1),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.entry_pool(x)
        y = self.entry_conv(y)

        # 1/2
        y, i1 = self.pool(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv1(y)
        a1 = self.se1(y)

        # 1/4
        y, i2 = self.pool(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv2(y)
        a2 = self.se2(y)

        # 1/8
        y, i3 = self.pool(y)
        y = self.bn3(y)
        y = self.relu(y)
        y = self.conv3(y)
        a3 = self.se3(y)

        # 1/16
        y, i4 = self.pool(y)
        y = self.bn4(y)
        y = self.relu(y)
        y = self.conv4(y)

        # 1/8
        y = F.max_unpool2d(y, i4, 2)
        y = torch.multiply(y, a3)
        y = self.dwconv1(y)
        # 1/4
        y = F.max_unpool2d(y, i3, 2)
        y = torch.multiply(y, a2)
        y = self.dwconv2(y)
        # 1/2
        y = F.max_unpool2d(y, i2, 2)
        y = torch.multiply(y, a1)
        y = self.dwconv3(y)
        # # 1
        y = F.max_unpool2d(y, i1, 2)
        y = F.interpolate(y, x.shape[2:])
        y = self.out(y)
        return y


class SegModel15Block(nn.Module):
    def __init__(self, in_channel, expand, kernel_size=3, **kwargs):
        super().__init__()
        out_channel = int(in_channel * expand)
        if "stride" in kwargs:
            self.stride = kwargs["stride"]
        else:
            self.stride = 1
        self.expand = nn.Conv2d(in_channel, out_channel, 1)
        self.dw = DSConv2d(out_channel, out_channel, kernel_size, **kwargs)
        self.proj = nn.Conv2d(out_channel, out_channel, 1)
        self.se = SqueezeBlock(out_channel)

    def forward(self, x):
        y = self.expand(x)
        y2 = self.se(y)
        if self.stride == 1:
            y1 = self.proj(self.dw(y))
            y = torch.multiply(y + y1, y2)
        else:
            y = self.proj(self.dw(y))
            y = torch.multiply(y, y2)
        return y


class SegModel15(nn.Module):
    def __init__(self, base=128):
        super(SegModel15, self).__init__()
        self.conv_entry = nn.Conv2d(3, 64, 1)

        for frac in [1, 2, 4, 8]:
            if isinstance(base, int):
                pool = nn.AdaptiveMaxPool2d(base//frac)
            else:
                pool = nn.AdaptiveMaxPool2d(base[0]//frac, base[1]//frac)
            setattr(self, f"pool_1_{frac}", pool)
        self.conv1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, padding=1)
        self.b1 = SegModel15Block(64, 1, padding=1)
        self.b2 = SegModel15Block(128, 1, padding=1)
        self.b3 = SegModel15Block(256, 1, padding=1)
        self.b4 = SegModel15Block(512, 1, padding=1)
        self.b5 = SegModel15Block(512, 0.125, padding=1)
        self.b6 = SegModel15Block(64, 0.5, padding=1)
        self.b7 = SegModel15Block(32, 0.5, padding=1)
        self.b8 = SegModel15Block(16, 0.5, padding=1)

        self.out = nn.Sequential(
            nn.Conv2d(8, 1, 1),
            HSigmoid()
        )

    def forward(self, x):

        y = self.pool_1_1(x)
        y = self.conv_entry(y)
        y = self.b1(y)
        y = self.conv1(y)

        y = self.pool_1_2(y)
        y = self.b2(y)
        y = self.conv2(y)

        y = self.pool_1_4(y)
        y = self.b3(y)
        y = self.conv3(y)

        y = self.pool_1_8(y)
        y = self.b4(y)
        y = self.conv4(y)

        y = self.b5(self.pool_1_4(y))
        y = self.b6(self.pool_1_2(y))
        y = self.b7(self.pool_1_1(y))

        y = F.interpolate(y, x.shape[2:])
        y = self.b8(y)
        y = self.out(y)
        return y


# Attempt to use full DWS Block
# -> upscale step got bad, same symptom as Model10
class SegModel16(nn.Module):
    def __init__(self, in_size=128):
        super(SegModel16, self).__init__()
        self.entry = nn.AdaptiveMaxPool2d(in_size)
        self.conv_entry = nn.Conv2d(3, 64, 7, padding=3)

        configs = [
                [64, 2, 5, 11, 2],
                [128, 2, 3, 7, 2],
                [256, 2, 1, 3, 2],
                [512, 1, 1, 3, 2],
                [512, 0.25, 1, 3, 1],
                [128, 0.25, 1, 3, 1],
                [32, 0.25, 1, 3, 1],
                [8, 0.5, 1, 3, 1],
        ]
        for i, (c, ex, pad, ksize, stride) in enumerate(configs):
            b = SegModel15Block(c, ex, kernel_size=ksize, padding=pad, stride=stride)
            setattr(self, f"b{(i)}", b)
            self.nblocks = i+1
        self.conv = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        y = self.entry(x)
        y = self.conv_entry(y)
        for i in range(self.nblocks):
            y = getattr(self, f"b{i}")(y)
        y = self.conv(y)
        y = F.interpolate(y, x.shape[2:])
        y = torch.sigmoid(y)
        return y


# Attempt at multi path structure
class SegModel17(nn.Module):
    def __init__(self, in_size=128):
        super(SegModel17, self).__init__()

        self.entry = nn.AdaptiveMaxPool2d(in_size)

        c1 = 64
        c2 = 64
        c3 = 128
        c4 = 64
        self.conv0_1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv0_2 = nn.Conv2d(c1, c1, 7, padding=3, groups=c1)
        self.conv0_3 = nn.Conv2d(c1, c2, 1)

        self.pool_2 = nn.AdaptiveMaxPool2d(in_size//2)
        self.pool_4 = nn.AdaptiveMaxPool2d(in_size//4)
        self.pool_8 = nn.AdaptiveMaxPool2d(in_size//8)

        self.conv_2 = nn.Conv2d(c2, c2, 3, padding=1, groups=c2)
        self.conv_4 = nn.Conv2d(c2, c3, 3, padding=1, groups=c2)
        self.conv_8 = nn.Conv2d(c2, c4, 3, padding=1, groups=c2)

        self.tconv_2 = nn.ConvTranspose2d(c2, c2, 2, stride=2, groups=c2)
        self.tconv_4 = nn.ConvTranspose2d(c3, c3, 4, stride=4, groups=c2)
        self.tconv_8 = nn.ConvTranspose2d(c4, c4, 8, stride=8, groups=c2)

        self.conv0_4 = nn.Conv2d(c1 + c2 + c3 + c4, c2, 1)
        self.conv0_5 = nn.Conv2d(c2, c2, 3, dilation=3, padding=3, groups=c2)
        self.conv0_6 = nn.Conv2d(c2, 1, 1)

        self.sq = SqueezeBlock(c2)
        self.info_block = nn.Conv2d(c2, c2, 3, padding=1)

        self.bn = nn.BatchNorm2d(c2)
        self.relu = HSwish()
        self.sigmoid = HSigmoid()

    def forward(self, x):
        y = self.entry(x)
        y = self.conv0_1(y)
        y = self.conv0_2(y)
        y = self.conv0_3(y)
        y_a = self.sq(y)
        y = torch.cat([
            y,
            self.tconv_2(self.conv_2(self.pool_2(y))),
            self.tconv_4(self.conv_4(self.pool_4(y))),
            self.tconv_8(self.conv_8(self.pool_8(y)))], 1)
        y = self.conv0_4(y)
        y_a = torch.multiply(y_a, self.info_block(y))
        y = self.conv0_5(y)
        y = self.relu(self.bn(y + y_a))
        y = self.conv0_6(y)
        y = F.interpolate(y, x.shape[2:])
        y = self.sigmoid(y)
        return y


class SegModel18(nn.Module):
    def __init__(self, in_size=256):
        super().__init__()
        self.entry = nn.Sequential(
                nn.AdaptiveMaxPool2d(in_size),
                nn.Conv2d(3, 64, 1)
        )
        self.pool = nn.MaxPool2d(2)

        c = [64, 64]
        self.conv1_1 = nn.Conv2d(c[0], c[1], 1)
        self.conv1_2 = nn.Conv2d(c[1], c[1], 3, padding=1, groups=c[1])
        self.conv1_3 = nn.Conv2d(c[1], c[1], 1)

        c = [64, 64]
        self.conv2_1 = nn.Conv2d(c[0], c[1], 1)
        self.conv2_2 = nn.Conv2d(c[1], c[1], 3, padding=1, groups=c[1])
        self.conv2_3 = nn.Conv2d(c[1], c[1], 1)

        c = [64, 256]
        self.conv3_1 = nn.Conv2d(c[0], c[1], 1)
        self.conv3_2 = nn.Conv2d(c[1], c[1], 3, padding=1, groups=c[1])
        self.conv3_3 = nn.Conv2d(c[1], c[1], 1)

        c = [256, 256]
        self.conv4_1 = nn.Conv2d(c[0], c[1], 1)
        self.conv4_2 = nn.Conv2d(c[1], c[1], 7, padding=3, groups=c[1])
        self.conv4_3 = nn.Conv2d(c[1], c[1], 1)

        self.relu = nn.ReLU()
        self.bn_64 = nn.BatchNorm2d(64)
        self.bn_256 = nn.BatchNorm2d(256)

        self.skip_1 = nn.Conv2d(64, 64, 1)
        self.skip_2 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.skip_3 = nn.Conv2d(64, 64, 1)

        self.bl_1 = nn.Conv2d(64, 64, 1)
        self.bl_2 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.bl_3 = nn.Conv2d(64, 64, 1)

        c = [256, 64]
        self.tconv1_1 = nn.ConvTranspose2d(c[0], c[1], 1)
        self.tconv1_2 = nn.ConvTranspose2d(c[1], c[1], 2, stride=2, groups=c[1])
        self.tconv1_3 = nn.ConvTranspose2d(c[1], c[1], 1)

        c = [64, 64]
        self.tconv2_1 = nn.ConvTranspose2d(c[0], c[1], 1)
        self.tconv2_2 = nn.ConvTranspose2d(c[1], c[1], 2, stride=2, groups=c[1])
        self.tconv2_3 = nn.ConvTranspose2d(c[1], c[1], 1)

        c = [64, 64]
        self.tconv3_1 = nn.ConvTranspose2d(c[0], c[1], 1)
        self.tconv3_2 = nn.ConvTranspose2d(c[1], c[1], 4, stride=4, groups=c[1])
        self.tconv3_3 = nn.ConvTranspose2d(c[1], c[1], 1)

        c = [64, 64]
        self.tconv4_1 = nn.ConvTranspose2d(c[0], c[1], 1)
        self.tconv4_2 = nn.ConvTranspose2d(c[1], c[1], 2, stride=2, groups=c[1])
        self.tconv4_3 = nn.ConvTranspose2d(c[1], c[1], 1)

        self.out_1 = nn.Conv2d(64, 64, 1)
        self.out_2 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.out_3 = nn.Conv2d(64, 1, 1)
        self.out_4 = nn.MaxPool2d(3, stride=1, padding=1)
        self.out_5 = nn.Sigmoid()

    def forward(self, x):
        y = self.entry(x)

        y = self.conv1_1(y)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool(y)
        y = self.relu(y)

        y = self.conv2_1(y)
        y = self.conv2_2(y)
        y = self.conv2_3(y)
        y = self.pool(y)
        y = self.bn_64(y)
        y = self.relu(y)

        y_skip = y
        y_skip = self.skip_1(y_skip)
        y_skip = self.skip_2(y_skip)
        y_skip = self.skip_3(y_skip)
        y_skip = self.pool(y_skip)
        y_skip = self.bn_64(y_skip)
        y_skip = self.relu(y_skip)

        y = self.conv3_1(y)
        y = self.conv3_2(y)
        y = self.conv3_3(y)
        y = self.pool(y)
        y = self.relu(y)

        y0 = y
        y = self.conv4_1(y)
        y = self.conv4_2(y)
        y = self.conv4_3(y)
        y = y + y0
        y = self.pool(y)
        y = self.bn_256(y)
        y = self.relu(y)

        y = self.tconv1_1(y)
        y = self.tconv1_2(y)
        y = self.tconv1_3(y)
        y = self.bn_64(y)
        y = self.relu(y)

        y_bl = y
        y_bl = self.bl_1(y_bl)
        y_bl = self.bl_2(y_bl)
        y_bl = self.bl_3(y_bl)
        y_bl = torch.multiply(y_bl, y_skip)

        y = y + y_bl
        y = self.tconv2_1(y)
        y = self.tconv2_2(y)
        y = self.tconv2_3(y)
        y = self.bn_64(y)
        y = self.relu(y)

        y = self.tconv3_1(y)
        y = self.tconv3_2(y)
        y = self.tconv3_3(y)
        y = self.bn_64(y)
        y = self.relu(y)

        y = self.out_1(y)
        y = self.out_2(y)
        y = self.out_3(y)
        y = self.out_4(y)
        y = F.interpolate(y, x.shape[2:], mode='bilinear')
        y = self.out_5(y)
        return y


# Use mobilenetV3 pretrained encoder
class SegModel19(nn.Module):
    def __init__(self):
        super(SegModel19, self).__init__()
        m = torchvision.models.mobilenet_v3_small(pretrained=True)

        self.in_pool = nn.AdaptiveMaxPool2d((224, 224))

        # Given 256x144 input
        # torch.Size([1, 3, 256, 144])
        # torch.Size([1, 16, 128, 72])
        # torch.Size([1, 16, 64, 36])
        # torch.Size([1, 24, 32, 18])
        # torch.Size([1, 24, 32, 18])
        # torch.Size([1, 40, 16, 9])
        # torch.Size([1, 40, 16, 9])
        # torch.Size([1, 40, 16, 9])
        # torch.Size([1, 48, 16, 9])
        # torch.Size([1, 48, 16, 9])
        # torch.Size([1, 96, 8, 5])
        # torch.Size([1, 96, 8, 5])
        # torch.Size([1, 96, 8, 5])
        # torch.Size([1, 576, 8, 5])
        self.features = m.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.tconv_1_2 = nn.Sequential(
               nn.ConvTranspose2d(16, 16, 2, stride=2),
               nn.BatchNorm2d(16),
               nn.Hardswish(inplace=True))
        self.tconv_1_4 = nn.Sequential(
               nn.ConvTranspose2d(16, 16, 2, stride=2),
               nn.BatchNorm2d(16),
               nn.Hardswish(inplace=True))
        self.tconv_1_8 = nn.Sequential(
               nn.ConvTranspose2d(24, 16, 2, stride=2),
               nn.BatchNorm2d(16),
               nn.Hardswish(inplace=True))
        self.tconv_1_16 = nn.Sequential(
               nn.ConvTranspose2d(48, 24, 2, stride=2),
               nn.BatchNorm2d(24),
               nn.Hardswish(inplace=True))
        self.tconv_1_32 = nn.Sequential(
                nn.ConvTranspose2d(576, 48, 2, stride=2),
                nn.BatchNorm2d(48),
                nn.Hardswish(inplace=True))

        self.se_1_2 = SqueezeBlock(16)
        self.se_1_4 = SqueezeBlock(16)
        self.se_1_8 = SqueezeBlock(24)

        self.conv_out = nn.Conv2d(16, 1, 1, groups=1)
        self.out = nn.Hardsigmoid(inplace=True)
        self.hrelu = nn.Hardrelu(inplace=True)

    def forward(self, x):
        y = x
        # y = self.in_pool(x)

        # 1/2
        y1 = y = self.features[0](y)
        a1 = self.se_1_2(y)

        # 1/4
        y2 = y = self.features[1](y)
        a2 = self.se_1_4(y)

        # 1/8
        y3 = y = self.features[2:4](y)
        a3 = self.se_1_8(y)

        # 1/16 -> 1/32
        y = self.features[4:](y)

        y = self.tconv_1_32(y)
        y = self.tconv_1_16(y) + y3
        y = self.hrelu(y)
        y = torch.multiply(y, a3)
        y = self.tconv_1_8(y) + y2
        y = self.hrelu(y)
        y = torch.multiply(y, a2)
        y = self.tconv_1_4(y) + y1
        y = self.hrelu(y)
        y = torch.multiply(y, a1)
        y = self.tconv_1_2(y)

        y = self.conv_out(y)
        y = F.interpolate(y, x.shape[2:])
        y = self.out(y)
        return y
