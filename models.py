import torch.nn as nn
import torch
import torch.nn.functional as F


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
        s.pool1 = nn.AvgPool2d(2)
        s.resi1 = ResidualBlock(64, 128)

        s.conv2 = nn.Conv2d(128, 128, 3, dilation=2)
        s.pool2 = nn.AvgPool2d(2)
        s.resi2 = ResidualBlock(128, 256)

        s.conv3 = nn.Conv2d(256, 256, 3, dilation=4)
        s.pool3 = nn.AvgPool2d(2)
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
        s.pool1 = nn.AvgPool2d(2)
        s.resi1 = ResidualBlock(64, 128)

        s.conv2 = nn.Conv2d(128, 128, 3, dilation=2)
        s.pool2 = nn.AvgPool2d(2)
        s.resi2 = ResidualBlock(128, 256)

        s.conv3 = nn.Conv2d(256, 256, 3, dilation=4)
        s.pool3 = nn.AvgPool2d(2)
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
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)

        # Squeeze
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.pool2x = nn.AvgPool2d((2, 1))
        self.pool2y = nn.AvgPool2d((1, 2))
        self.tconv1 = nn.ConvTranspose2d(64, 64, 4, stride=2)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)

        self.lin1 = nn.Conv2d(64, 64, 1)
        self.lin2 = nn.Conv2d(64, 64, 1)
        self.lin3 = nn.Conv2d(64, 64, 1)

        self.conv6 = nn.Conv2d(64, 64, 1)
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
        while y.shape[2] > 1 and y.shape[3] > 1:
            if y.shape[3] > 1:
                y = self.pool2x(y)
            if y.shape[3] > 1:
                y = self.pool2y(y)
        y = self.lin3(self.lin2(self.lin1(y)))
        y = self.tconv1(y)
        y = F.interpolate(y, (y_orig.shape[2], y_orig.shape[3]))
        y = torch.cat([self.conv6(y_orig), y], 1)

        # Rescale
        y = self.bn3(y)
        y = self.conv7(y)
        y = F.interpolate(y, (x.shape[2], x.shape[3]))
        y = self.sigmoid(y)
        return y


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
        s.pool1 = nn.AvgPool2d(2)

        s.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        s.pool2 = nn.AvgPool2d(2)

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
        y = F.interpolate(y, x.shape[2:])
        y = torch.sum(y, 1).unsqueeze(1)
        y = s.out(y)
        return y


class SegModel12(nn.Module):
    def __init__(s):
        super(SegModel12, s).__init__()
        s.sobel = FixedConv2d(sobel_kernel(), padding=1)
        s.edges = FixedConv2d(edge_detection_kernel(), padding=1)

        s.conv1 = nn.Conv2d(10, 64, 3, stride=2)
        s.pool1 = nn.AvgPool2d(2)

        s.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        s.pool2 = nn.AvgPool2d(2)

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
