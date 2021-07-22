import torch.nn as nn
import torch
from functools import reduce, singledispatch
import resnet


def TorchModel(f):
    class M(nn.Module):
        def __init__(self, *args, **kwargs):
            super(M, self).__init__()
            self.model = f(*args, **kwargs)

        def forward(self, x):
            return self.model(x)

    # Promote this class to top level, so that it is pickle-able
    M.__qualname__ = f.__name__
    return M


@TorchModel
def MLP1(n_in, n_out):
    layers = [
        nn.Flatten(),
        nn.Linear(n_in, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 218),
        nn.Sigmoid(),
        nn.Linear(218, n_out),
        # nn.Softmax(dim=1),
    ]
    return nn.Sequential(*layers)


@TorchModel
def MLP2(N, activations):
    if len(N) < 2:
        raise Exception("MLP2 expect at least 2 input")

    if activations is None:
        activations = []
    else:
        activations = [nn.__dict__[i]() for i in activations]

    layers = [nn.Flatten()]
    for k, (i, j) in enumerate(zip(N, N[1:])):
        layers = layers + [nn.Linear(i, j)]
        try:
            layers = layers + [activations[k]]
        except Exception as e:
            pass
    return nn.Sequential(*layers)


@TorchModel
def CNN(n_int, n_out):
    layers = [
        nn.Conv2d(3, 3, 3),
        nn.Flatten(),
        nn.Linear(2700, 100)
    ]
    return nn.Sequential(*layers)


class FCN(nn.Module):
    def __init__(self, in_channel, classes, imsize):
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
            return nn.ConvTranspose2d(classes, classes, 2, stride=2, bias=False)

        self.layer128 = makelayer(64, 128, 2)
        self.layer256 = makelayer(128, 256, 3)
        self.layer512 = makelayer(256, 512, 3)
        self.layer4096 = makelayer(512, 4096, 3)
        self.middle = nn.Conv2d(in_channel, classes, 1, bias=False)
        self.middle512 = nn.Conv2d(512, classes, 1, bias=False)
        self.middle256 = nn.Conv2d(256, classes, 1, bias=False)
        self.middle128 = nn.Conv2d(128, classes, 1, bias=False)
        self.middle4096 = nn.Sequential(
            nn.Conv2d(4096, 4096, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(4096, 4096, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(4096, 4096, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(4096, classes, 1, bias=False)
        )
        self.decode1 = upscale()
        self.decode2 = upscale()
        self.decode3 = upscale()
        self.decode4 = upscale()
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(imsize * imsize, classes)
        )

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


@TorchModel
def Seq(*args):
    """
    args = [
        ["Conv2d", "3, 3, 3"],
        ["AvgPool2d", "3, padding=1"],
        ["Flatten"]
        ["Linear", "1000, 1000"]
        ]
    """
    layer = []
    for arg in args:
        name = arg[0]
        astr = str(arg[1:]).strip("'()[]")
        layer = layer + [eval(f"nn.{name}({astr})")]
    return nn.Sequential(*layer)


def all_models():
    models = {}
    models.update(resnet.models)

    G = globals()
    for k in G:
        if type(G[k]) == type and issubclass(G[k], nn.Module):
            models[k] = G[k]
    return models


all_models = all_models()
