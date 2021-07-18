import torch.nn as nn
from functools import reduce, singledispatch


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
    G = globals()
    for k in G:
        if type(G[k]) == type and issubclass(G[k], nn.Module):
            models[k] = G[k]
    return models


all_models = all_models()
