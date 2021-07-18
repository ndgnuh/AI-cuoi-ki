import torch.nn as nn
import torchvision.ops as visionops
# C Error on import
# from pytorch_metric_learning import losses as pml_loss
# TODO: add custom loss functions
# - TripletLoss (available in torch?)
# - MultiTask Loss
# - ArcLoss


def is_loss_function(name: str):
    if name not in nn.modules.loss.__dict__:
        return False
    c = nn.modules.loss.__dict__[name]
    if type(c) == type and issubclass(c, nn.modules.loss._Loss):
        return True
    return False


def find_loss_function(name: str):
    if name in nn.__dict__ and is_loss_function(name):
        return nn.__dict__[name]()
    if name in visionops.__dict__ and is_loss_function(name):
        return visionops.__dict__[name]


def all_loss_functions():
    all = {}
    for lf in nn.modules.loss._Loss.__subclasses__():
        origname = lf.__name__
        name = lf.__name__
        if not name.startswith('_'):
            for i in range(10):
                if name not in all:
                    break
                name = f"{origname}_{i}"
            all[name] = lf
    return all


class ArcLoss(nn.modules.loss._Loss):
    pass
