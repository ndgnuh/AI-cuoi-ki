import json
import importlib
import torch
import typing
import os
from dataclasses import dataclass
from pprint import pprint
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
parser = ArgumentParser(prog="python3 *.py")

parser.add_argument("--config", "-c", dest="config_file")


# Get a path from dict
def get(d, path: str, v):
    keys = path.split("/")
    current = d
    for k in keys:
        if k not in current:
            return v
        current = current[k]
    return current


@dataclass
class Config:
    model: nn.Module
    model_path: typing.Union[None, str]
    train_data: DataLoader
    test_data: DataLoader
    device: torch.device
    optimizer: torch.optim.Optimizer
    loss_function: typing.Any
    lr: float
    batch_size: int
    decay_every: int
    decay_rate: float
    start_epoch: int
    end_epoch: int


def import_(path: str):
    # Import model from path
    path = path.split(".")
    module_name = (".").join(path[:-1])
    symbol_name = path[-1]
    Class = importlib.import_module(module_name).__dict__[symbol_name]
    return Class


def import_and_initialize(j, path: str, *extra_args, **extra_kwargs):
    Class = import_(j[path]["name"])
    args = get(j, f"{path}/args", [])
    kwargs = get(j, f"{path}/kwargs", {})
    kwargs.update(extra_kwargs)
    args = [*args, *extra_args]
    return Class(*args, **kwargs)


def parse_args(args=None):
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    j = None  # The json content
    with open(args.config_file) as io:
        j = json.load(io)

    # Import model from config
    model_path = get(j, "model/path", None)
    if model_path is not None and os.path.isfile(model_path):
        pprint(f"Loading model from {model_path}")
        model = torch.load(model_path)
    else:
        Model = import_(j["model"]["name"])
        model_args = get(j, "model/args", [])
        model_kwargs = get(j, "model/kwargs", {})
        model = Model(*model_args, **model_kwargs)

    if "device" in j["model"]:
        device = torch.device(j["model"]["device"])
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # Hyper params
    batch_size = get(j, "hyper/batch_size", 64)
    lr = get(j, "hyper/lr", 1e-3)

    # Import dataset from config
    transform = get(j, "dataset/kwargs/transform", ToTensor())
    train_data = import_and_initialize(
        j, "dataset", train=True, transform=transform)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = import_and_initialize(
        j, "dataset", train=False, transform=transform)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # import the optimizer or use default
    if "optimizer" not in j:
        j["optimizer"] = {
            "name": "torch.optim.Adam",
            "kwargs": {"lr": lr}
        }
    optimizer = import_and_initialize(
        j, "optimizer", model.parameters(), lr=lr)

    loss_function = import_and_initialize(j, "loss")
    if isinstance(loss_function, type):
        loss_function = loss_function()

    # hyper parameter stuffs

    config = Config(
        model=model,
        model_path=model_path,
        device=device,
        train_data=train_data,
        test_data=test_data,
        optimizer=optimizer,
        lr=lr,
        batch_size=batch_size,
        decay_every=get(j, "hyper/decay_every", 30),
        decay_rate=get(j, "hyper/decay_rate", 0.1),
        start_epoch=get(j, "hyper/start_epoch", 0),
        end_epoch=get(j, "hyper/end_epoch", 1000),
        loss_function=loss_function,
    )
    pprint(config)
    return config


if __name__ == "__main__":
    parse_args()
