import torch
import os
from torch.utils import data
from torchvision.transforms import ToTensor
from PIL import Image


class SegmentationData(data.Dataset):
    # Download option does nothing
    # it just there to ensure the method call does has the same attribute
    def __init__(self, datadir, train, download=None, transform=None, target_transform=None):
        if transform is None:
            transform = ToTensor()
        if target_transform is None:
            target_transform = transform
        suffix = "train" if train else "test"
        inputs = os.path.join(datadir, "x" + suffix)
        targets = os.path.join(datadir, "y" + suffix)
        # names should be in the same order
        # for both inputs and targets
        names = os.listdir(inputs)
        self.inputs = [os.path.join(inputs, name) for name in names]
        self.targets = [os.path.join(targets, name) for name in names]
        self.transform = transform
        self.target_transform = target_transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = Image.open(input_ID).convert(
            "RGB"), Image.open(target_ID).convert('L')

        # Preprocessing
        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
