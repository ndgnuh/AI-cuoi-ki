import torch
import cv2
import os
from torch.utils import data


class BaiduV2(data.Dataset):
    # Download option does nothing
    # it just there to ensure the method call does has the same attribute
    def __init__(self, datadir, train, download=None, transform=None):
        suffix = "train" if train else "test"
        inputs = os.path.join(datadir, "x" + suffix)
        targets = os.path.join(datadir, "y" + suffix)
        # names should be in the same order
        # for both inputs and targets
        names = os.listdir(inputs)
        self.inputs = [os.path.join(inputs, name) for name in names]
        self.targets = [os.path.join(targets, name) for name in names]
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = cv2.imread(input_ID), cv2.imread(target_ID)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(
            self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y
