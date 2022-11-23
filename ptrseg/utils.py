import torch
import numpy as np
import os
from os import path
from torch import Tensor
from torchvision.transforms import functional as FT
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from typing import Tuple
from PIL import Image
from functools import lru_cache, singledispatch


@singledispatch
def letterbox(image: Tensor, size, fill=0.5):
    # Thumbnail
    orig_size = image.shape[-2:]
    ratio = min([s / os for os, s in zip(orig_size, size)])
    new_size = [int(os * ratio) for os in orig_size]
    image = FT.resize(image, new_size)

    # padding to letterbox
    padding = [max(int((s - ns) / 2), 0)
               for s, ns in zip(size, new_size)]
    padding = tuple(reversed(padding))
    image = FT.pad(image, padding=padding, fill=fill)

    # Resize one last time
    image = FT.resize(image, size)
    return image

@letterbox.register(np.ndarray)
def letterbox_numpy(image: np.ndarray, *args, **kwargs):
    image = image.transpose((2, 0, 1))
    image = letterbox(torch.tensor(image), *args, **kwargs)
    image = image.numpy().transpose((1, 2, 0))
    return image


class SegmentDataset(VisionDataset):
    def __init__(self,
                 root: str,
                 image_size: Tuple[int, int] = (640, 640),
                 image_dir: str = "image",
                 mask_dir: str = "mask",
                 cached: bool = True,
                 reverse: bool = False,
                 **kwargs):
        super().__init__(root=root, **kwargs)
        self.root = root
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.imgs = os.listdir(path.join(root, image_dir))

        if reverse:
            self.images = list(reversed(self.imgs))

        if cached:
            self.load_sample = lru_cache(self.load_sample)

    def __len__(self):
        return len(self.imgs)

    def load_sample(self, image_path: str):
        image = Image.open(image_path)
        image = FT.to_tensor(image)
        image = letterbox(image, self.image_size)
        return image

    def __getitem__(self, index: int):
        name = self.imgs[index]
        image_path = path.join(self.root, self.image_dir, name)
        mask_path = path.join(self.root, self.mask_dir, name)
        image = self.load_sample(image_path)
        mask = self.load_sample(mask_path)
        mask = mask.squeeze(0) > 0
        mask = mask.type(torch.long)

        return image, mask
