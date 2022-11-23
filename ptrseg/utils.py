import torch
import os
from os import path
from torch import Tensor
from torchvision.transforms import functional as FT
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from typing import Tuple
from PIL import Image
from functools import lru_cache


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


class SegmentDataset(VisionDataset):
    def __init__(self,
                 root: str,
                 image_size: Tuple[int, int] = (640, 640),
                 image_dir: str = "image",
                 mask_dir: str = "mask",
                 cached: bool = True,
                 **kwargs):
        super().__init__(root=root, **kwargs)
        self.root = root
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.imgs = os.listdir(path.join(root, image_dir))

        if cached:
            self.load_sample = lru_cache(self.load_sample)

    def __len__(self):
        return len(self.imgs)

    def load_sample(self, image_path):
        image = Image.open(image_path)
        image = FT.to_tensor(image)
        image = letterbox(image, self.image_size)
        return image

    def __getitem__(self, index):
        name = self.imgs[index]
        image_path = path.join(self.root, self.image_dir, name)
        mask_path = path.join(self.root, self.mask_dir, name)
        image = self.load_sample(image_path)
        mask = self.load_sample(mask_path)
        return image, mask
