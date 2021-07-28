from PIL import Image
from config import parse_args
from torchvision.transfroms import ToTensor
from torch import unsqueeze

config = parse_args()
model = config.model


def gui():
    pass


def cli():
    pass


def get_mask(img):
    orig_img = img
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    if isinstance(img, Image):
        img = unsqueeze(ToTensor()(img), 0)

    return model(img)
