import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor

def rgb_bg(image, r, g, b):
    bg = np.ones_like(image)
    bg[:, :, 0] = b
    bg[:, :, 1] = g
    bg[:, :, 2] = r
    return bg

def mix_bg(model, image, bg, thr = 0.7, dilation=1, resize=None):
    if resize is None:
        resize = (128, 128)
    img = np.copy(image)
    image = cv2.resize(img, resize)
    image = ToTensor()(image)
    image = torch.unsqueeze(image, 0)

    mask = model(image)
    mask = mask > thr
    mask = tensor_to_image(mask).astype(np.uint8)
    oldshape = mask.shape

    if dilation > 0:
        mask = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8), iterations=1)
    mask = cv2.blur(mask, (10, 10))
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask = mask.reshape([*mask.shape, 1])
    print(mask.shape)

    image = img
    result = np.multiply(image, mask) + np.multiply(bg, 1 - mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result
