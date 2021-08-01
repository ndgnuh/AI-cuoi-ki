import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor
import accuracy_index as acc

def tensor_to_image(x):
    x = torch.squeeze(x, 0)
    x = x.transpose(0, 2).transpose(0, 1)
    return x.detach().numpy()

def rgb_bg(image, r, g, b):
    bg = np.ones_like(image)
    bg[:, :, 0] = b
    bg[:, :, 1] = g
    bg[:, :, 2] = r
    return bg


def conv(image, kernel):
    kernel = np.array(kernel)
    result = cv2.filter2D(image, -1, kernel=kernel)
    return result


def mix_bg(model, image, bg, threshold = 0.7, dilation=1, resize=None):
    img = np.copy(image)
    # image = conv(image, [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # if np.any(image > 1):
    #     image = (image / 255).astype(np.float32)
    # image = cv2.addWeighted(image, 1, np.ones_like(image) * 0, 0.3, 1)
    if resize is not None:
        image = cv2.resize(image, resize)
    image = ToTensor()(image)
    image = torch.unsqueeze(image, 0)

    mask = model(image)
    mask = mask > threshold
    mask = tensor_to_image(mask).astype(np.uint8)
    oldshape = mask.shape

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    if dilation > 0:
        mask = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8), iterations=1)
    mask = cv2.blur(mask, (30, 30))
    mask = mask.reshape([*mask.shape, 1])

    image = img
    result = np.multiply(image, mask) + np.multiply(bg, 1 - mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


def test_model(model, dataloader, index):
    device = "gpu" if torch.cuda.is_available() else "cpu"
    correct = 0
    model = model.to(device)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            correct += acc.accuracy(index, pred, y)
    correct /= len(dataloader.dataset)
    return correct
