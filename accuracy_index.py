import torch


def dice(pred, y):
    overlap = torch.sum(torch.logical_and(pred, y))
    return (2 * overlap) / (pred.sum() + y.sum())


def iou(pred, y):
    overlap = torch.sum(torch.logical_and(pred, y))
    union = torch.sum(torch.logical_or(pred, y))
    return (overlap / union)


def accuracy(index, yhat, y, thres = 0.7):
    fg = index(yhat > thres, y > thres)
    bg = index(yhat <= thres, y <= thres)
    return ((fg + bg) / 2).item()
