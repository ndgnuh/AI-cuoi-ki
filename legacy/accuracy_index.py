import torch


def accuracy(metric):
    def f(pred, y, thres=0.7):
        fg = metric(pred > thres, y > thres)
        bg = metric(pred <= thres, y <= thres)
        return (fg + bg) / 2
    f.__qualname__ = metric.__name__
    return f


@accuracy
def dice(pred, y):
    overlap = torch.sum(torch.logical_and(pred, y))
    return (2 * overlap) / (pred.sum() + y.sum())


@accuracy
def iou(pred, y, thres=0.7):
    # Will be zero if Truth=0 or Prediction=0
    overlap = torch.logical_and(pred, y).sum((2, 3))
    union = torch.logical_or(pred, y).sum((2, 3))
    result = (overlap + 1e-5) / (union + 1e-5)
    # Mean over batch
    result = result.sum() / result.shape[0]
    return result
