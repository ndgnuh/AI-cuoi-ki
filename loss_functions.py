import torch
import torch.nn.functional as F
import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    def forward(self, yhat, y):
        log_hat = torch.log2(yhat)
        loss = -torch.mean(torch.multiply(y, log_hat))
        return loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

ALPHA = 0.5
BETA = 0.5
GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma

        return FocalTversky

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss
class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, eps=1e-9):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
        return combo


# A kind of surface loss
# ported from
# https://viblo.asia/p/image-segmentation-sinet-extreme-lightweight-portrait-segmentation-sinet-paper-explaination-and-coding-implementation-ORNZq1MeZ0n#_63-loss-function-23
class SINetLoss(nn.Module):
    def __init__(self, ld=0.9):
        super(SINetLoss, self).__init__()
        self.ld = ld
        self.loss = nn.BCELoss()

    def dilation(self, y):
        y = F.max_pool2d(y, 15, padding=7, stride=1)
        return y

    def erosion(self, y):
        y = -F.max_pool2d(-y, 15, padding=7, stride=1)
        return y

    def boundary(self, y):
        return self.dilation(y) - self.erosion(y)

    def boundary_loss(self, ypred, y):
        loss = self.loss(self.boundary(ypred), self.boundary(y))
        return loss

    def forward(self, ypred, y):
        loss = self.loss(ypred, y)
        bd_loss = self.boundary_loss(ypred, y)
        loss = self.ld * bd_loss + loss
        # Multiply by batchsize, this is not in the original loss
        # loss = loss * y.shape[0]
        return loss


# Get the boundary of the mask
def boundary(mask, k=20):
    d = F.max_pool2d(mask, k, padding=k//2, stride=1)
    e = -F.max_pool2d(-mask, k, padding=k//2, stride=1)
    return d-e


class BDDiceBCELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BDDiceBCELoss, self).__init__()
        self.loss = DiceBCELoss(*args, **kwargs)

    def forward(self, inputs, targets):
        b_inputs = boundary(inputs)
        b_targets = boundary(targets)
        return self.loss(b_inputs, b_targets) + self.loss(inputs, targets)


class SurfaceLoss():
    def __call__(self, probs, dist_maps):
        pc = probs.type(torch.float32)
        dc = dist_maps.type(torch.float32)
        multipled = torch.einsum("bkwh,bkwh->bkwh", pc, dc)
        loss = multipled.mean()
        return loss
