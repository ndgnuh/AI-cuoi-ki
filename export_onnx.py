import torch
from ptrseg.model import SegmentLearner

model = SegmentLearner.load_from_checkpoint("latest.ckpt")
print(model)

model.eval()
model.model.to_onnx("segm.onnx", torch.rand(1, 3, 640, 640))
