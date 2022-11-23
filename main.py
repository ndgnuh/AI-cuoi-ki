import cv2
import numpy as np
import torch
from ptrseg.utils import letterbox
from onnxruntime import InferenceSession


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_mask(model, frame):
    frame = letterbox(frame, (640, 640))
    image = (frame / 255).astype('float32')
    image = image.transpose((2, 0, 1))[None, ...]
    prob, thres = model.run(None, {model.get_inputs()[0].name: image})
    prob = 1.0 * (prob[0][0] > thres[0][0])
    return prob, frame


capture = cv2.VideoCapture(0)
model = InferenceSession("segm.onnx", providers=["CUDAExecutionProvider"])

while True:
    ret, frame = capture.read()

    mask, frame = get_mask(model, frame)
    # mask = cv2.dilate(mask, np.ones((3, 3)))
    # mask = cv2.erode(mask, np.ones((3, 3)))
    bg = (190, 220, 0)
    for i in range(3):
        frame[:, :, i] = frame[:, :, i] * mask + (1 - mask) * bg[i]
    # cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
capture.release()
