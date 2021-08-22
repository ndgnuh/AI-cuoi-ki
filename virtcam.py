import cv2
import numpy as np
import torch
import os
import api
from pyfakewebcam import FakeWebcam
import argparse

def wh_from_str(s: str):
    wh = s.split("x")
    return int(wh[0]), int(wh[1])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", dest="model", default="checkpoint/SegModel7.pth")
    p.add_argument("--input-resize", "-r", dest="input_resize")
    p.add_argument("--input-size", "-s", dest="input_size", default="1280x720")
    p.add_argument("--device", "-d", dest="device", default="cpu")
    p.add_argument("--threshold", "-t", dest="threshold", default=0.7, type=float)
    arg = p.parse_args()
    arg.model = torch.load(arg.model, map_location=arg.device)
    if arg.input_resize:
        arg.input_resize = wh_from_str(arg.input_resize)
    arg.input_size = wh_from_str(arg.input_size)
    return arg

def main(arg):
    cap = cv2.VideoCapture("/dev/video0")
    width = arg.input_size[0]
    height = arg.input_size[1]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    fake = FakeWebcam('/dev/video2', width, height)

    background = cv2.imread("background.jpg")
    bg = cv2.resize(background, (width, height))

    while True:
        _, frame = cap.read()
        frame = api.mix_bg(
                arg.model,
                frame,
                bg,
                threshold=arg.threshold,
                dilation=0,
                resize=arg.input_resize)
        fake.schedule_frame(frame)


while True:
    print("Press Ctrl+\\ to stop, Ctrl+C to change bacground")
    try:
        arg = parse_args()
        main(arg)
    except KeyboardInterrupt:
        continue
