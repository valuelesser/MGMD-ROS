#!/usr/bin/env python3
"""11_bench_lad.py — 用真实视频帧测 LAD/LAC 的 GPU 延迟（含 warmup）。"""
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/gaodu/毕业设计/mgmd-linux")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MGMD_YOLOV5_DIR", str(ROOT / "third_party" / "yolov5-v6.0"))
os.environ.setdefault(
    "MGMD_ARD_ROOT",
    "/mnt/c/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV",
)

import cv2
import numpy as np

from mgmd import config as C
from mgmd_onnx import LacOnnx, LadOnnx

vid = C.ARD_VIDEOS / "phantom30.mp4"
cap = cv2.VideoCapture(str(vid))
for _ in range(60):
    ok, frame = cap.read()
cap.release()
assert ok, "读不到视频帧"
frame = cv2.resize(frame, (960, 720), interpolation=cv2.INTER_AREA)
print("frame", frame.shape)

lac = LacOnnx(device="cuda")
lad = LadOnnx(conf_thresh=0.5, device="cuda")
print("LAD providers", lad.session.get_providers())

crop = cv2.resize(frame[300:500, 400:600], (320, 320))
# warmup
for _ in range(5):
    lad.refine(frame, (480, 360))
    lac.mav_prob(crop)

t0 = time.perf_counter()
N = 20
for _ in range(N):
    r = lad.refine(frame, (480, 360))
t = (time.perf_counter() - t0) / N * 1000
print("LAD refine %.2f ms/call  (last conf=%s)" % (t, None if r is None else round(r.conf, 3)))

t0 = time.perf_counter()
for _ in range(N):
    lac.mav_prob(crop)
t = (time.perf_counter() - t0) / N * 1000
print("LAC mav_prob %.2f ms/call" % t)

# 纯 ORT 前向耗时（不含预处理/NMS）
blob = lad._preprocess(crop)
name = lad.session.get_inputs()[0].name
t0 = time.perf_counter()
for _ in range(N):
    lad.session.run(None, {name: blob})
t = (time.perf_counter() - t0) / N * 1000
print("ORT session.run %.2f ms" % t)
print("BENCH_OK")
