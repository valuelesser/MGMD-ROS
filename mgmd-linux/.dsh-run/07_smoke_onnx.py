#!/usr/bin/env python3
"""07_smoke_onnx.py — 在 .venv38 里单独验证 ONNX 推理链路（不经过 ROS）。"""
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

import numpy as np

from mgmd import config as C
print("ARD_VIDEOS =", C.ARD_VIDEOS, C.ARD_VIDEOS.is_dir())
print("WEIGHTS_DIR =", C.WEIGHTS_DIR)

from mgmd_onnx import LacOnnx, LadOnnx

t0 = time.perf_counter()
lac = LacOnnx(device="cuda")
lad = LadOnnx(conf_thresh=0.5, device="cuda")
print("providers LAC =", lac.session.get_providers())
print("providers LAD =", lad.session.get_providers())
print("init %.1f ms" % ((time.perf_counter() - t0) * 1000))

crop = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
probs = lac.mav_prob_batch([crop] * 8)
print("LAC batch8 ->", ["%.4f" % p for p in probs])

frame = (np.random.rand(720, 960, 3) * 255).astype(np.uint8)
t0 = time.perf_counter()
out = [lad.refine(frame, (480, 360)) for _ in range(5)]
print("LAD refine x5 ->", [None if r is None else round(r.conf, 3) for r in out])
print("LAD %.1f ms/call" % ((time.perf_counter() - t0) * 200))

print("SMOKE_OK")
