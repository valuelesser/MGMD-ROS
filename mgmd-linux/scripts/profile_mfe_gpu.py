#!/usr/bin/env python3
"""MFE 各子步骤在 GPU（torch CUDA）上的实测耗时。

背景：本机 OpenCV 无 CUDA 后端（cv2.cuda.getCudaEnabledDeviceCount()==0），
无法用 cv2.cuda 测。改用 torch 实现等价算子，测真实 GPU 耗时，
**包含 CPU→GPU / GPU→CPU 传输开销**（这是决定「值不值得上 GPU」的关键）。

对照组同时跑 CPU（cv2），并给出「零拷贝」与「每步往返」两种口径：
  - roundtrip : 每次调用都把 H2D 计入（最保守，接近真实流水线首尾传递）
  - resident  : 数据已常驻显存，只测 kernel（乐观上界）

用法：
    python3 scripts/profile_mfe_gpu.py --video <path> --frames 50 --w 960 --h 720
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def bench(fn, warmup=3, n=20, sync=False):
    """返回中位耗时(ms)。sync=True 时每次调用后 cuda.synchronize()。"""
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return st.median(ts)


# ---- torch 版算子 -------------------------------------------------------- #

def g_gray(x):                       # BGR uint8 [H,W,3] -> 灰度 uint8 [H,W]
    # 与 cv2.COLOR_BGR2GRAY 同权重（cv2 用定点系数，这里 float 近似后取整）
    b, g, r = x[..., 0].float(), x[..., 1].float(), x[..., 2].float()
    return (0.114 * b + 0.587 * g + 0.299 * r).round().clamp(0, 255).to(torch.uint8)


def g_absdiff(a, b):
    return (a.float() - b.float()).abs().to(torch.uint8)


def g_binarize(diff, t):
    return (diff > t).to(torch.uint8) * 255


def g_warp(gray, H, w, h):
    """等价 cv2.warpPerspective(gray, H, (w,h), WARP_INVERSE_MAP)。

    WARP_INVERSE_MAP 语义：dst(x,y) = src(H·(x,y))。
    用 grid_sample 实现：对每个输出像素 p，采样坐标 = H^-1 作用到 p 的归一化坐标。
    """
    Hinv = torch.linalg.inv(H)
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=gray.device),
        torch.arange(w, dtype=torch.float32, device=gray.device), indexing="ij")
    ones = torch.ones_like(xs)
    pts = torch.stack([xs, ys, ones], dim=-1)          # [h,w,3]
    q = pts @ Hinv.t()                                  # dst->src (因 WARP_INVERSE_MAP)
    qx, qy = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    gx = (qx + 0.5) / w * 2 - 1
    gy = (qy + 0.5) / h * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    src = gray.float().unsqueeze(0).unsqueeze(0)
    out = torch.nn.functional.grid_sample(src, grid, mode="bilinear",
                                          padding_mode="zeros", align_corners=False)
    return out[0, 0].round().clamp(0, 255).to(torch.uint8)


def g_morph(mask, ksize, iters, op):
    """方核形态学 = 可分离 max/min 池化。

    torch 2.4 的 F 没有 min_pool2d，用 -max_pool2d(-x) 实现 min。
    op="open_erode": 腐蚀 iters 次（对应 MORPH_OPEN）
    op="close_dilate": 膨胀 iters 次（对应 MORPH_CLOSE）
    """
    F = torch.nn.functional
    x = mask.float().unsqueeze(0).unsqueeze(0)
    p = ksize // 2
    for _ in range(iters):
        if op == "open_erode":
            x = -F.max_pool2d(-x, ksize, stride=1, padding=p)
        else:
            x = F.max_pool2d(x, ksize, stride=1, padding=p)
    return (x[0, 0] > 0).to(torch.uint8) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=720)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")

    cap = cv2.VideoCapture(a.video)
    frames = []
    for _ in range(a.frames + 1):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (a.w, a.h), interpolation=cv2.INTER_AREA))
    cap.release()
    print(f"帧 {len(frames)}  {a.w}x{a.h}")

    cur_bgr = frames[len(frames) // 2]
    prev_bgr = frames[len(frames) // 2 - 1]
    w, h = a.w, a.h

    # CPU 参考
    cur_gray_cv = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray_cv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    diff_cv = cv2.absdiff(cur_gray_cv, prev_gray_cv)
    bin_cv = (diff_cv > 5).astype(np.uint8) * 255
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # GPU 上常驻数据
    cur_t = torch.from_numpy(cur_bgr).to(dev)
    prev_t = torch.from_numpy(prev_bgr).to(dev)
    cur_g_t = g_gray(cur_t)
    prev_g_t = g_gray(prev_t)

    res = {}
    res["cvtColor"] = (bench(lambda: cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)),
                       bench(lambda: g_gray(cur_t), sync=True))
    res["absdiff"] = (bench(lambda: cv2.absdiff(cur_gray_cv, prev_gray_cv)),
                      bench(lambda: g_absdiff(cur_g_t, prev_g_t), sync=True))
    res["binarize"] = (bench(lambda: (diff_cv > 5).astype(np.uint8) * 255),
                       bench(lambda: g_binarize(g_absdiff(cur_g_t, prev_g_t), 5), sync=True))
    res["morph_open×1"] = (
        bench(lambda: cv2.morphologyEx(bin_cv, cv2.MORPH_OPEN, kern, iterations=1)),
        bench(lambda: g_morph(g_binarize(g_absdiff(cur_g_t, prev_g_t), 5), 3, 1, "open_erode"),
              sync=True))
    res["morph_close×3"] = (
        bench(lambda: cv2.morphologyEx(bin_cv, cv2.MORPH_CLOSE, kern, iterations=3)),
        bench(lambda: g_morph(g_binarize(g_absdiff(cur_g_t, prev_g_t), 5), 3, 3, "close_dilate"),
              sync=True))

    # 连通域（GPU 无直接等价，用 CPU 对照说明这是 GPU 的死区）
    bin_np = bin_cv
    res["connectedComponents"] = (
        bench(lambda: cv2.connectedComponentsWithStats(bin_np, connectivity=8)), None)

    # 单次 H2D / D2H 传输成本（决定「每步往返」是否可行）
    h2d = bench(lambda: torch.from_numpy(cur_gray_cv).to(dev), n=30)
    d2h = bench(lambda: cur_g_t.cpu(), n=30)

    print()
    print(f"{'算子':<22}{'CPU(cv2)':>11}{'GPU(torch)':>12}{'加速比':>9}")
    print("-" * 56)
    tot_c = tot_g = 0.0
    for k, (c, g) in res.items():
        if g is None:
            print(f"{k:<22}{c:>11.2f}{'不可用':>12}{'-':>9}")
            tot_c += c
            continue
        print(f"{k:<22}{c:>11.2f}{g:>12.2f}{c/g:>8.2f}x")
        tot_c += c
        tot_g += g
    print("-" * 56)
    print(f"{'合计':<22}{tot_c:>11.2f}{tot_g:>12.2f}{tot_c/tot_g:>8.2f}x")
    print(f"\n单次 H2D(960x720 灰度) = {h2d:.2f} ms")
    print(f"单次 D2H              = {d2h:.2f} ms")
    print(f"若每算子各往返一次，额外传输 = {(h2d + d2h) * 4:.2f} ms（4 个 GPU 算子）")
    print("\n注：connectedComponentsWithStats 在 torch 中无对应算子，"
          "GPU 侧显示『不可用』，这部分必须留在 CPU。")
    print("    GPU 侧算子需在 kernel 之间保持显存常驻，否则传输开销会吃掉收益。")


if __name__ == "__main__":
    main()
