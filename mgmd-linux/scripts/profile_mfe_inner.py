#!/usr/bin/env python3
"""把 MFE 拆成子步骤计时（SPEC-ROS §20.7）。

目的：判断 MFE 能不能上 GPU。只有先知道 46 ms 花在哪个子步骤上，
才能判断该步骤是「GPU 上本来就快（CUDA 有对应实现）」还是
「GPU 也帮不上（CPU→GPU 传输/同步开销就吃掉收益）」。

用法：
    python3 scripts/profile_mfe_inner.py --video <path> --frames 60 --w 960 --h 720
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def timeit(fn, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
    return (time.perf_counter() - t0) * 1000.0 / n, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    from mgmd.mfe import (MFEConfig, preprocess, align_to_reference,
                          _binarize, postprocess, _valid_region)

    cfg = MFEConfig()
    cfg.apply_recommended()
    cfg.validate()

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    frames = []
    for _ in range(a.frames + 2):
        ok, f = cap.read()
        if not ok:
            break
        if a.w and a.h:
            f = cv2.resize(f, (a.w, a.h), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if len(frames) < 3:
        raise SystemExit("视频帧不足")
    print(f"视频 {a.video}  帧 {len(frames)}  {frames[0].shape[1]}x{frames[0].shape[0]}")

    acc = {}

    def add(k, v):
        acc.setdefault(k, []).append(v)

    for i in range(1, len(frames) - 1):
        prev, cur, nxt = frames[i - 1], frames[i], frames[i + 1]

        ms, cur_gray = timeit(lambda: preprocess(cur, cfg))
        add("1_preprocess_cur", ms)
        ms, prev_gray = timeit(lambda: preprocess(prev, cfg))
        ms2, next_gray = timeit(lambda: preprocess(nxt, cfg))
        add("1_preprocess_prev_next", ms + ms2)

        ms, (prev_al, prev_valid, prev_H) = timeit(
            lambda: align_to_reference(prev_gray, cur_gray, cfg))
        add("2_align_prev", ms)
        ms, (_next_al, next_valid, next_H) = timeit(
            lambda: align_to_reference(next_gray, cur_gray, cfg))
        add("2_align_next", ms)

        ms, d12 = timeit(lambda: _binarize(cv2.absdiff(cur_gray, prev_al),
                                           cfg.threshold, prev_valid, cfg))
        add("3_absdiff+binarize_12", ms)

        # 单独看 warpPerspective 占 align 多少（关键：GPU 能否加速）
        if prev_H is not None:
            h, w = cur_gray.shape
            ms, _ = timeit(lambda: cv2.warpPerspective(
                prev_gray, prev_H, (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0))
            add("2a_warpPerspective", ms)

        ms, _ = timeit(lambda: align_to_reference(prev_gray, cur_gray, cfg))
        # 其中 LK 光流 + RANSAC 的纯关键点部分
        from mgmd.mfe import grid_keypoints
        h, w = cur_gray.shape
        pts = grid_keypoints(w, h, cfg.grid_cols, cfg.grid_rows)
        lk = dict(winSize=(15, 15), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        ms_lk, (tracked, status, _e) = timeit(
            lambda: cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, **lk))
        add("2b_calcOpticalFlowPyrLK", ms_lk)
        ok = status.ravel().astype(bool)
        src_pts = pts.reshape(-1, 2)[ok]
        ref_pts = tracked.reshape(-1, 2)[ok]
        ms_ho, _ = timeit(lambda: cv2.findHomography(ref_pts, src_pts, cv2.RANSAC,
                                                     cfg.ransac_thresh))
        add("2c_findHomography", ms_ho)

        # valid_region（也是 warp/fillPoly，GPU 可加速项）
        if prev_H is not None:
            ms_vr, _ = timeit(lambda: _valid_region(prev_H, w, h))
            add("2d_valid_region", ms_vr)

        ms, (closed, boxes) = timeit(lambda: postprocess(
            cv2.bitwise_or(d12, _binarize(cv2.absdiff(
                cur_gray, align_to_reference(next_gray, cur_gray, cfg)[0]),
                cfg.threshold, next_valid, cfg)), cfg))
        add("4_postprocess", ms)

    print()
    print(f"{'子步骤':<28}{'p50':>9}{'p95':>9}{'均值':>9}{'占比':>8}")
    print("-" * 64)
    # 单次 MFE 的等效总耗时 = 各步骤均值之和
    sub = ("2a_", "2b_", "2c_", "2d_")
    total = sum(st.mean(v) for k, v in acc.items() if not k.startswith(sub))
    order = ["1_preprocess_cur", "1_preprocess_prev_next", "2_align_prev",
             "2_align_next", "3_absdiff+binarize_12", "4_postprocess",
             "2a_warpPerspective", "2b_calcOpticalFlowPyrLK",
             "2c_findHomography", "2d_valid_region"]
    for k in order:
        v = acc.get(k)
        if not v:
            continue
        s = sorted(v)
        p50 = st.median(v)
        p95 = s[int(len(s) * .95)]
        note = "  ← align 的子项" if k.startswith(("2a", "2b", "2c", "2d")) else ""
        share = st.mean(v) / total * 100 if not k.startswith(("2a", "2b", "2c", "2d")) else 0
        print(f"{k:<28}{p50:>9.2f}{p95:>9.2f}{st.mean(v):>9.2f}"
              f"{share:>7.1f}%{note}")
    print("-" * 64)
    print(f"{'等效总计':<28}{'':>9}{'':>9}{total:>9.2f}")
    print("\n注：2a~2d 是 align 的子项，只作『GPU 可加速哪部分』参考，不计入总计。")


if __name__ == "__main__":
    main()
