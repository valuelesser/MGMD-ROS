#!/usr/bin/env python3
"""验证 step_H 对齐对 TF 的影响（SPEC-ROS §22，改代码前先跑）。

回答三个问题：
  Q1  step_H 能补齐多少？compose_H 的成功率是多少？
  Q2  对齐前后 d_value 变化多大？跨过 d_min 的比例差多少？
  Q3  align_missing 的真实占比（当前生产代码里恒为 100%）

方法：离线逐帧跑 MFE→SORT，对同一份 SORT 输出分别用
      align=True/False 两种口径算 classify，**不修改任何生产代码**。

用法：
    python3 scripts/verify_step_H.py --video <path> --frames 300 --w 960 --h 720
    python3 scripts/verify_step_H.py --video <path> --no-align-h   # 模拟当前生产行为
"""
from __future__ import annotations

import argparse
import collections
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(video, n_frames, w, h, d_min, span, feed_h, d_mode):
    from mgmd.mfe import MFEConfig, enhance
    from mgmd.tf import SortTracker

    cfg = MFEConfig()
    cfg.apply_recommended()
    cfg.validate()
    k = cfg.k

    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < n_frames + 2 * k + 1:
        ok, f = cap.read()
        if not ok:
            break
        if w and h:
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit("读不到视频")

    tr_align = SortTracker(span=span, d_mode=d_mode, d_min=d_min)   # 对齐口径
    tr_plain = SortTracker(span=span, d_mode=d_mode, d_min=d_min)   # 不对齐口径

    stats = collections.Counter()
    d_align, d_plain, d_delta = [], [], []
    h_ok = 0
    h_total = 0

    for t in range(k, len(frames) - k):
        res = enhance(frames[t - k], frames[t], frames[t + k], cfg, t)
        boxes = res.boxes
        if not boxes:
            stats["empty_frames"] += 1
            continue
        pool = [((float(x), float(y), float(x + ww), float(y + hh)), float(ww * hh))
                for (x, y, ww, hh) in boxes]
        dets = [b for b, _ in pool]
        scores = [s for _, s in pool]

        # step_H 的语义：key=帧 f，值 = H(f-1 -> f)。MFE 的 prev_H 正是 t-1 -> t。
        step = res.info.get("prev_H") if feed_h else None
        if feed_h:
            h_total += 1
            if step is not None:
                h_ok += 1

        tr_align.update(dets, t, scores=scores, step_H=step)
        tr_plain.update(dets, t, scores=scores, step_H=None)

        for tid, s in tr_align.states.items():
            p = tr_plain.states.get(tid)
            if p is None:
                continue
            stats["cmp_" + s.reason] += 1
            # reason 会被 classify 末尾覆盖（见 tests/test_tf.py），
            # 故对齐是否真的生效必须直接看 compose_H。
            a_pair = s.window(span)
            if a_pair is not None:
                (f0, _b0), (f1, _b1) = a_pair
                stats["compose_total"] += 1
                if s.compose_H(f0, f1) is not None:
                    stats["compose_ok"] += 1
            if s.reason != "lifetime_lt_3" and s.reason != "no_window":
                d_align.append(s.d_value)
                d_plain.append(p.d_value)
                d_delta.append(abs(s.d_value - p.d_value))
        stats["frames"] += 1

    n = max(1, len(d_align))
    print(f"\n--- 处理 {stats['frames']} 帧, 空帧 {stats['empty_frames']} ---")
    print(f"Q1 step_H 提供率: {h_ok}/{h_total} = {100*h_ok/max(1,h_total):.1f}%")
    ct, co = stats["compose_total"], stats["compose_ok"]
    print(f"   compose_H 成功率: {co}/{ct} = {100*co/max(1,ct):.1f}%"
          f"   <- 对齐是否生效的真实指标（reason 字段不可信）")

    tot = sum(v for kk, v in stats.items() if kk.startswith("cmp_"))
    print(f"\nQ3 classify 结果分布（对齐口径）, 共 {tot} 次: ")
    for kk, v in stats.most_common():
        if kk.startswith("cmp_"):
            print(f"    {kk[4:]:<24}{v:>7}  {100*v/max(1,tot):>5.1f}%")

    if d_align:
        print(f"\nQ2 d_value 对比（{len(d_align)} 次可比较）:")
        print(f"    对齐   p50={st.median(d_align):.4f}  mean={st.mean(d_align):.4f}")
        print(f"    不对齐 p50={st.median(d_plain):.4f}  mean={st.mean(d_plain):.4f}")
        print(f"    |Δd|   p50={st.median(d_delta):.4f}  "
              f"p95={sorted(d_delta)[int(len(d_delta)*.95)]:.4f}  "
              f"max={max(d_delta):.4f}")
        cross_a = sum(1 for v in d_align if v >= d_min)
        cross_p = sum(1 for v in d_plain if v >= d_min)
        print(f"    >= d_min({d_min})  对齐 {cross_a} ({100*cross_a/n:.1f}%)  "
              f"不对齐 {cross_p} ({100*cross_p/n:.1f}%)")
        flips = sum(1 for a, p in zip(d_align, d_plain)
                    if (a >= d_min) != (p >= d_min))
        print(f"    **判定翻转** {flips} 次 ({100*flips/n:.1f}%)")
    else:
        print("\nQ2 无可比较样本")

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--d-min", type=float, default=0.02)
    ap.add_argument("--span", type=int, default=3)
    ap.add_argument("--d-mode", default="self")
    ap.add_argument("--no-align-h", action="store_true",
                    help="不喂 step_H（模拟当前生产行为，用于确认『恒 missing』）")
    a = ap.parse_args()
    run(a.video, a.frames, a.w, a.h, a.d_min, a.span, feed_h=not a.no_align_h,
        d_mode=a.d_mode)


if __name__ == "__main__":
    main()
