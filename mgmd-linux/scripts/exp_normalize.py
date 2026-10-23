"""scripts/exp_normalize.py — 差分图归一化方案对照（SPEC §8.17）。

论文原文只说"a threshold of 5 is applied on E_1,2"，未提归一化。
但绝对阈值 5 在不同光照/对比度下行为差异极大，本脚本对比几种归一化方案。

方案：
    abs5       : 基线，diff > 5（论文原文）
    norm255    : diff / diff.max() * 255，再 >5      ← 全局归一化
    norm_p99   : diff / P99(diff) * 255，再 >5       ← 分位归一化（抗异常值）
    norm_local : 分块局部归一化（16x16 块各自除以块内 P99）
    zscore     : (diff - mean) / std > 5             ← z 分数

评估：候选数/帧、MFE 候选层召回（GT IoU>0.25）、跑 4 个视频排除单视频偏差。

用法：
    python3 scripts/exp_normalize.py
    python3 scripts/exp_normalize.py --videos phantom41 phantom63 phantom09 phantom70
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from mgmd import config as C
from mgmd.annotations import load_video_gt
from mgmd.io_video import VideoReader
from mgmd.metrics import box_iou
from mgmd.mfe import (MFEConfig, _binarize, align_to_reference, grid_keypoints,
                      postprocess, preprocess)


def norm_diff(diff: np.ndarray, scheme: str, valid: np.ndarray,
              base_threshold: int = 5) -> tuple[np.ndarray, float]:
    """按方案归一化差分图。

    关键：归一化会改变量纲，阈值必须同比换算，否则等价于把门槛降了 10 倍以上。
    返回 (归一化后的图, 应当使用的阈值)。

    阈值换算原则：让归一化后的阈值等价于"原始空间里的 base_threshold"。
    """
    d = diff.astype(np.float32)
    if scheme == "abs":
        return diff, float(base_threshold)

    if scheme == "norm255":
        # 缩放到 0-255（除以全局最大值）
        m = d.max()
        if m <= 0:
            return np.zeros_like(diff), float(base_threshold)
        scale = 255.0 / m
        return np.clip(d * scale, 0, 255).astype(np.uint8), base_threshold * scale

    if scheme == "norm_p99":
        # 缩放到 0-255（除以 P99，抗异常值）
        p = np.percentile(d, 99)
        if p <= 0:
            return np.zeros_like(diff), float(base_threshold)
        scale = 255.0 / p
        return np.clip(d * scale, 0, 255).astype(np.uint8), base_threshold * scale

    if scheme == "zscore":
        # z 分数映射：1 个标准差 = 25.5（这样 z=0.2 对应 5.1）
        mu, sd = d.mean(), d.std()
        if sd <= 1e-6:
            return np.zeros_like(diff), float(base_threshold)
        z = (d - mu) / sd
        out = np.clip(z * 25.5, 0, 255)
        # 原始 diff > base 等价于 z > (base-mu)/sd，归一化后阈值 = 该 z * 25.5
        return out.astype(np.uint8), max(0.0, (base_threshold - mu) / sd * 25.5)

    if scheme == "norm_local":
        # 16x16 分块局部归一化：每块缩放到 0-255（除以块内 P90）
        h, w = d.shape
        bs = 16
        out = np.zeros_like(d)
        thr_vals = []
        for y in range(0, h, bs):
            for x in range(0, w, bs):
                blk = d[y:y + bs, x:x + bs]
                if blk.size == 0:
                    continue
                p = max(np.percentile(blk, 90), 1.0)
                out[y:y + bs, x:x + bs] = np.clip(blk / p * 255.0, 0, 255)
                thr_vals.append(base_threshold / p * 255.0)
        thr = float(np.median(thr_vals)) if thr_vals else float(base_threshold)
        return out.astype(np.uint8), thr

    raise ValueError(scheme)


def run_scheme(video: str, scheme: str, cfg: MFEConfig, start: int,
               frames: int, gt: dict) -> dict:
    n_vis = n_hit = n_cand = n_frame = 0
    hits = []
    k = cfg.k

    with VideoReader(video) as vr:
        buf = {}
        for idx, fr in vr.iter_frames(start=start, limit=frames + 2 * k + 2):
            buf[idx] = fr
            t = idx - k
            if t < start or (t - k) not in buf:
                continue
            if (t + k) not in buf:
                continue

            cur = preprocess(buf[t], cfg)
            prev = preprocess(buf[t - k], cfg)
            nxt = preprocess(buf[t + k], cfg)

            pa, vp, _ = align_to_reference(prev, cur, cfg)
            na, vn, _ = align_to_reference(nxt, cur, cfg)

            d12 = cv2.absdiff(cur, pa)
            d23 = cv2.absdiff(cur, na)
            n12, thr12 = norm_diff(d12, scheme, vp, cfg.threshold)
            n23, thr23 = norm_diff(d23, scheme, vn, cfg.threshold)
            b12 = _binarize(n12, int(round(thr12)), vp)
            b23 = _binarize(n23, int(round(thr23)), vn)
            motion = cv2.bitwise_or(b12, b23)

            _, boxes = postprocess(motion, cfg)
            n_frame += 1
            n_cand += len(boxes)

            g = gt.get(t)
            best = 0.0
            if g is not None and g.visible:
                n_vis += 1
                cand = [(x, y, x + w, y + h) for (x, y, w, h) in boxes]
                for o in g.objects:
                    for b in cand:
                        v = box_iou(b, (o.x1, o.y1, o.x2, o.y2))
                        if v > best:
                            best = v
                if best > C.IOU_THRESH:
                    n_hit += 1
            hits.append(best > C.IOU_THRESH)
            for kk in [b for b in buf if b < t - 2 * k - 2]:
                del buf[kk]

    longest = cur_m = 0
    for h in hits:
        cur_m = 0 if h else cur_m + 1
        longest = max(longest, cur_m)

    return {
        "frames": n_frame, "visible": n_vis, "hit": n_hit,
        "recall": n_hit / n_vis if n_vis else 0.0,
        "cand_per_frame": n_cand / max(1, n_frame),
        "longest_miss": longest,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+",
                    default=["phantom41", "phantom63", "phantom09", "phantom70"])
    ap.add_argument("--start", type=int, default=500)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--out", default=str(C.LOGS / "exp_normalize.json"))
    args = ap.parse_args(argv)

    schemes = ["abs", "norm255", "norm_p99", "zscore", "norm_local"]
    cfg = MFEConfig(k=1, mode="3frame_or", threshold=args.threshold)
    gts = {v: load_video_gt(v) for v in args.videos}

    print(f"阈值固定 {args.threshold}，各归一化方案对照", flush=True)
    print(f"{'video':<11}{'scheme':<12}{'cand/f':>8}{'recall':>8}{'miss':>6}", flush=True)
    rows = []
    for v in args.videos:
        for s in schemes:
            r = run_scheme(v, s, cfg, args.start, args.frames, gts[v])
            print(f"{v:<11}{s:<12}{r['cand_per_frame']:>8.0f}"
                  f"{r['recall']*100:>7.1f}%{r['longest_miss']:>6}", flush=True)
            rows.append({"video": v, "scheme": s, **r})
        print("", flush=True)

    # 汇总：各方案跨视频平均
    print("=== 跨视频平均 ===", flush=True)
    print(f"{'scheme':<12}{'cand/f':>9}{'recall':>9}{'miss':>7}", flush=True)
    for s in schemes:
        sub = [r for r in rows if r["scheme"] == s]
        if not sub:
            continue
        print(f"{s:<12}{np.mean([r['cand_per_frame'] for r in sub]):>9.0f}"
              f"{np.mean([r['recall'] for r in sub])*100:>8.1f}%"
              f"{np.mean([r['longest_miss'] for r in sub]):>7.1f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "threshold": args.threshold, "start": args.start, "frames": args.frames,
        "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n已写入 {args.out}", flush=True)
    return C.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
