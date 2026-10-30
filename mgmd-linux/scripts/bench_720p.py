"""scripts/bench_720p.py — 720p 真实分辨率端到端实测（S3.5 交付物，SPEC-ROS §14.4）。

目的：把 §14.7 里 720p 的**估算值**换成实测值。
TT 相机是 960×720，而 ARD-MAV 测试视频是 1920×1080。
此处把 1080p 视频降采样到 960×720（分辨率真实，画面内容非 TT）。

    source scripts/env_setup_ros38.sh && $PY38 scripts/bench_720p.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from mgmd import config as C
from mgmd.io_video import VideoReader
from mgmd.lac import crop_box
from mgmd.mfe import MFEConfig, enhance
from mgmd_onnx import LacOnnx, LadOnnx
from mgmd_onnx.tf_fast import FastSortTracker


def run(video: str, size, device: str, n: int, warm: int = 25) -> dict:
    cfg = MFEConfig()
    cfg.apply_recommended()
    cfg.validate()
    k = cfg.k

    lac = LacOnnx(device=device)
    lad = LadOnnx(conf_thresh=0.5, device=device)
    tr = FastSortTracker(iou_thresh=0.3, max_age=3, min_hits=1, span=3,
                         d_mode="self", d_min=0.02)

    buf = {}
    with VideoReader(video) as vr:
        for i, f in vr.iter_frames(start=1, limit=n + 2 * k + 2):
            buf[i] = cv2.resize(f, size, interpolation=cv2.INTER_AREA) if size else f

    t_mfe, t_tf, t_lac, t_lad, t_tot = [], [], [], [], []
    n_tr, n_lac, n_lad = [], [], []
    done = 0
    for t in range(2, n + 1):
        if (t - k) not in buf or (t + k) not in buf:
            continue
        a = time.perf_counter()
        res = enhance(buf[t - k], buf[t], buf[t + k], cfg, t)
        b = time.perf_counter()
        pool = [((float(x), float(y), float(x + w), float(y + h)), float(w * h))
                for (x, y, w, h) in res.boxes]

        tracks = tr.update([bb for bb, _ in pool], t, scores=[s for _, s in pool])
        c = time.perf_counter()
        pool = [(x.box, x.score) for x in tracks]
        n_tr.append(len(pool))

        if pool:
            crops, keep = [], []
            for bb, s in pool:
                cc = crop_box(buf[t], bb)
                if cc is not None:
                    crops.append(cc)
                    keep.append((bb, s))
            if crops:
                pr = lac.mav_prob_batch(crops)
                pool = [(bb, p) for (bb, _s), p in zip(keep, pr) if p >= 0.5]
            else:
                pool = []
        d = time.perf_counter()
        n_lac.append(len(pool))

        if pool:
            out = []
            for bb, _s in pool:
                rb = lad.refine(buf[t], ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2))
                if rb is not None and rb.conf >= 0.5:
                    out.append((rb.box, rb.conf))
            pool = out
        e = time.perf_counter()
        n_lad.append(len(pool))

        done += 1
        if done > warm:
            t_mfe.append((b - a) * 1000)
            t_tf.append((c - b) * 1000)
            t_lac.append((d - c) * 1000)
            t_lad.append((e - d) * 1000)
            t_tot.append((e - a) * 1000)

    def st(v):
        s = sorted(v)
        return dict(p50=round(statistics.median(s), 2),
                    mean=round(statistics.mean(s), 2),
                    p95=round(s[int(len(s) * 0.95)], 2))

    return dict(
        video=video,
        resolution=f"{size[0]}x{size[1]}" if size else "native",
        frames=len(t_tot),
        stages_ms=dict(MFE=st(t_mfe), TF=st(t_tf), LAC=st(t_lac), LAD=st(t_lad)),
        end_to_end_ms=st(t_tot),
        hz=dict(p50=round(1000 / statistics.median(t_tot), 2),
                p95=round(1000 / sorted(t_tot)[int(len(t_tot) * 0.95)], 2)),
        avg_tracks=round(float(np.mean(n_tr)), 1),
        avg_lac=round(float(np.mean(n_lac)), 2),
        avg_lad=round(float(np.mean(n_lad)), 2),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom30")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/logs/bench_720p.json")
    args = ap.parse_args()

    results = []
    for name, size in [("1080p（测试集原生）", None), ("720p（TT 实际）", (960, 720))]:
        r = run(args.video, size, args.device, args.frames)
        r["label"] = name
        results.append(r)
        print(f"\n=== {name} ===")
        print(f"  端到端 p50={r['end_to_end_ms']['p50']:.1f} ms  "
              f"p95={r['end_to_end_ms']['p95']:.1f} ms  -> {r['hz']['p50']:.1f} Hz")
        for kk, v in r["stages_ms"].items():
            print(f"    {kk:4s} p50={v['p50']:7.2f} ms  p95={v['p95']:7.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
