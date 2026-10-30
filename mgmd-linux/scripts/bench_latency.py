"""scripts/bench_latency.py — 端到端延迟与吞吐实测（SPEC-ROS §13.11）。

目的：为 S3 的实时性指标提供**实测依据**，而不是拍脑袋定指标。

    source scripts/env_setup_ros38.sh
    $PY38 scripts/bench_latency.py --video phantom30 --device cuda

输出：MFE / TF / LAC / LAD 各自的耗时与占比，以及端到端 ms/帧、可达 Hz。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mgmd import config as C
from mgmd.io_video import VideoReader
from mgmd.lac import crop_box
from mgmd.mfe import MFEConfig, enhance
from mgmd.tf import SortTracker


def bench(video: str, device: str, n_frames: int, warmup: int = 30) -> dict:
    cfg = MFEConfig()
    cfg.apply_recommended()
    cfg.validate()

    from mgmd_onnx import LacOnnx, LadOnnx
    lac = LacOnnx(device=device)
    lad = LadOnnx(conf_thresh=0.5, device=device)

    tracker = SortTracker(iou_thresh=0.3, max_age=3, min_hits=1,
                          span=3, d_mode="self", d_min=0.02)
    t_mfe, t_tf, t_lac, t_lad = [], [], [], []
    n_cand, n_track, n_lac, n_lad = [], [], [], []

    buf = {}
    k = cfg.k
    done = 0
    with VideoReader(video) as vr:
        for idx, fr in vr.iter_frames(start=1, limit=n_frames + 2 * k + 2):
            buf[idx] = fr
            t = idx - k
            if t < 1 or (t - k) not in buf or (t + k) not in buf:
                continue
            # ---- MFE
            a = time.perf_counter()
            res = enhance(buf[t - k], buf[t], buf[t + k], cfg, t)
            b = time.perf_counter()
            pool = [((float(x), float(y), float(x + w), float(y + h)), float(w * h))
                    for (x, y, w, h) in res.boxes]
            n_cand.append(len(pool))

            # ---- TF
            tracks = tracker.update([bb for bb, _ in pool], t,
                                    scores=[s for _, s in pool])
            c = time.perf_counter()
            pool = [(tr.box, tr.score) for tr in tracks]
            n_track.append(len(pool))

            # ---- LAC
            d = e = c
            if pool:
                crops, keep = [], []
                for bb, s in pool:
                    cc = crop_box(buf[t], bb)
                    if cc is not None:
                        crops.append(cc)
                        keep.append((bb, s))
                if crops:
                    probs = lac.mav_prob_batch(crops)
                    pool = [(bb, p) for (bb, _s), p in zip(keep, probs) if p >= 0.5]
                else:
                    pool = []
            d = time.perf_counter()
            n_lac.append(len(pool))

            # ---- LAD
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
            if done > warmup:
                t_mfe.append((b - a) * 1000)
                t_tf.append((c - b) * 1000)
                t_lac.append((d - c) * 1000)
                t_lad.append((e - d) * 1000)

            for kk in [x for x in buf if x < t - 2 * k - 2]:
                del buf[kk]

    def stat(v):
        return dict(p50=round(statistics.median(v), 2),
                    mean=round(statistics.mean(v), 2),
                    p95=round(sorted(v)[int(len(v) * 0.95)], 2),
                    sum=round(sum(v), 2)) if v else None

    total = [a + b + c_ + d_ for a, b, c_, d_ in
             zip(t_mfe, t_tf, t_lac, t_lad)]
    nf = len(total)
    return dict(
        video=video, device=device, frames=nf,
        各阶段ms=dict(MFE=stat(t_mfe), TF=stat(t_tf),
                     LAC=stat(t_lac), LAD=stat(t_lad)),
        端到端=dict(p50=round(statistics.median(total), 2),
                    mean=round(statistics.mean(total), 2),
                    p95=round(sorted(total)[int(nf * 0.95)], 2)),
        可达Hz=dict(p50=round(1000 / statistics.median(total), 2),
                    p95=round(1000 / sorted(total)[int(nf * 0.95)], 2)),
        平均候选数=round(float(np.mean(n_cand)), 1),
        平均轨迹数=round(float(np.mean(n_track)), 1),
        平均过LAC=round(float(np.mean(n_lac)), 2),
        平均过LAD=round(float(np.mean(n_lad)), 2),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom30")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r = bench(args.video, args.device, args.frames)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
