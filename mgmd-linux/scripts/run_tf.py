"""scripts/run_tf.py — 在真实视频上跑 MFE → TF 并统计（SPEC T05）。

用法：
    python3 scripts/run_tf.py --video phantom76 --start 300 --frames 120
    python3 scripts/run_tf.py --video phantom76 --frames 120 --d-mode enclosing --d-min 1.0
    python3 scripts/run_tf.py --video phantom76 --frames 120 --d-mode self --d-min 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mgmd import config as C
from mgmd.annotations import load_video_gt
from mgmd.io_video import VideoReader
from mgmd.metrics import box_iou
from mgmd.mfe import MFEConfig, enhance
from mgmd.tf import SortTracker, distance_metric

def run(video, mfe_cfg, start, frames, d_mode, d_min, span, iou_thresh, max_age):
    gt = load_video_gt(video)
    tracker = SortTracker(iou_thresh=iou_thresh, max_age=max_age, min_hits=1,
                          span=span, d_mode=d_mode, d_min=d_min)
    rows = []
    d_values = []
    n_visible = n_hit = 0
    n_cand = n_out = 0
    hits = []
    t_ms = []

    with VideoReader(video) as vr:
        buf = {}
        k = mfe_cfg.k
        end = start + frames - 1 + k
        for idx, fr in vr.iter_frames(start=start, limit=end - start + k + 1):
            buf[idx] = fr
            t = idx - k
            if t < start or (t - k) not in buf or (C.needs_next if False else (mfe_cfg.needs_next and (t + k) not in buf)):
                continue
            t0 = time.perf_counter()
            res = enhance(buf[t - k], buf[t], buf.get(t + k) if mfe_cfg.needs_next else None, mfe_cfg, t)
            boxes4 = [(x, y, x + w, y + h) for (x, y, w, h) in res.boxes]
            out = tracker.update(boxes4, t)
            t_ms.append((time.perf_counter() - t0) * 1000.0)
            n_cand += len(boxes4)
            n_out += len(out)
            d_values += [tr.d_value for tr in out]

            g = gt.get(t)
            best = 0.0
            if g is not None and g.visible:
                n_visible += 1
                o = g.objects[0]
                for tr in out:
                    best = max(best, box_iou(tr.box, (o.x1, o.y1, o.x2, o.y2)))
                if best > C.IOU_THRESH:
                    n_hit += 1
            hits.append(best > C.IOU_THRESH)
            rows.append({"frame": t, "n_cand": len(boxes4), "n_out": len(out),
                         "best_iou": round(best, 4)})
            for kk in [b for b in buf if b < t - 2 * k - 2]:
                del buf[kk]

    longest = cur = 0
    for h in hits:
        cur = 0 if h else cur + 1
        longest = max(longest, cur)

    return {
        "video": video, "d_mode": d_mode, "d_min": d_min, "span": span,
        "frames": len(rows), "gt_visible": n_visible, "hit": n_hit,
        "recall": n_hit / n_visible if n_visible else 0.0,
        "cand_per_frame": n_cand / len(rows) if rows else 0,
        "out_per_frame": n_out / len(rows) if rows else 0,
        "longest_miss_streak": longest,
        "d_median": float(np.median(d_values)) if d_values else 0.0,
        "d_max": float(np.max(d_values)) if d_values else 0.0,
        "mean_ms": float(np.mean(t_ms)) if t_ms else 0.0,
        "per_frame": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom76")
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--mode", default="3frame_or",
                    choices=["2frame", "3frame_and", "3frame_or"])
    ap.add_argument("--d-mode", default="enclosing", choices=["enclosing", "self"])
    ap.add_argument("--d-min", type=float, default=1.0)
    ap.add_argument("--span", type=int, default=3)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-age", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    try:
        mfe_cfg = MFEConfig(k=args.k, mode=args.mode)
        r = run(args.video, mfe_cfg, args.start, args.frames, args.d_mode,
                args.d_min, args.span, args.iou, args.max_age)
        print(f"{r['d_mode']:10s} d_min={r['d_min']:<6} 召回={r['recall']*100:6.2f}%  "
              f"输出/帧={r['out_per_frame']:6.2f}  最长漏检={r['longest_miss_streak']:3d}  "
              f"D中位={r['d_median']:.3f} D最大={r['d_max']:.3f}  {r['mean_ms']:.0f}ms")
        out = args.out or str(C.LOGS / f"tf_{args.video}_{args.start}_{args.frames}_{args.d_mode}.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"已写入 {out}")
        return C.EXIT_OK
    except C.MgmdError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
