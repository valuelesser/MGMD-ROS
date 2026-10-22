"""scripts/run_mfe.py — 在真实视频上跑 MFE 并统计候选召回（SPEC T04 / G2）。

用法：
    python3 scripts/run_mfe.py --video phantom76 --start 100 --frames 120
    python3 scripts/run_mfe.py --video phantom76 --frames 120 --mode 2frame
    python3 scripts/run_mfe.py --video phantom63 --frames 120 --compare-modes

召回定义：GT 框与任一候选框 IoU > 0.25 即算该帧召回成功。
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


def run_video(video: str, cfg: MFEConfig, start: int, limit: int | None,
              save_dir: Path | None = None) -> dict:
    gt = load_video_gt(video)
    results = []
    n_gt_visible = 0
    n_hit = 0
    n_cand_total = 0
    n_empty = 0
    hits = []          # 逐帧是否命中（供连续漏检统计）
    timings = []

    with VideoReader(video) as vr:
        buf = {}
        k = cfg.k
        need = k if cfg.needs_next else 0
        # 顺序读，保留环形缓冲
        for idx, frame in vr.iter_frames(start=max(1, start), limit=(limit or 0) + 2 * k + 1 if limit else None):
            buf[idx] = frame
            t = idx - k
            if limit and idx > start + limit - 1 + need:
                break
            if t < start or t < 1:
                continue
            if t - k not in buf:
                continue
            cur = buf[t]
            prev = buf.get(t - k)
            nxt = buf.get(t + k) if cfg.needs_next else None

            t0 = time.perf_counter()
            res = enhance(prev, cur, nxt, cfg, frame_index=t)
            timings.append((time.perf_counter() - t0) * 1000.0)

            cands = res.boxes
            n_cand_total += len(cands)

            g = gt.get(t)
            if g is not None and g.visible:
                n_gt_visible += 1
                best = 0.0
                for b in cands:
                    x, y, w, h = b
                    iou = box_iou((x, y, x + w, y + h), (g.objects[0].x1, g.objects[0].y1,
                                                         g.objects[0].x2, g.objects[0].y2))
                    best = max(best, iou)
                hit = best > C.IOU_THRESH
                hits.append(hit)
                if hit:
                    n_hit += 1
            else:
                hits.append(False)
            if not cands:
                n_empty += 1
            results.append({"frame": t, "n_cand": len(cands)})

            # 清理过老缓冲
            for key in [k2 for k2 in buf if k2 < t - 2 * k - 2]:
                del buf[key]

    # 连续漏检最长长度
    longest_miss = 0
    cur_miss = 0
    for h in hits:
        if h:
            cur_miss = 0
        else:
            cur_miss += 1
            longest_miss = max(longest_miss, cur_miss)

    n_frames = len(hits)
    return {
        "video": video,
        "mode": cfg.mode,
        "k": cfg.k,
        "frames": n_frames,
        "gt_visible": n_gt_visible,
        "hit": n_hit,
        "recall": (n_hit / n_gt_visible) if n_gt_visible else 0.0,
        "cand_per_frame": (n_cand_total / n_frames) if n_frames else 0.0,
        "empty_frames": n_empty,
        "longest_miss_streak": longest_miss,
        "mean_ms": float(np.mean(timings)) if timings else 0.0,
        "per_frame": results,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MFE 真实视频候选召回")
    ap.add_argument("--video", default="phantom76")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--mode", default="3frame_or",
                    choices=["2frame", "3frame_and", "3frame_or"])
    ap.add_argument("--compare-modes", action="store_true", help="三种模式并排对比")
    ap.add_argument("--extras", action="store_true",
                    help="开启四项额外约束（默认全关，SPEC D-04~D-07）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    try:
        modes = ["2frame", "3frame_and", "3frame_or"] if args.compare_modes else [args.mode]
        rows = []
        for m in modes:
            cfg = MFEConfig(k=args.k, threshold=args.threshold, mode=m)
            if args.extras:
                cfg.preprocess_blur = 11
                cfg.disp_filter = True
                cfg.max_area = 3000
                cfg.aspect_filter = True
            r = run_video(args.video, cfg, args.start, args.frames)
            rows.append(r)
            print(f"{m:12s} k={args.k} 召回={r['recall']*100:6.2f}%  "
                  f"候选/帧={r['cand_per_frame']:7.1f}  空帧={r['empty_frames']:4d}  "
                  f"最长连续漏检={r['longest_miss_streak']:3d}  {r['mean_ms']:6.1f} ms/帧")

        out = args.out or str(C.LOGS / f"mfe_{args.video}_{args.start}_{args.frames}.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        print(f"\n结果已写入 {out}")
        return C.EXIT_OK
    except C.MgmdError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
