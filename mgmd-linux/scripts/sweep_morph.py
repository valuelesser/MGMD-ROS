"""scripts/sweep_morph.py — MFE 形态学参数扫描（SPEC T06 / G2）。

目标：解决 phantom76 上"框合并"导致的连续漏检（帧 350/360/363/392/398/401）。
扫描维度：close_iters、morph_size、min_area、max_area（M11）
评估：召回 + 最长连续漏检 + 候选/帧

用法：
    python3 scripts/sweep_morph.py --video phantom76 --start 300 --frames 120
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mgmd import config as C
from mgmd.annotations import load_video_gt
from mgmd.io_video import VideoReader
from mgmd.metrics import box_iou
from mgmd.mfe import MFEConfig, enhance


def eval_config(video: str, cfg: MFEConfig, start: int, frames: int) -> dict:
    """跑一段视频，返回召回/候选/连续漏检。"""
    gt = load_video_gt(video)
    n_vis = n_hit = n_cand = 0
    hits = []
    with VideoReader(video) as vr:
        buf = {}
        k = cfg.k
        for idx, fr in vr.iter_frames(start=start, limit=frames + 2 * k + 2):
            buf[idx] = fr
            t = idx - k
            if t < start or (t - k) not in buf:
                continue
            if cfg.needs_next and (t + k) not in buf:
                continue
            res = enhance(buf[t - k], buf[t], buf.get(t + k) if cfg.needs_next else None, cfg, t)
            n_cand += len(res.boxes)
            g = gt.get(t)
            best = 0.0
            if g is not None and g.visible:
                n_vis += 1
                o = g.objects[0]
                for (x, y, w, h) in res.boxes:
                    best = max(best, box_iou((x, y, x + w, y + h),
                                             (o.x1, o.y1, o.x2, o.y2)))
                if best > C.IOU_THRESH:
                    n_hit += 1
            hits.append(best > C.IOU_THRESH)
            for kk in [b for b in buf if b < t - 2 * k - 2]:
                del buf[kk]

    longest = cur = 0
    for h in hits:
        cur = 0 if h else cur + 1
        longest = max(longest, cur)
    return {
        "recall": n_hit / n_vis if n_vis else 0.0,
        "cand_per_frame": n_cand / len(hits) if hits else 0.0,
        "longest_miss": longest,
        "frames": len(hits),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom76")
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    try:
        grid = []
        for close_iters in (1, 2, 3):
            for morph_size in (3, 5):
                for max_area in (0, 3000):
                    grid.append(dict(close_iters=close_iters, morph_size=morph_size,
                                     max_area=max_area))

        rows = []
        print(f"{'close':>6}{'morth':>6}{'maxarea':>9}{'recall':>9}{'cand/f':>9}{'miss':>6}")
        for p in grid:
            cfg = MFEConfig(k=args.k, mode="3frame_or", **p)
            r = eval_config(args.video, cfg, args.start, args.frames)
            rows.append({**p, **r})
            print(f"{p['close_iters']:>6}{p['morph_size']:>6}{p['max_area']:>9}"
                  f"{r['recall']*100:>8.2f}%{r['cand_per_frame']:>9.1f}{r['longest_miss']:>6}")

        # 按 (召回降序, 漏检升序) 排序推荐
        best = sorted(rows, key=lambda r: (-r["recall"], r["longest_miss"]))[:5]
        print("\n最佳 5 组（按召回优先）:")
        for r in best:
            print(f"  close={r['close_iters']} morph={r['morph_size']} max_area={r['max_area']}"
                  f" → 召回 {r['recall']*100:.2f}% 漏检 {r['longest_miss']}")

        out = args.out or str(C.LOGS / f"sweep_morph_{args.video}_{args.start}_{args.frames}.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"video": args.video, "rows": rows, "best": best},
                                        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n已写入 {out}")
        return C.EXIT_OK
    except C.MgmdError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
