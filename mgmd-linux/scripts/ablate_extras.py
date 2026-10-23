"""scripts/ablate_extras.py — 四项额外约束的逐项消融（SPEC D-04~D-07）。

四项约束（论文均未提及，旧工程默认全开）：
    M4  preprocess_blur   高斯模糊 11×11
    M5  disp_filter       关键点位移过滤（>0.25·max(w,h) 丢弃）
    M11 max_area          连通域面积上限 3000
    M12 aspect_filter     长宽比过滤（0.2~5）

输出：MFE 候选层的召回 / 精确率上界 / 候选数 / 连通域数。
MFE 层是最干净的观察点（不受 LAC/LAD 干扰）。

用法：
    python3 scripts/ablate_extras.py --video phantom09
    python3 scripts/ablate_extras.py --all-small      # Small 组全部 5 个
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

import numpy as np

from mgmd import config as C
from mgmd.annotations import load_video_gt
from mgmd.io_video import VideoReader
from mgmd.metrics import box_iou
from mgmd.mfe import MFEConfig, enhance


CONFIGS = {
    "baseline(全关)":   dict(),
    "M4 blur":          dict(preprocess_blur=11),
    "M5 disp":          dict(disp_filter=True),
    "M11 max_area":     dict(max_area=3000),
    "M12 aspect":       dict(aspect_filter=True),
    "全开(旧工程)":     dict(preprocess_blur=11, disp_filter=True,
                             max_area=3000, aspect_filter=True),
}


def run_config(video: str, extra: dict, start: int, frames: int,
               base: MFEConfig) -> dict:
    cfg = MFEConfig(k=base.k, mode=base.mode, threshold=base.threshold, **extra)
    gt = load_video_gt(video)
    n_vis = n_hit = n_cand = n_frame = 0
    hits = []
    t0 = time.perf_counter()

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
            res = enhance(buf[t - k], buf[t],
                          buf.get(t + k) if cfg.needs_next else None, cfg, t)
            n_frame += 1
            n_cand += len(res.boxes)
            g = gt.get(t)
            best = 0.0
            if g is not None and g.visible:
                n_vis += 1
                boxes = [(x, y, x + w, y + h) for (x, y, w, h) in res.boxes]
                for gg in g.objects:
                    for b in boxes:
                        v = box_iou(b, (gg.x1, gg.y1, gg.x2, gg.y2))
                        if v > best:
                            best = v
                if best > C.IOU_THRESH:
                    n_hit += 1
            hits.append(best > C.IOU_THRESH)
            for kk in [b for b in buf if b < t - 2 * k - 2]:
                del buf[kk]

    dt = time.perf_counter() - t0
    longest = cur = 0
    for h in hits:
        cur = 0 if h else cur + 1
        longest = max(longest, cur)

    return {
        "frames": n_frame,
        "visible": n_vis,
        "hit": n_hit,
        "recall": n_hit / n_vis if n_vis else 0.0,
        "cand_per_frame": n_cand / max(1, n_frame),
        "longest_miss": longest,
        "ms_per_frame": dt / max(1, n_frame) * 1000,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--group", default=None,
                    choices=["Ordinary", "Complex", "Small"])
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--frames", type=int, default=0, help="0=全视频")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--mode", default="3frame_or")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.video:
        videos = [args.video]
    elif args.group:
        videos = [C.video_id_to_name(v) for v in C.CONDITION_GROUPS[args.group]]
    else:
        videos = [C.video_id_to_name(v) for v in C.TEST_IDS]

    base = MFEConfig(k=args.k, mode=args.mode)
    all_rows = {}

    for v in videos:
        gt = load_video_gt(v)
        last = max(gt)
        frames = args.frames if args.frames else (last - args.start + 1)
        print(f"\n=== {v} ({frames} 帧) ===", flush=True)
        print(f"{'config':<18}{'recall':>8}{'cand/f':>8}{'miss':>6}{'ms/f':>7}", flush=True)
        rows = {}
        for name, extra in CONFIGS.items():
            r = run_config(v, extra, args.start, frames, base)
            rows[name] = r
            print(f"{name:<18}{r['recall']*100:>7.1f}%{r['cand_per_frame']:>8.1f}"
                  f"{r['longest_miss']:>6}{r['ms_per_frame']:>7.0f}", flush=True)
        all_rows[v] = rows

    out = args.out or str(C.LOGS / "ablate_extras.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "k": args.k, "mfe_mode": args.mode,
        "configs": list(CONFIGS), "results": all_rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n已写入 {out}", flush=True)
    return C.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
