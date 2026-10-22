"""scripts/run_full_video.py — 单视频单组合全量评测（SPEC T11 前置）。

比 run_ablation 快：只跑一个模块组合，边跑边打印进度。
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

from mgmd import config as C
from mgmd.annotations import load_video_gt
from mgmd.metrics import Detection, evaluate, summary
from mgmd.mfe import MFEConfig
from mgmd.pipeline import parse_modules, run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom40")
    ap.add_argument("--modules", default="mfe,lac,lad")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--frames", type=int, default=0, help="0=到视频末尾")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--mode", default="3frame_or")
    ap.add_argument("--d-mode", default="self")
    ap.add_argument("--d-min", type=float, default=0.4038)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    gt = load_video_gt(args.video)
    last = max(gt)
    frames = args.frames if args.frames else (last - args.start + 1)
    mods = parse_modules(args.modules)

    print(f"video={args.video} frames={args.start}..{args.start + frames - 1} "
          f"({frames}) mods={args.modules}", flush=True)
    t0 = time.perf_counter()
    cfg = MFEConfig(k=args.k, mode=args.mode)
    outs = run(args.video, cfg, mods, start=args.start, limit=frames,
               d_mode=args.d_mode, d_min=args.d_min, progress=True)
    dt = time.perf_counter() - t0
    print(f"pipeline 完成: {len(outs)} 帧, {dt:.1f}s, "
          f"{dt / max(1, len(outs)) * 1000:.0f} ms/帧", flush=True)

    dets = {o.frame: [Detection(b[0], b[1], b[2], b[3], conf=s)
                      for b, s in zip(o.boxes, o.scores)] for o in outs}
    fr = set(dets)
    gt_scoped = {f: g for f, g in gt.items() if f in fr}
    res = evaluate(dets, gt_scoped)
    s = summary(res)

    print("\n" + "=" * 60, flush=True)
    print(f"P={s['precision']:.4f}  R={s['recall']:.4f}  F1={s['f1']:.4f}  "
          f"APall={s['ap_allpoint']:.4f}  AP11={s['ap_11point']:.4f}", flush=True)
    print(f"TP={s['tp']} FP={s['fp']} FN={s['fn']}", flush=True)
    print(f"检测数分布: {s['det_count_hist']}", flush=True)
    print(f"GT数分布:   {s['gt_count_hist']}", flush=True)
    print(f"按GT数分组: {json.dumps(s['by_gt_count'], ensure_ascii=False)}", flush=True)

    out = args.out or str(C.LOGS / f"full_{args.video}_{args.modules.replace(',', '-')}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "video": args.video, "start": args.start, "frames": frames,
        "modules": args.modules, "k": args.k, "mfe_mode": args.mode,
        "d_mode": args.d_mode, "d_min": args.d_min,
        "elapsed_s": round(dt, 2),
        "ms_per_frame": round(dt / max(1, len(outs)) * 1000, 1),
        "metrics": s,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n已写入 {out}", flush=True)
    return C.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
