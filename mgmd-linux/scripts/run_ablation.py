"""scripts/run_ablation.py — 模块消融（SPEC F1 / T06）。

支持任意模块子集，对齐论文 Table 3 的 6 种组合：
    mfe / mfe,tf / mfe,lac / mfe,tf,lac / mfe,lac,lad / mfe,tf,lac,lad

用法：
    python3 scripts/run_ablation.py --video phantom76 --frames 120
    python3 scripts/run_ablation.py --video phantom76 --frames 120 --modules mfe,tf,lac
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mgmd import config as C
from mgmd.annotations import load_video_gt
from mgmd.metrics import Detection, evaluate, summary
from mgmd.mfe import MFEConfig
from mgmd.pipeline import parse_modules, run, write_jsonl

# 论文 Table 3 的 6 种组合（LAD 已接入，SPEC T09）
TABLE3_COMBOS = [
    "mfe",
    "mfe,tf",
    "mfe,lac",
    "mfe,tf,lac",
    "mfe,lac,lad",
    "mfe,tf,lac,lad",
]


def eval_outputs(outs, gt: dict) -> dict:
    """评测，**只在实际跑过的帧上算**（关键：否则分母含未处理的 GT 帧）。

    这是 bug 修复记录（SPEC §7）：此前把整段视频的 GT 全算进召回分母，
    导致召回被低估 17 倍（1049 帧 GT vs 60 帧实跑）。
    """
    dets_by_frame = {}
    for o in outs:
        dets_by_frame[o.frame] = [
            Detection(b[0], b[1], b[2], b[3], conf=s)
            for b, s in zip(o.boxes, o.scores)
        ]
    frames_run = set(dets_by_frame)
    gt_scoped = {f: g for f, g in gt.items() if f in frames_run}
    res = evaluate(dets_by_frame, gt_scoped)
    return summary(res)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom76")
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--mode", default="3frame_or",
                    choices=["2frame", "3frame_and", "3frame_or"])
    ap.add_argument("--modules", default=None, help="逗号分隔；省略则跑全部组合")
    ap.add_argument("--d-mode", default="enclosing", choices=["enclosing", "self"])
    ap.add_argument("--d-min", type=float, default=1.0)
    ap.add_argument("--lac-thresh", type=float, default=0.5)
    ap.add_argument("--lad-conf", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    try:
        combos = [args.modules] if args.modules else TABLE3_COMBOS
        gt = load_video_gt(args.video)
        rows = []
        print(f"{'modules':<18}{'P':>8}{'R':>8}{'F1':>8}{'AP11':>8}{'APall':>8}{'det/f':>8}")
        for combo in combos:
            mods = parse_modules(combo)
            cfg = MFEConfig(k=args.k, mode=args.mode)
            outs = run(args.video, cfg, mods, start=args.start, limit=args.frames,
                       d_mode=args.d_mode, d_min=args.d_min, lac_thresh=args.lac_thresh,
                       lad_conf=args.lad_conf)
            s = eval_outputs(outs, gt)
            n_det = sum(len(o.boxes) for o in outs) / max(1, len(outs))
            rows.append({"modules": combo, **s, "det_per_frame": n_det})
            print(f"{combo:<18}{s['precision']:>8.3f}{s['recall']:>8.3f}"
                  f"{s['f1']:>8.3f}{s['ap_11point']:>8.3f}{s['ap_allpoint']:>8.3f}{n_det:>8.2f}")

        out = args.out or str(C.LOGS / f"ablation_{args.video}_{args.start}_{args.frames}.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({
            "video": args.video, "start": args.start, "frames": args.frames,
            "k": args.k, "mfe_mode": args.mode,
            "d_mode": args.d_mode, "d_min": args.d_min,
            "lac_thresh": args.lac_thresh,
            "rows": rows,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n已写入 {out}")
        return C.EXIT_OK
    except C.MgmdError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
