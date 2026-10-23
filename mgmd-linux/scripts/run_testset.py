"""scripts/run_testset.py — 全 15 个测试视频评测（SPEC T11b）。

逐视频跑完整 pipeline → 评测 → 汇总。支持 GPU/CPU 自动切换、断点续跑。

用法：
    python3 scripts/run_testset.py                    # 全部 15 个视频
    python3 scripts/run_testset.py --video phantom09  # 单个
    python3 scripts/run_testset.py --device cuda      # 强制 GPU
    python3 scripts/run_testset.py --force            # 重跑已完成的
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
from mgmd.annotations import load_video_gt, video_stats
from mgmd.metrics import Detection, evaluate, summary
from mgmd.mfe import MFEConfig
from mgmd.pipeline import parse_modules, run

RESULTS_DIR = C.LOGS / "testset"


def eval_one(video: str, mods: frozenset, cfg: MFEConfig, device: str,
             lad_batch: int, d_mode: str, d_min: float,
             lac_thresh: float, lad_topk: int = 0, min_hits: int = 1) -> dict:
    gt = load_video_gt(video)
    last = max(gt)
    t0 = time.perf_counter()
    outs = run(video, cfg, mods, start=1, limit=last,
               d_mode=d_mode, d_min=d_min, lac_thresh=lac_thresh,
               device=device, lad_batch=lad_batch, lad_topk=lad_topk,
               min_hits=min_hits)
    dt = time.perf_counter() - t0

    dets = {o.frame: [Detection(b[0], b[1], b[2], b[3], conf=s)
                      for b, s in zip(o.boxes, o.scores)] for o in outs}
    fr = set(dets)
    gt_scoped = {f: g for f, g in gt.items() if f in fr}
    res = evaluate(dets, gt_scoped)
    s = summary(res)

    return {
        "video": video,
        "frames_gt": last,
        "frames_run": len(outs),
        "elapsed_s": round(dt, 1),
        "ms_per_frame": round(dt / max(1, len(outs)) * 1000, 1),
        "metrics": s,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="只跑指定视频")
    ap.add_argument("--modules", default="mfe,lac,lad")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--lad-batch", type=int, default=8)
    ap.add_argument("--lad-topk", type=int, default=0,
                    help="只精修 score 最高前 k 个候选（0=全部）")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--mode", default="3frame_or")
    ap.add_argument("--d-mode", default="self")
    ap.add_argument("--d-min", type=float, default=0.4038)
    ap.add_argument("--lac-thresh", type=float, default=0.5)
    ap.add_argument("--min-hits", type=int, default=1,
                    help="SORT min_hits（论文 1；调大可减少轨迹池膨胀）")
    ap.add_argument("--out-suffix", default="", help="结果子目录后缀")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    mods = parse_modules(args.modules)
    cfg = MFEConfig(k=args.k, mode=args.mode)
    global RESULTS_DIR
    if args.out_suffix:
        RESULTS_DIR = RESULTS_DIR / args.out_suffix
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 解析设备实际值
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[config] device={device} modules={args.modules} lad_batch={args.lad_batch} "
          f"lad_topk={args.lad_topk} lac_thresh={args.lac_thresh} "
          f"d={args.d_mode}:{args.d_min}", flush=True)

    videos = ([args.video] if args.video
              else [C.video_id_to_name(v) for v in C.TEST_IDS])

    rows = []
    total_t0 = time.perf_counter()
    for i, v in enumerate(videos, 1):
        out_file = RESULTS_DIR / f"{v}.json"
        if out_file.exists() and not args.force:
            print(f"[{i}/{len(videos)}] {v} 已存在，跳过（--force 重跑）", flush=True)
            rows.append(json.loads(out_file.read_text(encoding="utf-8")))
            continue

        st = video_stats(v)
        print(f"[{i}/{len(videos)}] {v}  {st['frames_annotated']} 帧  "
              f"目标中位 {st['median_w']:.0f}x{st['median_h']:.0f}px ...", flush=True)
        try:
            r = eval_one(v, mods, cfg, device, args.lad_batch,
                         args.d_mode, args.d_min, args.lac_thresh, args.lad_topk,
                         args.min_hits)
        except Exception as e:
            print(f"    ✗ 失败: {type(e).__name__}: {e}", flush=True)
            continue

        m = r["metrics"]
        print(f"    P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
              f"AP={m['ap_allpoint']:.4f}  {r['ms_per_frame']:.0f}ms/帧  "
              f"({r['elapsed_s']:.0f}s)", flush=True)
        out_file.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        rows.append(r)

    total_dt = time.perf_counter() - total_t0

    if not rows:
        print("没有成功的结果", file=sys.stderr)
        return C.EXIT_RUNTIME

    # 汇总（按帧加权，不是简单平均）
    tp = sum(r["metrics"]["tp"] for r in rows)
    fp = sum(r["metrics"]["fp"] for r in rows)
    fn = sum(r["metrics"]["fn"] for r in rows)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
    ap = sum(r["metrics"]["ap_allpoint"] for r in rows) / len(rows)
    frames = sum(r["frames_run"] for r in rows)

    print("\n" + "=" * 72, flush=True)
    print(f"{'video':<12}{'frames':>8}{'P':>8}{'R':>8}{'F1':>8}{'AP':>8}{'ms/f':>8}", flush=True)
    print("-" * 72, flush=True)
    for r in sorted(rows, key=lambda x: x["video"]):
        m = r["metrics"]
        print(f"{r['video']:<12}{r['frames_run']:>8}{m['precision']:>8.3f}"
              f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['ap_allpoint']:>8.3f}"
              f"{r['ms_per_frame']:>8.0f}", flush=True)
    print("-" * 72, flush=True)
    print(f"{'汇总(帧级)':<12}{frames:>8}{p:>8.3f}{rc:>8.3f}{f1:>8.3f}{ap:>8.3f}", flush=True)
    print(f"\n总耗时 {total_dt/60:.1f} 分钟，共 {frames} 帧", flush=True)
    print(f"\n论文 Table 1 MGMD: P=0.84 R=0.59 F1=0.69 AP=0.55", flush=True)
    print(f"本工程(测试集):    P={p:.2f} R={rc:.2f} F1={f1:.2f} AP={ap:.2f}", flush=True)

    summary_file = RESULTS_DIR / "_summary.json"
    summary_file.write_text(json.dumps({
        "device": device, "modules": args.modules,
        "k": args.k, "mfe_mode": args.mode,
        "d_mode": args.d_mode, "d_min": args.d_min,
        "lac_thresh": args.lac_thresh, "lad_batch": args.lad_batch,
        "lad_topk": args.lad_topk,
        "n_videos": len(rows), "frames": frames,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": rc, "f1": f1, "ap_allpoint": ap,
        "elapsed_min": round(total_dt / 60, 1),
        "per_video": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n汇总已写入 {summary_file}", flush=True)
    return C.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
