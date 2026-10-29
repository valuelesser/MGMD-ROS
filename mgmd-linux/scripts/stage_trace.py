"""scripts/stage_trace.py — 逐级追踪 MFE→TF→LAC→LAD 每步的 P/R/F1（SPEC §10.11）。

回答"在哪一步崩掉的"：在每个模块的**出口**都算一遍指标，观察指标如何变化。

与 `pipeline.run` 的区别：
    - 不提前 return，每级都留快照
    - 各级用同一套评测（IoU≥0.25 + 匈牙利一对一）
    - 额外记录每级的"帧覆盖率"（有框的帧占比）

用法：
    source scripts/env_setup.sh
    $PY scripts/stage_trace.py --video phantom30 --frames 400
    $PY scripts/stage_trace.py --video phantom30 --frames 400 --d-min 0.20
    $PY scripts/stage_trace.py --all          # 15 个测试视频汇总
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
from mgmd.metrics import Detection, evaluate, summary
from mgmd.mfe import MFEConfig, enhance
from mgmd.tf import SortTracker

STAGES = ["MFE", "TF", "LAC", "LAD"]


def trace_one(video: str, cfg: MFEConfig, d_mode: str, d_min: float,
              lac_thresh: float, lad_conf: float, device: str,
              lad_batch: int, max_frames: int | None, iou_thresh: float = 0.3,
              max_age: int = 3, span: int = 3, min_hits: int = 1,
              use_stages: tuple[str, ...] = ("MFE", "TF", "LAC", "LAD")) -> dict:
    gt = load_video_gt(video)
    last = max(gt)
    end = last if max_frames is None else min(last, max_frames)
    k = cfg.k

    from mgmd.lac import LacClassifier, crop_box
    from mgmd.lad import LadDetector
    clf = LacClassifier(device=device) if "LAC" in use_stages else None
    lad = LadDetector(conf_thresh=lad_conf, device=device) if "LAD" in use_stages else None
    tracker = SortTracker(iou_thresh=iou_thresh, max_age=max_age, min_hits=min_hits,
                          span=span, d_mode=d_mode, d_min=d_min) if "TF" in use_stages else None

    # 每级收集 (frame -> [(box, score)])
    acc: dict[str, dict[int, list]] = {s: {} for s in STAGES}
    nframes = 0
    t_start = time.perf_counter()

    buf: dict[int, np.ndarray] = {}
    with VideoReader(video) as vr:
        for idx, fr in vr.iter_frames(start=1, limit=end + k):
            buf[idx] = fr
            t = idx - k
            if t < 1 or (t - k) not in buf:
                continue
            if cfg.needs_next and (t + k) not in buf:
                continue
            nframes += 1

            res = enhance(buf[t - k], buf[t],
                          buf.get(t + k) if cfg.needs_next else None, cfg, t)
            pool = [((float(x), float(y), float(x + w), float(y + h)), float(w * h))
                    for (x, y, w, h) in res.boxes]
            acc["MFE"][t] = list(pool)

            if tracker is not None:
                tracks = tracker.update([b for b, _ in pool], t,
                                        scores=[s for _, s in pool])
                pool = [(tr.box, tr.score) for tr in tracks]
                acc["TF"][t] = list(pool)

            if clf is not None and pool:
                crops, keep = [], []
                for b, s in pool:
                    c = crop_box(buf[t], b)
                    if c is not None:
                        crops.append(c)
                        keep.append((b, s))
                if crops:
                    probs = clf.mav_prob_batch(crops)
                    pool = [(b, p) for (b, _s), p in zip(keep, probs) if p >= lac_thresh]
                else:
                    pool = []
            if clf is not None:
                acc["LAC"][t] = list(pool)

            if lad is not None and pool:
                centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
                           for b, _ in pool]
                rbs = lad.refine_batch(buf[t], centers, batch_size=lad_batch)
                pool = [(rb.box, rb.conf) for rb in rbs
                        if rb is not None and rb.conf >= lad_conf]
            if lad is not None:
                acc["LAD"][t] = list(pool)

            for kk in [b for b in buf if b < t - 2 * k - 2]:
                del buf[kk]

    elapsed = time.perf_counter() - t_start

    # 逐级评测（只评测实际跑到的帧）
    gt_scoped = {t: gt[t] for t in range(1, end + 1) if t in gt}
    rows = []
    for s in STAGES:
        if s not in acc or not acc[s]:
            continue
        dets = {t: [Detection(b[0], b[1], b[2], b[3], conf=sc) for b, sc in v]
                for t, v in acc[s].items() if v}
        # 缺失的帧按空检测计入（否则会高估召回）
        for t in gt_scoped:
            dets.setdefault(t, [])
        m = summary(evaluate(dets, gt_scoped))
        n_box = sum(len(v) for v in acc[s].values())
        nframes_with = sum(1 for v in acc[s].values() if v)
        rows.append(dict(
            stage=s, tp=m["tp"], fp=m["fp"], fn=m["fn"],
            P=m["precision"], R=m["recall"], F1=m["f1"], AP=m["ap_allpoint"],
            boxes=n_box, box_per_frame=n_box / max(1, nframes),
            frames_with_box=nframes_with,
            frame_cover=nframes_with / max(1, nframes),
        ))
    return dict(video=video, frames=nframes, elapsed_s=elapsed,
                ms_per_frame=elapsed / max(1, nframes) * 1000, rows=rows)


def fmt_table(rows: list[dict]) -> str:
    out = [f"{'阶段':>6} {'TP':>7} {'FP':>7} {'FN':>7} {'P':>7} {'R':>7} {'F1':>7} "
           f"{'AP':>7} {'框/帧':>8} {'有框帧%':>8}", "-" * 84]
    for r in rows:
        out.append(f"{r['stage']:>6} {r['tp']:>7} {r['fp']:>7} {r['fn']:>7} "
                   f"{r['P']:>7.3f} {r['R']:>7.3f} {r['F1']:>7.3f} {r['AP']:>7.3f} "
                   f"{r['box_per_frame']:>8.1f} {r['frame_cover']*100:>7.1f}%")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom30")
    ap.add_argument("--all", action="store_true", help="跑全部 15 个测试视频")
    ap.add_argument("--frames", type=int, default=None, help="每视频最多帧数")
    ap.add_argument("--d-mode", default="self")
    ap.add_argument("--d-min", type=float, default=0.4038)
    ap.add_argument("--lac-thresh", type=float, default=0.5)
    ap.add_argument("--lad-conf", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lad-batch", type=int, default=8)
    ap.add_argument("--out", default=None, help="结果 json 路径")
    args = ap.parse_args()

    cfg = MFEConfig()
    cfg.apply_recommended()
    cfg.validate()
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    videos = ([C.video_id_to_name(v) for v in C.TEST_IDS] if args.all else [args.video])
    results = []
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v} ...", flush=True)
        r = trace_one(v, cfg, args.d_mode, args.d_min, args.lac_thresh,
                      args.lad_conf, device, args.lad_batch, args.frames)
        results.append(r)
        print(fmt_table(r["rows"]), flush=True)
        print(f"  {r['frames']} 帧, {r['ms_per_frame']:.1f} ms/帧\n", flush=True)

    if len(results) == 1:
        payload = results[0]
    else:
        # 汇总：把每级所有帧的 TP/FP/FN 相加
        agg = {}
        for s in STAGES:
            tp = sum(next(x for x in r["rows"] if x["stage"] == s)["tp"]
                     for r in results if any(x["stage"] == s for x in r["rows"]))
            fp = sum(next(x for x in r["rows"] if x["stage"] == s)["fp"]
                     for r in results if any(x["stage"] == s for x in r["rows"]))
            fn = sum(next(x for x in r["rows"] if x["stage"] == s)["fn"]
                     for r in results if any(x["stage"] == s for x in r["rows"]))
            ap = np.mean([next(x for x in r["rows"] if x["stage"] == s)["ap_allpoint"]
                          if any(x["stage"] == s for x in r["rows"]) else 0
                          for r in results])
            boxes = sum(next(x for x in r["rows"] if x["stage"] == s)["boxes"]
                        for r in results if any(x["stage"] == s for x in r["rows"]))
            nf = sum(r["frames"] for r in results)
            P = tp / (tp + fp) if tp + fp else 0.0
            R = tp / (tp + fn) if tp + fn else 0.0
            agg[s] = dict(stage=s, tp=tp, fp=fp, fn=fn, P=P, R=R,
                          F1=2 * P * R / (P + R) if P + R else 0.0,
                          AP=float(ap), boxes=boxes, box_per_frame=boxes / max(1, nf),
                          frames_with_box=0, frame_cover=0.0)
        rows = [agg[s] for s in STAGES if s in agg]
        payload = dict(video="ALL(15)", frames=sum(r["frames"] for r in results),
                       elapsed_s=sum(r["elapsed_s"] for r in results),
                       ms_per_frame=sum(r["elapsed_s"] for r in results)
                       / max(1, sum(r["frames"] for r in results)) * 1000,
                       rows=rows, per_video=results)
        print("=" * 84)
        print("汇总（15 视频，TP/FP/FN 相加）")
        print("=" * 84)
        print(fmt_table(rows))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
