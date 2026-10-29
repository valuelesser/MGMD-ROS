"""scripts/run_testset_onnx.py — 用 ONNX 后端跑测试集（SPEC-ROS §13.7 验证门槛）。

目标：ONNX 路径的 F1 必须与 torch 路径的 0.6752 完全一致。

    source scripts/env_setup_ros38.sh
    $PY38 scripts/run_testset_onnx.py --out outputs/logs/ros_onnx_testset

与 run_testset.py 的差别：LAC/LAD 走 mgmd_onnx，其余（MFE/TF/评测）完全复用。
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
from mgmd.lac import crop_box
from mgmd.metrics import Detection, evaluate, summary
from mgmd.mfe import MFEConfig, enhance
from mgmd.tf import SortTracker
from mgmd_onnx import LacOnnx, LadOnnx

_DUMP: dict = {}


def run_video(video: str, cfg: MFEConfig, lac, lad, *,
              d_mode: str, d_min: float, lac_thresh: float, lad_conf: float,
              iou_thresh: float, max_age: int, span: int, min_hits: int,
              limit: int | None):
    gt = load_video_gt(video)
    end = min(max(gt), limit) if limit else max(gt)
    k = cfg.k
    tracker = SortTracker(iou_thresh=iou_thresh, max_age=max_age,
                          min_hits=min_hits, span=span, d_mode=d_mode, d_min=d_min)
    dets = {}
    n_frames = 0
    t0 = time.perf_counter()

    buf = {}
    with VideoReader(video) as vr:
        for idx, fr in vr.iter_frames(start=1, limit=end + k):
            buf[idx] = fr
            t = idx - k
            if t < 1 or (t - k) not in buf:
                continue
            if cfg.needs_next and (t + k) not in buf:
                continue
            n_frames += 1
            res = enhance(buf[t - k], buf[t],
                          buf.get(t + k) if cfg.needs_next else None, cfg, t)
            pool = [((float(x), float(y), float(x + w), float(y + h)), float(w * h))
                    for (x, y, w, h) in res.boxes]

            tracks = tracker.update([b for b, _ in pool], t, scores=[s for _, s in pool])
            pool = [(tr.box, tr.score) for tr in tracks]

            if pool:
                crops, keep = [], []
                for b, s in pool:
                    c = crop_box(buf[t], b)
                    if c is not None:
                        crops.append(c)
                        keep.append((b, s))
                if crops:
                    probs = lac.mav_prob_batch(crops)
                    pool = [(b, p) for (b, _s), p in zip(keep, probs) if p >= lac_thresh]
                else:
                    pool = []

            if pool:
                out = []
                for b, _s in pool:
                    rb = lad.refine(buf[t], ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
                    if rb is not None and rb.conf >= lad_conf:
                        out.append((rb.box, rb.conf))
                pool = out

            if pool:
                dets[t] = [Detection(b[0], b[1], b[2], b[3], conf=s) for b, s in pool]
            for kk in [b for b in buf if b < t - 2 * k - 2]:
                del buf[kk]

    dt = time.perf_counter() - t0
    gt_scoped = {t: gt[t] for t in range(1, end + 1) if t in gt}
    m = summary(evaluate(dets, gt_scoped))
    if _DUMP.get("dir"):
        import pickle
        Path(_DUMP["dir"]).mkdir(parents=True, exist_ok=True)
        (Path(_DUMP["dir"]) / f"{video}.pkl").write_bytes(
            pickle.dumps({t: [(d.x1, d.y1, d.x2, d.y2, d.conf) for d in v]
                          for t, v in dets.items()}))
    return dict(video=video, frames=n_frames, elapsed_s=dt,
                ms_per_frame=dt / max(1, n_frames) * 1000, **m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/logs/ros_onnx_testset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--video", default=None, help="只跑指定视频")
    ap.add_argument("--dump-dir", default=None, help="dump 原始检测用于对拍")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-min", type=float, default=0.02)
    ap.add_argument("--d-mode", default="self")
    ap.add_argument("--lac-thresh", type=float, default=0.5)
    ap.add_argument("--lad-conf", type=float, default=0.5)
    args = ap.parse_args()

    _DUMP["dir"] = args.dump_dir
    cfg = MFEConfig()
    cfg.apply_recommended()
    cfg.validate()

    print(f"[onnx] device={args.device} d_min={args.d_min}")
    lac = LacOnnx(device=args.device)
    lad = LadOnnx(device=args.device)
    print(f"[onnx] providers LAC={lac.session.get_providers()[0]} "
          f"LAD={lad.session.get_providers()[0]}")

    videos = ([args.video] if args.video
              else [C.video_id_to_name(v) for v in C.TEST_IDS])
    per = []
    t_all = time.perf_counter()
    for i, v in enumerate(videos, 1):
        r = run_video(v, cfg, lac, lad, d_mode=args.d_mode, d_min=args.d_min,
                      lac_thresh=args.lac_thresh, lad_conf=args.lad_conf,
                      iou_thresh=0.3, max_age=3, span=3, min_hits=1, limit=args.limit)
        per.append(r)
        print(f"[{i}/{len(videos)}] {v:12s} P={r['precision']:.4f} "
              f"R={r['recall']:.4f} F1={r['f1']:.4f} AP={r['ap_allpoint']:.4f} "
              f"{r['ms_per_frame']:.0f}ms", flush=True)

    tp = sum(r["tp"] for r in per)
    fp = sum(r["fp"] for r in per)
    fn = sum(r["fn"] for r in per)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    AP = float(np.mean([r["ap_allpoint"] for r in per]))
    nf = sum(r["frames"] for r in per)
    el = time.perf_counter() - t_all

    out = dict(backend="onnx", device=args.device, d_min=args.d_min,
               n_videos=len(per), frames=nf, precision=P, recall=R, f1=F1,
               ap_allpoint=AP, elapsed_min=el / 60, per_video=per)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"ONNX 全链路: P={P:.4f} R={R:.4f} F1={F1:.4f} AP={AP:.4f}")
    print(f"torch 基准: P=0.8970 R=0.5413 F1=0.6752 AP=0.5239")
    d = abs(F1 - 0.6752)
    print(f"F1 差值: {d:.6f} — {'✅ 一致' if d < 1e-4 else '❌ 不一致'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
