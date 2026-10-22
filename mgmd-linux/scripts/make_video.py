"""scripts/make_video.py — 逐帧可视化视频（SPEC §3）。

输出一个 mp4：左边全图（GT 绿框 + 检测红框），右边目标附近放大（看清像素级目标），
底部叠加每帧指标。用来看"框到底标在哪、什么时候丢、丢的时候发生了什么"。

三种模式：
    --mode pipeline   走完整 pipeline（默认 mfe,lac,lad）
    --mode mfe        只看 MFE 候选
    --mode compare    左右并排对比两种模块组合

用法：
    bash scripts/make_video.sh                      # 一键（含环境）
    python3 scripts/make_video.py --video phantom76 --start 300 --frames 200
    python3 scripts/make_video.py --video phantom76 --start 370 --frames 60 --modules mfe,lac,lad
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from mgmd import config as C
from mgmd import viz
from mgmd.annotations import load_video_gt
from mgmd.io_video import VideoReader
from mgmd.metrics import box_iou
from mgmd.mfe import MFEConfig
from mgmd.pipeline import parse_modules, run

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)


def draw_hud(canvas, t, gt_boxes, det_boxes, best_iou, hit, info: str = "", n_cand: int = 0):
    """在画布顶部画信息条。"""
    h, w = canvas.shape[:2]
    bar = np.zeros((76, w, 3), np.uint8)
    bar[:] = (28, 28, 28)

    status = "HIT" if hit else "MISS"
    color = GREEN if hit else RED
    cv2.putText(bar, f"frame {t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
    cv2.putText(bar, status, (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(bar, f"IoU={best_iou:.3f}", (280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)
    cv2.putText(bar, f"GT={len(gt_boxes)}  CAND={n_cand}  DET={len(det_boxes)}", (470, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2, cv2.LINE_AA)
    # 图例
    cv2.rectangle(bar, (10, 46), (26, 60), GREEN, -1)
    cv2.putText(bar, "GT", (32, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(bar, (80, 46), (96, 60), RED, -1)
    cv2.putText(bar, "detection", (102, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    if info:
        cv2.putText(bar, info, (240, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    return np.vstack([bar, canvas])


def make_zoom(frame, gt_boxes, det_boxes, size=480, patch=100, cand_boxes=None):
    """目标附近放大视图。返回 (图像, best_iou)。"""
    best_iou = 0.0
    center = None
    if gt_boxes:
        for d in det_boxes:
            for g in gt_boxes:
                v = box_iou(d, g)
                if v > best_iou:
                    best_iou = v
        g = gt_boxes[0]
        center = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
    elif det_boxes:
        d = det_boxes[0]
        center = ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2)
    else:
        center = (frame.shape[1] / 2, frame.shape[0] / 2)

    h, w = frame.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    x1 = max(0, min(cx - patch // 2, max(0, w - patch)))
    y1 = max(0, min(cy - patch // 2, max(0, h - patch)))
    x2, y2 = min(w, x1 + patch), min(h, y1 + patch)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8), 0.0

    scale = size / float(max(crop.shape[1], crop.shape[0]))
    out = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                     interpolation=cv2.INTER_NEAREST)

    def m(b):
        a = (b[0] - x1) * scale
        bb = (b[1] - y1) * scale
        c = (b[2] - x1) * scale
        d = (b[3] - y1) * scale
        return (a, bb, max(c, a + 4), max(d, bb + 4))

    for b in (cand_boxes or []):
        if b[2] > x1 and b[0] < x2 and b[3] > y1 and b[1] < y2:
            out = viz.draw_boxes(out, [m(b)], YELLOW, 1, None)
    for b in gt_boxes:
        if b[2] > x1 and b[0] < x2 and b[3] > y1 and b[1] < y2:
            out = viz.draw_boxes(out, [m(b)], GREEN, 2, None)
    for b in det_boxes:
        if b[2] > x1 and b[0] < x2 and b[3] > y1 and b[1] < y2:
            out = viz.draw_boxes(out, [m(b)], RED, 2, None)

    cv2.putText(out, f"zoom x{int(round(scale))}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)
    return out, best_iou


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom76")
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--modules", default="mfe,lac,lad")
    ap.add_argument("--mode", default="3frame_or",
                    choices=["2frame", "3frame_and", "3frame_or"])
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--d-mode", default="self", choices=["enclosing", "self"])
    ap.add_argument("--d-min", type=float, default=0.4038)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--show-candidates", action="store_true", default=True,
                    help="同时显示 MFE 原始候选（黄细框）")
    ap.add_argument("--no-candidates", dest="show_candidates", action="store_false")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    mods = parse_modules(args.modules)
    out_path = Path(args.out) if args.out else (
        C.FIGURES / f"{args.video}_{args.start}_{args.frames}_{args.modules.replace(',', '-')}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"运行 pipeline: video={args.video} frames={args.start}..{args.start+args.frames-1} "
          f"modules={args.modules}")
    cfg = MFEConfig(k=args.k, mode=args.mode)
    outs = run(args.video, cfg, mods, start=args.start, limit=args.frames,
               d_mode=args.d_mode, d_min=args.d_min)
    by_frame = {o.frame: o for o in outs}
    print(f"pipeline 完成，{len(outs)} 帧有输出")

    # 额外跑一次纯 MFE，用于显示原始候选（黄框），便于看清各级筛掉了什么
    show_cand = args.show_candidates
    cand_by_frame = {}
    if show_cand:
        print("额外运行纯 MFE 以显示原始候选...")
        mfe_outs = run(args.video, cfg, frozenset({"mfe"}),
                       start=args.start, limit=args.frames)
        cand_by_frame = {o.frame: o.boxes for o in mfe_outs}
        print(f"  原始候选帧数 {len(cand_by_frame)}")

    gt = load_video_gt(args.video)

    # 全图缩放 + 放大图，并排
    full_w, full_h = 960, 540
    zoom_size = 480
    canvas_w = full_w + zoom_size
    canvas_h = full_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, args.fps, (canvas_w, canvas_h + 76))
    if not vw.isOpened():
        print(f"无法创建视频: {out_path}", file=sys.stderr)
        return C.EXIT_RUNTIME

    n_hit = n_vis = 0
    with VideoReader(args.video) as vr:
        for idx, frame in vr.iter_frames(start=args.start, limit=args.frames):
            t = idx
            g = gt.get(t)
            gt_boxes = [(o.x1, o.y1, o.x2, o.y2) for o in g.objects] if g else []
            o = by_frame.get(t)
            det_boxes = o.boxes if o else []

            # 全图
            full = cv2.resize(frame, (full_w, full_h), interpolation=cv2.INTER_AREA)
            sx, sy = full_w / frame.shape[1], full_h / frame.shape[0]

            def sc(b):
                return (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)

            cand_boxes = cand_by_frame.get(t, [])
            for b in cand_boxes:
                full = viz.draw_boxes(full, [sc(b)], YELLOW, 1, None)
            for b in gt_boxes:
                full = viz.draw_boxes(full, [sc(b)], GREEN, 2, None)
            for b in det_boxes:
                full = viz.draw_boxes(full, [sc(b)], RED, 2, None)

            zoom, best_iou = make_zoom(frame, gt_boxes, det_boxes, zoom_size,
                                       cand_boxes=cand_boxes)
            hit = best_iou > C.IOU_THRESH
            if gt_boxes:
                n_vis += 1
                if hit:
                    n_hit += 1

            # 补齐高度
            if zoom.shape[0] < canvas_h:
                zoom = cv2.copyMakeBorder(zoom, 0, canvas_h - zoom.shape[0], 0, 0,
                                          cv2.BORDER_CONSTANT, value=(20, 20, 20))
            combo = np.hstack([full, zoom[:canvas_h]])

            info = f"{args.modules}  k={args.k}  d={args.d_mode}:{args.d_min}"
            combo = draw_hud(combo, t, gt_boxes, det_boxes, best_iou, hit, info,
                             n_cand=len(cand_boxes))
            vw.write(combo)

    vw.release()

    recall = n_hit / n_vis if n_vis else 0.0
    print(f"\n视频已写出: {out_path}")
    print(f"大小: {out_path.stat().st_size/1e6:.1f} MB")
    print(f"该段命中: {n_hit}/{n_vis} = 召回 {recall*100:.1f}%")
    return C.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
