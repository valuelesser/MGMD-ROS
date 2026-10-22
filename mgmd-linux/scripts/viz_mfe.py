"""scripts/viz_mfe.py — MFE 逐阶段可视化（SPEC §3）。

ARD-MAV 目标只有 10–20 px，全图尺寸下肉眼不可见，因此每个阶段都配一个
以 GT 为中心的放大视图（默认 x8）。这是判断"目标是否产生信号"的唯一方式。

输出布局（3 行 × 2 列）：全图 / 放大，逐阶段对照
    输入+GT          | 放大：输入 + GT(绿) + 最佳候选(红)
    运动掩膜 E_t      | 放大：运动掩膜
    形态学后掩膜      | 放大：形态学后掩膜

用法：
    python3 scripts/viz_mfe.py --video phantom76 --start 300 --frames 4
    python3 scripts/viz_mfe.py --video phantom76 --start 375 --frames 6
    python3 scripts/viz_mfe.py --video phantom63 --start 200 --frames 4 --mode 2frame
    python3 scripts/viz_mfe.py --video phantom76 --frames 4 --extras
"""
from __future__ import annotations

import argparse
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
from mgmd.mfe import MFEConfig, enhance

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)


def zoom_at(img: np.ndarray, center, maps, out_size: int = 620,
            patch: int = 80) -> np.ndarray:
    """以 center 为中心裁 patch×patch 放大，并按比例映射框坐标。"""
    h, w = img.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    x1 = max(0, min(cx - patch // 2, max(0, w - patch)))
    y1 = max(0, min(cy - patch // 2, max(0, h - patch)))
    x2, y2 = min(w, x1 + patch), min(h, y1 + patch)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), np.uint8)

    scale = out_size / float(max(crop.shape[1], crop.shape[0]))
    out = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                            max(1, int(crop.shape[0] * scale))),
                     interpolation=cv2.INTER_NEAREST)

    for color, thick, boxes in maps:
        sel = []
        for b in boxes:
            if b[2] > x1 and b[0] < x2 and b[3] > y1 and b[1] < y2:
                a = (b[0] - x1) * scale
                bb = (b[1] - y1) * scale
                c = (b[2] - x1) * scale
                d = (b[3] - y1) * scale
                sel.append((a, bb, max(c, a + 4), max(d, bb + 4)))
        if sel:
            out = viz.draw_boxes(out, sel, color, thick, None)
    return out


def render_frame(frame, gt_boxes, cand_boxes, motion, closed, cfg, t,
                 zoom_size=620, patch=80) -> np.ndarray:
    """生成 6 格对比图（3 阶段 × 全图/放大）。"""
    best_iou, best_idx = 0.0, -1
    if gt_boxes:
        for i, cb in enumerate(cand_boxes):
            v = max(box_iou(cb, g) for g in gt_boxes)
            if v > best_iou:
                best_iou, best_idx = v, i

    if gt_boxes:
        gb = gt_boxes[0]
        center = ((gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2)
    elif cand_boxes:
        cb = cand_boxes[0]
        center = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
    else:
        center = (frame.shape[1] / 2, frame.shape[0] / 2)

    r = patch // 2
    cx, cy = int(center[0]), int(center[1])

    full_in = viz.draw_boxes(frame, gt_boxes, GREEN, 2, None)
    if best_idx >= 0:
        full_in = viz.draw_boxes(full_in, [cand_boxes[best_idx]], RED, 2, None)
    cv2.rectangle(full_in, (cx - r, cy - r), (cx + r, cy + r), CYAN, 2)

    full_mask = viz.draw_boxes(viz.mask_to_bgr(motion), gt_boxes, GREEN, 1, None)
    cv2.rectangle(full_mask, (cx - r, cy - r), (cx + r, cy + r), CYAN, 2)

    full_closed = viz.draw_boxes(viz.mask_to_bgr(closed), gt_boxes, GREEN, 1, None)
    cv2.rectangle(full_closed, (cx - r, cy - r), (cx + r, cy + r), CYAN, 2)

    maps_in = [(GREEN, 2, gt_boxes)]
    if best_idx >= 0:
        maps_in.append((RED, 2, [cand_boxes[best_idx]]))
    zoom_in = zoom_at(frame, center, maps_in, zoom_size, patch)
    zoom_mask = zoom_at(viz.mask_to_bgr(motion), center,
                        [(GREEN, 2, gt_boxes)], zoom_size, patch)
    zoom_closed = zoom_at(viz.mask_to_bgr(closed), center,
                          [(GREEN, 2, gt_boxes)], zoom_size, patch)

    zm = zoom_size // patch
    return viz.make_panel([
        (f"input + GT  frame {t}  GT={len(gt_boxes)}", full_in),
        (f"ZOOM x{zm}  green=GT  red=best candidate", zoom_in),
        (f"motion E_t  {cfg.mode}  k={cfg.k}  th={cfg.threshold}", full_mask),
        ("ZOOM  motion mask", zoom_mask),
        (f"morphology  open={cfg.open_iters} close={cfg.close_iters}", full_closed),
        ("ZOOM  after morphology", zoom_closed),
    ], cols=2, tile_w=620)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom76")
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--mode", default="3frame_or",
                    choices=["2frame", "3frame_and", "3frame_or"])
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--patch", type=int, default=80, help="放大裁剪边长（像素）")
    ap.add_argument("--extras", action="store_true", help="开启四项额外约束")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    cfg = MFEConfig(k=args.k, mode=args.mode, threshold=args.threshold)
    if args.extras:
        cfg.preprocess_blur = 11
        cfg.disp_filter = True
        cfg.max_area = 3000
        cfg.aspect_filter = True

    tag = "extras" if args.extras else "bare"
    outdir = Path(args.outdir) if args.outdir else \
        C.FIGURES / f"mfe_{args.video}_{args.start}_{args.mode}_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    gt = load_video_gt(args.video)
    k = cfg.k
    stats = []

    with VideoReader(args.video) as vr:
        buf = {}
        for idx, fr in vr.iter_frames(start=args.start, limit=args.frames + 2 * k + 2):
            buf[idx] = fr
            t = idx - k
            if t < args.start or (t - k) not in buf:
                continue
            if cfg.needs_next and (t + k) not in buf:
                continue

            res = enhance(buf[t - k], buf[t],
                          buf.get(t + k) if cfg.needs_next else None, cfg, t)
            cand = [(x, y, x + w, y + h) for (x, y, w, h) in res.boxes]
            g = gt.get(t)
            gt_boxes = ([(o.x1, o.y1, o.x2, o.y2) for o in g.objects] if g else [])

            panel = render_frame(buf[t], gt_boxes, cand, res.motion,
                                 res.closed, cfg, t, patch=args.patch)
            viz.write_image(outdir / f"frame_{t:05d}.jpg", panel)

            best = max([box_iou(c, gg) for c in cand for gg in gt_boxes], default=0.0)
            stats.append((t, len(cand), round(best, 3),
                          int(gt_boxes[0][2] - gt_boxes[0][0]) if gt_boxes else 0,
                          int(gt_boxes[0][3] - gt_boxes[0][1]) if gt_boxes else 0))
            for kk in [b for b in buf if b < t - 2 * k - 2]:
                del buf[kk]

    print(f"输出目录: {outdir}")
    print(f"{'frame':>6}{'cands':>7}{'bestIoU':>9}{'GT wh':>10}  判定")
    for t, nc, bi, gw, gh in stats:
        print(f"{t:>6}{nc:>7}{bi:>9.3f}{f'{gw}x{gh}':>10}  {'HIT' if bi > 0.25 else 'MISS'}")
    return C.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
