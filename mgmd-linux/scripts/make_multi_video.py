"""scripts/make_multi_video.py — 多目标诊断可视化（SPEC §10.8）。

展示 phantom20 帧 24-56 的双目标场景，四栏并列：

    左1  全图：GT 绿框 + MFE 候选黄框 + 确认轨迹红框，两个 GT 打不同标签
    左2  GT0 区域放大 120x120（下方快速目标）
    左3  GT1 区域放大 120x120（上方缓慢目标）
    底部  逐帧状态条：两个目标各自的 MFE 命中 / 轨迹 ID / 是否确认

目的：让"第二个目标在哪一环丢失"一眼可见。

用法：
    source scripts/env_setup.sh
    $PY scripts/make_multi_video.py
    $PY scripts/make_multi_video.py --video phantom20 --start 24 --frames 30
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
from mgmd.annotations import load_video_gt
from mgmd.io_video import VideoReader
from mgmd.mfe import MFEConfig, enhance
from mgmd.tf import SortTracker, iou

GREEN = (0, 255, 0)      # GT
YELLOW = (0, 255, 255)   # MFE 候选
RED = (0, 0, 255)        # 确认轨迹
CYAN = (255, 255, 0)     # 未被任何轨迹覆盖的 GT
WHITE = (255, 255, 255)
GRAY = (140, 140, 140)
BLACK = (0, 0, 0)


def put(img, text, org, color=WHITE, scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def zoom_patch(frame, center, size, out_size=240, label="", hit=None):
    """以 center 为中心裁 size×size 并放大到 out_size。"""
    h, w = frame.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    half = size // 2
    x1 = max(0, min(cx - half, w - size))
    y1 = max(0, min(cy - half, h - size))
    x2, y2 = x1 + size, y1 + size
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        patch = np.zeros((out_size, out_size, 3), np.uint8)
    else:
        patch = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    # 中心十字
    c = out_size // 2
    cv2.line(patch, (c - 12, c), (c + 12, c), GRAY, 1)
    cv2.line(patch, (c, c - 12), (c, c + 12), GRAY, 1)
    # 标题条
    bar = np.full((22, out_size, 3), 30, np.uint8)
    col = GREEN if hit else RED
    put(bar, label, (4, 16), col, 0.45)
    return np.vstack([bar, patch])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom20")
    ap.add_argument("--start", type=int, default=24)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="outputs/figures/multi_phantom20.mp4")
    ap.add_argument("--d-min", type=float, default=0.5)
    ap.add_argument("--mode", default="recommended",
                    choices=["recommended", "official"])
    args = ap.parse_args()

    cfg = MFEConfig()
    if args.mode == "official":
        cfg.apply_official()
    else:
        cfg.apply_recommended()
    cfg.validate()
    k = cfg.k

    vid = C.ARD_VIDEOS / args.video
    if not vid.with_suffix(".mp4").is_file():
        raise SystemExit(f"视频不存在: {vid}")
    gt = load_video_gt(args.video)

    end = args.start + args.frames - 1
    buf: dict[int, np.ndarray] = {}
    with VideoReader(vid) as vr:
        for idx, fr in vr.iter_frames(start=args.start, limit=end - args.start + 1):
            buf[idx] = fr
    if not buf:
        raise SystemExit("没读到帧")

    tr = SortTracker(d_mode="enclosing", d_min=args.d_min,
                     iou_thresh=0.3, max_age=3, span=3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    # 统计
    stat = {"both": 0, "one": 0, "none": 0, "n_multi": 0}
    n_hit = [0, 0]
    n_total = [0, 0]

    for t in sorted(buf):
        cur = buf.get(t)
        if cur is None:
            continue
        prev = buf.get(t - k)
        nxt = buf.get(t + k)

        # MFE（缺前后帧时退化为 2 帧差分）
        res = enhance(prev, cur, nxt if nxt is not None else cur, cfg, t)
        cands = [(float(x), float(y), float(x + w), float(y + h))
                 for (x, y, w, h) in res.boxes]
        tracks = tr.update(cands, t)

        g = gt.get(t)
        objs = []
        if g:
            for o in g.objects:
                objs.append((o.x1, o.y1, o.x2, o.y2))

        vis = cur.copy()
        h0, w0 = vis.shape[:2]

        # 候选：黄色细框
        for c in cands:
            cv2.rectangle(vis, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])),
                          YELLOW, 1)

        # GT：每个目标判断是否被轨迹覆盖
        g_color = []
        for gi, gb in enumerate(objs):
            covered = any(iou(tk.box, gb) >= 0.25 for tk in tracks)
            hit_mfe = any(iou(c, gb) >= 0.25 for c in cands)
            if len(objs) == 2:
                n_total[gi] += 1
                if hit_mfe:
                    n_hit[gi] += 1
            col = GREEN if covered else CYAN
            g_color.append(col)
            x1, y1, x2, y2 = (int(v) for v in gb)
            cv2.rectangle(vis, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), col, 1)
            tag = f"GT{gi}"
            if not hit_mfe:
                tag += " miss"
            put(vis, tag, (x1 - 2, max(11, y1 - 5)), col, 0.4)

        # 轨迹：红色
        for tk in tracks:
            b = tk.box
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          RED, 2)
            put(vis, f"T{tk.track_id}", (int(b[2]) + 3, int(b[1]) + 8), RED, 0.4)

        # 统计画幅
        if len(objs) >= 2:
            stat["n_multi"] += 1
            covered_idx = set()
            for gi, gb in enumerate(objs):
                if any(iou(tk.box, gb) >= 0.25 for tk in tracks):
                    covered_idx.add(gi)
            if len(covered_idx) >= 2:
                stat["both"] += 1
            elif len(covered_idx) == 1:
                stat["one"] += 1
            else:
                stat["none"] += 1

        # 缩放主图到 640 宽
        scale = 640 / w0
        main_img = cv2.resize(vis, (640, int(h0 * scale)))

        # 两个放大窗
        zsize = 120
        if len(objs) >= 1:
            c0 = ((objs[0][0] + objs[0][2]) / 2, (objs[0][1] + objs[0][3]) / 2)
            h0_ = any(iou(c, objs[0]) >= 0.25 for c in cands)
            p0 = zoom_patch(cur, c0, zsize, 232,
                            f"GT0 ({objs[0][2]-objs[0][0]:.0f}x{objs[0][3]-objs[0][1]:.0f}) "
                            f"{'MFE hit' if h0_ else 'MFE MISS'}", h0_)
        else:
            p0 = np.zeros((254, 232, 3), np.uint8)
        if len(objs) >= 2:
            c1 = ((objs[1][0] + objs[1][2]) / 2, (objs[1][1] + objs[1][3]) / 2)
            h1_ = any(iou(c, objs[1]) >= 0.25 for c in cands)
            p1 = zoom_patch(cur, c1, zsize, 232,
                            f"GT1 ({objs[1][2]-objs[1][0]:.0f}x{objs[1][3]-objs[1][1]:.0f}) "
                            f"{'MFE hit' if h1_ else 'MFE MISS'}", h1_)
        else:
            p1 = np.zeros((254, 232, 3), np.uint8)

        right = np.vstack([p0, p1])
        if right.shape[0] < main_img.shape[0]:
            pad = np.zeros((main_img.shape[0] - right.shape[0], right.shape[1], 3), np.uint8)
            right = np.vstack([right, pad])
        else:
            right = right[:main_img.shape[0]]

        frame_out = np.hstack([main_img, right])

        # HUD 条
        hud_h = 76
        hud = np.full((hud_h, frame_out.shape[1], 3), 24, np.uint8)
        put(hud, f"frame {t}  ({args.video})  MFE cand={len(cands)}  tracks={len(tracks)}"
                 f"  GT={len(objs)}", (8, 20), WHITE, 0.55)
        put(hud, f"D={args.d_min} max_age=3 mode={args.mode}  |  "
                 f"multi-frames both={stat['both']} one={stat['one']} none={stat['none']}",
            (8, 42), GRAY, 0.45)
        tot = max(1, stat["n_multi"])
        put(hud, f"GT0 MFE hit {n_hit[0]}/{n_total[0]}    "
                 f"GT1 MFE hit {n_hit[1]}/{n_total[1]}    "
                 f"双目标帧={stat['n_multi']}",
            (8, 64), CYAN, 0.45)
        frame_out = np.vstack([frame_out, hud])

        if writer is None:
            H, W = frame_out.shape[:2]
            writer = cv2.VideoWriter(str(tmp), fourcc, args.fps, (W, H))
        writer.write(frame_out)

    if writer is not None:
        writer.release()

    # 转码 H.264（浏览器可播放）
    final = out_path
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(final)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        tmp.unlink(missing_ok=True)
    else:
        print("ffmpeg 转码失败，保留 mp4v:", r.stderr[:200])
        final = tmp

    print(f"输出: {final}")
    print(f"双目标帧 {stat['n_multi']}: both={stat['both']} one={stat['one']} none={stat['none']}")
    print(f"GT0 MFE 命中 {n_hit[0]}/{n_total[0]}   GT1 {n_hit[1]}/{n_total[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
