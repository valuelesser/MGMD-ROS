"""scripts/live_track.py — 实时摄像头 MAV 检测与追踪（SPEC §8.30）。

摄像头 → MFE → LAC → LAD → 画面叠加显示。

两种运行模式：
    --display     弹出窗口实时显示（需 GUI 版 opencv-python）
    --record      无头模式，写 mp4（适用于无显示的机器）

用法：
    python3 scripts/live_track.py --display              # 摄像头实时显示
    python3 scripts/live_track.py --record out.mp4       # 录制
    python3 scripts/live_track.py --camera 0 --display
    python3 scripts/live_track.py --video x.mp4 --display  # 用视频文件模拟

按键（--display 模式）：
    q / ESC   退出
    s         截图保存
    SPACE     暂停/继续
    d         切换 debug 视图（显示候选框）

重要限制（SPEC §8.30）：
    摄像头的分辨率/视角与 ARD-MAV 完全不同。LAC/LAD 权重在 ARD-MAV
    （机载俯视、1920×1080）上训练，**直接用于网络摄像头大概率检不出**。
    本脚本主要用于验证"实时通路是否跑通"，检测效果不代表模型能力。
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from mgmd import config as C
from mgmd.mfe import MFEConfig, enhance
from mgmd.lac import LacClassifier, crop_box
from mgmd.lad import LadDetector

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)


class Track:
    """极简轨迹：用于让检测框在漏检帧也能短暂延续，并在画面上标 ID。"""

    def __init__(self, tid: int, box, frame_idx: int, max_age: int = 8):
        self.id = tid
        self.box = box
        self.last_seen = frame_idx
        self.max_age = max_age
        self.path: deque = deque(maxlen=30)
        self.path.append(self._center(box))

    @staticmethod
    def _center(box):
        return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

    def update(self, box, frame_idx: int):
        self.box = box
        self.last_seen = frame_idx
        self.path.append(self._center(box))

    def expired(self, frame_idx: int) -> bool:
        return frame_idx - self.last_seen > self.max_age


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = x2 - x1, y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class LiveTracker:
    """把 MFE→LAC→LAD 串成逐帧处理，并维护简易轨迹。"""

    def __init__(self, cfg: MFEConfig, device: str, lad_batch: int,
                 trt_engine: str | None, lac_thresh: float = 0.5):
        self.cfg = cfg
        self.k = cfg.k
        self.buf: dict[int, np.ndarray] = {}
        self.frame_idx = 0
        self.clf = LacClassifier(device=device)
        self.lad = LadDetector(device=device, trt_engine=trt_engine)
        self.lad_batch = lad_batch
        self.lac_thresh = lac_thresh
        self.tracks: list[Track] = []
        self._next_id = 1
        self.timings: deque = deque(maxlen=60)

    def process(self, frame: np.ndarray) -> dict:
        """处理一帧。返回 {boxes, cands, ms, fps}。"""
        t0 = time.perf_counter()
        idx = self.frame_idx
        self.frame_idx += 1
        self.buf[idx] = frame
        k = self.k

        # 需要 t-k 与 t+k 都在缓冲里（OR 模式）
        center = idx - k
        if center < k or (center - k) not in self.buf or (center + k) not in self.buf:
            self._evict(center)
            return {"boxes": [], "cands": [], "ms": 0.0,
                    "fps": self.fps(), "warming": True}

        cur = self.buf[center]
        res = enhance(self.buf[center - k], cur,
                      self.buf[center + k] if self.cfg.needs_next else None,
                      self.cfg, center)
        cands = [(float(x), float(y), float(x + w), float(y + h))
                 for (x, y, w, h) in res.boxes]

        pool = cands
        if pool:
            crops = [crop_box(cur, b) for b in pool]
            pairs = [(b, c) for b, c in zip(pool, crops) if c is not None]
            if pairs:
                ps = self.clf.mav_prob_batch([c for _, c in pairs])
                pool = [b for (b, _), p in zip(pairs, ps) if p >= self.lac_thresh]
            else:
                pool = []

        if pool:
            rbs = self.lad.refine_batch(
                cur, [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in pool],
                batch_size=self.lad_batch)
            pool = [rb.box for rb in rbs if rb is not None and rb.conf >= 0.5]

        self._update_tracks(pool, center)
        self._evict(center)
        ms = (time.perf_counter() - t0) * 1000
        self.timings.append(ms)
        return {"boxes": pool, "cands": cands, "ms": ms,
                "fps": self.fps(), "warming": False}

    def fps(self) -> float:
        if not self.timings:
            return 0.0
        m = float(np.mean(self.timings))
        return 1000.0 / m if m > 0 else 0.0

    def _update_tracks(self, boxes, frame_idx: int):
        used = set()
        for b in boxes:
            best, best_iou = None, 0.3
            for tr in self.tracks:
                if tr.id in used:
                    continue
                v = iou(tr.box, b)
                if v > best_iou:
                    best, best_iou = tr, v
            if best is not None:
                best.update(b, frame_idx)
                used.add(best.id)
            else:
                self.tracks.append(Track(self._next_id, b, frame_idx))
                used.add(self._next_id)
                self._next_id += 1
        self.tracks = [t for t in self.tracks if not t.expired(frame_idx)]

    def _evict(self, center: int):
        for key in [f for f in self.buf if f < center - 2 * self.k - 2]:
            del self.buf[key]


def draw(frame: np.ndarray, result: dict, tracker: LiveTracker,
         debug: bool, paused: bool) -> np.ndarray:
    out = frame.copy()
    if debug:
        for b in result["cands"]:
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          YELLOW, 1)
    # 轨迹
    for tr in tracker.tracks:
        b = tr.box
        fresh = (tr.last_seen >= tracker.frame_idx - 1)
        color = RED if fresh else (120, 120, 120)
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      color, 2)
        cv2.putText(out, f"MAV#{tr.id}", (int(b[0]), max(12, int(b[1]) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        pts = list(tr.path)
        for i in range(1, len(pts)):
            cv2.line(out, pts[i - 1], pts[i], GREEN, 1, cv2.LINE_AA)

    h, w = out.shape[:2]
    bar = np.full((58, w, 3), 28, np.uint8)
    cv2.putText(bar, f"FPS {result['fps']:5.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2, cv2.LINE_AA)
    cv2.putText(bar, f"proc {result['ms']:5.1f}ms", (150, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2, cv2.LINE_AA)
    cv2.putText(bar, f"cand {len(result['cands'])}  track {len(tracker.tracks)}",
                (330, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
    cv2.putText(bar, f"th={tracker.cfg.threshold} k={tracker.k} "
                     f"median={tracker.cfg.median_blur}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    if paused:
        cv2.putText(bar, "PAUSED", (w - 130, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
    if debug:
        cv2.putText(bar, "DEBUG", (w - 130, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="摄像头索引")
    src.add_argument("--video", default=None, help="用视频文件代替摄像头")
    ap.add_argument("--display", action="store_true", help="弹窗实时显示")
    ap.add_argument("--record", default=None, help="录制到 mp4（无头模式）")
    ap.add_argument("--limit", type=int, default=0, help="处理帧数上限（0=不限）")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--min-area", type=int, default=15)
    ap.add_argument("--preset", default="recommended",
                    choices=["paper", "recommended", "mvdt"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lad-batch", type=int, default=8)
    ap.add_argument("--lad-trt-engine", default=None)
    ap.add_argument("--lac-thresh", type=float, default=0.5)
    args = ap.parse_args(argv)

    cfg = MFEConfig(k=args.k, mode="3frame_or")
    if args.preset == "recommended":
        cfg.apply_recommended()
    elif args.preset == "mvdt":
        cfg.apply_recommended()
        cfg.threshold, cfg.min_area, cfg.k = 2, 5, 2
    cfg.threshold = args.threshold if args.threshold != 5 else cfg.threshold
    cfg.min_area = args.min_area if args.min_area != 15 else cfg.min_area
    cfg.k = args.k

    # 打开源
    if args.video:
        cap = cv2.VideoCapture(args.video)
        src_name = Path(args.video).name
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        src_name = f"camera{args.camera}"
    if not cap.isOpened():
        print(f"[mgmd] E_DATA_MISSING: 无法打开 {src_name}", file=sys.stderr)
        return C.EXIT_DATA

    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[mgmd] 源={src_name} {w0}x{h0}  preset={args.preset} "
          f"th={cfg.threshold} k={cfg.k} device={args.device}")
    if args.lad_trt_engine:
        print(f"[mgmd] TRT engine: {args.lad_trt_engine}")

    tracker = LiveTracker(cfg, args.device, args.lad_batch,
                          args.lad_trt_engine, args.lac_thresh)

    vw = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(args.record, fourcc, 15.0, (w0, h0 + 58))
        if not vw.isOpened():
            print(f"[mgmd] 无法创建 {args.record}", file=sys.stderr)
            cap.release()
            return C.EXIT_RUNTIME
        print(f"[mgmd] 录制到 {args.record}")

    if not args.display and not args.record:
        print("[mgmd] 需指定 --display 或 --record", file=sys.stderr)
        cap.release()
        return C.EXIT_CONFIG

    debug = False
    paused = False
    saved = 0
    n = 0
    t_start = time.perf_counter()
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("[mgmd] 视频结束 / 摄像头读取失败")
                    break
                result = tracker.process(frame)
                vis = draw(frame, result, tracker, debug, paused)
                n += 1
                if vw is not None:
                    vw.write(vis)
                if args.limit and n >= args.limit:
                    print(f"[mgmd] 达到帧上限 {args.limit}")
                    break
            else:
                vis = draw(frame, result, tracker, debug, paused)

            if args.display:
                cv2.imshow("MGMD Live", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("d"):
                    debug = not debug
                elif key == ord("s"):
                    fn = f"outputs/live_snap_{saved:03d}.jpg"
                    Path(fn).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(fn, vis)
                    print(f"[mgmd] 截图 {fn}")
                    saved += 1
    except KeyboardInterrupt:
        print("\n[mgmd] 中断")
    finally:
        cap.release()
        if vw is not None:
            vw.release()
        if args.display:
            cv2.destroyAllWindows()

    dt = time.perf_counter() - t_start
    avg_fps = n / dt if dt > 0 else 0
    print(f"\n[mgmd] 处理 {n} 帧，{dt:.1f}s，平均 {avg_fps:.1f} FPS "
          f"（含采集/显示开销）")
    print(f"[mgmd] 内部处理 FPS: {tracker.fps():.1f}")
    return C.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
