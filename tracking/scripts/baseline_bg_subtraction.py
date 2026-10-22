"""scripts/baseline_bg_subtraction.py — 静止相机强基线（SPEC §5.13）。

滑动中位背景 + 暗目标帧差 + 连通域 + 最近邻/Kalman 跟踪；输出与 run_sequence 同 schema
子集（信置 null，reason=BASELINE_*）并双口径评测（同 assoc_frame 关联 + 全帧级）。

用法：
    python scripts/baseline_bg_subtraction.py --config configs/D1_no_overlap.yaml \
        --gt data_derived/gt/D1_cam2.csv --out outputs/runs_baselines/bgsub_D1_no_overlap \
        [--tracker nn|kalman]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spa2d import io_mvdt as io  # noqa: E402
from spa2d import metrics as MET  # noqa: E402
from spa2d import records as R  # noqa: E402
from spa2d import timing as T  # noqa: E402
from spa2d.config import ConfigError, load_config  # noqa: E402

BG_WINDOW = 50
KAPPA = 3.0
GATE = 50.0


def _gray(frame):
    import cv2

    return (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if frame.ndim == 3 else frame.astype(np.float32))


class NNTrail:
    def __init__(self):
        self.pos = None
        self.vel = np.zeros(2)

    def step(self, dets):
        pred = None if self.pos is None else self.pos + self.vel
        best = None
        if dets:
            if pred is None:
                best = max(dets, key=lambda d: d[2]) if dets else None
            else:
                c = min(dets, key=lambda d: np.hypot(d[0] - pred[0], d[1] - pred[1]))
                if np.hypot(c[0] - pred[0], c[1] - pred[1]) <= GATE:
                    best = c
        if best is None:
            return None
        newpos = np.array([best[0], best[1]], float)
        if self.pos is not None:
            self.vel = newpos - self.pos
        self.pos = newpos
        return (float(newpos[0]), float(newpos[1]), float(self.vel[0]), float(self.vel[1]))


class KalmanTrack:
    """匀速模型恒速 Kalman（位置/速度 2D）。"""

    def __init__(self):
        self.x = None
        self.P = np.eye(4) * 10.0
        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.Q = np.eye(4) * 0.5
        self.Rm = np.eye(2) * 4.0

    def step(self, dets):
        if self.x is not None:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
        pred = None if self.x is None else self.x[:2]
        best = None
        if dets:
            c = (min(dets, key=lambda d: np.hypot(d[0] - pred[0], d[1] - pred[1]))
                 if pred is not None else max(dets, key=lambda d: d[2]))
            if pred is None or np.hypot(c[0] - pred[0], c[1] - pred[1]) <= GATE:
                best = c
        if best is None:
            return None
        z = np.array([best[0], best[1]], float)
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
        else:
            S = self.H @ self.P @ self.H.T + self.Rm
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ (z - self.H @ self.x)
            self.P = (np.eye(4) - K @ self.H) @ self.P
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3]))


def _open_source(cfg):
    if cfg.sequence.video and Path(cfg.sequence.video).is_file():
        return io.open_sequence(video=cfg.sequence.video)
    fd = Path(cfg.sequence.frames_dir)
    if fd.is_dir():
        return io.open_sequence(frames_dir=str(fd))
    if fd.is_file():
        return io.open_sequence(video=str(fd))
    raise FileNotFoundError(f"帧源不存在: {fd}")


def detect(src, F, N, stride, tracker_name, max_target_area=4096):
    import cv2

    info = src.probe()
    buf = deque(maxlen=BG_WINDOW)
    tracker = KalmanTrack() if tracker_name == "kalman" else NNTrail()
    per_frame = []          # (frame, present, x, y, vx, vy)
    for f in range(F):
        gray = _gray(src.read(f))
        buf.append(gray)
        bg = np.median(np.stack(list(buf)), axis=0)
        diff = bg - gray
        thr = diff.mean() + KAPPA * diff.std()
        mask = (diff > thr).astype(np.uint8)
        num, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        dets = []
        for c in range(1, num):
            area = stats[c, cv2.CC_STAT_AREA]
            if 1 <= area <= 4 * max_target_area:
                dets.append((float(cents[c][0]), float(cents[c][1]), int(area)))
        res = tracker.step(dets)
        if res is None:
            per_frame.append((f, False, None, None, None, None))
        else:
            per_frame.append((f, True, *res))
    return per_frame, info


def _rec(i, t, N, mode, present, x, y, vx, vy):
    est = None if not present else R.Estimate(x, y, vx, vy)
    return R.WindowRecord(
        window_index=i, t_start=t, t_center=t + (N - 1) / 2.0,
        assoc_frame=MET.associate_frame(t, N, mode), search_mode="baseline",
        state_in="BASELINE", state_out="BASELINE" if present else "BASELINE_ABSENT",
        accepted=present, target_present=present,
        reason="BASELINE_OK" if present else "BASELINE_ABSENT", est=est,
        timing_ms=R.TimingMs(t_pre=0.0, t_search=0.0, t_state=0.0, t_total=0.0))


def run_baseline(cfg, gt, out_dir: Path, tracker="nn"):
    src = _open_source(cfg)
    F = len(src)
    cap = int(os.environ.get("SPA2D_FRAME_LIMIT", "0") or 0)
    if cap > 0:
        F = min(F, cap)
    N, stride = cfg.window.N, cfg.window.stride
    per_frame, info = detect(src, F, N, stride, tracker)
    frame_state = {f: (present, x, y, vx, vy) for (f, present, x, y, vx, vy) in per_frame}

    # 同 assoc_frame 关联（窗口级）
    windows = []
    for i, t in enumerate(range(0, F - N + 1, stride)):
        f = MET.associate_frame(t, N, cfg.metrics.association)
        present, x, y, vx, vy = frame_state.get(f, (False, None, None, None, None))
        windows.append(_rec(i, t, N, cfg.metrics.association, present, x, y, vx, vy))

    # 全帧级（N=1 记录，assoc=f）
    full = []
    for f in range(F):
        present, x, y, vx, vy = frame_state[f]
        rec = _rec(f, f, 1, "round_half_up", present, x, y, vx, vy)
        rec.t_center = float(f)
        rec.assoc_frame = f
        full.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    R.write_jsonl(windows, out_dir / "windows.jsonl")
    res_w = MET.evaluate_windows(windows, gt, config_name=f"bgsub_{cfg.config_name}")
    MET.write_metrics_json(res_w, out_dir / "metrics.json")
    res_f = MET.evaluate_windows(full, gt, config_name=f"bgsub_{cfg.config_name}_fullframe")
    MET.write_metrics_json(res_f, out_dir / "metrics_fullframe.json")
    meta = {"baseline": "median_bg", "tracker": tracker, "bg_window": BG_WINDOW, "kappa": KAPPA,
            "config": {"name": cfg.config_name, "path": str(cfg.source_path)},
            "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                            "numpy": np.__version__},
            "sequence": cfg.sequence.name, "n_windows": len(windows), "n_frames": F}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True),
                                           encoding="utf-8")
    return res_w, res_f, len(windows)


def build_parser():
    p = argparse.ArgumentParser(prog="baseline_bg_subtraction.py",
                                description="静止相机背景建模基线（SPEC §5.13）")
    p.add_argument("--config", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tracker", choices=["nn", "kalman"], default="nn")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc.format(), file=sys.stderr)
        return exc.exit_code
    gt_path = Path(args.gt)
    if not gt_path.is_file():
        print(f"[spa2d] E_DATA_MISSING: GT 不存在 {gt_path}", file=sys.stderr)
        return 3
    try:
        gt = R.read_gt_csv(gt_path)
        res_w, res_f, n = run_baseline(cfg, gt, Path(args.out), args.tracker)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[spa2d] E_RUNTIME: {exc}", file=sys.stderr)
        return 4
    print(f"baseline {cfg.config_name} tracker={args.tracker}: windows={n} "
          f"recall@100={res_w.recall.get(100)} precision@100={res_w.precision.get(100)} "
          f"(full-frame precision@100={res_f.precision.get(100)}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
