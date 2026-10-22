"""scripts/make_visuals_d1d2.py — D1/D2 匹配滤波结果可视化（轨迹/误差/帧叠加）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spa2d import io_mvdt as io, records as R  # noqa: E402

RUNS = {
    "D1/cam2": ("outputs/runs/D1_matched", "data_derived/gt/D1_cam2.csv",
                "data_raw/mvdt/D1/cam2/cam2.mp4"),
    "D2/cam1": ("outputs/runs/D2_matched", "data_derived/gt/D2_cam1.csv",
                "data_raw/mvdt/D2/cam1/cam1.mp4"),
}
OUT = Path("outputs/figures")


def assoc(rec):
    n = int(round(2.0 * (rec.t_center - rec.t_start) + 1.0))
    return rec.t_start + (n - 1) // 2


def traj(name, run_dir, gt_path):
    recs = R.read_jsonl(f"{run_dir}/windows.jsonl")
    gt = R.read_gt_csv(gt_path)
    gx = [g.cx for g in gt if g.visible == 1 and g.cx is not None]
    gy = [g.cy for g in gt if g.visible == 1 and g.cx is not None]
    ex, ey, err = [], [], []
    for r in recs:
        if not r.target_present:
            continue
        f = assoc(r); ex.append(r.est.x); ey.append(r.est.y)
        err.append(float(np.hypot(r.est.x - gt[f].cx, r.est.y - gt[f].cy)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(gx, gy, "-", color="green", lw=1.2, label="GT")
    a1.plot(ex, ey, ".", color="red", ms=3, label="SPA2D est")
    a1.invert_yaxis(); a1.set_title(f"{name} trajectory (GT vs est)")
    a1.set_xlabel("x px"); a1.set_ylabel("y px"); a1.legend()
    a2.plot(range(len(err)), err, "-", color="C0")
    a2.axhline(100, color="k", ls="--", lw=0.8)
    a2.set_title(f"{name} per-window position error (px)")
    a2.set_xlabel("accepted window #"); a2.set_ylabel("error px")
    fig.tight_layout()
    p = OUT / f"fig8_{name.replace('/', '_')}.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    return p, err


def overlay(name, run_dir, gt_path, video):
    recs = R.read_jsonl(f"{run_dir}/windows.jsonl")
    gt = R.read_gt_csv(gt_path)
    src = io.open_sequence(video=video)
    picks = [r for r in recs if r.target_present][::max(1, len(recs) // 6)][:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), picks):
        f = assoc(r)
        bgr = src.read(f)
        h, w = bgr.shape[:2]
        cx, cy = gt[f].cx, gt[f].cy
        half = 90
        x0, x1 = max(0, int(cx) - half), min(w, int(cx) + half)
        y0, y1 = max(0, int(cy) - half), min(h, int(cy) + half)
        patch = bgr[y0:y1, x0:x1, ::-1]
        ax.imshow(patch)
        ax.plot(cx - x0, cy - y0, "o", mfc="none", mec="lime", ms=14, mew=2)      # GT
        ax.plot(r.est.x - x0, r.est.y - y0, "+", color="red", ms=14, mew=2)       # est
        ax.set_title(f"f{f}  err={np.hypot(r.est.x-cx, r.est.y-cy):.1f}px", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{name} overlay (green=GT, red=SPA2D est)")
    fig.tight_layout()
    p = OUT / f"overlay_{name.replace('/', '_')}.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (run_dir, gt, video) in RUNS.items():
        if not Path(run_dir).is_dir() or not Path(video).is_file():
            print("skip", name); continue
        p1, err = traj(name, run_dir, gt)
        p2 = overlay(name, run_dir, gt, video)
        print(f"{name}: {p1}  {p2}  RMSE={np.sqrt(np.mean(np.square(err))):.2f}px "
              f"max={max(err):.1f}px n={len(err)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
