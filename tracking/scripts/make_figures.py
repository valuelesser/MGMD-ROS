"""scripts/make_figures.py — 图表与失败拼图（SPEC §4.1 / PRD §11）。

用法：
    python scripts/make_figures.py --runs outputs/runs --out outputs/figures [--failures 10] [--no-images]

产出（命名稳定，供 report.md 引用）：
    fig7_recall_fps.png    Fig7 风格 recall + FPS 双轴柱状图
    fig8_trajectories.png  Fig8 风格 GT vs 估计轨迹
    failures_fa.png / failures_miss.png  失败案例拼图（≥failures，标注 reason；--no-images 跳过）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import make_tables as MT  # noqa: E402
from spa2d import io_mvdt as io  # noqa: E402
from spa2d import metrics as MET  # noqa: E402


def _recall_fps(rows):
    labels = [r["config"] for r in rows]
    recall = [100.0 * (r["recall@100"] or 0.0) for r in rows]
    fps = [r["effective_frame_fps"] or 0.0 for r in rows]
    return labels, recall, fps


def fig7(rows, out_dir: Path) -> Path:
    labels, recall, fps = _recall_fps(rows)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(rows)), 4.5))
    ax.bar(x, recall, color="#4c72b0", label="recall@100px (%)")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Recall@100px (%)", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax2 = ax.twinx()
    ax2.plot(x, fps, "k.-", label="effective FPS")
    ax2.set_ylabel("effective FPS")
    ax.set_title("SPA2D recall@100px and effective FPS")
    fig.tight_layout()
    path = out_dir / "fig7_recall_fps.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _pick_runs(runs, keys=("D1/cam2", "dataset1/cam2", "D4/cam3", "dataset4/cam3")):
    picked = []
    for k in keys:
        for r in runs:
            if r.seq == k and r.gt is not None:
                picked.append(r)
                break
    return picked or [r for r in runs if r.gt is not None][:2]


def fig8(runs, out_dir: Path) -> Path:
    picked = _pick_runs(runs)
    n = max(1, len(picked))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, run in zip(axes[0], picked):
        gt = [(g.cx, g.cy) for g in run.gt if g.visible == 1 and g.cx is not None]
        if gt:
            gx, gy = zip(*gt)
            ax.plot(gx, gy, "-", color="green", lw=1.2, label="GT")
        ex = [(r.est.x, r.est.y) for r in run.windows if r.target_present and r.est]
        if ex:
            ax.plot([p[0] for p in ex], [p[1] for p in ex], ".", color="red", ms=2,
                    label="SPA2D est")
        ax.invert_yaxis()
        ax.set_title(f"{run.seq} trajectory")
        ax.set_xlabel("x px")
        ax.set_ylabel("y px")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = out_dir / "fig8_trajectories.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _collect_failures(runs, want: int):
    fa, miss = [], []
    for run in runs:
        if run.gt is None:
            continue
        frames_dir = _frames_dir_from_meta(run)
        if frames_dir is None:
            continue
        try:
            src = io.open_sequence(frames_dir=str(frames_dir))
        except (FileNotFoundError, ValueError):
            continue
        for rec in run.windows:
            n = MET._N_from(rec)
            f = MET.associate_frame(rec.t_start, n, "round_half_up")
            g = run.gt[f] if 0 <= f < len(run.gt) else None
            visible = bool(g is not None and g.visible == 1 and g.cx is not None)
            if rec.target_present:
                d = float("inf")
                if visible:
                    d = ((rec.est.x - g.cx) ** 2 + (rec.est.y - g.cy) ** 2) ** 0.5
                if not visible or d > 100:
                    cx, cy = (rec.est.x, rec.est.y)
                    fa.append((src, f, cx, cy, rec.reason))
            else:
                if visible:
                    miss.append((src, f, g.cx, g.cy, rec.reason))
            if len(fa) >= want and len(miss) >= want:
                return fa[:want], miss[:want]
    return fa[:want], miss[:want]


def _frames_dir_from_meta(run):
    import json

    mp = Path(run.dir) / "run_meta.json"
    if not mp.is_file():
        return None
    meta = json.loads(mp.read_text(encoding="utf-8"))
    fd = meta.get("config", {}).get("snapshot", {}).get("sequence", {}).get("frames_dir")
    return Path(fd) if fd else None


def _montage(items, title, path: Path, half: int = 32) -> Path:
    cols = min(5, max(1, len(items)))
    rows = max(1, (len(items) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (src, f, cx, cy, reason) in zip(axes.ravel(), items):
        try:
            bgr = src.read(int(f))
            h, w = bgr.shape[:2]
            x0, x1 = max(0, int(cx) - half), min(w, int(cx) + half)
            y0, y1 = max(0, int(cy) - half), min(h, int(cy) + half)
            patch = bgr[y0:y1, x0:x1, ::-1]
            ax.imshow(patch)
        except Exception:  # noqa: BLE001
            ax.text(0.5, 0.5, "n/a", ha="center")
        ax.set_title(f"f{f} {reason}", fontsize=7)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="make_figures.py", description="图表与失败拼图")
    p.add_argument("--runs", default="outputs/runs")
    p.add_argument("--gt", default="data_derived/gt")
    p.add_argument("--out", default="outputs/figures")
    p.add_argument("--failures", type=int, default=10)
    p.add_argument("--no-images", action="store_true", help="license 禁用图像：仅统计图")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = MT.load_runs(Path(args.runs), Path(args.gt))
    rows = MT.summary_rows(runs)
    if not rows:
        print("[spa2d] E_DATA_MISSING: 无可用 runs → 先运行 run_sequence.py", file=sys.stderr)
        return 3
    made = [fig7(rows, out), fig8(runs, out)]
    if not args.no_images:
        fa, miss = _collect_failures(runs, args.failures)
        if fa:
            made.append(_montage(fa, f"False accepts (n={len(fa)})", out / "failures_fa.png"))
        if miss:
            made.append(_montage(miss, f"Misses (n={len(miss)})", out / "failures_miss.png"))
        print(f"failure montage: FA={len(fa)} miss={len(miss)} (requested {args.failures})")
    for m in made:
        print("figure:", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
