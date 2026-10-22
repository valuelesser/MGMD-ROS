"""scripts/make_tables.py — 汇总指标表（SPEC §4.1 / §5.8 / PRD §11）。

用法：
    python scripts/make_tables.py --runs outputs/runs --gt data_derived/gt --out outputs/metrics

产出：summary.md / summary.csv、sensitivity.csv、paper_vs_ours.md、stratified.csv。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spa2d import metrics as MET  # noqa: E402
from spa2d import records as R  # noqa: E402

# 论文目标值（PRD §6，用于对照而非调参）
PAPER = {
    "D1_no_overlap": dict(fps=27.91, valid=99.50, recall=100.00, precision=98.99, rmse_p=4.43, rmse_v=0.26),
    "D1_overlap": dict(fps=9.49, valid=99.41, recall=100.00, precision=98.82, rmse_p=4.48, rmse_v=1.22),
    "D2_no_overlap": dict(fps=16.53, valid=97.79, recall=97.38, precision=99.55, rmse_p=12.43, rmse_v=1.42),
    "D2_overlap": dict(fps=5.36, valid=98.69, recall=99.21, precision=98.98, rmse_p=11.64, rmse_v=4.65),
    "D3_overlap": dict(fps=10.57, valid=93.86, recall=92.41, precision=93.72, rmse_p=4.28, rmse_v=0.58),
    "D4_overlap": dict(fps=1.93, valid=85.84, recall=84.64, precision=93.36, rmse_p=23.11, rmse_v=5.69),
}


@dataclass
class Run:
    dir: str
    name: str
    seq: str
    backend: str
    windows: list
    timing: dict
    gt: list | None


def load_runs(runs_dir: Path, gt_dir: Path) -> list[Run]:
    runs: list[Run] = []
    if not runs_dir.is_dir():
        return runs
    for run in sorted(runs_dir.iterdir()):
        wpath = run / "windows.jsonl"
        if not wpath.is_file():
            continue
        meta = {}
        mp = run / "run_meta.json"
        if mp.is_file():
            meta = json.loads(mp.read_text(encoding="utf-8"))
        snap = meta.get("config", {}).get("snapshot", {}) or {}
        seq = (snap.get("sequence", {}) or {}).get("name", run.name)
        name = meta.get("config", {}).get("name", run.name)
        backend = meta.get("backend", "unknown")
        timing = {}
        tp = run / "timing.json"
        if tp.is_file():
            timing = json.loads(tp.read_text(encoding="utf-8"))
        windows = R.read_jsonl(wpath)
        gtp = gt_dir / f"{seq.replace('/', '_')}.csv"
        gt = R.read_gt_csv(gtp) if gtp.is_file() else None
        runs.append(Run(run.name, name, seq, backend, windows, timing, gt))
    return runs


def summary_rows(runs: list[Run]) -> list[dict]:
    rows = []
    for run in runs:
        fps = run.timing.get("fps", {})
        row = {
            "run": run.dir, "config": run.name, "sequence": run.seq, "backend": run.backend,
            "n_windows": len(run.windows),
            "window_fps": fps.get("window_fps"),
            "effective_frame_fps": fps.get("effective_frame_fps"),
            "valid_outcome": None, "recall@100": None, "precision@100": None,
            "rmse_p": None, "rmse_v": None, "tier": "B",
        }
        if run.gt is not None:
            res = MET.evaluate_windows(run.windows, run.gt, config_name=run.name)
            row.update({
                "valid_outcome": res.valid_outcome,
                "recall@100": res.recall.get(100),
                "precision@100": res.precision.get(100),
                "rmse_p": res.rmse_p, "rmse_v": res.rmse_v,
            })
        rows.append(row)
    return rows


def sensitivity_rows(runs: list[Run]) -> list[dict]:
    out = []
    for run in runs:
        if run.gt is None:
            continue
        for mode in MET.ASSOC_MODES:
            res = MET.evaluate_windows(run.windows, run.gt, assoc_mode=mode, config_name=run.name)
            for r in MET.sensitivity_rows(res):
                r = dict(r)
                r["run"] = run.dir
                r["config"] = run.name
                r["association"] = mode
                out.append({"run": r["run"], "config": r["config"], "association": mode,
                            "radius_px": r["radius_px"], "recall": r["recall"],
                            "precision": r["precision"], "correct": r["correct"], "fp": r["fp"]})
    return out


_WIDTH_BINS = ((0, 3, "tiny<=3"), (4, 8, "small4-8"), (9, 16, "medium9-16"),
               (17, 10 ** 9, "large>16"))


def _width_bin(w):
    if w is None:
        return "unknown"
    for lo, hi, label in _WIDTH_BINS:
        if lo <= w <= hi:
            return label
    return "unknown"


def stratified_rows(runs: list[Run]) -> list[dict]:
    """按目标宽度（GT width_px）分层统计 recall/precision@100。"""
    acc: dict[str, dict] = {}
    for run in runs:
        if run.gt is None:
            continue
        for rec in run.windows:
            n = MET._N_from(rec)
            f = MET.associate_frame(rec.t_start, n, "round_half_up")
            g = run.gt[f] if 0 <= f < len(run.gt) else None
            visible = bool(g is not None and g.visible == 1 and g.cx is not None)
            if not rec.target_present:
                continue
            w = None if g is None else g.width_px
            key = _width_bin(w)
            d = float("inf")
            if visible:
                d = ((rec.est.x - g.cx) ** 2 + (rec.est.y - g.cy) ** 2) ** 0.5
            b = acc.setdefault(key, {"correct": 0, "fp": 0})
            if visible and d <= 100:
                b["correct"] += 1
            else:
                b["fp"] += 1
    rows = []
    for key, b in sorted(acc.items()):
        rows.append({"stratum_width": key, "correct": b["correct"], "fp": b["fp"],
                     "precision@100": (b["correct"] / (b["correct"] + b["fp"])
                                       if (b["correct"] + b["fp"]) else None)})
    return rows


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def write_summary_md(rows: list[dict], path: Path) -> None:
    lines = ["# Summary", "",
             "FPS 为报告项（RTX 5060 Ti 相对分析；绝对 FPS 不上论文对照）。Tier B 协议复现。", "",
             "| run | config | backend | windows | eff FPS | valid% | recall@100 | precision@100 | RMSEp | RMSEv |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| {run} | {config} | {backend} | {n} | {fps} | {v} | {rc} | {pr} | {rp} | {rv} |".format(
            run=r["run"], config=r["config"], backend=r["backend"], n=r["n_windows"],
            fps=_fmt(r["effective_frame_fps"]),
            v=_fmt(None if r["valid_outcome"] is None else r["valid_outcome"] * 100),
            rc=_fmt(None if r["recall@100"] is None else r["recall@100"] * 100),
            pr=_fmt(None if r["precision@100"] is None else r["precision@100"] * 100),
            rp=_fmt(r["rmse_p"]), rv=_fmt(r["rmse_v"])))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_vs_ours(rows: list[dict], path: Path) -> None:
    lines = ["# Paper vs Ours", "",
             "论文目标值来自 PRD §6（Jetson AGX Orin 50W）。本环境为 RTX 5060 Ti，FPS 不做绝对对照。",
             "Tier A 需要作者 artifact（当前未获取）→ 下列 Tier A 列记为 N/A，结论按 **Tier B 协议复现**。", "",
             "| config | metric | paper | ours | tier |",
             "|---|---|---:|---:|---|"]
    by = {r["config"]: r for r in rows}
    for cfg, p in PAPER.items():
        ours = by.get(cfg)
        for key, pk, ok in (("effective_frame_fps", "fps", "effective_frame_fps"),
                            ("valid_outcome", "valid", "valid_outcome"),
                            ("recall@100", "recall", "recall@100"),
                            ("precision@100", "precision", "precision@100")):
            ov = None if ours is None else ours.get(ok)
            if key == "valid_outcome" and ov is not None:
                ov *= 100
            if key in ("recall@100", "precision@100") and ov is not None:
                ov *= 100
            lines.append(f"| {cfg} | {key} | {p[pk]} | {_fmt(ov)} | B (A=N/A) |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="make_tables.py", description="汇总指标表（SPEC §5.8）")
    p.add_argument("--runs", default="outputs/runs")
    p.add_argument("--gt", default="data_derived/gt")
    p.add_argument("--out", default="outputs/metrics")
    args = p.parse_args(argv)
    runs = load_runs(Path(args.runs), Path(args.gt))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    srows = summary_rows(runs)
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0]) if srows else
                           ["run", "config", "sequence", "backend"])
        w.writeheader()
        w.writerows(srows)
    write_summary_md(srows, out / "summary.md")
    write_paper_vs_ours(srows, out / "paper_vs_ours.md")
    sens = sensitivity_rows(runs)
    with (out / "sensitivity.csv").open("w", encoding="utf-8", newline="") as fh:
        cols = ["run", "config", "association", "radius_px", "recall", "precision",
                "correct", "fp"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sens)
    strat = stratified_rows(runs)
    with (out / "stratified.csv").open("w", encoding="utf-8", newline="") as fh:
        cols = ["stratum_width", "correct", "fp", "precision@100"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(strat)
    print(f"tables: {len(runs)} runs -> {out} (summary/sensitivity/paper_vs_ours/stratified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
