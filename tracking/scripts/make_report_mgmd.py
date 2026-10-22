"""scripts/make_report_mgmd.py — MGMD 指标汇总 + 消融表 + SPA2D 对比（SPEC §5.7/§9.3，US-007）。

用法：
    python scripts/make_report_mgmd.py --runs outputs/runs_mgmd --gt data_derived/gt \
        [--spa2d outputs/metrics/summary.csv] [--out outputs/report_mgmd.md] \
        [--metrics-out outputs/metrics_mgmd]

产出：
- <metrics-out>/mgmd_summary.csv  — 每个 run × 半径的 P/R/F1（长表）
- <metrics-out>/mgmd_ablation.csv — 消融四档 × 序列（recall/precision/F1@主半径）
- <out>（report_mgmd.md）        — 主指标表 + 消融表 + SPA2D 对比表 + AP(stretch)

口径（FR-10）：bbox 中心到 GT 点距离 ≤ r（r∈{10,20,50,100}，主指标 100px）判 TP，
逐帧 N=1 记录，每帧至多一个 TP；复用 spa2d.metrics.evaluate_windows 从零复算
（报告数值不依赖 run 内 metrics.json，可与 run 记录互验）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spa2d import metrics as MET  # noqa: E402
from spa2d import records as R  # noqa: E402

RADII = (10, 20, 50, 100)
PRIMARY_RADIUS = 100
TIERS = (("mfe", "MFE"),
         ("mfe,tf", "MFE+TF"),
         ("mfe,tf,lac", "MFE+TF+LAC"),
         ("mfe,tf,lac,lad", "MFE+TF+LAC+LAD（全流水线）"))
SEQ_ORDER = ("D1", "D2", "D3", "D4")


@dataclass
class MgmdRun:
    dir: str
    sequence: str          # 如 D1/cam2
    modules: tuple[str, ...]
    records: list
    gt: list | None
    meta: dict = field(default_factory=dict)

    @property
    def tier_key(self) -> str:
        """规范档位键（按 mfe→tf→lac→lad 顺序；run_meta 中 modules 为字母序）。"""
        order = ("mfe", "tf", "lac", "lad")
        return ",".join(m for m in order if m in self.modules)

    @property
    def seq_short(self) -> str:
        return self.sequence.split("/")[0]


def load_runs(runs_dir: Path, gt_dir: Path) -> list[MgmdRun]:
    """发现 run（读 run_meta.json 的 mgmd.modules 与 sequence.name，不依赖目录命名）。"""
    runs: list[MgmdRun] = []
    if not runs_dir.is_dir():
        return runs
    for run in sorted(runs_dir.iterdir()):
        wpath = run / "windows.jsonl"
        mpath = run / "run_meta.json"
        if not wpath.is_file() or not mpath.is_file():
            continue
        meta = json.loads(mpath.read_text(encoding="utf-8"))
        seq = (meta.get("config", {}).get("snapshot", {}).get("sequence", {})
               or {}).get("name", run.name)
        modules = tuple(meta.get("mgmd", {}).get("modules") or ["mfe"])
        gt_path = gt_dir / f"{seq.replace('/', '_')}.csv"
        gt = R.read_gt_csv(gt_path) if gt_path.is_file() else None
        runs.append(MgmdRun(dir=run.name, sequence=seq, modules=modules,
                            records=R.read_jsonl(wpath), gt=gt, meta=meta))
    return runs


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else float("nan")


def recompute(run: MgmdRun, radii=RADII, primary=PRIMARY_RADIUS):
    """从 windows.jsonl + GT 复算指标（报告不依赖 run 内 metrics.json）。"""
    if run.gt is None:
        return None
    return MET.evaluate_windows(
        run.records, run.gt, radii=radii,
        config_name=run.meta.get("config", {}).get("name", run.dir),
        primary_radius=primary)


def verify_against_metrics_json(run: MgmdRun, runs_dir: Path, res) -> list[str]:
    """复算结果 vs run 内 metrics.json；不一致 → warning 列表（数值可复算复现的自检）。"""
    msgs: list[str] = []
    mpath = runs_dir / run.dir / "metrics.json"
    if not mpath.is_file() or res is None:
        return msgs
    stored = json.loads(mpath.read_text(encoding="utf-8"))
    for key in ("recall", "precision"):
        for r, v in res.__getattribute__(key).items():
            sv = stored.get(key, {}).get(str(r))
            if sv is not None and not math.isclose(v, float(sv), rel_tol=1e-9, abs_tol=1e-12):
                msgs.append(f"{run.dir} {key}@{r}: 复算 {v} != metrics.json {sv}")
    return msgs


def radius_ap(records: list, gt: list, radius: int = PRIMARY_RADIUS) -> float | None:
    """半径匹配口径 AP（stretch）：按选择得分（confidence.pnr）降序扫描，
    101 点插值 AP。每帧至多一个输出的约定保持不变。"""
    pts: list[tuple[float, bool]] = []
    visible_den = sum(1 for g in gt
                      if g.visible == 1 and g.cx is not None and g.cy is not None)
    if visible_den == 0:
        return None
    for rec in records:
        if not rec.target_present or rec.est is None:
            continue
        f = int(rec.assoc_frame)
        g = gt[f] if 0 <= f < len(gt) else None
        tp = bool(g is not None and g.visible == 1 and g.cx is not None
                  and math.hypot(rec.est.x - g.cx, rec.est.y - g.cy) <= radius)
        score = rec.confidence.pnr if rec.confidence is not None else 0.0
        pts.append((score, tp))
    if not pts:
        return None
    pts.sort(key=lambda p: -p[0])
    tp_c = fp_c = 0
    prec_at_recall: list[float] = []
    recalls: list[float] = []
    for _score, tp in pts:
        tp_c += int(tp)
        fp_c += int(not tp)
        recalls.append(tp_c / visible_den)
        prec_at_recall.append(tp_c / (tp_c + fp_c))
    ap = 0.0
    for k in range(101):
        r_thr = k / 100.0
        ap += max((p for p, r in zip(prec_at_recall, recalls) if r >= r_thr),
                  default=0.0)
    return ap / 101.0


def _fmt(v, pct: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v * 100:.2f}" if pct else f"{v:.4f}"


def build_tables(runs: list[MgmdRun]):
    """汇总：summary 长表行 + 消融矩阵 + AP。"""
    summary_rows: list[dict] = []
    ablation: dict[str, dict[str, dict]] = {}   # seq → tier_key → row
    ap_rows: list[dict] = []
    for run in runs:
        res = recompute(run)
        for radius in RADII:
            row = {"run": run.dir, "sequence": run.sequence, "modules": run.tier_key,
                   "radius_px": radius}
            if res is not None:
                p, r = res.precision.get(radius, float("nan")), res.recall.get(radius, float("nan"))
                row.update(precision=p, recall=r, f1=f1(p, r),
                           correct=res.correct.get(radius, 0), fp=res.fp.get(radius, 0),
                           visible_den=res.visible_den)
            summary_rows.append(row)
        if res is not None:
            p100, r100 = res.precision.get(PRIMARY_RADIUS, float("nan")), \
                res.recall.get(PRIMARY_RADIUS, float("nan"))
            ablation.setdefault(run.seq_short, {})[run.tier_key] = {
                "precision": p100, "recall": r100, "f1": f1(p100, r100),
                "fp": res.fp.get(PRIMARY_RADIUS, 0), "visible_den": res.visible_den}
            ap = radius_ap(run.records, run.gt)
            if ap is not None:
                ap_rows.append({"run": run.dir, "sequence": run.sequence,
                                "modules": run.tier_key, "ap_radius_px": PRIMARY_RADIUS,
                                "ap": ap})
    return summary_rows, ablation, ap_rows


def write_csvs(summary_rows, ablation, ap_rows, metrics_out: Path) -> None:
    metrics_out.mkdir(parents=True, exist_ok=True)
    fields = ("run", "sequence", "modules", "radius_px", "precision", "recall", "f1",
              "correct", "fp", "visible_den")
    with (metrics_out / "mgmd_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k) for k in fields})
    ab_rows = []
    for seq, tiers in sorted(ablation.items()):
        for tier_key, label in TIERS:
            cell = tiers.get(tier_key)
            if cell is None:
                continue
            ab_rows.append({"sequence": seq, "tier": label, "modules": tier_key,
                            "precision@100": cell["precision"], "recall@100": cell["recall"],
                            "f1@100": cell["f1"], "fp@100": cell["fp"],
                            "visible_den": cell["visible_den"]})
    with (metrics_out / "mgmd_ablation.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sequence", "tier", "modules", "precision@100",
                                           "recall@100", "f1@100", "fp@100", "visible_den"])
        w.writeheader()
        w.writerows(ab_rows)
    if ap_rows:
        with (metrics_out / "mgmd_ap.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["run", "sequence", "modules",
                                               "ap_radius_px", "ap"])
            w.writeheader()
            w.writerows(ap_rows)


def _md_main_table(summary_rows) -> list[str]:
    lines = ["## 主指标表（P/R/F1 @ 各半径，bbox 中心→点 GT 半径匹配口径）", ""]
    lines.append("| sequence | modules | r=10 P/R/F1 | r=20 P/R/F1 | r=50 P/R/F1 | r=100 P/R/F1 |")
    lines.append("|---|---|---|---|---|---|")
    by = {(r["run"], r["radius_px"]): r for r in summary_rows}
    seen_runs = []
    for r in summary_rows:
        if r["run"] not in seen_runs:
            seen_runs.append(r["run"])
    for run in seen_runs:
        cells = []
        for radius in RADII:
            row = by.get((run, radius), {})
            p, rc, fv = row.get("precision"), row.get("recall"), row.get("f1")
            cells.append(f"{_fmt(p)}/{_fmt(rc)}/{_fmt(fv)}"
                         if p is not None else "N/A")
        base = by.get((run, RADII[0]), {})
        lines.append(f"| {base.get('sequence', '')} | {base.get('modules', '')} "
                     f"| {' | '.join(cells)} |")
    return lines


def _md_ablation_table(ablation) -> list[str]:
    lines = ["## 模块消融表（对齐论文 Table 3 结构；指标为 recall/precision/F1@100px，%）", ""]
    seqs = [s for s in SEQ_ORDER if s in ablation] + \
        sorted(s for s in ablation if s not in SEQ_ORDER)
    for seq in seqs:
        lines += [f"### {seq}", "",
                  "| 模块组合 | recall@100 | precision@100 | F1@100 | FP@100 |",
                  "|---|---:|---:|---:|---:|"]
        for tier_key, label in TIERS:
            cell = ablation[seq].get(tier_key)
            if cell is None:
                lines.append(f"| {label} | — | — | — | — |")
            else:
                lines.append(f"| {label} | {_fmt(cell['recall'])} "
                             f"| {_fmt(cell['precision'])} | {_fmt(cell['f1'])} "
                             f"| {cell['fp']} |")
        lines.append("")
    lines.append("> 口径说明：本表为 MVDT 点 GT 半径匹配 Tier B 口径；与论文 Table 3 "
                 "（ARD-MAV bbox GT，IoU 口径）仅做趋势定性对照，不做数值等价。")
    return lines


def _md_env_config(runs: list[MgmdRun]) -> list[str]:
    """配置与环境快照（可复现性）：每序列的调参后关键字段。"""
    lines = ["## 配置与环境快照（可复现性）", "",
             "| sequence | frame_gap | blur | area_min | OR | overflow | coherence | alg1_d_min | lac_thresh | downscale | LAC权重sha256(前12) |",
             "|---|---:|---:|---:|:--:|:--:|:--:|---:|---:|---:|---|"]
    seen = set()
    for run in runs:
        snap = (run.meta.get("config", {}) or {}).get("snapshot", {}) or {}
        seq = (snap.get("sequence", {}) or {}).get("name", run.sequence)
        if seq in seen or not snap:
            continue
        seen.add(seq)
        m = snap.get("mfe", {}) or {}
        tf = snap.get("tf", {}) or {}
        lac = snap.get("lac", {}) or {}
        rt = snap.get("runtime", {}) or {}
        sha = ((run.meta.get("mgmd", {}) or {}).get("weights", {}) or {}).get("lac", {}) or {}
        sha12 = (sha.get("sha256") or "—")[:12]
        lines.append(
            f"| {seq} | {m.get('frame_gap', '—')} | {m.get('blur_kernel', '—')} | "
            f"{m.get('area_min', '—')} | {m.get('or_aggregation', '—')} | "
            f"{m.get('overflow', '—')} | {m.get('motion_coherence', '—')} | "
            f"{tf.get('alg1_d_min', '—')} | {lac.get('prob_thresh', '—')} | "
            f"{rt.get('downscale', '—')} | {sha12} |")
    lines.append("")
    return lines


def _md_tier_b_section() -> list[str]:
    """域差与 Tier B 声明（SPEC §1.1 / PRD §3 域差声明）。"""
    return [
        "## 域差与 Tier B 声明", "",
        "MGMD 论文面向**运动相机空对空**场景：单应帧对齐消除相机自身运动，MFE 的"
        "运动一致性过滤与 TF 的 Algorithm 1 用于剔除视差/摇摆植被假阳性。本工程评测"
        "数据 MVDT 为**静态地面相机**（单应≈恒等，无视差假阳性）、**点 GT**（非 bbox）、"
        "**单目标**，与论文 ARD-MAV 域存在系统性差异：",
        "",
        "1. **帧差分灵敏度**：MVDT 目标帧间位移中位仅 1.13–7.56 px（50→21 fps 不等），"
        "官方 k=1 两帧差在 D1 上 GT 处峰值 3 < 阈值 5 → 结构性漏检。已按 PRD US-003"
        "「k 可配」引入 `frame_gap`（D11），D1 启用论文口径 OR 聚合（D12）。",
        "2. **小目标形态学**：目标 blob 面积 8–17 px²，低于官方 ARD-MAV 常数 "
        "`area_min=30`；已按序调至 5（D11 同批实测调参）。",
        "3. **运动一致性过滤**：官方逐框 ShiTomasi+LK 一致性为移动相机视差设计，"
        "静态相机下弱纹理目标无角点 → 误杀；MVDT 关闭（D14），假阳性交由 TF+LAC/LAD。",
        "4. **LAC 外观域失效**：官方 Net_best.pth 在 MVDT 上 P(MAV)≈0（GT 裁剪 0.000 vs "
        "背景 0.004，呈反相关），硬阈值 0.5 使全流水线零输出 → MVDT 配 `prob_thresh=0`"
        "（直通/排序），消融表如实反映 LAC 域失效。LAD（YOLOv5s-crop）域内有效"
        "（GT 中心裁剪 8/8 检出，conf≈0.89）。",
        "5. **TF 阈值域适配**：Algorithm 1（ρ/d≥1）针对 ARD-MAV 快目标；MVDT 慢目标 "
        "D≈0.003–0.6，按序下调 `alg1_d_min`（D1/D2=0.003–0.01，D3=0.1，D4=0.002）。",
        "",
        "**Tier B 协议**：以上结论为 MVDT 静态相机 + 点 GT + 半径匹配口径下的复现，"
        "**不外推到 onboard / 运动相机场景**；与论文 Table 3（ARD-MAV bbox GT / IoU 口径）"
        "仅做趋势定性对照，不做数值等价。所有阈值调整记录于 `run_meta.json`（config 快照）"
        "与 `tasks/mgmd_mvdt_reproduction_spec.md`（D11–D14）。",
        "",
    ]


def _run_diag(run: MgmdRun) -> dict:
    """全流水线档逐序列诊断：输出覆盖与命中分解。"""
    import math
    n = len(run.records)
    outputs = [r for r in run.records if r.target_present and r.est is not None]
    vis_den = 0
    correct = fp_out = fp_absent = 0
    dists = []
    gt = run.gt or []
    for r in run.records:
        f = int(r.assoc_frame)
        g = gt[f] if 0 <= f < len(gt) else None
        visible = bool(g is not None and g.visible == 1 and g.cx is not None)
        if visible:
            vis_den += 1
        if r.target_present and r.est is not None:
            if visible:
                d = math.hypot(r.est.x - g.cx, r.est.y - g.cy)
                dists.append(d)
                if d <= PRIMARY_RADIUS:
                    correct += 1
                else:
                    fp_out += 1
            else:
                fp_absent += 1
    return {"sequence": run.sequence, "modules": run.tier_key, "n_frames": n,
            "visible_den": vis_den, "outputs": len(outputs), "correct@100": correct,
            "fp_out_of_radius": fp_out, "fp_on_absent": fp_absent,
            "no_output_frames": n - len(outputs),
            "median_dist_px": (float(np.median(dists)) if dists else None),
            "align_fallback": run.meta.get("mgmd", {}).get("align_fallback_frames")}


def _md_failure_analysis(runs: list[MgmdRun]) -> list[str]:
    """自动失败分析（AC：至少 D3/D4 各一节）。取全流水线档。"""
    full = [r for r in runs if r.tier_key == "mfe,tf,lac,lad" and r.gt is not None]
    lines = ["## 失败案例分析（全流水线档，自动统计）", "",
             "| sequence | 帧数 | 可见帧 | 输出帧 | 命中@100 | 越径FP | 缺席FP | 无输出帧 | 输出中位距(px) | 单应回退帧 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    diags = {}
    for run in sorted(full, key=lambda r: r.sequence):
        d = _run_diag(run)
        diags[d["sequence"]] = d
        md = "—" if d["median_dist_px"] is None else f"{d['median_dist_px']:.1f}"
        lines.append(f"| {d['sequence']} | {d['n_frames']} | {d['visible_den']} | "
                     f"{d['outputs']} | {d['correct@100']} | {d['fp_out_of_radius']} | "
                     f"{d['fp_on_absent']} | {d['no_output_frames']} | {md} | "
                     f"{d['align_fallback'] if d['align_fallback'] is not None else '—'} |")
    lines.append("")
    for seq in ("D3", "D4"):
        d = next((v for k, v in diags.items() if k.startswith(seq)), None)
        if d is None:
            continue
        miss = max(0, d["visible_den"] - d["correct@100"])
        recall = d["correct@100"] / max(d["visible_den"], 1)
        med = d["median_dist_px"]
        med_s = "—" if med is None else f"{med:.1f}px"
        hint = "（D3 背景纹理弱/目标快，回退后仍以恒等补偿差分）" if seq == "D3" else ""
        lines += [f"### {seq}", "",
                  f"- 可见帧 {d['visible_den']}，命中 {d['correct@100']}"
                  f"（recall={recall:.3f}），漏检 {miss} 帧、"
                  f"越径 FP {d['fp_out_of_radius']}、缺席期 FP {d['fp_on_absent']}、"
                  f"无输出 {d['no_output_frames']} 帧。",
                  f"- 单应回退 {d['align_fallback']} 帧{hint}；输出中位距 {med_s}。",
                  ]
    return lines


def _md_targets(summary_rows) -> list[str]:
    """指标达标核验（issue-024 AC）。"""
    best: dict[str, dict] = {}
    for r in summary_rows:
        if r["radius_px"] != PRIMARY_RADIUS or r.get("recall") is None:
            continue
        seq = r["sequence"].split("/")[0]
        cur = best.get(seq)
        if cur is None or (r["recall"] or 0) > (cur["recall"] or 0):
            best[seq] = r
    targets = {"D1": 0.95, "D2": 0.95, "D3": 0.50, "D4": None}
    lines = ["## 指标达标核验（AC）", "",
             "| sequence | 最佳档 recall@100 | 目标 | 结论 |", "|---|---:|---:|---|"]
    for seq in SEQ_ORDER:
        r = best.get(seq)
        got = None if r is None else r["recall"]
        tgt = targets.get(seq)
        tgt_s = "≥95%" if tgt == 0.95 else ("≥50%" if tgt == 0.50 else "跑通出指标")
        ok = "✅ 达标" if (got is not None and (tgt is None or got >= tgt)) else (
            "❌ 未达标" if got is not None else "❌ 无结果")
        lines.append(f"| {seq} | {_fmt(got)} | {tgt_s} | {ok} |")
    lines.append("")
    return lines


def _md_spa2d_table(spa2d_csv: Path, summary_rows) -> list[str]:
    import csv as _csv
    lines = ["## 与 SPA2D 基线对比（@100px，%）", "",
             "| sequence | MGMD recall | MGMD precision | SPA2D recall | SPA2D precision |",
             "|---|---:|---:|---:|---:|"]
    spa2d: dict[str, dict] = {}
    with spa2d_csv.open("r", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            seq = str(row.get("sequence", ""))
            spa2d.setdefault(seq.split("/")[0], row)
    by_seq: dict[str, dict] = {}
    for r in summary_rows:
        if r["radius_px"] == PRIMARY_RADIUS and r["modules"] == "mfe,tf,lac,lad" \
                and r.get("recall") is not None:
            by_seq[r["sequence"].split("/")[0]] = r

    def _spa2d_val(row, key):
        if not row or row.get(key) in (None, ""):
            return None
        return float(row[key])

    for seq in SEQ_ORDER:                       # 仅四条正式序列（排除 smoke 等）
        m, s = by_seq.get(seq), spa2d.get(seq)
        lines.append(f"| {seq} | {_fmt(m['recall']) if m else '—'} "
                     f"| {_fmt(m['precision']) if m else '—'} "
                     f"| {_fmt(_spa2d_val(s, 'recall@100'))} "
                     f"| {_fmt(_spa2d_val(s, 'precision@100'))} |")
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="make_report_mgmd.py",
                                description="MGMD 指标汇总+消融+对比报告（SPEC §5.7/§9.3）")
    p.add_argument("--runs", default="outputs/runs_mgmd")
    p.add_argument("--gt", default="data_derived/gt")
    p.add_argument("--spa2d", default=None, help="SPA2D summary.csv（可选对比）")
    p.add_argument("--out", default="outputs/report_mgmd.md")
    p.add_argument("--metrics-out", default="outputs/metrics_mgmd")
    args = p.parse_args(argv)

    runs_dir, gt_dir = Path(args.runs), Path(args.gt)
    runs = load_runs(runs_dir, gt_dir)
    if not runs:
        print(f"[mgmd] E_DATA_MISSING: 无 run 记录: {runs_dir} → "
              "先运行 scripts/run_mgmd.py 产出 outputs/runs_mgmd/*", file=sys.stderr)
        return 3

    summary_rows, ablation, ap_rows = build_tables(runs)

    mismatches: list[str] = []
    for run in runs:
        res = recompute(run)
        mismatches += verify_against_metrics_json(run, runs_dir, res)
    for m in mismatches:
        warnings.warn(f"[mgmd] 复算不一致: {m}")

    write_csvs(summary_rows, ablation, ap_rows, Path(args.metrics_out))

    lines = ["# MGMD × MVDT 复现报告（指标与消融）", "",
             "> 自动生成：scripts/make_report_mgmd.py；数值全部由 windows.jsonl + GT 复算，",
             "> 可用 `--runs/--gt` 复现。Tier B 协议口径（静态相机、点 GT、半径匹配）。", ""]
    lines += _md_targets(summary_rows) + [""]
    lines += _md_main_table(summary_rows) + [""]
    lines += _md_ablation_table(ablation) + [""]
    lines += _md_tier_b_section()
    lines += _md_failure_analysis(runs) + [""]
    lines += _md_env_config(runs) + [""]
    if ap_rows:
        lines += ["## AP（半径匹配口径，stretch）", "",
                  "| sequence | modules | AP@100 |", "|---|---|---:|"]
        for r in ap_rows:
            lines.append(f"| {r['sequence']} | {r['modules']} | {r['ap']:.4f} |")
        lines.append("")
    if args.spa2d and Path(args.spa2d).is_file():
        lines += _md_spa2d_table(Path(args.spa2d), summary_rows) + [""]
    if mismatches:
        lines += ["## ⚠️ 复算不一致", ""] + [f"- {m}" for m in mismatches]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {len(runs)} runs -> {out}（summary/ablation csv -> {args.metrics_out}）"
          + (f"；⚠️ {len(mismatches)} 处复算不一致" if mismatches else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
