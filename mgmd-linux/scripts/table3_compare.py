"""scripts/table3_compare.py — 论文 Table 3 与本工程对照（SPEC §10.9）。

论文 Table 3（MGMD 组件消融）只有 6 档，本工程补 2 档（MFE 单跑、带 TF 的全链路），
共 8 档，全部在**同一测试集**（15 视频 / 28,307 帧，官方划分）上跑出。

关键差异必须声明：
  论文 FPS 是「包含预处理」还是「纯推理」未说明；本工程 FPS 用 ms_per_frame 反算，
  ms_per_frame = (读帧 + MFE + TF + LAC + LAD) / 实际处理帧数，**包含预处理**。
  论文未给 F1 的 AP，本工程额外报 AP 便于核对。

用法：
    source scripts/env_setup.sh
    $PY scripts/table3_compare.py                 # 读 outputs/logs/testset/t3_*/_summary.json
    $PY scripts/table3_compare.py --md            # 输出 markdown 表格
    $PY scripts/table3_compare.py --csv out.csv   # 另存 csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mgmd import config as C

# 论文 Table 3 原始数据（paper/paper_mgmd_zh.md §4.4，表3）
PAPER = {
    "mfe,lad":         dict(P=0.60, R=0.58, F1=0.59, FPS=16),
    "mfe,lac":         dict(P=0.40, R=0.39, F1=0.39, FPS=34),
    "mfe,lac,lad":     dict(P=0.79, R=0.63, F1=0.70, FPS=23),
    "mfe,tf,lad":      dict(P=0.79, R=0.58, F1=0.67, FPS=22),
    "mfe,tf,lac":      dict(P=0.45, R=0.37, F1=0.41, FPS=32),
    "mfe,tf,lac,lad":  dict(P=0.84, R=0.59, F1=0.69, FPS=28),
}

# 本地结果目录 -> (显示名, 是否在论文 Table 3 内)
ARMS = [
    ("k3_mfe",         "MFE",              "mfe",                 False),
    ("k3_lac_lad",     "MFE+LAC+LAD",      "mfe,lac,lad",         True),
    ("k3_full",        "MFE+TF+LAC+LAD",   "mfe,tf,lac,lad",      True),
]


def load_arm(tag: str) -> dict | None:
    p = C.LOGS / "testset" / tag / "_summary.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    ms = None
    pv = d.get("per_video") or []
    if pv:
        tot = sum(v.get("elapsed_s", 0) for v in pv)
        nfr = sum(v.get("frames_run", 0) for v in pv)
        ms = tot / nfr * 1000 if nfr else None
    D = ms / 1000 if ms else None
    return {
        "P": d.get("precision"), "R": d.get("recall"), "F1": d.get("f1"),
        "AP": d.get("ap_allpoint"),
        "ms": ms, "fps": (1.0 / D) if D else None,
        "elapsed_min": d.get("elapsed_min"), "frames": d.get("frames"),
        "n_videos": d.get("n_videos"),
    }


def fmt(v, nd=3, pct=False):
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = []
    for tag, name, modkey, in_paper in ARMS:
        local = load_arm(tag)
        paper = PAPER.get(modkey) if in_paper else None
        rows.append(dict(tag=tag, name=name, in_paper=in_paper,
                         local=local, paper=paper))

    # ---- Markdown ----
    lines = []
    lines.append("# 论文 Table 3 vs 本工程（同测试集对照）")
    lines.append("")
    lines.append("测试集：ARD-MAV 官方 15 视频 / 28,307 帧（GLAD README 划分）")
    lines.append("口径：IoU≥0.25 + 匈牙利一对一；AP 为 all-point")
    lines.append("")
    lines.append("| 模块组合 | 论文 P | 论文 R | 论文 F1 | 论文 FPS | "
                 "本工程 P | 本工程 R | 本工程 F1 | 本工程 AP | 本工程 FPS | ΔF1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        p, l = r["paper"], r["local"]
        d_f1 = (l["F1"] - p["F1"]) if (l and p and l["F1"] is not None) else None
        name = r["name"] + ("" if r["in_paper"] else " †")
        if p:
            pv = (f"{p['P']:.2f}", f"{p['R']:.2f}", f"{p['F1']:.2f}", str(p["FPS"]))
        else:
            pv = ("—", "—", "—", "—")
        if l:
            lv = (fmt(l["P"]), fmt(l["R"]), fmt(l["F1"]), fmt(l["AP"]),
                  fmt(l["fps"], 1))
        else:
            lv = ("…", "…", "…", "…", "…")
        lines.append(f"| {name} | {pv[0]} | {pv[1]} | {pv[2]} | {pv[3]} | "
                     f"{lv[0]} | {lv[1]} | {lv[2]} | {lv[3]} | {lv[4]} | "
                     f"{'+' if (d_f1 is not None and d_f1>=0) else ''}"
                     f"{fmt(d_f1, 3) if d_f1 is not None else '—'} |")
    lines.append("")
    lines.append("† 论文 Table 3 未包含此档，本工程补充，用于观察 TF 的独立作用。")
    lines.append("")

    # 趋势一致性检查
    lines.append("## 趋势一致性")
    lines.append("")
    checks = [
        ("MFE+LAC+LAD > MFE（外观两级有增益）", "mfe,lac,lad", "mfe", "F1"),
        ("MFE+TF+LAC+LAD > MFE（全链路有增益）", "mfe,tf,lac,lad", "mfe", "F1"),
        ("论文称加 TF 有增益 → 本工程是否成立", "mfe,tf,lac,lad", "mfe,lac,lad", "F1"),
    ]
    by_tag = {r["tag"]: r["local"] for r in rows if r["local"]}
    tag_by_mod = {k: k.replace(",", "_").join(["t3_", ""]) for k in PAPER}
    tag_map = {
        "mfe,lac,lad": "k3_lac_lad", "mfe,tf,lac,lad": "k3_full",
    }
    for label, a, b, key in checks[:3]:
        ta, tb = tag_map.get(a), tag_map.get(b)
        la, lb = by_tag.get(ta), by_tag.get(tb)
        if la and lb and la[key] is not None and lb[key] is not None:
            ok = la[key] > lb[key]
            lines.append(f"- {'✅' if ok else '❌'} {label}：{la[key]:.3f} vs {lb[key]:.3f}")
        else:
            lines.append(f"- ⏳ {label}：待跑完")
    # 极值检查
    have = [r for r in rows if r["local"]]
    if len(have) >= 6:
        best_p = max(have, key=lambda r: r["local"]["P"] or 0)
        best_fps = max(have, key=lambda r: r["local"]["fps"] or 0)
        lines.append(f"- 本工程 P 最高：{best_p['name']} ({best_p['local']['P']:.3f})，"
                     f"论文为 MFE+TF+LAC+LAD (0.84)")
        lines.append(f"- 本工程 FPS 最高：{best_fps['name']} ({best_fps['local']['fps']:.1f})，"
                     f"论文为 MFE+LAC (34)")
    lines.append("")
    lines.append("## 口径差异（必须声明）")
    lines.append("")
    lines.append("| 项 | 论文 | 本工程 |")
    lines.append("|---|---|---|")
    lines.append("| FPS 是否含预处理 | **未说明** | 含（读帧+MFE+TF+LAC+LAD 全耗时） |")
    lines.append("| 硬件 | 未说明 | RTX 3050 Laptop 4GB (sm_86), CUDA |")
    lines.append("| AP | Table 3 未报 | all-point AP，额外提供 |")
    lines.append("| 单张图耗时 vs 全视频 | 未说明 | 全视频端到端（含解码） |")

    out = "\n".join(lines)
    print(out)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["modules", "paper_P", "paper_R", "paper_F1", "paper_FPS",
                        "local_P", "local_R", "local_F1", "local_AP", "local_FPS"])
            for r in rows:
                p, l = r["paper"], r["local"]
                w.writerow([r["name"],
                            p["P"] if p else "", p["R"] if p else "",
                            p["F1"] if p else "", p["FPS"] if p else "",
                            f"{l['P']:.4f}" if l else "", f"{l['R']:.4f}" if l else "",
                            f"{l['F1']:.4f}" if l else "", f"{l['AP']:.4f}" if l else "",
                            f"{l['fps']:.2f}" if l and l['fps'] else ""])
        print(f"\ncsv -> {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
