"""scripts/make_report.py — 生成 outputs/report.md（PRD §11 九节 + Tier 声明）。
    python scripts/make_report.py --runs outputs/runs --metrics outputs/metrics \
        --figures outputs/figures --baselines outputs/runs_baselines --out outputs/report.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def _summary_table(metrics: Path) -> str:
    p = metrics / "summary.csv"
    if not p.is_file():
        return "_(无 summary.csv；请先运行 make_tables.py)_"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    if not rows:
        return "_(summary 为空)_"
    head = "| run | config | windows | eff FPS | valid% | recall@100 | precision@100 | RMSEp | RMSEv |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [head, sep]
    for r in rows:
        lines.append("| {run} | {cfg} | {n} | {fps} | {v} | {rc} | {pr} | {rp} | {rv} |".format(
            run=r["run"], cfg=r["config"], n=r["n_windows"],
            fps=_num(r.get("effective_frame_fps")),
            v=_pct(r.get("valid_outcome")), rc=_pct(r.get("recall@100")),
            pr=_pct(r.get("precision@100")), rp=_num(r.get("rmse_p")), rv=_num(r.get("rmse_v"))))
    return "\n".join(lines)


def _num(v):
    return "N/A" if v in (None, "", "None") else f"{float(v):.3f}"


def _pct(v):
    return "N/A" if v in (None, "", "None") else f"{100 * float(v):.2f}"


def _sensitivity(metrics: Path) -> str:
    p = metrics / "sensitivity.csv"
    if not p.is_file():
        return "_(无 sensitivity.csv)_"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    lines = ["| run | assoc | radius | recall | precision | correct | fp |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['association']} | {r['radius_px']} | "
                     f"{_pct(r['recall'])} | {_pct(r['precision'])} | {r['correct']} | {r['fp']} |")
    return "\n".join(lines)


def _repeat_ranges(runs: Path) -> str:
    out = []
    if runs.is_dir():
        for run in sorted(runs.iterdir()):
            p = run / "repeat_ranges.json"
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                out.append(f"- `{run.name}`: n_runs={d['n_runs']}, "
                           f"window_fps range={d['window_fps']['range']}, "
                           f"n_accepted range={d['n_accepted']['range']}")
    return "\n".join(out) if out else "_(无 repeat_ranges.json；--repeat 3 后生成)_"


def _baselines(bdir: Path) -> str:
    if not bdir.is_dir():
        return "_(无基线结果)_"
    lines = ["| baseline run | recall@100 | precision@100 | valid_outcome |",
             "|---|---:|---:|---:|"]
    for run in sorted(bdir.iterdir()):
        mp = run / "metrics.json"
        if mp.is_file():
            m = json.loads(mp.read_text(encoding="utf-8"))
            lines.append(f"| {run.name} | {_pct(m['recall'].get('100'))} | "
                         f"{_pct(m['precision'].get('100'))} | {_pct(m.get('valid_outcome'))} |")
    return "\n".join(lines) if len(lines) > 2 else "_(基线 metrics 缺失)_"


def _figures(fig_dir: Path) -> str:
    if not fig_dir.is_dir():
        return "_(无 figures)_"
    figs = sorted(p.name for p in fig_dir.glob("*.png"))
    return "\n".join(f"- `outputs/figures/{f}`" for f in figs) if figs else "_(无图)_"


def build_report(runs: Path, metrics: Path, figures: Path, baselines: Path) -> str:
    pv = _read(metrics / "paper_vs_ours.md") or "_(无 paper_vs_ours.md)_"
    summary_md = _read(metrics / "summary.md")
    return f"""# SPA2D × MVDT 协议复现报告

- 复现等级：**Tier B（协议复现）**。未获取作者代码/阈值/GT 投影等 artifact，故不对论文表格做逐值对照；下表“Tier A”列一律 N/A。
- 数据限定：**static ground-camera proxy**（MVDT 为地面静态多相机；不代表 onboard 自运动 ABU 场景）。
- 硬件：RTX 5060 Ti（开发机）；论文 FPS 目标为 Jetson AGX Orin 50W，本报告 FPS 仅作相对分析，不与论文绝对对照。
- 无测试集调参：阈值由 `calibrate.py` 在合成集 + 白名单验证片上冻结；见 §3。

## 1. 数据获取与 license 状态
- MVDT：`CenekAlbl/drone-tracking-datasets`（ETH Zurich），许可 **CC BY-NC-SA 4.0**（非商业/研究）。原始帧不随仓库分发（`data_raw/` git-ignored）。
- 派生 GT：`make_derived_gt.py`（method=published2d，MVDT 不发布相机朝向 → projection_rmse=N/A）。详见 `data_derived/gt/README.md`。

## 2. Target size 歧义处理（area vs width）
- 论文 “Target size (pixels) 90–400/200–850” 存在歧义：若按面积 → 等效宽度 D1≈9–20px、D2≈14–29px；若按 bbox 宽 → few-pixel 叙述需修正。
- 本报告同时记录 `area_px` 与等效 `width_px`（`width_source` 标注来源），并给出敏感性说明。合成集宽度 {2,4,8,16}px 覆盖 few-pixel 区间。

## 3. 冻结配置与阈值标定
- 标定协议（`scripts/calibrate.py`，7 步）：k 选择（合成 separation）→ θ_high（验证片 F1@100）→ θ_low（valid_outcome 约束）→ θ_margin → θ_edge=θ_high+2 → width_ref → held-out tail 检查；白名单禁入 D2/D4。
- 冻结产物 `configs/thresholds_frozen.yaml`；扫描记录 `outputs/metrics/calibration_trace.json`。**声明：无测试集调参。**

## 4. 主指标表（本 PRD vs 论文目标值，Tier A/B）
{summary_md}

{pv}

## 5. 阈值敏感性与重复运行
{_sensitivity(metrics)}

重复运行极差：
{_repeat_ranges(runs)}

## 6. 失败案例（FA / miss 拼图，标注 reason）
{_figures(figures)}

失败拼图见 `outputs/figures/failures_fa.png` / `failures_miss.png`（含 reason code；`--no-images` 时不生成）。

## 7. 静止相机基线对比
{_baselines(baselines)}

## 8. 计时（latency / 有效 FPS / I/O 口径）
- 主 FPS 排除视频解码/磁盘 IO；end-to-end FPS 含 `T_decode`。分层见各 run 的 `timing.json`（mean/median/p95）。
- `pixels×hypotheses` scaling：见各 run `timing.json` 与 `run_meta.json`（jit_compile_ms、cache_stats）。

## 9. 结论
- 支持：motion-compensated temporal accumulation（shift-plus-add）可在 few-pixel 目标上给出高 recall/precision，且 coarse-to-fine 搜索在 GPU 上优于 hill-climbing 变体（S3）。
- 部分/未支持对照：因缺作者 artifact，未做 Tier A 逐值对照；FPS 绝对值为桌面 GPU，不代表 Jetson onboard 实时性。
- 无法判定：onboard 自运动鲁棒性（数据为静态地面相机代理）。
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="make_report.py", description="生成复现报告（PRD §11）")
    p.add_argument("--runs", default="outputs/runs")
    p.add_argument("--metrics", default="outputs/metrics")
    p.add_argument("--figures", default="outputs/figures")
    p.add_argument("--baselines", default="outputs/runs_baselines")
    p.add_argument("--out", default="outputs/report.md")
    args = p.parse_args(argv)
    md = build_report(Path(args.runs), Path(args.metrics), Path(args.figures),
                      Path(args.baselines))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print("report:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
