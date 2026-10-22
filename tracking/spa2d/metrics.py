"""spa2d.metrics — 窗口级指标（SPEC §5.8）。

- 关联：associate_frame（round_half_up / floor）。
- evaluate_windows：recall/precision@{radii}px、valid_outcome、RMSEp、RMSEv（仅 correct@primary）、
  visible_den、valid_absent、reason 分布、FP 来源、all-accepted-visible 诊断 RMSE。
- 敏感性：distance_distribution（d/max(width_ref,1) 的 p50/p90/p95）、repeat_ranges。
- 写出：write_metrics_json / sensitivity_rows。
"""
from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from spa2d.records import GTFrame, MetricsResult

ASSOC_MODES = ("round_half_up", "floor")
DEFAULT_RADII = (10, 20, 50, 100)
PRIMARY_RADIUS = 100
FP_SOURCES = ("insect", "bird", "cloud", "clutter", "edge", "reacq", "unknown")


def associate_frame(t_start: int, N: int, mode: str = "round_half_up") -> int:
    center = t_start + (N - 1) / 2.0
    if mode == "round_half_up":
        return int(math.floor(center + 0.5))
    if mode == "floor":
        return int(math.floor(center))
    raise ValueError(f"未知关联模式: {mode!r}（可选 {ASSOC_MODES}）")


def gt_velocity(gt: Sequence[GTFrame]) -> np.ndarray:
    """GT 速度：中心差分；端点单侧；不可见帧为 NaN（PRD §4.4）。"""
    n = len(gt)
    pos = np.full((n, 2), np.nan, dtype=np.float64)
    for i, g in enumerate(gt):
        if g.visible == 1 and g.cx is not None and g.cy is not None:
            pos[i] = (g.cx, g.cy)
    vel = np.full((n, 2), np.nan, dtype=np.float64)
    for i in range(n):
        if np.isnan(pos[i]).any():
            continue
        if i == 0:
            if n > 1 and not np.isnan(pos[1]).any():
                vel[0] = pos[1] - pos[0]
        elif i == n - 1:
            if not np.isnan(pos[i - 1]).any():
                vel[i] = pos[i] - pos[i - 1]
        elif not np.isnan(pos[i - 1]).any() and not np.isnan(pos[i + 1]).any():
            vel[i] = (pos[i + 1] - pos[i - 1]) / 2.0
    return vel


def _N_from(rec) -> int:
    return int(round(2.0 * (rec.t_center - rec.t_start) + 1.0))


def _gt_at(gt, f):
    return gt[f] if 0 <= f < len(gt) else None


def _fp_source(rec, overlay: dict | None) -> str:
    if overlay and rec.window_index in overlay:
        src = str(overlay[rec.window_index])
        return src if src in FP_SOURCES else "unknown"
    if rec.flags.edge:
        return "edge"
    if rec.search_mode == "global" or rec.state_in in ("REACQ", "LOST", "OUT_OF_FRAME") \
            or rec.reason == "NO_PRIOR_GLOBAL":
        return "reacq"
    return "unknown"


def evaluate_windows(records, gt, radii=DEFAULT_RADII, assoc_mode: str = "round_half_up",
                     config_name: str = "", width_ref: float | None = None,
                     primary_radius: int = PRIMARY_RADIUS,
                     fp_sources: dict | None = None) -> MetricsResult:
    """按 §5.8 计算窗口级指标。fp_sources 为人工标注覆盖（window_index→source），不改变指标。"""
    radii = list(radii)
    correct = {R: 0 for R in radii}
    fp = {R: 0 for R in radii}
    visible_den = 0
    valid_absent = 0
    total = len(records)
    gtv = gt_velocity(gt)
    err_sq: list[float] = []
    vel_sq: list[float] = []
    all_visible_sq: list[float] = []
    reason_counts: Counter = Counter()
    fp_src: Counter = Counter()

    for rec in records:
        reason_counts[rec.reason] += 1
        n = _N_from(rec)
        f = associate_frame(rec.t_start, n, assoc_mode)
        g = _gt_at(gt, f)
        visible = bool(g is not None and g.visible == 1
                       and g.cx is not None and g.cy is not None)
        if visible:
            visible_den += 1
        if rec.target_present:
            if visible:
                d = math.hypot(rec.est.x - g.cx, rec.est.y - g.cy)
                for R in radii:
                    if d <= R:
                        correct[R] += 1
                    else:
                        fp[R] += 1
                all_visible_sq.append(d * d)
                if d <= primary_radius:
                    err_sq.append(d * d)
                    if 0 <= f < len(gtv) and not np.isnan(gtv[f]).any():
                        dv = math.hypot(rec.est.vx - gtv[f, 0], rec.est.vy - gtv[f, 1])
                        vel_sq.append(dv * dv)
            else:
                for R in radii:
                    fp[R] += 1
                fp_src[_fp_source(rec, fp_sources)] += 1
        else:
            if g is None or g.visible == 0 or g.out_of_frame == 1:
                valid_absent += 1

    def _ratio(a, b):
        return float(a / b) if b > 0 else float("nan")

    if visible_den == 0:
        warnings.warn("visible_den=0：recall 未定义（NaN）")
    recall = {R: _ratio(correct[R], visible_den) for R in radii}
    precision = {R: _ratio(correct[R], correct[R] + fp[R]) for R in radii}
    valid_outcome = _ratio(correct[primary_radius] + valid_absent, total) if total else float("nan")
    rmse_p = math.sqrt(sum(err_sq) / len(err_sq)) if err_sq else None
    rmse_v = math.sqrt(sum(vel_sq) / len(vel_sq)) if vel_sq else None
    diag = math.sqrt(sum(all_visible_sq) / len(all_visible_sq)) if all_visible_sq else None

    return MetricsResult(
        config_name=config_name, association=assoc_mode, radii_px=tuple(radii),
        recall=recall, precision=precision, correct=correct, fp=fp,
        valid_outcome=valid_outcome, rmse_p=rmse_p, rmse_v=rmse_v,
        visible_den=visible_den, valid_absent=valid_absent, total_window_outputs=total,
        diagnostic_rmse_all_visible=diag, reason_counts=dict(reason_counts),
        fp_sources=dict(fp_src))


def evaluate_both_modes(records, gt, radii=DEFAULT_RADII, config_name="",
                        width_ref=None) -> dict[str, MetricsResult]:
    return {mode: evaluate_windows(records, gt, radii, mode, config_name, width_ref)
            for mode in ASSOC_MODES}


def distance_distribution(records, gt, assoc_mode="round_half_up",
                          width_ref: float = 1.0) -> dict:
    """已接受且 GT 可见窗口的距离 d/max(width_ref,1) 分布（p50/p90/p95）。"""
    denom = max(float(width_ref), 1.0)
    vals = []
    for rec in records:
        if not rec.target_present:
            continue
        f = associate_frame(rec.t_start, _N_from(rec), assoc_mode)
        g = _gt_at(gt, f)
        if g is None or g.visible != 1 or g.cx is None or g.cy is None:
            continue
        vals.append(math.hypot(rec.est.x - g.cx, rec.est.y - g.cy) / denom)
    if not vals:
        return {"n": 0, "p50": None, "p90": None, "p95": None}
    arr = np.asarray(vals)
    return {"n": len(vals), "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95))}


def repeat_ranges(results: Sequence[MetricsResult]) -> dict:
    """多次重复运行的指标极差（与 run_sequence --repeat 联动）。"""
    def rng(fn):
        vals = [fn(r) for r in results]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not vals:
            return {"min": None, "max": None, "range": None}
        return {"min": float(min(vals)), "max": float(max(vals)),
                "range": float(max(vals) - min(vals))}
    return {
        "n_runs": len(results),
        "recall@100": rng(lambda r: r.recall.get(100)),
        "precision@100": rng(lambda r: r.precision.get(100)),
        "valid_outcome": rng(lambda r: r.valid_outcome),
        "rmse_p": rng(lambda r: r.rmse_p),
        "rmse_v": rng(lambda r: r.rmse_v),
    }


def sensitivity_rows(result: MetricsResult) -> list[dict]:
    return [{"radius_px": R, "recall": result.recall.get(R),
             "precision": result.precision.get(R), "correct": result.correct.get(R),
             "fp": result.fp.get(R)} for R in result.radii_px]


def write_metrics_json(result: MetricsResult, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False),
                 encoding="utf-8")


def load_metrics_json(path) -> MetricsResult:
    return MetricsResult.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
