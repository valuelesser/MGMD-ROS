"""spa2d.confidence — 置信度与连续性（SPEC §5.5）。

集中定义全部公式与常数（供状态机/配置复用，避免硬编码重复）：
    ρ            = max(8, ceil(2·width_ref))
    PNR          = (S − μ_bg) / max(σ_bg, 1e-6)，背景 = {M ≥ 1} 去掉半径 ρ 圆盘
    margin 排除半径 = max(2·δ_used, 0.2)
    margin       = (S − S2) / max(S, 1e-6)；无剩余候选 → 1.0 + single_candidate
    r_cont_high  = max(25, 3·width_ref)
    r_cont_low   = max(10, 1.5·width_ref)
    p_pred       = p_prev + v_prev·(c − c_prev)；r_cont = ‖p_center − p_pred‖₂
无 prior → r_cont = NaN、exempt = True。
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from spa2d.records import Continuity

__all__ = [
    "INSUFFICIENT_BG",
    "SIGMA_EPS",
    "MARGIN_MIN_RADIUS",
    "rho_for",
    "cont_radii",
    "margin_exclusion_radius",
    "background_stats",
    "pnr",
    "peak_pnr",
    "velocity_margin",
    "Prior",
    "continuity",
]

INSUFFICIENT_BG = 100      # 背景像素下限
SIGMA_EPS = 1e-6
MARGIN_MIN_RADIUS = 0.2


def rho_for(width_ref: float) -> int:
    """PNR 背景排除圆盘半径：max(8, ceil(2·width_ref))。"""
    return max(8, math.ceil(2.0 * width_ref))


def cont_radii(width_ref: float) -> tuple[float, float]:
    """连续性半径 (r_cont_high, r_cont_low) = (max(25,3·w), max(10,1.5·w))。"""
    return max(25.0, 3.0 * width_ref), max(10.0, 1.5 * width_ref)


def margin_exclusion_radius(delta_used: float) -> float:
    """速度 margin 的排除半径：max(2·δ_used, 0.2)。"""
    return max(2.0 * delta_used, MARGIN_MIN_RADIUS)


def background_stats(A, M, center, rho: int) -> tuple[float, float, int]:
    """返回 (μ_bg, σ_bg, count)：{M≥1} 去掉中心半径 ρ 圆盘，float64 聚合。

    背景像素 < INSUFFICIENT_BG(100) 时发出警告并返回 (0.0, 0.0, count)。
    """
    arr = np.asarray(A, dtype=np.float64)
    msk = np.asarray(M, dtype=np.float64)
    h, w = arr.shape
    cx = int(round(float(center[0])))
    cy = int(round(float(center[1])))
    yy, xx = np.mgrid[0:h, 0:w]
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= float(rho) ** 2
    valid = (msk >= 1.0) & (~disk)
    vals = arr[valid]
    if vals.size < INSUFFICIENT_BG:
        warnings.warn(f"背景像素不足（{vals.size} < {INSUFFICIENT_BG}），PNR=0")
        return 0.0, 0.0, int(vals.size)
    return float(vals.mean()), float(vals.std()), int(vals.size)


def pnr(peak_value: float, mu: float, sigma: float) -> float:
    return float((peak_value - mu) / max(sigma, SIGMA_EPS))


def peak_pnr(A, M, peak_xy, rho: int) -> float:
    """在给定整数峰位置计算 PNR；背景不足时返回 0。"""
    arr = np.asarray(A, dtype=np.float64)
    x = int(round(float(peak_xy[0])))
    y = int(round(float(peak_xy[1])))
    mu, sigma, count = background_stats(A, M, (x, y), rho)
    if count < INSUFFICIENT_BG:
        return 0.0
    return pnr(float(arr[y, x]), mu, sigma)


def velocity_margin(scores, cands, v_best, delta_used: float) -> tuple[float, bool]:
    """速度 margin：排除 v_best 近邻（Chebyshev ≤ max(2δ_used,0.2)）后的次高分比。

    返回 (margin, single_candidate)；无剩余候选 → margin=1.0、single_candidate=True。
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    c = np.asarray(cands, dtype=np.float64).reshape(-1, 2)
    vb = np.asarray(v_best, dtype=np.float64).reshape(2)
    radius = margin_exclusion_radius(delta_used)
    far = np.max(np.abs(c - vb[None, :]), axis=1) > radius
    peak = float(s.max())
    if not np.any(far):
        return 1.0, True
    s2 = float(s[far].max())
    return (peak - s2) / max(peak, SIGMA_EPS), False


@dataclass(frozen=True)
class Prior:
    c_prev: float
    p_prev: np.ndarray
    v_prev: np.ndarray


def continuity(p_center, c: float, width_ref: float,
               prior: Prior | None = None) -> Continuity:
    """连续性残差 r_cont = ‖p_center − p_pred‖₂（无 prior → NaN + exempt）。"""
    high, low = cont_radii(width_ref)
    if prior is None:
        return Continuity(r_cont=None, r_cont_high=high, r_cont_low=low, exempt=True)
    pc = np.asarray(p_center, dtype=np.float64).reshape(2)
    p_pred = np.asarray(prior.p_prev, dtype=np.float64).reshape(2) + \
        np.asarray(prior.v_prev, dtype=np.float64).reshape(2) * (float(c) - float(prior.c_prev))
    r = float(np.linalg.norm(pc - p_pred))
    return Continuity(r_cont=r, r_cont_high=high, r_cont_low=low, exempt=False)
