"""spa2d.peak — 峰值提取、亚像素细化与边缘保护（SPEC §5.4 / §5.3）。

流程（在最优假设累加图 ``A_vbest`` 上）：

1. 整数峰 ``(x0,y0) = argmax``（并列取 ``(y,x)`` 字典序最小）。
2. 若 3×3 邻域 ``M ≥ 1`` 全部成立，且 ``A(x0,y0)`` 为严格最大 → 二维二次曲面
   （可分离抛物面）亚像素细化；``|分母| ≤ 1e-12`` 或 ``|f| > 0.5`` 时该轴回退 0；
   结果 clip 到 ``[0,W-1]×[0,H-1]``。否则使用整数位置。
3. 记录 ``peak_value``、``xy_int``、``xy_sub``、``subpixel_used``、``valid_count``、
   ``edge_candidate``。

边缘保护：``edge_margin = max(2, ceil((N-1)·max(|vx|,|vy|)) + 1)``；
``edge_candidate = peak 在边带内 ∨ M(peak) < N``。

中心帧位置（论文 Eq.9）：``p_center = p0 + (N-1)/2 · v``。
"""
from __future__ import annotations

import math

import numpy as np

from spa2d.records import PeakResult

_DENOM_EPS = 1e-12
_OFFSET_MAX = 0.5


def _axis_offset(left: float, center: float, right: float) -> float:
    denom = left - 2.0 * center + right
    if abs(denom) <= _DENOM_EPS:
        return 0.0
    offset = 0.5 * (left - right) / denom
    if abs(offset) > _OFFSET_MAX:
        return 0.0
    return float(offset)


def subpixel_refine(A: np.ndarray, xy_int: tuple[int, int]) -> tuple[float, float] | None:
    """给定整数峰位置返回亚像素偏移 ``(fx, fy)``；无法细化（越界无法取 3×3）返回 None。

    单轴退化（分母≈0 或 |offset|>0.5）该轴取 0。
    """
    arr = np.asarray(A, dtype=np.float64)
    h, w = arr.shape
    x0, y0 = int(xy_int[0]), int(xy_int[1])
    if not (1 <= x0 < w - 1 and 1 <= y0 < h - 1):
        return None
    fx = _axis_offset(arr[y0, x0 - 1], arr[y0, x0], arr[y0, x0 + 1])
    fy = _axis_offset(arr[y0 - 1, x0], arr[y0, x0], arr[y0 + 1, x0])
    return fx, fy


def _strict_max(A: np.ndarray, x0: int, y0: int) -> bool:
    center = A[y0, x0]
    window = A[y0 - 1:y0 + 2, x0 - 1:x0 + 2]
    return bool(np.count_nonzero(window >= center) == 1)


def extract_peak(A: np.ndarray, M: np.ndarray, N: int, edge_margin: int) -> PeakResult:
    """提取整数峰并条件性亚像素细化（见模块 docstring）。"""
    arr = np.asarray(A, dtype=np.float64)
    msk = np.asarray(M, dtype=np.float64)
    if arr.shape != msk.shape:
        raise ValueError(f"A 与 M 形状不一致: {arr.shape} vs {msk.shape}")
    h, w = arr.shape

    flat = int(np.argmax(arr))
    y0, x0 = divmod(flat, w)  # C 序最小索引 = (y,x) 字典序最小
    peak_value = float(arr[y0, x0])
    valid_count = int(round(float(msk[y0, x0])))

    interior = 1 <= x0 < w - 1 and 1 <= y0 < h - 1
    all_valid = interior and bool(np.all(msk[y0 - 1:y0 + 2, x0 - 1:x0 + 2] >= 1.0))
    refined = interior and all_valid and _strict_max(arr, x0, y0)

    xy_sub: tuple[float, float] | None = None
    if refined:
        offsets = subpixel_refine(arr, (x0, y0))
        if offsets is not None:
            fx, fy = offsets
            sx = min(max(x0 + fx, 0.0), w - 1.0)
            sy = min(max(y0 + fy, 0.0), h - 1.0)
            xy_sub = (float(sx), float(sy))
    subpixel_used = xy_sub is not None
    if xy_sub is None:
        xy_sub = (float(x0), float(y0))

    in_band = (x0 < edge_margin or x0 >= w - edge_margin
               or y0 < edge_margin or y0 >= h - edge_margin)
    edge_candidate = bool(in_band or valid_count < N)

    return PeakResult(
        value=peak_value,
        xy_int=(x0, y0),
        xy_sub=xy_sub,
        subpixel_used=subpixel_used,
        valid_count=valid_count,
        edge_candidate=edge_candidate,
    )


def edge_margin_for(N: int, v) -> int:
    """edge_margin = max(2, ceil((N-1)·max(|vx|,|vy|)) + 1)（SPEC §5.3）。"""
    speed = max(abs(float(v[0])), abs(float(v[1])))
    return max(2, math.ceil((N - 1) * speed) + 1)


def center_frame_position(p0, v, N: int) -> tuple[float, float]:
    """中心帧位置 p_center = p0 + (N-1)/2 · v（论文 Eq.9）。"""
    factor = (N - 1) / 2.0
    return (float(p0[0]) + factor * float(v[0]), float(p0[1]) + factor * float(v[1]))


def center_frame_position_accel(p0, v, a, N: int) -> tuple[float, float]:
    """中心帧位置（含加速度）：p0 + (N-1)/2·v + ½·((N-1)/2)²·a。"""
    f = (N - 1) / 2.0
    return (float(p0[0]) + f * float(v[0]) + 0.5 * f * f * float(a[0]),
            float(p0[1]) + f * float(v[1]) + 0.5 * f * f * float(a[1]))
