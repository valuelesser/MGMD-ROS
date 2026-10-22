"""spa2d.preprocess — 暗目标增强预处理（SPEC §5.1，对齐论文 Eq.6）。

步骤（严格按序，禁止任何额外增强）：
    1. 灰度化：BGR→GRAY 或 RGB→GRAY（2D 输入视为已灰度），转 float32 [0,255]。
    2. 形态学闭运算背景：``C = morphologyEx(G, MORPH_CLOSE, B)``。
    3. 暗目标增强：``D = max(C - G, 0)``。
    4. 归一化：``I = D / 255.0``（float32，范围 [0,1]）。

结构元 ``B`` 为椭圆（``MORPH_ELLIPSE``），尺寸 k ∈ {3,5,7,9,11}（默认 5，来自冻结阈值）。

边界行为：闭运算使用 **OpenCV 默认边界**（``morphologyEx`` 的默认 ``borderType`` 与
``morphologyDefaultBorderValue()``），本模块不做自定义填充。累加阶段的双线性重采样
边界是另一个约定，见 SPEC §5.2（BORDER_CONSTANT=0）与 ``spa2d.accumulate``。

明确禁止：全图直方图均衡、CLAHE、任何学习式/滤波式增强（PRD §5.1 明令）。
"""
from __future__ import annotations

import cv2
import numpy as np

ALLOWED_KERNEL_SIZES = (3, 5, 7, 9, 11)
DEFAULT_KERNEL_SIZE = 5
_KERNEL_SHAPES = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}
_COLOR_CODES = {"BGR": cv2.COLOR_BGR2GRAY, "RGB": cv2.COLOR_RGB2GRAY}


def build_kernel(k: int = DEFAULT_KERNEL_SIZE, shape: str = "ellipse") -> np.ndarray:
    """返回 OpenCV 结构元。k ∈ {3,5,7,9,11}，shape ∈ {ellipse,rect,cross}。"""
    if int(k) not in ALLOWED_KERNEL_SIZES:
        raise ValueError(f"kernel 尺寸必须是 {ALLOWED_KERNEL_SIZES} 之一，实际 {k}")
    if shape not in _KERNEL_SHAPES:
        raise ValueError(f"kernel 形状必须是 {tuple(_KERNEL_SHAPES)} 之一，实际 {shape!r}")
    return cv2.getStructuringElement(_KERNEL_SHAPES[shape], (int(k), int(k)))


def to_gray(frame: np.ndarray, color_order: str = "BGR") -> np.ndarray:
    """灰度化并转 float32（范围 [0,255]）。2D 输入按已灰度处理。"""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"输入必须是 2D 灰度或 3D (H,W,1/3/4)，实际 shape={arr.shape}")
    if arr.shape[2] == 1:
        return arr[:, :, 0].astype(np.float32)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    order = color_order.upper()
    if order == "GRAY":
        return arr[:, :, 0].astype(np.float32)
    if order not in _COLOR_CODES:
        raise ValueError(f"color_order 必须是 BGR/RGB/GRAY 之一，实际 {color_order!r}")
    return cv2.cvtColor(arr, _COLOR_CODES[order]).astype(np.float32)


def preprocess_frame(frame: np.ndarray, kernel: np.ndarray,
                     color_order: str = "BGR") -> np.ndarray:
    """暗目标增强：灰度化 → MORPH_CLOSE → max(C−G,0) → /255.0。

    返回 float32 数组，shape=(H,W)，范围 [0,1]。``color_order`` 记录输入色彩顺序。
    """
    gray = to_gray(frame, color_order=color_order)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    enhanced = np.maximum(background - gray, 0.0)
    return (enhanced / 255.0).astype(np.float32)


def matched_filter_frame(frame: np.ndarray, sigma: float,
                         color_order: str = "BGR") -> np.ndarray:
    """few-pixel 暗斑匹配滤波：DoG（宽-窄高斯）于灰度图，暗小斑为正响应。

    ``sigma`` 为尺度先验（匹配目标等效半径）。大尺度暗结构被宽-窄差分抑制；
    用于区分小目标与大杂波（SPEC §5.1 的补充候选表示，仅在选择候选表示时启用）。
    """
    gray = to_gray(frame, color_order=color_order)
    narrow = cv2.GaussianBlur(gray, (0, 0), float(sigma))
    wide = cv2.GaussianBlur(gray, (0, 0), float(sigma) * 1.6)
    return (wide - narrow).astype(np.float32)


def matched_filter_multiscale(frame: np.ndarray, sigmas,
                              color_order: str = "BGR") -> np.ndarray:
    """多尺度 DoG（尺度自适应）：对同一像素取各 σ 的最大响应。

    ``sigmas`` 应限制在目标尺度附近（避免大尺度杂波取胜）。
    """
    gray = to_gray(frame, color_order=color_order)
    best = None
    for s in sigmas:
        d = (cv2.GaussianBlur(gray, (0, 0), float(s))
             - cv2.GaussianBlur(gray, (0, 0), float(s) * 1.6))
        best = d if best is None else np.maximum(best, d)
    return best.astype(np.float32)
