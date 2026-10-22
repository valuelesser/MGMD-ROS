"""mgmd.align — 网格关键点 LK + RANSAC 单应帧对齐（SPEC §5.1）。

移植官方 Functions.motion_compensate：网格间距 grid_spacing=(96,72)，点位于网格中心；
LK winSize 15 / maxLevel 3 / criteria (EPS|COUNT, 30, 0.01)；跟踪位移 >max_flow_px(50)
丢弃；有效点 <min_inliers(9) 或 findHomography/求逆失败 → 回退恒等矩阵并记 warning
（FR-3，不崩溃）。补偿用 WARP_INVERSE_MAP；掩膜 = 图像四角经 inv(H) 变换后的无效区。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np

from mgmd.config import MfeConfig

IDENTITY_H = np.array([[0.999, 0.0, 0.0], [0.0, 0.999, 0.0], [0.0, 0.0, 1.0]],
                      dtype=np.float64)


@dataclass
class AlignResult:
    compensated: np.ndarray   # Î_{t-1}（对齐到当前帧坐标系的上一帧灰度图）
    mask: np.ndarray          # 无效边缘区（255=无效，差分后扣除）
    homography: np.ndarray    # 3×3 单应（回退时为近似恒等）
    inliers: int              # 有效跟踪点数
    fallback: bool            # True=回退恒等（已发 warning）


def grid_points(width: int, height: int, spacing: tuple[int, int]) -> np.ndarray:
    """网格关键点（间距 grid_spacing，取网格中心；官方 Functions.motion_compensate 口径）。"""
    gw, gh = spacing
    nw, nh = int(width / gw - 1), int(height / gh - 1)
    pts = [(np.float32(i * gw + gw / 2.0), np.float32(j * gh + gh / 2.0))
           for i in range(max(nw, 0)) for j in range(max(nh, 0))]
    if not pts:
        return np.zeros((0, 1, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def _fallback(prev_gray: np.ndarray, reason: str, inliers: int) -> AlignResult:
    warnings.warn(f"[mgmd] 单应估计失败回退恒等（{reason}）", UserWarning, stacklevel=2)
    h, w = prev_gray.shape[:2]
    return AlignResult(compensated=prev_gray.copy(),
                       mask=np.zeros((h, w), dtype=np.uint8),
                       homography=IDENTITY_H.copy(), inliers=inliers, fallback=True)


def align(prev_gray: np.ndarray, cur_gray: np.ndarray, cfg: MfeConfig) -> AlignResult:
    """网格 LK + RANSAC 单应；失败回退恒等矩阵并记 warning（FR-3）。"""
    h, w = cur_gray.shape[:2]
    lk_params = dict(winSize=(cfg.lk_win_size, cfg.lk_win_size),
                     maxLevel=cfg.lk_max_level,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    pts_prev = grid_points(w, h, cfg.grid_spacing)
    if len(pts_prev) < cfg.min_inliers:
        return _fallback(prev_gray, f"网格点数 {len(pts_prev)} < min_inliers", 0)

    pts_cur, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts_prev, None,
                                                 **lk_params)
    if pts_cur is None or st is None:
        return _fallback(prev_gray, "LK 跟踪失败", 0)

    good_new = pts_cur[st == 1]
    good_old = pts_prev[st == 1]

    points_new: list[np.ndarray] = []
    points_old: list[np.ndarray] = []
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        if np.hypot(a - c, b - d) > cfg.max_flow_px:
            continue
        points_new.append(np.array([a, b]))
        points_old.append(np.array([c, d]))

    inliers = len(points_old)
    if inliers < cfg.min_inliers:
        return _fallback(prev_gray, f"有效点 {inliers} < min_inliers={cfg.min_inliers}",
                         inliers)

    try:
        H, _status = cv2.findHomography(np.array(points_new), np.array(points_old),
                                        cv2.RANSAC, cfg.ransac_thresh)
    except cv2.error as exc:
        return _fallback(prev_gray, f"findHomography 异常: {exc}", inliers)
    if H is None:
        return _fallback(prev_gray, "findHomography 返回 None", inliers)

    compensated = cv2.warpPerspective(prev_gray, H, (w, h),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # 掩膜：四角经 inv(H) 变换后的多边形外为无效区（255）
    try:
        homo_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return _fallback(prev_gray, "单应矩阵奇异无法求逆", inliers)
    vertex = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
    vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
    im = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(im, vertex_transformed, 255)
    mask = 255 - im

    return AlignResult(compensated=compensated, mask=mask, homography=H,
                       inliers=inliers, fallback=False)
