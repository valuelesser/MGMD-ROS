"""mgmd.diff — 自适应阈值两帧差分 + 形态学 + 连通域候选提取（SPEC §5.2）。

移植官方 MOD2_global（SPEC D1 官方两帧口径）：
- 预处理：GaussianBlur(blur_kernel) → 灰度；
- 阈值：T = diff_thresh_base + int(mean(diff))（diff_adaptive=false 时 T=base）；
- 掩膜扣除用 cv2.subtract 饱和减法（SPEC D8，官方 numpy uint8 回绕减法不采用）；
- medianBlur 5 → 3×3 开 iter1 → 3×3 闭 iter3；
- 连通域：30 < contourArea < 3000 且 0.6 < w/h < 3；候选 >max_boxes(50) 整帧弃检；
- 逐框运动一致性：enlarge a=2 → ShiTomasi(30/0.15/3/3) + LK(15/3/(30,0.03))，
  avg_dist≥motion_avg_dist_min 且 std/mean(θ)≤ratio_theta_max 且
  std/mean(dist)≤ratio_dist_max；通过者 score=avg_dist。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mgmd.config import MfeConfig


@dataclass
class Candidate:
    rect: tuple[int, int, int, int]  # (x, y, w, h)，处理尺度坐标
    score: float                     # 运动一致性 avg_dist（每帧选择规则用，SPEC §5.6）


def preprocess_gray(bgr: np.ndarray, cfg: MfeConfig) -> np.ndarray:
    """GaussianBlur(blur_kernel) → 灰度（SPEC §5.1 step 1）。"""
    blur = cv2.GaussianBlur(bgr, (cfg.blur_kernel, cfg.blur_kernel), 0)
    return cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)


def frame_threshold(frame_diff: np.ndarray, mask: np.ndarray, cfg: MfeConfig) -> np.ndarray:
    """自适应阈值二值化 + 掩膜饱和扣除（SPEC §5.2 steps 2-3，D8）。"""
    t = cfg.diff_thresh_base + (int(np.mean(frame_diff)) if cfg.diff_adaptive else 0)
    _rv, thresh = cv2.threshold(frame_diff, t, 255, cv2.THRESH_BINARY)
    return cv2.subtract(thresh, mask)


def morphology(binary: np.ndarray, cfg: MfeConfig) -> np.ndarray:
    """medianBlur → 开（去噪）→ 闭（填孔）（SPEC §5.2 step 4）。"""
    out = cv2.medianBlur(binary, cfg.median_blur)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_kernel,) * 2)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=cfg.open_iter)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=cfg.close_iter)
    return out


def filter_contours(binary: np.ndarray, cfg: MfeConfig) -> list[tuple[int, int, int, int]]:
    """连通域 → 面积/宽高比过滤 → 外接矩形列表（SPEC §5.2 steps 5-6）。

    候选超 max_boxes 时：overflow=drop → 返回 []（官方整帧弃检）；
    overflow=topk → 按 contourArea 降序截断到 max_boxes（SPEC D13）。
    """
    contours, _hier = cv2.findContours(binary.copy(), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
    rects: list[tuple[int, int, int, int]] = []
    areas: list[float] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0:
            continue
        area = cv2.contourArea(contour)
        ratio = w / h
        if cfg.area_min < area < cfg.area_max and cfg.ratio_min < ratio < cfg.ratio_max:
            rects.append((int(x), int(y), int(w), int(h)))
            areas.append(float(area))
    if len(rects) > cfg.max_boxes:
        if cfg.overflow == "topk":
            order = np.argsort(np.asarray(areas))[::-1][: cfg.max_boxes]
            return [rects[i] for i in sorted(order.tolist())]
        return []
    return rects


def enlargebox(x: int, y: int, w: int, h: int, a: int,
               width: int, height: int) -> tuple[int, int, int, int] | None:
    """官方 Functions.enlargebox：四周扩 a px，边界 clamp；退化（≤0）返回 None。"""
    x1, y1 = x - a, y - a
    w1, h1 = w + a * 2, h + a * 2
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    if x1 + w1 >= width:
        w1 = width - x1 - 1
    if y1 + h1 >= height:
        h1 = height - y1 - 1
    if w1 <= 0 or h1 <= 0:
        return None
    return int(x1), int(y1), int(w1), int(h1)


def motion_coherence(cur_gray: np.ndarray, compensated: np.ndarray,
                     rect: tuple[int, int, int, int], cfg: MfeConfig) -> float | None:
    """逐框运动一致性（官方 MOD2 'motion classifier'，SPEC §5.2 step 7）。

    通过返回 score=avg_dist；被过滤返回 None。

    扩框量 = enlarge_px + 2·(frame_gap−1)（SPEC D12：frame_gap>1 时位移为 k·v，
    官方 a=2 的裁剪容不下；frame_gap=1 时严格等于官方 a=2）。
    """
    height, width = cur_gray.shape[:2]
    x0, y0, w0, h0 = rect
    a = cfg.enlarge_px + 2 * (cfg.frame_gap - 1)
    box = enlargebox(x0, y0, w0, h0, a, width, height)
    if box is None:
        return None
    x1, y1, w1, h1 = box
    ratio1 = w1 / h1
    if ratio1 < 0.6 or ratio1 > 3:
        return None

    crop_cur = cur_gray[y1:y1 + h1, x1:x1 + w1]
    crop_comp = compensated[y1:y1 + h1, x1:x1 + w1]
    if crop_cur.size == 0 or crop_comp.size == 0:
        return None

    feature_params = dict(maxCorners=30, qualityLevel=0.15, minDistance=3, blockSize=3)
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    pts1 = cv2.goodFeaturesToTrack(crop_cur, mask=None, **feature_params)
    if pts1 is None:
        return None

    pts2, st1, _err = cv2.calcOpticalFlowPyrLK(crop_comp, crop_cur, pts1, None, **lk_params)
    if pts2 is None or st1 is None:
        return None
    good_new = pts2[st1 == 1]
    good_old = pts1[st1 == 1]
    if len(good_new) < 1:
        return None

    motion_dist: list[float] = []
    motion_theta: list[float] = []
    for new, old in zip(good_new, good_old):
        a1, b1 = new.ravel()
        c1, d1 = old.ravel()
        motion_dist.append(float(np.hypot(a1 - c1, b1 - d1)))
        motion_theta.append(float(57.3 * np.arctan2(d1 - b1, c1 - a1)))

    dist = np.array(motion_dist)
    theta = np.array(motion_theta)
    avg_dist = float(np.mean(dist))
    avg_theta = float(np.mean(theta))
    ratio_theta = float(np.std(theta) / avg_theta) if abs(avg_theta) > 1e-12 else float("inf")
    ratio_dist = float(np.std(dist) / avg_dist) if avg_dist > 1e-12 else float("inf")

    if avg_dist < cfg.motion_avg_dist_min:
        return None
    if ratio_theta > cfg.motion_ratio_theta_max or ratio_dist > cfg.motion_ratio_dist_max:
        return None
    return avg_dist


def candidates(cur_bgr: np.ndarray, compensated: np.ndarray, mask: np.ndarray,
               cfg: MfeConfig) -> list[Candidate]:
    """两帧差分候选提取（官方口径 = 单差分对）。候选溢出策略见 overflow（SPEC D13）。"""
    return candidates_or(cur_bgr, [(compensated, mask)], cfg)


def candidates_or(cur_bgr: np.ndarray,
                  pairs: list[tuple[np.ndarray, np.ndarray]],
                  cfg: MfeConfig) -> list[Candidate]:
    """OR 聚合候选提取（SPEC D12）：每个 (compensated, mask) 对独立自适应阈值二值化后
    bitwise_or 聚合 → 形态学 → 连通域；逐框运动一致性取各对最大 score（任一过即保留）。

    单对时等价官方两帧口径；两对 [(Î_{t−k},mask), (Î_{t+k},mask)] 即论文三帧 OR。
    """
    cur_gray = preprocess_gray(cur_bgr, cfg)
    binary = np.zeros(cur_gray.shape, dtype=np.uint8)
    dmax: np.ndarray | None = None
    for compensated, mask in pairs:
        frame_diff = cv2.absdiff(cur_gray, compensated)
        dmax = frame_diff if dmax is None else np.maximum(dmax, frame_diff)
        binary = cv2.bitwise_or(binary, frame_threshold(frame_diff, mask, cfg))
    binary = morphology(binary, cfg)
    rects = filter_contours(binary, cfg)
    if not rects:
        return []
    out: list[Candidate] = []
    for rect in rects:
        if not cfg.motion_coherence:
            # SPEC D14：静态相机无视差假阳性，跳过一致性过滤；
            # score = blob 内差分峰值（运动证据强度，跨帧可比，供选择/AP 排序）
            x, y, w, h = rect
            patch = dmax[y:y + h, x:x + w] if dmax is not None else None
            score = float(patch.max()) if (patch is not None and patch.size) else 0.0
            out.append(Candidate(rect=rect, score=score))
            continue
        best: float | None = None
        for compensated, _mask in pairs:
            score = motion_coherence(cur_gray, compensated, rect, cfg)
            if score is not None and (best is None or score > best):
                best = score
        if best is not None:
            out.append(Candidate(rect=rect, score=best))
    return out
