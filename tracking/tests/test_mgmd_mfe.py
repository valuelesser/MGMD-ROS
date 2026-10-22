"""tests/test_mgmd_mfe.py — MFE 帧对齐 + 差分候选提取（SPEC §5.1/§5.2，US-003）。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from mgmd.align import align, grid_points
from mgmd.config import MfeConfig
from mgmd.diff import (candidates, enlargebox, filter_contours, frame_threshold,
                       morphology, motion_coherence, preprocess_gray)

W, H = 1024, 768


def _cfg(**over) -> MfeConfig:
    base = dict(blur_kernel=11, grid_spacing=(96, 72), lk_win_size=15, lk_max_level=3,
                max_flow_px=50, min_inliers=9, ransac_thresh=3.0, frame_gap=1,
                or_aggregation=False, overflow="drop", motion_coherence=True, diff_thresh_base=5,
                diff_adaptive=True, median_blur=5, open_iter=1, close_iter=3,
                morph_kernel=3, area_min=30, area_max=3000, ratio_min=0.6, ratio_max=3.0,
                max_boxes=50, enlarge_px=2, motion_avg_dist_min=1.0,
                motion_ratio_theta_max=0.8, motion_ratio_dist_max=0.8)
    return MfeConfig(**{**base, **over})


def _weak_texture(seed: int = 0) -> np.ndarray:
    """中等纹理背景：LK 网格点可跟踪（单应可估），但角点响应远低于亮方块。"""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 160, size=(H, W), dtype=np.uint8)
    blur = cv2.GaussianBlur(noise, (9, 9), 0)
    return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)


def _put_square(img: np.ndarray, x: int, y: int, s: int = 12) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + s, y + s), (255, 255, 255), -1)
    return out


# ---------------- align.py ----------------

def test_grid_points_official_spacing():
    pts = grid_points(1920, 1080, (96, 72))
    assert pts.shape == (19 * 14, 1, 2)           # 官方 1920×1080 → 19×14
    assert tuple(pts[0, 0]) == (48.0, 36.0)       # 网格中心


def test_grid_points_tiny_frame_empty():
    assert len(grid_points(100, 80, (96, 72))) == 0


def test_align_static_scene_identity():
    cfg = _cfg()
    f = _weak_texture()
    g = preprocess_gray(f, cfg)
    res = align(g, g.copy(), cfg)
    assert not res.fallback
    assert res.inliers >= cfg.min_inliers
    assert np.allclose(res.homography, np.eye(3), atol=1e-2)
    assert int(res.mask.sum()) == 0               # 恒等变换无无效边缘
    assert np.array_equal(res.compensated, g)


def test_align_fallback_on_uniform_frames():
    cfg = _cfg()
    g = np.full((H, W), 128, dtype=np.uint8)
    with pytest.warns(UserWarning, match="回退恒等"):
        res = align(g, g.copy(), cfg)
    assert res.fallback
    assert np.array_equal(res.compensated, g)     # 恒等回退：compensated=prev
    assert int(res.mask.sum()) == 0


def test_align_fallback_on_tiny_frame():
    cfg = _cfg()
    g = np.full((100, 80), 128, dtype=np.uint8)
    with pytest.warns(UserWarning):
        res = align(g, g.copy(), cfg)
    assert res.fallback and res.inliers == 0


# ---------------- diff.py 子步骤 ----------------

def test_preprocess_gray_shape():
    cfg = _cfg()
    gray = preprocess_gray(np.zeros((60, 80, 3), np.uint8), cfg)
    assert gray.shape == (60, 80) and gray.dtype == np.uint8


def test_threshold_adaptive_and_fixed():
    diff = np.full((50, 50), 10, dtype=np.uint8)
    mask = np.zeros((50, 50), np.uint8)
    out_ad = frame_threshold(diff, mask, _cfg(diff_adaptive=True))
    assert int(out_ad.sum()) == 0                 # T=5+10=15 > 10
    out_fx = frame_threshold(diff, mask, _cfg(diff_adaptive=False))
    assert int((out_fx == 255).sum()) == 50 * 50  # T=5 < 10


def test_threshold_mask_saturating_no_wraparound():
    """SPEC D8：cv2.subtract 饱和扣除，掩膜区不得出现 numpy 回绕值 1。"""
    diff = np.zeros((40, 40), np.uint8)
    mask = np.zeros((40, 40), np.uint8)
    mask[:, 20:] = 255
    out = frame_threshold(diff, mask, _cfg(diff_adaptive=False))
    assert set(np.unique(out).tolist()) <= {0, 255}
    assert int(out[:, 20:].sum()) == 0


def test_morphology_cleans_speckle():
    binary = np.zeros((60, 60), np.uint8)
    binary[10, 10] = 255                          # 单像素噪点
    cv2.rectangle(binary, (30, 30), (45, 45), 255, -1)
    out = morphology(binary, _cfg())
    assert out[10, 10] == 0                       # 开运算去噪
    assert out[37, 37] == 255                     # 块保留


def test_filter_contours_area_ratio():
    binary = np.zeros((200, 300), np.uint8)
    cv2.rectangle(binary, (10, 10), (16, 16), 255, -1)      # 7×7=49 ✓
    cv2.rectangle(binary, (50, 10), (52, 12), 255, -1)      # 3×3=9 太小
    cv2.rectangle(binary, (100, 10), (180, 80), 255, -1)    # 81×71 太大
    cv2.rectangle(binary, (10, 100), (13, 140), 255, -1)    # 4×41 ratio<0.6
    rects = filter_contours(binary, _cfg())
    assert len(rects) == 1
    x, y, w, h = rects[0]
    assert (x, y) == (10, 10) and 30 < w * h < 3000


def test_filter_contours_too_many_boxes_returns_empty():
    binary = np.zeros((100, 400), np.uint8)
    for i in range(6):
        cv2.rectangle(binary, (i * 60 + 5, 40), (i * 60 + 15, 50), 255, -1)
    assert len(filter_contours(binary, _cfg(max_boxes=3))) == 6 - 6  # 整帧弃检
    assert len(filter_contours(binary, _cfg(max_boxes=50))) == 6


def test_enlargebox_clamp_and_degenerate():
    assert enlargebox(0, 0, 5, 5, 2, 100, 100) == (0, 0, 9, 9)
    assert enlargebox(90, 90, 8, 8, 2, 100, 100) == (88, 88, 11, 11)
    assert enlargebox(99, 99, 5, 5, 2, 100, 100) == (97, 97, 2, 2)   # 边界 clamp
    assert enlargebox(105, 105, 5, 5, 2, 100, 100) is None           # 退化 → None


# ---------------- 端到端候选提取 ----------------

def test_candidates_static_background_zero():
    cfg = _cfg()
    f = _weak_texture()
    g = preprocess_gray(f, cfg)
    res = align(g, g.copy(), cfg)
    assert candidates(f, res.compensated, res.mask, cfg) == []


def test_candidates_moving_square_detected():
    cfg = _cfg()
    bg = _weak_texture()
    x0, y0, dx, dy = 400, 300, 4, 3               # 小位移（LK winSize 覆盖）
    f1 = _put_square(bg, x0, y0)
    f2 = _put_square(bg, x0 + dx, y0 + dy)
    g1, g2 = preprocess_gray(f1, cfg), preprocess_gray(f2, cfg)
    res = align(g1, g2, cfg)
    cands = candidates(f2, res.compensated, res.mask, cfg)
    assert cands, "运动目标未检出"
    cx_t, cy_t = x0 + dx + 6, y0 + dy + 6
    best = min(cands, key=lambda c: np.hypot(c.rect[0] + c.rect[2] / 2 - cx_t,
                                             c.rect[1] + c.rect[3] / 2 - cy_t))
    err = np.hypot(best.rect[0] + best.rect[2] / 2 - cx_t,
                   best.rect[1] + best.rect[3] / 2 - cy_t)
    assert err <= 5, f"框心误差 {err:.2f}px > 5px"
    assert best.score >= cfg.motion_avg_dist_min
    # 旧位置静态过滤由 test_motion_coherence_filters_static_blob 覆盖


def test_overflow_topk_caps_largest():
    """SPEC D13：候选超 max_boxes 时 topk 按面积截断（而非官方整帧弃检）。"""
    binary = np.zeros((100, 400), np.uint8)
    for i in range(6):
        s = 5 + i                                    # 面积递增
        cv2.rectangle(binary, (i * 60 + 5, 40), (i * 60 + 5 + s, 40 + s), 255, -1)
    cfg = _cfg(max_boxes=3, overflow="topk")
    rects = filter_contours(binary, cfg)
    assert len(rects) == 3
    assert min(r[2] * r[3] for r in rects) >= 9 * 9  # 保留最大的三个
    assert filter_contours(binary, _cfg(max_boxes=3, overflow="drop")) == []


def test_candidates_or_future_frame_pair():
    """SPEC D12 OR 聚合：目标仅在 t..t+k 段运动（t−k..t 静止）时，
    两帧口径无候选，OR 口径经未来帧检出。"""
    from mgmd.diff import candidates_or
    cfg = _cfg(blur_kernel=3, area_min=5, overflow="topk", frame_gap=3)
    bg = _weak_texture()
    f_static = _put_square(bg, 400, 300)           # t−k 与 t 相同（静止段）
    f_moved = _put_square(bg, 408, 305)            # t+k（之后跳变 8,5px）
    g_static = preprocess_gray(f_static, cfg)
    g_moved = preprocess_gray(f_moved, cfg)
    mask = np.zeros(g_static.shape, np.uint8)
    # 两帧（过去对）：无差分 → 无候选
    assert candidates_or(f_static, [(g_static, mask)], cfg) == []
    # OR（过去 + 未来对）：未来差分出 blob
    cands = candidates_or(f_static, [(g_static, mask), (g_moved, mask)], cfg)
    assert cands, "OR 聚合未检出"
    # blob 应在旧/新位置附近（合并 blob 覆盖轨迹）
    assert any(395 <= c.rect[0] <= 425 for c in cands)


def test_motion_coherence_filters_static_blob():
    cfg = _cfg()
    bg = _weak_texture()
    f1 = _put_square(bg, 400, 300)
    f2 = _put_square(bg, 404, 303)
    g1, g2 = preprocess_gray(f1, cfg), preprocess_gray(f2, cfg)
    # 纯静态背景区域（远离运动目标）→ avg_dist≈0，一致性不通过
    assert motion_coherence(g2, g1, (100, 100, 14, 14), cfg) is None
    # 运动目标新位置 → 通过且 score≈hypot(4,3)=5
    score = motion_coherence(g2, g1, (402, 301, 14, 14), cfg)
    assert score is not None and 3.0 <= score <= 7.0
