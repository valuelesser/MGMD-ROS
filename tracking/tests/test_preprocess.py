"""test_preprocess — 暗目标增强预处理（SPEC §5.1/§9.1，I-03 验收）。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from spa2d import preprocess as P


def test_build_kernel_sizes_and_shapes():
    for k in (3, 5, 7, 9, 11):
        ker = P.build_kernel(k)
        assert ker.shape == (k, k)
        assert ker.dtype == np.uint8
        assert ker[(k - 1) // 2, (k - 1) // 2] == 1
        if k >= 5:
            assert ker[0, 0] == 0  # 椭圆角为 0
    rect = P.build_kernel(5, "rect")
    assert np.all(rect == 1)
    cross = P.build_kernel(5, "cross")
    assert cross[0, 0] == 0 and cross[2, 2] == 1


def test_build_kernel_invalid():
    for bad in (1, 2, 4, 6, 13):
        with pytest.raises(ValueError):
            P.build_kernel(bad)
    with pytest.raises(ValueError):
        P.build_kernel(5, "disk")


def test_output_dtype_shape_range(small_bgr_image):
    ker = P.build_kernel(5)
    out = P.preprocess_frame(small_bgr_image, ker)
    assert out.dtype == np.float32
    assert out.shape == small_bgr_image.shape[:2]
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_d_nonnegative_random(small_bgr_image):
    out = P.preprocess_frame(small_bgr_image, P.build_kernel(3))
    assert np.all(out >= 0.0)


def test_matches_manual_pipeline(small_bgr_image):
    ker = P.build_kernel(5)
    out = P.preprocess_frame(small_bgr_image, ker)
    gray = cv2.cvtColor(small_bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    expect = np.maximum(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, ker) - gray, 0.0) / 255.0
    assert np.allclose(out, expect.astype(np.float32))


def test_color_order_bgr_vs_rgb_differs():
    img = np.full((32, 32, 3), 200, np.uint8)
    img[14:17, 14:17] = (0, 0, 255)  # BGR: 红斑；按 RGB 解释则为蓝斑
    ker = P.build_kernel(7)
    assert not np.allclose(P.to_gray(img, "BGR"), P.to_gray(img, "RGB"))
    bgr = P.preprocess_frame(img, ker, color_order="BGR")
    rgb = P.preprocess_frame(img, ker, color_order="RGB")
    assert not np.allclose(bgr, rgb)


def test_gray_input_path(small_gray_float):
    out = P.preprocess_frame(small_gray_float, P.build_kernel(3))
    assert out.dtype == np.float32 and out.shape == small_gray_float.shape


def test_invalid_channel_count():
    with pytest.raises(ValueError):
        P.preprocess_frame(np.zeros((8, 8, 2), np.uint8), P.build_kernel(3))


def test_closing_removes_small_dark_feature():
    canvas = np.full((64, 64), 200, np.uint8)
    canvas[30:33, 30:33] = 50  # 3x3 暗斑，小于 kernel
    out = P.preprocess_frame(canvas, P.build_kernel(7))
    iy, ix = np.unravel_index(int(np.argmax(out)), out.shape)
    assert abs(iy - 31) <= 1 and abs(ix - 31) <= 1
    assert out[31, 31] > 0.3
    bg_median = float(np.median(out))
    assert out[31, 31] > 10 * max(bg_median, 1e-6)


def test_closing_keeps_large_dark_region():
    canvas = np.full((64, 64), 200, np.uint8)
    canvas[20:40, 20:40] = 50  # 20x20 暗区，远大于 kernel → 残差小
    out = P.preprocess_frame(canvas, P.build_kernel(7))
    assert out[30, 30] < 0.05


def test_no_forbidden_enhancement_in_source():
    import ast

    src = Path(P.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    # 主预处理链路（preprocess_frame/to_gray）禁止直方图均衡/CLAHE/滤波增强；
    # 仅 matched_filter_frame（few-pixel 匹配滤波候选表示）允许 GaussianBlur(DoG)。
    forbidden = {"equalizeHist", "createCLAHE", "bilateralFilter", "filter2D",
                 "adaptiveThreshold", "normalize"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "matched_filter_frame":
            continue
        for sub in ast.walk(node) if isinstance(node, ast.FunctionDef) else []:
            if isinstance(sub, ast.Call):
                fn = sub.func
                name = fn.attr if isinstance(fn, ast.Attribute) else (
                    fn.id if isinstance(fn, ast.Name) else "")
                assert name not in forbidden, f"预处理链路不应调用 {name}"
    # matched_filter_frame 必须存在（尺度先验候选表示）
    assert "matched_filter_frame" in src


def test_docstring_documents_boundary():
    doc = P.__doc__ or ""
    assert "边界" in doc
    assert "MORPH_CLOSE" in doc


def test_matched_filter_selects_small_over_large_blob():
    import numpy as np

    h, w = 96, 128
    img = np.full((h, w), 200.0, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    small = np.exp(-(((xx - 40) ** 2 + (yy - 30) ** 2) / (2 * 1.5 ** 2))) * 80.0
    large = np.exp(-(((xx - 90) ** 2 + (yy - 60) ** 2) / (2 * 12.0 ** 2))) * 120.0
    img -= (small + large)
    resp = P.matched_filter_frame(img, sigma=2.0)
    iy, ix = np.unravel_index(int(np.argmax(resp)), resp.shape)
    assert abs(ix - 40) <= 3 and abs(iy - 30) <= 3   # 选中小斑
    # 纯 max 暗度会被大斑主导
    assert np.argmin(img) != np.argmax(resp)
