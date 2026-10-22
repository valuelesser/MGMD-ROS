"""tests.test_lad — LAD 验收（SPEC T09）。"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from mgmd.lad import crop_region


def test_crop_region_center():
    x1, y1, w, h = crop_region((1080, 1920), (960, 540), 320)
    assert (w, h) == (320, 320)
    assert x1 == 800 and y1 == 380


def test_crop_region_clamps_at_left_edge():
    x1, y1, w, h = crop_region((1080, 1920), (10, 540), 320)
    assert x1 == 0, f"左边应 clamp 到 0，实际 {x1}"
    assert w == 320


def test_crop_region_clamps_at_right_edge():
    x1, y1, w, h = crop_region((1080, 1920), (1910, 540), 320)
    assert x1 + w == 1920, f"右边应贴边，实际 {x1 + w}"
    assert w == 320


def test_crop_region_clamps_top_bottom():
    x1, y1, w, h = crop_region((1080, 1920), (960, 5), 320)
    assert y1 == 0
    x1, y1, w, h = crop_region((1080, 1920), (960, 1075), 320)
    assert y1 + h == 1080


def test_crop_region_small_frame():
    """帧小于 crop_size 时取整幅。"""
    x1, y1, w, h = crop_region((100, 150), (75, 50), 320)
    assert w == 150 and h == 100


def test_lad_loads_with_correct_classes():
    """官方权重应加载为单类 Drone（与 ARD-MAV 训练域一致）。"""
    from mgmd.lad import LadDetector
    d = LadDetector()
    assert d.nc == 1, f"类别数应为 1，实际 {d.nc}"
    assert "Drone" in str(d.class_names)


def test_lad_refine_returns_valid_box():
    """推理返回的框应在图像范围内且 conf 合理。"""
    from mgmd.lad import LadDetector
    import cv2
    d = LadDetector()
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[500:560, 900:960] = 200
    rb = d.refine(frame, (930, 530))
    if rb is not None:
        x1, y1, x2, y2 = rb.box
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080
        assert 0.0 <= rb.conf <= 1.0
