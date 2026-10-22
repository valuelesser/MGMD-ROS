"""tests.test_mfe — MFE 验收（SPEC T04）。

重点：四项额外约束的开关必须真的生效（默认关闭）。
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mgmd.mfe import (MFEConfig, align_to_reference, enhance, grid_keypoints,
                      postprocess, preprocess)


def _blank(h=240, w=320):
    return np.zeros((h, w, 3), np.uint8)


def _textured(h=240, w=320):
    """带静态纹理的背景。

    关键：纯黑背景上只有一个移动方块时，LK 会把方块本身当特征跟踪，
    单应会把方块对齐，差分归零（这是正确行为，不是 bug）。真实场景
    纹理铺满全图，单应只能对齐背景，移动目标才会留下残差。
    """
    rng = np.random.RandomState(0)
    base = rng.randint(0, 90, (h, w, 3), dtype=np.uint8)
    # 加一些结构性条纹，便于 LK 找角点
    for x in range(10, w, 40):
        base[:, x:x + 3] = 160
    for y in range(10, h, 40):
        base[y:y + 3, :] = 160
    return base


def _with_square(x, y, s=20, h=240, w=320, val=255):
    img = np.zeros((h, w, 3), np.uint8)
    img[y:y + s, x:x + s] = val
    return img


def _textured_with_square(x, y, s=20, h=240, w=320):
    """纹理背景 + 亮方块（模拟真实：背景纹理 + 运动小目标）。"""
    img = _textured(h, w)
    img[y:y + s, x:x + s] = 255
    return img


def test_config_validate_rejects_bad():
    with pytest.raises(ValueError):
        MFEConfig(mode="bogus").validate()
    with pytest.raises(ValueError):
        MFEConfig(k=0).validate()
    with pytest.raises(ValueError):
        MFEConfig(preprocess_blur=10).validate()   # 偶数
    with pytest.raises(ValueError):
        MFEConfig(min_area=100, max_area=50).validate()
    MFEConfig().validate()  # 默认合法


def test_grid_keypoints_count_and_bounds():
    pts = grid_keypoints(320, 240, 30, 20)
    assert pts.shape == (30 * 20, 1, 2)
    xs = pts[:, 0, 0]
    ys = pts[:, 0, 1]
    assert xs.min() > 0 and xs.max() < 320
    assert ys.min() > 0 and ys.max() < 240


def test_static_background_no_motion():
    """静止背景：差分应为全零 → 无候选。"""
    cfg = MFEConfig(mode="2frame")
    img = _blank()
    res = enhance(img, img, None, cfg)
    assert res.boxes == [], f"静止帧不应有候选，实际 {len(res.boxes)}"
    assert int(res.motion.sum()) == 0


def test_single_difference_detected():
    """2 帧模式下，背景静止 + 目标移动 → 应产生候选。"""
    cfg = MFEConfig(mode="2frame")
    prev = _textured_with_square(100, 100)
    cur = _textured_with_square(112, 100)
    res = enhance(prev, cur, None, cfg)
    assert len(res.boxes) >= 1, f"应有至少一个候选，实际 {len(res.boxes)}"


def test_pure_black_background_single_object_is_aligned_away():
    """回归：纯黑背景 + 唯一运动方块，LK 会把方块对齐 → 差分为零。

    这不是 bug，是单应的正确行为（方块是唯一可跟踪特征）。
    记录此用例，防止后人误以为漏检而改坏单应。
    """
    cfg = MFEConfig(mode="2frame")
    res = enhance(_with_square(100, 100), _with_square(112, 100), None, cfg)
    assert res.boxes == [], "唯一特征被单应对齐，差分归零属预期"


def test_or_mode_returns_more_or_equal_than_and():
    """OR 召回 >= AND（论文核心论点：OR 增强小目标运动特征）。"""
    prev = _textured_with_square(100, 100)
    cur = _textured_with_square(112, 100)
    nxt = _textured_with_square(124, 100)
    b_or = len(enhance(prev, cur, nxt, MFEConfig(mode="3frame_or")).boxes)
    b_and = len(enhance(prev, cur, nxt, MFEConfig(mode="3frame_and")).boxes)
    assert b_or >= b_and, f"OR({b_or}) 应 >= AND({b_and})"


def test_extra_constraint_m11_max_area_toggle():
    """M11 面积上限开关必须生效。"""
    prev = _textured_with_square(100, 100, s=60)
    cur = _textured_with_square(112, 100, s=60)
    off = enhance(prev, cur, None, MFEConfig(mode="2frame", max_area=0))
    on = enhance(prev, cur, None, MFEConfig(mode="2frame", max_area=200))
    assert len(off.boxes) >= 1
    assert len(on.boxes) < len(off.boxes), "开启 max_area 后候选应减少"


def test_extra_constraint_m12_aspect_toggle():
    """M12 长宽比过滤开关必须生效。"""
    # 构造一个很扁的矩形（宽 60 高 5），长宽比 12 > max_aspect 5
    prev = _blank()
    cur = _blank()
    cur[100:105, 100:160] = 255
    off = enhance(prev, cur, None, MFEConfig(mode="2frame", aspect_filter=False))
    on = enhance(prev, cur, None, MFEConfig(mode="2frame", aspect_filter=True))
    assert len(off.boxes) >= 1
    assert len(on.boxes) == 0, f"长宽比 12 应被过滤，实际保留 {len(on.boxes)}"


def test_extra_constraint_m4_blur_toggle():
    """M4 高斯模糊开关可切换，不崩溃。"""
    prev = _textured_with_square(100, 100)
    cur = _textured_with_square(112, 100)
    r0 = enhance(prev, cur, None, MFEConfig(mode="2frame", preprocess_blur=0))
    r1 = enhance(prev, cur, None, MFEConfig(mode="2frame", preprocess_blur=11))
    assert isinstance(r0.boxes, list) and isinstance(r1.boxes, list)


def test_extra_constraint_m5_disp_filter_toggle():
    """M5 位移过滤开关可切换，不崩溃。"""
    prev = _textured_with_square(100, 100)
    cur = _textured_with_square(112, 100)
    r0 = enhance(prev, cur, None, MFEConfig(mode="2frame", disp_filter=False))
    r1 = enhance(prev, cur, None, MFEConfig(mode="2frame", disp_filter=True))
    assert isinstance(r0.boxes, list) and isinstance(r1.boxes, list)


def test_default_all_extra_constraints_off():
    """SPEC D-04~D-07：四项额外约束默认全部关闭。"""
    cfg = MFEConfig()
    assert cfg.preprocess_blur == 0
    assert cfg.disp_filter is False
    assert cfg.max_area == 0
    assert cfg.aspect_filter is False


def test_first_frame_no_prev_returns_empty():
    res = enhance(None, _with_square(100, 100), None, MFEConfig())
    assert res.boxes == []
    assert res.info.get("note") == "no_prev"


def test_postprocess_drops_tiny_blob():
    """面积 < min_area 的连通域被丢弃（论文 15 px）。"""
    cfg = MFEConfig(min_area=15)
    m = np.zeros((100, 100), np.uint8)
    m[10:12, 10:12] = 255      # 4 px
    _closed, boxes = postprocess(m, cfg)
    assert boxes == []


def test_postprocess_keeps_large_blob():
    cfg = MFEConfig(min_area=15)
    m = np.zeros((100, 100), np.uint8)
    m[10:20, 10:20] = 255      # 100 px
    _closed, boxes = postprocess(m, cfg)
    assert len(boxes) == 1


def test_align_identity_on_identical_frames():
    cfg = MFEConfig()
    gray = cv2.cvtColor(_with_square(100, 100), cv2.COLOR_BGR2GRAY)
    aligned, valid, H = align_to_reference(gray, gray, cfg)
    assert aligned.shape == gray.shape
    assert valid.shape == gray.shape
    # 相同帧：对齐后应与原图接近
    assert np.abs(aligned.astype(int) - gray.astype(int)).mean() < 5
