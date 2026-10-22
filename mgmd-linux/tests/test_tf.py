"""tests.test_tf — TF 验收（SPEC T05）。

覆盖：D 的数学性质、SORT 关联、Algorithm 1 生命周期门控、双模式阈值。
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from mgmd.tf import (SortTracker, TrackState, box_diagonal, distance_metric,
                     iou, transform_box)


def test_iou_and_diagonal():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(box_diagonal((0, 0, 3, 4)) - 5.0) < 1e-9


def test_d_enclosing_never_reaches_one():
    """SPEC D-02 核心：enclosing 定义下 D<1 恒成立。"""
    rng = np.random.RandomState(0)
    worst = 0.0
    for _ in range(5000):
        x1, y1 = rng.uniform(0, 500, 2)
        w1, h1 = rng.uniform(1, 50, 2)
        a = (x1, y1, x1 + w1, y1 + h1)
        x2, y2 = rng.uniform(0, 500, 2)
        w2, h2 = rng.uniform(1, 50, 2)
        b = (x2, y2, x2 + w2, y2 + h2)
        worst = max(worst, distance_metric(a, b, "enclosing"))
    assert worst < 1.0, f"enclosing D 应恒 <1，实际最大 {worst}"


def test_d_self_can_exceed_one():
    """self 定义下 D 可以 >1（位移大、框小时）。"""
    a = (0.0, 0.0, 10.0, 10.0)
    b = (100.0, 0.0, 110.0, 10.0)
    assert distance_metric(a, b, "self") > 1.0


def test_d_zero_for_identical_boxes():
    a = (10.0, 10.0, 20.0, 20.0)
    assert distance_metric(a, a, "enclosing") == 0.0
    assert distance_metric(a, a, "self") == 0.0


def test_lifetime_gate_rejects_short_track():
    """Algorithm 1：寿命 <3 帧时 P=0。"""
    st = TrackState(track_id=1, buffer_size=4)
    st.add(1, (0, 0, 10, 10))
    assert st.classify(3, "self", 0.0) == 0
    assert st.reason == "lifetime_lt_3" or st.reason == "no_window"


def test_window_uses_bounded_buffer():
    """S 是有界缓冲：满后淘汰最老，window 取倒数第 span 个。"""
    st = TrackState(track_id=1, buffer_size=4)
    for f in range(1, 11):
        st.add(f, (f * 10.0, 0, f * 10.0 + 10, 10))
    assert len(st.boxes) == 4, f"缓冲应为 4，实际 {len(st.boxes)}"
    pair = st.window(3)
    assert pair is not None
    (f_old, _b_old), (f_new, _b_new) = pair
    assert f_new == 10
    assert f_old == 7, f"应取帧 7（N-3），实际 {f_old}"


def test_sort_confirms_moving_track():
    """匀速运动轨迹应被确认并输出。"""
    tr = SortTracker(d_mode="self", d_min=0.1)
    outs = []
    for f in range(1, 12):
        outs = tr.update([(f * 5.0, 50.0, f * 5.0 + 10, 60.0)], f)
    assert len(outs) >= 1, "运动轨迹应被确认"
    assert outs[0].d_value >= 0.1


def test_sort_rejects_static_track():
    """静止轨迹 self 模式 D=0，应被拒（D<d_min）。"""
    tr = SortTracker(d_mode="self", d_min=0.1)
    outs = []
    for f in range(1, 12):
        outs = tr.update([(50.0, 50.0, 60.0, 60.0)], f)
    assert outs == [], f"静止轨迹不应输出，实际 {len(outs)}"


def test_sort_literal_d_min_confirms_nothing():
    """SPEC D-02 字面模式：d_min=1.0 + enclosing → 零确认。"""
    tr = SortTracker(d_mode="enclosing", d_min=1.0)
    outs = []
    for f in range(1, 30):
        outs = tr.update([(f * 20.0, 50.0, f * 20.0 + 10, 60.0)], f)
    assert outs == [], "enclosing + d_min=1.0 不应确认任何轨迹（论文判据不可达）"


def test_sort_max_age_deletes_track():
    """max_age=3：连续 4 帧未匹配应删除轨迹。"""
    tr = SortTracker(d_mode="self", d_min=0.0)
    for f in range(1, 8):
        tr.update([(f * 3.0, 0.0, f * 3.0 + 10, 10.0)], f)
    n_before = len(tr.states)
    assert n_before >= 1
    for f in range(8, 14):
        tr.update([], f)          # 无检测
    assert len(tr.states) == 0, f"轨迹应被删除，剩余 {len(tr.states)}"


def test_no_coasting_output():
    """SPEC D-12：不做滑行输出，未匹配帧不输出轨迹。"""
    tr = SortTracker(d_mode="self", d_min=0.0)
    for f in range(1, 8):
        tr.update([(f * 5.0, 0.0, f * 5.0 + 10, 10.0)], f)
    outs = tr.update([], 8)      # 本帧无检测
    assert outs == [], "未匹配帧不应输出（不滑行）"


def test_two_targets_tracked_separately():
    """两个分离目标应得到两条轨迹。"""
    tr = SortTracker(d_mode="self", d_min=0.0)
    outs = []
    for f in range(1, 15):
        a = (f * 4.0, 20.0, f * 4.0 + 10, 30.0)
        b = (200.0 - f * 4.0, 200.0, 210.0 - f * 4.0, 210.0)
        outs = tr.update([a, b], f, scores=[0.9, 0.8])
    assert len(outs) == 2, f"应输出 2 条轨迹，实际 {len(outs)}"


def test_transform_box_identity():
    H = np.eye(3)
    out = transform_box((10.0, 20.0, 30.0, 40.0), H)
    assert out is not None
    assert all(abs(a - b) < 1e-6 for a, b in zip(out, (10.0, 20.0, 30.0, 40.0)))


def test_transform_box_translation():
    H = np.array([[1.0, 0, 5.0], [0, 1.0, -3.0], [0, 0, 1.0]])
    out = transform_box((10.0, 20.0, 30.0, 40.0), H)
    assert out is not None
    assert abs(out[0] - 15.0) < 1e-6 and abs(out[1] - 17.0) < 1e-6


def test_calibrated_threshold_from_train_only():
    """标定阈值必须能区分运动与静止（供 T05 标定脚本用）。"""
    moving, static = [], []
    for f in range(1, 20):
        a = (f * 6.0, 0.0, f * 6.0 + 12, 12.0)
        b = ((f + 1) * 6.0, 0.0, (f + 1) * 6.0 + 12, 12.0)
        moving.append(distance_metric(a, b, "enclosing"))
        static.append(distance_metric((0, 0, 12, 12), (0.5, 0, 12.5, 12), "enclosing"))
    assert min(moving) > max(static), "运动轨迹 D 应整体高于静止轨迹"
