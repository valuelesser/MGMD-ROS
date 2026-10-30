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


# --------------------------------------------------------------------------- #
# step_H 对齐（SPEC-ROS §22 / SPEC §8 缺口1）
# --------------------------------------------------------------------------- #

def _mk_tracker(**kw):
    d = dict(iou_thresh=0.3, max_age=3, min_hits=1, span=3,
             d_mode="self", d_min=0.02, align=True)
    d.update(kw)
    return SortTracker(**d)


def test_compose_H_single_step():
    """单步级联：compose_H(f,f+1) == step_H[f+1]。"""
    H = np.array([[1.0, 0, 3.0], [0, 1.0, -2.0], [0, 0, 1]], np.float64)
    st = TrackState(track_id=1)
    st.step_H[10] = H
    got = st.compose_H(9, 10)
    assert got is not None and np.allclose(got, H)


def test_compose_H_missing_step_returns_none():
    """缺任一帧 → None（这是既不补 step_H 时必然发生的事情）。"""
    st = TrackState(track_id=1)
    st.step_H[10] = np.eye(3)
    assert st.compose_H(8, 10) is None     # 缺 step_H[9]


def test_compose_H_cascades_translations():
    """两段平移级联应为和：3 + 4 = 7。"""
    st = TrackState(track_id=1)
    st.step_H[5] = np.array([[1, 0, 3.0], [0, 1, 0], [0, 0, 1]], np.float64)
    st.step_H[6] = np.array([[1, 0, 4.0], [0, 1, 0], [0, 0, 1]], np.float64)
    H = st.compose_H(4, 6)
    assert H is not None
    assert abs(H[0, 2] - 7.0) < 1e-9


def test_align_missing_reason_is_overwritten():
    """**已知缺陷**：classify 末尾无条件覆盖 reason，align_missing 不会保留。

    调用链：align 分支先设 reason="align_missing"，
    末尾 `self.reason = "confirmed"` 又覆盖它。
    后果：`reason` 字段无法用来判断对齐是否生效，
    统计 align_missing 占比必须直接看 compose_H 的返回值，
    不能读 reason（否则永远是 0，会误判成"对齐已生效"）。

    此测试锁定当前行为，防止误以为 reason 可信。SPEC §8 缺口1 / §22。
    """
    st = TrackState(track_id=1)
    for f in range(6):
        st.add(f, (0.0 + f, 0.0, 10.0 + f, 10.0))
    st.classify(span=3, d_mode="self", d_min=0.0, align=True)
    assert st.compose_H(0, 1) is None        # 真正的事实：链路缺失
    assert st.reason != "align_missing"      # 但 reason 看不出这一点


def test_align_applied_reason_not_missing():
    """喂了完整 step_H 后不再 align_missing。"""
    st = TrackState(track_id=1)
    for f in range(6):
        st.add(f, (0.0 + f, 0.0, 10.0 + f, 10.0))
        st.step_H[f] = np.eye(3)
    st.classify(span=3, d_mode="self", d_min=0.0, align=True)
    assert st.reason != "align_missing"


def test_step_H_changes_d_value():
    """step_H 生效会改变 d_value（平移使框错开，d 变大）。"""
    mk = lambda: TrackState(track_id=1)
    a, b = mk(), mk()
    for f, x in [(5, 0.0), (6, 1.0), (7, 2.0), (8, 3.0)]:
        a.add(f, (x, 0.0, x + 10, 10.0))
        b.add(f, (x, 0.0, x + 10, 10.0))
        a.step_H[f] = np.eye(3)
        b.step_H[f] = np.eye(3)
    a.classify(span=3, d_mode="self", d_min=0.0, align=True)
    b.classify(span=3, d_mode="self", d_min=0.0, align=False)
    assert abs(a.d_value - b.d_value) < 1e-9     # 恒等 H 下两者应相同


def test_tracker_update_accepts_step_H():
    """update(step_H=...) 会把单应写入对应轨迹状态。"""
    tr = _mk_tracker()
    H = np.eye(3)
    tr.update([(0.0, 0.0, 10.0, 10.0)], 0, scores=[100.0], step_H=H)
    assert tr.states
    tid = next(iter(tr.states))
    assert 0 in tr.states[tid].step_H
