"""tests/test_mgmd_tf.py — SORT + Algorithm 1 轨迹分类（SPEC §5.3/§5.4，US-004）。"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("filterpy", reason="filterpy 未装：TF 用例 skip（FR-11）")
pytest.importorskip("scipy", reason="scipy 未装：TF 用例 skip（FR-11）")

from mgmd.config import TfConfig          # noqa: E402
from mgmd.track import SortTracker, iou, rect_to_z, x_to_rect  # noqa: E402


def _cfg(**over) -> TfConfig:
    base = dict(iou_thresh=0.3, max_age=3, min_hits=1, alg1_age=3, alg1_window=3,
                alg1_d_min=1.0)
    return TfConfig(**{**base, **over})


def _rect(x: float, y: float, w: float = 20, h: float = 10) -> tuple[int, int, int, int]:
    return (int(x), int(y), int(w), int(h))


def _feed(tracker: SortTracker, rects_per_frame: list[list]) -> list:
    out = []
    for f, rects in enumerate(rects_per_frame):
        out.append(tracker.update(rects, f))
    return out


# ---------------- 基础几何 ----------------

def test_rect_z_roundtrip():
    rect = (10.0, 20.0, 30.0, 15.0)
    assert x_to_rect(np.concatenate([rect_to_z(rect).ravel(), [0, 0, 0]])) == \
        pytest.approx(rect, abs=1e-6)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


# ---------------- SORT 生命周期 ----------------

def test_constant_velocity_track_confirmed():
    """匀速 8px/帧、框 20×10：ρ(3帧)=24 ≥ d=22.36 → Algorithm 1 confirmed（US-004）。"""
    trk = SortTracker(_cfg())
    trk.reset_ids()
    outs = _feed(trk, [[_rect(100 + 8 * t, 100)] for t in range(6)])
    assert all(len(o) == 1 for o in outs)
    ids = {o[0].id for o in outs}
    assert len(ids) == 1                      # 轨迹 id 稳定续接
    assert not outs[2][0].confirmed or outs[2][0].age < 3
    assert outs[4][0].confirmed               # age≥3 且 D≥1
    assert outs[5][0].confirmed
    vx, _vy = outs[5][0].velocity
    assert vx > 1.0                           # Kalman 学到正速度


def test_static_track_filtered():
    """静止框轨迹 ρ≈0 → Algorithm 1 不确认（假阳性过滤）。"""
    trk = SortTracker(_cfg())
    outs = _feed(trk, [[_rect(100, 100)] for _ in range(6)])
    assert all(len(o) == 1 for o in outs)
    assert not any(o[0].confirmed for o in outs)


def test_jitter_track_filtered():
    """±1px 抖动轨迹 D<1 → 不确认。"""
    rng = np.random.default_rng(0)
    frames = [[_rect(100 + int(rng.integers(-1, 2)), 100 + int(rng.integers(-1, 2)))]
              for _ in range(8)]
    trk = SortTracker(_cfg())
    outs = _feed(trk, frames)
    assert all(len(o) == 1 for o in outs)
    assert not any(o[0].confirmed for o in outs)


def test_slow_track_below_d_min():
    """匀速但位移不足（2px/帧，3 帧 ρ=6 < d=22.36）→ 不确认。"""
    trk = SortTracker(_cfg())
    outs = _feed(trk, [[_rect(100 + 2 * t, 100)] for t in range(6)])
    assert not any(o[0].confirmed for o in outs)


def test_gap_reacquire_within_max_age():
    """断帧 2 帧内（≤max_age=3）轨迹可续接且 id 不变（US-004 AC）。"""
    trk = SortTracker(_cfg())
    seq = [[_rect(100 + 8 * t, 100)] for t in range(3)] + [[], []] + \
          [[_rect(100 + 8 * 5, 100)]]
    outs = _feed(trk, seq)
    assert len(outs[0]) == 1
    tid = outs[0][0].id
    assert outs[3] == [] and outs[4] == []    # 断帧期无输出
    assert len(outs[5]) == 1
    assert outs[5][0].id == tid               # 同一轨迹续接
    assert outs[5][0].hits >= 4


def test_track_deleted_after_max_age():
    trk = SortTracker(_cfg())
    seq = [[_rect(100, 100)]] + [[] for _ in range(5)]
    outs = _feed(trk, seq)
    assert all(o == [] for o in outs[1:])
    assert trk.trackers == []                 # time_since_update>3 → 删除


def test_two_parallel_tracks_keep_ids():
    """IoU 关联正确性：两条平行运动轨迹 id 不串。"""
    trk = SortTracker(_cfg())
    frames = [[_rect(100 + 8 * t, 100), _rect(300 - 8 * t, 300)] for t in range(5)]
    outs = _feed(trk, frames)
    assert all(len(o) == 2 for o in outs)
    ids0 = [o[0].id for o in outs]
    ids1 = [o[1].id for o in outs]
    assert len(set(ids0)) == 1 and len(set(ids1)) == 1 and ids0[0] != ids1[0]


def test_alg1_threshold_configurable():
    """阈值可配（FR-5）：alg1_d_min 提高后匀速 8px/帧不再确认。"""
    trk = SortTracker(_cfg(alg1_d_min=2.0))
    outs = _feed(trk, [[_rect(100 + 8 * t, 100)] for t in range(6)])
    assert not any(o[0].confirmed for o in outs)


def test_min_hits_one_outputs_first_frame():
    """min_hits=1：首帧即有轨迹输出（PRD FR-4 口径）。"""
    trk = SortTracker(_cfg())
    outs = trk.update([_rect(100, 100)], 0)
    assert len(outs) == 1 and outs[0].hits == 1


def test_history_alignment_is_prev_to_current():
    """回归：align_H 语义为上一帧→当前帧，历史点应被搬到当前帧（不得反向）。"""
    trk = SortTracker(_cfg())
    trk.reset_ids()
    trk.update([_rect(100, 100)], 0)           # 历史首点 = 中心 (110,105)
    H_prev_cur = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
    outs = trk.update([_rect(100, 100)], 1, align_H=H_prev_cur)
    assert len(outs) == 1
    assert outs[0].history[0] == pytest.approx((113.0, 104.0), abs=1e-6)
