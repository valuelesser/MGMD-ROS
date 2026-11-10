"""detector_node 帧缓冲推进逻辑（SPEC-ROS §25）。

**背景**：`ready_frame_id()` 原来要求 `t-k` 与 `t+k` **精确存在于缓冲**。
一旦帧号不是连续整数（stride 抽帧、帧号跳变），该条件**永不成立**，
`ready_frame_id()` 恒返回 `None`，整条流水线**静默停在 0 帧**。

本测试锁定四件事：
  ① 连续输入下与旧实现**逐帧一致**（零回归）
  ② stride 不整除 k 时不再卡死（旧实现卡死，这是 bug）
  ③ 窗口语义：邻居距离 ∈ **[k, max_gap]**，取最接近 k 的，
     既不能缩小基线（k=2 不能退化成 t±1），也不能无限放大
  ④ 推进不了时要能**喊出来**（stall 计数，不静默）

**注意**：测试用「追平」语义（每次 push 后 while 处理到追平），
与真实 `on_image` 的 `while True: ready_frame_id()` 循环一致。

运行：
    .venv/bin/python -m pytest tests/test_frame_buffer.py -q
"""

import pytest

from collections import OrderedDict, deque


# --------------------------------------------------------------------------- #
# 目标实现（修复后 detector_node.FrameBuffer 的推进逻辑）
# --------------------------------------------------------------------------- #
class FrameBuffer:
    """最小复刻：只保留帧推进相关部分，不含图像/ONNX。

    ready_frame_id 返回 (t, prev, nxt)，step 用 prev/nxt 做 MFE 对齐。
    """

    def __init__(self, k: int = 1, buffer_size: int = 6, max_gap: int = 0):
        self.k = k
        self.buffer_size = max(buffer_size, 2 * k + 1)
        self.frames: "OrderedDict[int, object]" = OrderedDict()
        self.last_done = -(10 ** 9)
        self._gaps: "deque[int]" = deque(maxlen=32)
        self._last_push = None
        self._max_gap_cfg = max_gap
        self.stall_pushes = 0

    # -- 推入 ---------------------------------------------------------------
    def push(self, frame_id: int) -> None:
        if self.frames and frame_id <= next(reversed(self.frames)):
            # 帧号回绕（循环回放）→ 清空重来
            self.frames.clear()
            self._gaps.clear()
            self._last_push = None
            self.stall_pushes = 0
        if self._last_push is not None:
            gap = frame_id - self._last_push
            if gap > 0:
                self._gaps.append(gap)
        self._last_push = frame_id
        self.frames[frame_id] = True
        while len(self.frames) > self.buffer_size:
            self.frames.popitem(last=False)

    # -- 窗口 ---------------------------------------------------------------
    def max_gap(self) -> int:
        """允许的最大邻居距离。默认自适应 = max(k, 观测帧间隔中位数)。"""
        if self._max_gap_cfg > 0:
            return self._max_gap_cfg
        if not self._gaps:
            return self.k
        ordered = sorted(self._gaps)
        return max(self.k, ordered[len(ordered) // 2])   # 中位数，抗偶发丢帧

    def window(self, t: int):
        """返回 (prev, nxt)。距离 ∈ [k, max_gap]，取最接近 k 的。"""
        if t not in self.frames:
            return None
        mg = self.max_gap()
        prev = nxt = None
        for d in range(self.k, mg + 1):
            if prev is None and (t - d) in self.frames:
                prev = t - d
            if nxt is None and (t + d) in self.frames:
                nxt = t + d
            if prev is not None and nxt is not None:
                break
        if prev is None or nxt is None:
            return None
        return prev, nxt

    # -- 推进 ---------------------------------------------------------------
    def ready_frame_id(self):
        """返回 (t, prev, nxt) 或 None。"""
        if len(self.frames) < 2 * self.k + 1:
            return None
        for t in list(self.frames.keys()):
            if t <= self.last_done:
                continue
            w = self.window(t)
            if w is not None:
                return (t, w[0], w[1])
        return None

    def mark_done(self, t: int) -> None:
        self.last_done = t


# --------------------------------------------------------------------------- #
# 旧实现（有 bug 的那个），保留原样用于对照
# --------------------------------------------------------------------------- #
class OldFrameBuffer:
    def __init__(self, k: int = 1, buffer_size: int = 6):
        self.k = k
        self.buffer_size = max(buffer_size, 2 * k + 1)
        self.frames: "OrderedDict[int, object]" = OrderedDict()
        self.last_done = -(10 ** 9)

    def push(self, frame_id: int) -> None:
        self.frames[frame_id] = True
        while len(self.frames) > self.buffer_size:
            self.frames.popitem(last=False)

    def ready_frame_id(self):
        if len(self.frames) < 2 * self.k + 1:
            return None
        for t in list(self.frames.keys()):
            if t <= self.last_done:
                continue
            if (t - self.k) in self.frames and (t + self.k) in self.frames:
                return t
        return None


# --------------------------------------------------------------------------- #
# 统一「追平」语义：模拟 on_image 的 while 循环
# --------------------------------------------------------------------------- #
def _follow(buf, frames) -> "list[int]":
    """依次 push 每个帧号，每次 push 后 while 追平。返回产出帧号列表。"""
    out = []
    for f in frames:
        buf.push(f)
        while True:
            r = buf.ready_frame_id()
            if r is None:
                break
            t = r[0] if isinstance(r, tuple) else r
            buf.last_done = t
            out.append(t)
    return out


def _follow_new(buf, frames) -> "list[int]":
    """FrameBuffer 版：额外维护 stall 计数。"""
    out = []
    for f in frames:
        buf.push(f)
        processed = 0
        while True:
            r = buf.ready_frame_id()
            if r is None:
                break
            buf.mark_done(r[0])
            out.append(r[0])
            processed += 1
        if processed == 0:
            buf.stall_pushes += 1
        else:
            buf.stall_pushes = 0
    return out


# --------------------------------------------------------------------------- #
# ① 零回归：连续输入逐帧一致
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 3])
def test_continuous_input_matches_old_implementation(k):
    new, old = FrameBuffer(k=k), OldFrameBuffer(k=k)
    seq_new = _follow_new(new, range(0, 200))
    seq_old = _follow(old, range(0, 200))
    assert seq_new == seq_old, (
        f"k={k} 连续输入必须与旧实现逐帧一致，否则是回归")


def test_continuous_input_processes_almost_all_frames():
    n_in = 200
    seq = _follow_new(FrameBuffer(k=1), range(0, n_in))
    assert len(seq) >= n_in - 3, f"连续输入应几乎全处理，实际 {len(seq)}/{n_in}"


# --------------------------------------------------------------------------- #
# ② 修 bug：stride 不整除 k 时不再卡死
# --------------------------------------------------------------------------- #
# 只保留 k % stride != 0 的组合（这些才是旧实现会卡死的场景）
STRIDE_STALL_CASES = [
    (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
    (3, 2), (5, 2), (6, 2),
]


@pytest.mark.parametrize("stride,k", STRIDE_STALL_CASES)
def test_stride_no_longer_stalls(stride, k):
    assert k % stride != 0, f"前置条件：stride={stride} 应不整除 k={k}"

    frames = list(range(0, 200, stride))
    n_in = len(frames)

    seq_old = _follow(OldFrameBuffer(k=k), frames)
    seq_new = _follow_new(FrameBuffer(k=k), frames)

    assert len(seq_old) == 0, (
        f"旧实现在 stride={stride},k={k} 应卡死在 0 帧，实际 {len(seq_old)}")
    # 边界损失：首尾各 k/stride 附近的帧无邻居，约 2*ceil(k/stride) 帧
    assert len(seq_new) >= n_in * 0.90, (
        f"stride={stride},k={k} 修复后应 >=90%，实际 "
        f"{len(seq_new)}/{n_in} = {100*len(seq_new)/n_in:.0f}%")


@pytest.mark.parametrize("stride,k", [(1, 2), (2, 2), (1, 3), (3, 3)])
def test_stride_dividing_k_is_unchanged(stride, k):
    """stride 整除 k 时旧实现本来就能跑，修复后必须行为一致。"""
    assert k % stride == 0
    frames = list(range(0, 200, stride))
    seq_new = _follow_new(FrameBuffer(k=k), frames)
    seq_old = _follow(OldFrameBuffer(k=k), frames)
    assert seq_new == seq_old, (
        f"stride={stride},k={k} 修复后必须与旧实现一致")


# --------------------------------------------------------------------------- #
# ③ 窗口语义：距离 ∈ [k, max_gap]
# --------------------------------------------------------------------------- #
def test_window_uses_nearest_within_bounds_when_gaps_exist():
    buf = FrameBuffer(k=1, max_gap=2)
    for f in (0, 2, 4, 6):
        buf.push(f)
    assert buf.window(2) == (0, 4), "应取最近邻（距离 2）"
    assert buf.window(4) == (2, 6)


def test_window_never_shrinks_baseline_below_k():
    """k=2 时绝不能返回 t±1 —— 那会把 3 帧基线跨度从 2 缩到 1。"""
    buf = FrameBuffer(k=2, max_gap=4)
    for f in range(0, 7):          # 只推 7 帧，避免把 t=4 挤出缓冲
        buf.push(f)
    assert buf.window(4) == (2, 6), "k=2 必须返回距离 2 的邻居"


def test_window_prefers_exact_k_over_larger_gap():
    buf = FrameBuffer(k=1, max_gap=4)
    for f in (0, 1, 2, 3, 4):
        buf.push(f)
    assert buf.window(2) == (1, 3), "t±1 都在时应选它们，而不是更远"


def test_window_rejects_gap_exceeding_max_gap():
    buf = FrameBuffer(k=1, max_gap=1)
    for f in (0, 5, 10):
        buf.push(f)
    assert buf.window(5) is None, "间隔 5 > max_gap=1，必须拒绝"


def test_window_rejects_missing_neighbor_at_edges():
    buf = FrameBuffer(k=1, max_gap=4)
    for f in (0, 4):
        buf.push(f)
    assert buf.window(0) is None, "首帧无前驱"
    assert buf.window(4) is None, "末帧无后继"


# --------------------------------------------------------------------------- #
# ④ 自适应 max_gap：连续输入退化为精确 k（零行为变化）
# --------------------------------------------------------------------------- #
def test_auto_max_gap_equals_k_for_continuous_input():
    buf = FrameBuffer(k=1)
    for f in range(0, 20):
        buf.push(f)
    assert buf.max_gap() == 1, "连续输入时 max_gap 应等于 k"


def test_auto_max_gap_follows_observed_stride():
    buf = FrameBuffer(k=1)
    for f in range(0, 40, 4):
        buf.push(f)
    assert buf.max_gap() == 4, "观测到 stride=4 时 max_gap 应自适应为 4"


def test_auto_max_gap_never_below_k():
    buf = FrameBuffer(k=3)
    for f in range(0, 20):
        buf.push(f)
    assert buf.max_gap() >= 3


def test_auto_max_gap_uses_median_against_occasional_drops():
    """偶发丢帧不应把 max_gap 抬上去（用中位数而非最大值）。"""
    buf = FrameBuffer(k=1)
    for f in (0, 1, 2, 4, 5):      # 3 丢了，gap = [1,2,1]
        buf.push(f)
    assert buf.max_gap() == 1, "偶发丢帧应被中位数过滤，max_gap 仍为 1"


# --------------------------------------------------------------------------- #
# ⑤ 可观测性：推进不了必须能喊出来
# --------------------------------------------------------------------------- #
def test_stall_counter_increases_when_not_advancing():
    buf = FrameBuffer(k=1, max_gap=1)
    _follow_new(buf, (0, 5, 10, 15, 20))
    assert buf.stall_pushes >= 3, "无法推进时计数必须增长，供节点打 WARN"


def test_stall_counter_resets_after_recovery():
    buf = FrameBuffer(k=1, max_gap=1)
    _follow_new(buf, (0, 5, 10))
    assert buf.stall_pushes > 0
    _follow_new(buf, (11, 12, 13))
    assert buf.stall_pushes == 0, "恢复推进后计数必须归零"


# --------------------------------------------------------------------------- #
# ⑥ 回绕：帧号回退时清空（循环回放）
# --------------------------------------------------------------------------- #
def test_frame_id_rewind_clears_buffer():
    buf = FrameBuffer(k=1)
    for f in (100, 101, 102, 103):
        buf.push(f)
    buf.push(0)                       # 视频循环，帧号回绕
    assert list(buf.frames.keys()) == [0], "回绕后缓冲应清空，只含新帧"


# --------------------------------------------------------------------------- #
# ⑦ buffer_size 下限保护
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k,requested,expected", [
    (1, 2, 3), (2, 2, 5), (2, 6, 6), (3, 4, 7),
])
def test_buffer_size_never_below_two_k_plus_one(k, requested, expected):
    buf = FrameBuffer(k=k, buffer_size=requested)
    assert buf.buffer_size == expected
