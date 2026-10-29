"""mgmd.tf — 轨迹过滤（论文 III-B，SPEC §8 T05）。

两部分：
    III-B1 轨迹生成   : SORT（Kalman 匀速模型 + 匈牙利 IoU 关联）
    III-B2 轨迹分类   : Algorithm 1 / 式(5) 的 D = ρ / d

**SORT 参数严格按论文**：IoU 0.3、max_age 3、min_hits 1。

**D 的两种定义（SPEC D-02）**：
    enclosing : 论文原文 —— d = 两框最小外包框的对角线
    self      : 替代定义 —— d = 两框各自对角线的均值
论文的 ξε(5) 在 enclosing 下恒有 D<1（ρ<d 恒成立），因此 D≥1 永不可达。
用 §6.0 的实测证据（28,096 对真实 GT，0 通过）支撑此结论，故提供双模式：
    literal  : d_min = 1.0，如实记录"0 条通过"
    calibrated : d_min 从**训练集**标定，测试集不参与

**不使用**旧工程/tracking 的 `ever_confirmed` 粘性标志和"每序列一个阈值"做法（SPEC D-11/D-12）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# 几何
# --------------------------------------------------------------------------- #

def box_centre(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def box_diagonal(b: Box) -> float:
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def distance_metric(a: Box, b: Box, mode: str = "enclosing") -> float:
    """式(5)：D = ρ / d。

    mode="enclosing"：d = 最小外包框对角线（论文原文）
    mode="self"    ：d = 两框对角线均值（替代定义）
    """
    ca, cb = box_centre(a), box_centre(b)
    rho = float(np.hypot(cb[0] - ca[0], cb[1] - ca[1]))
    if mode == "enclosing":
        ex1, ey1 = min(a[0], b[0]), min(a[1], b[1])
        ex2, ey2 = max(a[2], b[2]), max(a[3], b[3])
        d = float(np.hypot(ex2 - ex1, ey2 - ey1))
    elif mode == "self":
        d = 0.5 * (box_diagonal(a) + box_diagonal(b))
    else:
        raise ValueError(f"未知 d mode: {mode}")
    return rho / d if d > 0 else 0.0


def transform_box(b: Box, H: np.ndarray) -> Box | None:
    """经单应变换后取轴对齐外接框。"""
    import cv2
    pts = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]],
                   dtype=np.float32).reshape(-1, 1, 2)
    m = cv2.perspectiveTransform(pts, H)
    if not np.isfinite(m).all():
        return None
    m = m.reshape(4, 2)
    x1, y1 = float(m[:, 0].min()), float(m[:, 1].min())
    x2, y2 = float(m[:, 0].max()), float(m[:, 1].max())
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (x1, y1, x2, y2)


# --------------------------------------------------------------------------- #
# Algorithm 1
# --------------------------------------------------------------------------- #

@dataclass
class TrackState:
    """一条轨迹在 Algorithm 1 意义下的状态。

    按 Algorithm 1 step 2，S 是**有界滑动缓冲**：新框追加到末尾，满则淘汰最老。
    `lifetime` 是轨迹存活帧数（f(S)≥3 的门控），不受缓冲淘汰影响。
    """
    track_id: int
    buffer_size: int = 4          # span+1，保证能取到 N-3
    boxes: list = field(default_factory=list)       # [(frame, box)]，oldest first
    step_H: dict = field(default_factory=dict)      # frame f: f-1 -> f 的单应
    total_updates: int = 0

    probability: int = 0
    d_value: float = 0.0
    reason: str = "cold_start"

    def add(self, frame: int, box: Box) -> None:
        self.boxes.append((frame, box))
        self.total_updates += 1
        while len(self.boxes) > self.buffer_size:
            self.boxes.pop(0)

    @property
    def lifetime(self) -> int:
        return self.total_updates

    def window(self, span: int = 3):
        """取 [s_{N-span}, s_N] 对。返回 (older, newest) 或 None。"""
        if len(self.boxes) < 2:
            return None
        newest_frame, newest_box = self.boxes[-1]
        target = newest_frame - span
        older = None
        for f, b in self.boxes[:-1]:
            if f <= target and (older is None or f > older[0]):
                older = (f, b)
        if older is None:
            older = self.boxes[0]
            if older[0] >= newest_frame:
                return None
        return older, (newest_frame, newest_box)

    def compose_H(self, f0: int, f1: int):
        """级联单应 f0 -> f1。缺任一步 → None。"""
        H = None
        for f in range(f0 + 1, f1 + 1):
            step = self.step_H.get(f)
            if step is None:
                return None
            H = step if H is None else step @ H
        return H

    def classify(self, span: int, d_mode: str, d_min: float, align: bool = True) -> int:
        """跑 Algorithm 1 step 3-8。"""
        if self.lifetime < 3:
            self.probability, self.d_value, self.reason = 0, 0.0, "lifetime_lt_3"
            return 0
        pair = self.window(span)
        if pair is None:
            self.probability, self.d_value, self.reason = 0, 0.0, "no_window"
            return 0

        (f_old, b_old), (f_new, b_new) = pair
        b_old_use = b_old
        if align and f_old != f_new:
            H = self.compose_H(f_old, f_new)
            if H is not None:
                tb = transform_box(b_old, H)
                if tb is not None:
                    b_old_use = tb
                else:
                    self.reason = "align_degenerate"
            else:
                self.reason = "align_missing"

        self.d_value = distance_metric(b_old_use, b_new, d_mode)
        self.probability = 1 if self.d_value >= d_min else 0
        if self.probability == 0:
            self.reason = "d_below_threshold"
        else:
            self.reason = "confirmed"
        return self.probability


# --------------------------------------------------------------------------- #
# SORT
# --------------------------------------------------------------------------- #

@dataclass
class Track:
    track_id: int
    box: Box
    age: int
    hits: int
    velocity: tuple[float, float]
    confirmed: bool = False
    d_value: float = 0.0
    score: float = 0.0
    miss_count: int = 0


class SortTracker:
    """SORT + Algorithm 1 轨迹分类。

    sort 参数（论文）：iou_thresh=0.3, max_age=3, min_hits=1
    alg1 参数（论文）：span=3（s_{N-3}）, 寿命门控 3, d_min 由 D 模式决定
    """

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 3, min_hits: int = 1,
                 span: int = 3, d_mode: str = "enclosing", d_min: float = 1.0,
                 align: bool = True, init_velocity: bool = False,
                 init_v_search: float = 1.5):
        self.iou_thresh = iou_thresh
        # 方案 B（SPEC §10.10.6）：新轨迹用上一帧未关联候选初始化速度。
        # 论文为 False（KalmanBox 初速度恒为 0）；开启后偏离论文，需成对报告（D-16）。
        self.init_velocity = init_velocity
        self.init_v_search = init_v_search   # 搜索半径 = 该倍数 × 新框对角线
        self._prev_free: list[Box] = []      # 上一帧未被任何轨迹关联的候选
        self.max_age = max_age
        self.min_hits = min_hits
        self.span = span
        self.d_mode = d_mode
        self.d_min = d_min
        self.align = align
        self.states: dict[int, TrackState] = {}
        self._kal: dict[int, object] = {}
        self._next_id = 1
        self._age: dict[int, int] = {}
        self._hits: dict[int, int] = {}
        self._miss: dict[int, int] = {}

    def _init_kalman_velocity(self, kal, box: Box, history: list[Box]) -> None:
        """用上一帧最近的未关联候选，给新轨迹一个初速度（SPEC §10.10.6 方案 B）。

        论文口径下 KalmanBox 初速度恒为 0，新轨迹首次 predict() 框停在原地，
        位移超过 0.538×框宽即关联失败（§10.10.3）。此处用最近邻候选补偿。

        只改新轨迹的初始化状态，不改 SORT 的关联阈值与 IoU 度量。
        """
        import numpy as np
        cx, cy = box_centre(box)
        diag = box_diagonal(box)
        radius = self.init_v_search * max(diag, 1.0)
        best, best_d = None, radius
        for pb in history:
            px, py = box_centre(pb)
            d = float(np.hypot(cx - px, cy - py))
            if d < best_d:
                best, best_d = pb, d
        if best is None:
            return
        px, py = box_centre(best)
        # 速度单位：像素/帧（SORT 状态里 vx,vy 即帧间位移）
        kal.x[4, 0] = cx - px
        kal.x[5, 0] = cy - py

    def update(self, dets: list[Box], frame: int, scores: list[float] | None = None,
               step_H: np.ndarray | None = None) -> list[Track]:
        """关联 + 更新。返回本帧输出轨迹（已过 min_hits 与 Algorithm 1）。"""
        from scipy.optimize import linear_sum_assignment
        from mgmd.kalman import KalmanBox

        scores = scores or [1.0] * len(dets)
        active = list(self.states.keys())
        # 本帧候选快照（用于下一帧的新轨迹速度初始化）
        prev_free_snapshot = list(self._prev_free) if self.init_velocity else None

        # 预测
        preds = {}
        for tid in active:
            preds[tid] = self._kal[tid].predict()

        # 匈牙利关联（IoU 收益最大化）
        matched: list[tuple[int, int]] = []
        if dets and active:
            gain = np.zeros((len(active), len(dets)))
            for i, tid in enumerate(active):
                for j, d in enumerate(dets):
                    gain[i, j] = iou(preds[tid], d)
            rows, cols = linear_sum_assignment(-gain)
            for r, c in zip(rows, cols):
                if gain[r, c] >= self.iou_thresh:
                    matched.append((active[r], int(c)))

        matched_tids = {t for t, _ in matched}
        matched_dets = {d for _, d in matched}

        # 更新匹配
        for tid, dj in matched:
            box = dets[dj]
            self._kal[tid].update(box)
            self._hits[tid] += 1
            self._miss[tid] = 0
            self.states[tid].add(frame, box)
            if step_H is not None:
                self.states[tid].step_H[frame] = step_H
            self.states[tid].classify(self.span, self.d_mode, self.d_min, self.align)

        # 未匹配轨迹：老化
        for tid in active:
            if tid in matched_tids:
                continue
            self._miss[tid] += 1
            self._age[tid] += 1
            if step_H is not None:
                self.states[tid].step_H[frame] = step_H

        # 新建轨迹
        for dj, box in enumerate(dets):
            if dj in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self.states[tid] = TrackState(track_id=tid, buffer_size=max(4, self.span + 1))
            self.states[tid].add(frame, box)
            if step_H is not None:
                self.states[tid].step_H[frame] = step_H
            self._kal[tid] = KalmanBox(box)
            if self.init_velocity and prev_free_snapshot:
                self._init_kalman_velocity(self._kal[tid], box, prev_free_snapshot)
            self._age[tid] = 1
            self._hits[tid] = 1
            self._miss[tid] = 0
            self.states[tid].classify(self.span, self.d_mode, self.d_min, self.align)

        # 删除超龄
        for tid in list(self.states.keys()):
            if self._miss.get(tid, 0) > self.max_age:
                del self.states[tid]
                self._kal.pop(tid, None)
                self._age.pop(tid, None)
                self._hits.pop(tid, None)
                self._miss.pop(tid, None)

        # 本帧候选存回，供下一帧初始化速度（用全部候选，理由见 §10.10.6）
        if self.init_velocity:
            self._prev_free = list(dets)

        # 收集输出：本帧有匹配、过 min_hits、Algorithm 1 确认
        out: list[Track] = []
        det_score = {dj: scores[dj] for dj in range(len(dets))}
        tid_to_det = {tid: dj for tid, dj in matched}
        for tid, st in self.states.items():
            if self._hits.get(tid, 0) < self.min_hits:
                continue
            if self._miss.get(tid, 0) > 0:
                continue                      # 只输出本帧有检测的（不滑行，SPEC D-12）
            if st.probability != 1:
                continue                      # Algorithm 1 未确认
            out.append(Track(
                track_id=tid, box=st.boxes[-1][1], age=self._age.get(tid, 0),
                hits=self._hits.get(tid, 0), velocity=self._kal[tid].velocity,
                confirmed=True, d_value=st.d_value,
                score=det_score.get(tid_to_det.get(tid, -1), 0.0),
                miss_count=self._miss.get(tid, 0),
            ))
        return out
