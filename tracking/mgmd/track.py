"""mgmd.track — SORT 跟踪 + 论文 Algorithm 1 轨迹分类（SPEC §5.3/§5.4）。

SORT：7 维状态 [cx,cy,s,r,vx,vy,vs] 匀速 Kalman（filterpy），Hungarian 关联
（scipy linear_sum_assignment），IoU 阈值/生命周期参数见 TfConfig
（iou_thresh=0.3 / max_age=3 / min_hits=1，FR-4）。

Algorithm 1（FR-5）：轨迹 age≥alg1_age 且对齐后 D(s_{N−w},s_N)=ρ/d ≥ alg1_d_min
判为真实运动目标；ρ=历史窗口两端点位移，d=当前框对角线（SPEC 假设 A2）。
位置对齐：update 接受可选 align_H（上一帧→当前帧坐标系的累计单应），
存储前用它把全部历史点变换到当前帧坐标系；静态相机 ≈ 恒等。

依赖懒加载：scipy/filterpy 缺失 → E_DEP_MISSING（退出码 4；FR-11）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mgmd.config import MgmdError, TfConfig


def _require_tf_deps():
    """懒加载 scipy/filterpy；缺失 → E_DEP_MISSING（退出码 4，SPEC §6.1）。"""
    try:
        import scipy.optimize  # noqa: F401
        import filterpy.kalman  # noqa: F401
    except ImportError as exc:
        raise MgmdError("E_DEP_MISSING",
                        f"TF 依赖缺失: {exc.name}",
                        ".venv/bin/pip install -r env/requirements.txt（scipy/filterpy 组）",
                        4) from exc


@dataclass
class Track:
    id: int
    rect: tuple[float, float, float, float]  # (x, y, w, h)
    age: int
    hits: int
    velocity: tuple[float, float]
    confirmed: bool                          # Algorithm 1 通过（SPEC §5.4）
    history: list[tuple[float, float]] = field(default_factory=list)
    score: float = 0.0                       # 最近一次关联候选的运动得分（选择/AP 用）


def rect_to_z(rect: tuple[float, float, float, float]) -> np.ndarray:
    """(x,y,w,h) → [cx, cy, s=w*h, r=w/h]。"""
    x, y, w, h = (float(v) for v in rect)
    s = w * h
    r = w / max(h, 1e-6)
    return np.array([x + w / 2.0, y + h / 2.0, s, r], dtype=np.float64).reshape(4, 1)


def x_to_rect(x: np.ndarray) -> tuple[float, float, float, float]:
    """Kalman 状态 [cx,cy,s,r,...] → (x,y,w,h)；s 负值/NaN 防护。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    cx, cy, s, r = float(x[0]), float(x[1]), max(float(x[2]), 1e-6), float(x[3])
    r = max(r, 1e-6)
    w = float(np.sqrt(s * r))
    h = s / max(w, 1e-6)
    return (cx - w / 2.0, cy - h / 2.0, w, h)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """(x,y,w,h) IoU（SORT 关联代价）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class _KalmanBoxTracker:
    """单目标 Kalman 跟踪器（SORT 经典实现口径）。"""

    count = 0

    def __init__(self, rect: tuple[float, float, float, float]) -> None:
        from filterpy.kalman import KalmanFilter

        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.eye(7)
        kf.F[0, 4] = kf.F[1, 5] = kf.F[2, 6] = 1.0
        kf.H = np.zeros((4, 7))
        kf.H[:4, :4] = np.eye(4)
        kf.R[2:, 2:] *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        kf.x[:4] = rect_to_z(rect)
        self.kf = kf
        self.id = _KalmanBoxTracker.count
        _KalmanBoxTracker.count += 1
        self.age = 0
        self.hits = 1
        self.hit_streak = 0
        self.time_since_update = 0
        self.last_score = 0.0
        self.ever_confirmed = False   # Algorithm 1 轨迹级粘性确认（SWE 见 _alg1_confirmed）
        self.history: list[tuple[float, float]] = [self.center()]

    def center(self) -> tuple[float, float]:
        x = self.kf.x.ravel()
        return (float(x[0]), float(x[1]))

    def predict(self) -> tuple[float, float, float, float]:
        x = self.kf.x.ravel()
        if float(x[6]) + float(x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_rect(self.kf.x)

    def update(self, rect: tuple[float, float, float, float], score: float = 0.0) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_score = float(score)
        self.kf.update(rect_to_z(rect))
        self.history.append(self.center())

    @property
    def velocity(self) -> tuple[float, float]:
        x = self.kf.x.ravel()
        return (float(x[4]), float(x[5]))

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return x_to_rect(self.kf.x)


class SortTracker:
    """SORT 管理器 + Algorithm 1 轨迹分类（SPEC §5.3/§5.4）。"""

    def __init__(self, cfg: TfConfig) -> None:
        _require_tf_deps()
        self.cfg = cfg
        self.trackers: list[_KalmanBoxTracker] = []
        self.frame_count = 0

    def _alg1_confirmed(self, trk: _KalmanBoxTracker) -> bool:
        """论文 Algorithm 1 轨迹分类（SPEC §5.4）：轨迹级判定，一旦满足即粘性保持
        （MVDT 慢目标存在 ρ→0 的相位；逐帧判定会使已确认轨迹反复失格）。"""
        if trk.ever_confirmed:
            return True
        if trk.age < self.cfg.alg1_age or len(trk.history) < 2:
            return False
        w = min(self.cfg.alg1_window, len(trk.history) - 1)
        p_now = trk.history[-1]
        p_ref = trk.history[-1 - w]
        rho = float(np.hypot(p_now[0] - p_ref[0], p_now[1] - p_ref[1]))
        _x, _y, bw, bh = trk.rect
        d = float(np.hypot(bw, bh))
        if d > 0 and rho / d >= self.cfg.alg1_d_min:
            trk.ever_confirmed = True
        return trk.ever_confirmed

    def update(self, rects: list[tuple[int, int, int, int]], frame_idx: int,
               align_H: np.ndarray | None = None,
               scores: list[float] | None = None) -> list[Track]:
        """SORT 关联更新；align_H 为历史位置→当前帧坐标系的累计单应（None=恒等）。

        scores：与 rects 对齐的候选运动得分（用于选择/AP 排序键，缺省 0）。
        返回本帧活跃（min_hits 达标且本帧有匹配/新建）轨迹，含 Algorithm 1 confirmed。
        """
        from scipy.optimize import linear_sum_assignment

        self.frame_count += 1
        # 历史位置对齐到当前帧坐标系（align_H 语义：上一帧→当前帧；
        # pipeline 已对 align() 返回的当前→上一帧 H 取逆）
        if align_H is not None and not np.allclose(align_H, np.eye(3)):
            for trk in self.trackers:
                pts = np.array(trk.history, dtype=np.float64).reshape(-1, 1, 2)
                import cv2
                trk.history = [(float(p[0]), float(p[1])) for p in
                               cv2.perspectiveTransform(pts, align_H).reshape(-1, 2)]

        # 预测
        trk_preds: list[tuple[float, float, float, float]] = []
        alive: list[_KalmanBoxTracker] = []
        for trk in self.trackers:
            pred = trk.predict()
            if np.any(np.isnan(pred)):
                trk.time_since_update = self.cfg.max_age + 1  # NaN 预测直接淘汰
                continue
            trk_preds.append(pred)
            alive.append(trk)
        self.trackers = alive

        # Hungarian 关联（IoU 代价）
        dets = [tuple(map(float, r)) for r in rects]
        matched: list[tuple[int, int]] = []
        unmatched_dets = set(range(len(dets)))
        unmatched_trks = set(range(len(self.trackers)))
        if dets and self.trackers:
            cost = np.zeros((len(self.trackers), len(dets)), dtype=np.float64)
            for t, trk in enumerate(self.trackers):
                for d, det in enumerate(dets):
                    cost[t, d] = 1.0 - iou(trk_preds[t], det)
            row_ind, col_ind = linear_sum_assignment(cost)
            for t, d in zip(row_ind, col_ind):
                if 1.0 - cost[t, d] < self.cfg.iou_thresh:
                    continue
                matched.append((int(t), int(d)))
                unmatched_trks.discard(int(t))
                unmatched_dets.discard(int(d))

        # 更新匹配轨迹
        for t, d in matched:
            self.trackers[t].update(dets[d], (scores[d] if scores is not None else 0.0))
        # 新建轨迹
        for d in unmatched_dets:
            new_trk = _KalmanBoxTracker(dets[d])
            if scores is not None:
                new_trk.last_score = float(scores[d])
            self.trackers.append(new_trk)
        # 删除超龄轨迹
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.cfg.max_age]

        out: list[Track] = []
        for trk in self.trackers:
            if trk.hits < self.cfg.min_hits:
                continue
            # SPEC D16：SORT 滑行输出——本帧有匹配（tsu==0）输出；已确认轨迹在
            # 短暂未匹配帧（tsu≤max_age）输出 Kalman 预测框，填补帧差无信号相位
            if trk.time_since_update == 0 or trk.ever_confirmed:
                out.append(Track(id=trk.id, rect=trk.rect, age=trk.age, hits=trk.hits,
                                 velocity=trk.velocity, confirmed=self._alg1_confirmed(trk),
                                 history=list(trk.history), score=trk.last_score))
        return out

    def reset_ids(self) -> None:
        _KalmanBoxTracker.count = 0
