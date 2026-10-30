"""mgmd_onnx.tf_fast — SortTracker 向量化加速（SPEC-ROS §14.5）。

**瓶颈来源**：`SortTracker.update` 中
    for i, tid in enumerate(active):
        for j, d in enumerate(dets):
            gain[i, j] = iou(preds[tid], d)
实测 15 轨迹 × 131 候选 × 290 帧 = **1127 万次 `iou()`，占 TF 耗时 69%**
（cProfile: cumtime 16.1s / 23.4s）。

**做法**：继承 `SortTracker`，覆写 `update`，**只把 gain 矩阵改成一次向量化调用**，
其余逻辑逐行照抄原实现。`mgmd/` 不改（V7）。

**验收**：SPEC-ROS §14.6 S3-c —— 与 SortTracker 逐帧输出差必须为 0。
"""
from __future__ import annotations

import numpy as np

from mgmd.tf import SortTracker as _SortTracker
from mgmd.tf import Track, TrackState
from mgmd.kalman import KalmanBox


def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """向量化 IoU。A (N,4), B (M,4) → (N,M)，与标量 iou 逐元素语义一致。"""
    ix1 = np.maximum(A[:, None, 0], B[None, :, 0])
    iy1 = np.maximum(A[:, None, 1], B[None, :, 1])
    ix2 = np.minimum(A[:, None, 2], B[None, :, 2])
    iy2 = np.minimum(A[:, None, 3], B[None, :, 3])
    iw = ix2 - ix1
    ih = iy2 - iy1
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    ua = np.maximum(0.0, A[:, 2] - A[:, 0]) * np.maximum(0.0, A[:, 3] - A[:, 1])
    ub = np.maximum(0.0, B[:, 2] - B[:, 0]) * np.maximum(0.0, B[:, 3] - B[:, 1])
    u = ua[:, None] + ub[None, :] - inter
    return np.where(u > 0, inter / np.where(u > 0, u, 1.0), 0.0)


class FastSortTracker(_SortTracker):
    """输出与 SortTracker 逐帧一致；IoU 关联矩阵向量化。"""

    def update(self, dets, frame, scores=None, step_H=None):
        from scipy.optimize import linear_sum_assignment

        scores = scores or [1.0] * len(dets)
        active = list(self.states.keys())
        prev_free_snapshot = list(self._prev_free) if self.init_velocity else None

        # 预测
        preds = {}
        for tid in active:
            preds[tid] = self._kal[tid].predict()

        # ---- 唯一改动：gain 向量化 ----
        matched: list[tuple[int, int]] = []
        if dets and active:
            A = np.array([preds[tid] for tid in active], dtype=np.float64)
            B = np.array(dets, dtype=np.float64)
            gain = iou_matrix(A, B)
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
            self.states[tid] = TrackState(track_id=tid,
                                          buffer_size=max(4, self.span + 1))
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

        if self.init_velocity:
            self._prev_free = list(dets)

        # 收集输出
        out: list[Track] = []
        det_score = {dj: scores[dj] for dj in range(len(dets))}
        tid_to_det = {tid: dj for tid, dj in matched}
        for tid, st in self.states.items():
            if self._hits.get(tid, 0) < self.min_hits:
                continue
            if self._miss.get(tid, 0) > 0:
                continue
            if st.probability != 1:
                continue
            out.append(Track(
                track_id=tid, box=st.boxes[-1][1], age=self._age.get(tid, 0),
                hits=self._hits.get(tid, 0), velocity=self._kal[tid].velocity,
                confirmed=True, d_value=st.d_value,
                score=det_score.get(tid_to_det.get(tid, -1), 0.0),
                miss_count=self._miss.get(tid, 0),
            ))
        return out
