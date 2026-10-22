"""mgmd.kalman — SORT 用的 7 维匀速 Kalman 滤波（SORT 标准实现口径）。

状态 [cx, cy, s, r, vx, vy, vs]，s=面积，r=宽高比。
纯 numpy，不依赖 filterpy，便于审计（SPEC §8 T05）。
过程/测量噪声沿用 SORT 论文参考实现（Bewley et al. 2016）的超参。
"""
from __future__ import annotations

import numpy as np


class KalmanBox:
    """单目标 Kalman 跟踪器。"""

    def __init__(self, box: tuple[float, float, float, float]):
        self.dim_x, self.dim_z = 7, 4
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01

        self.x = np.zeros((7, 1))
        self.x[:4] = self._box_to_z(box)

    @staticmethod
    def _box_to_z(box) -> np.ndarray:
        x1, y1, x2, y2 = box
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        return np.array([[(x1 + x2) / 2.0], [(y1 + y2) / 2.0], [w * h], [w / h]])

    def _z_to_box(self, z: np.ndarray) -> tuple[float, float, float, float]:
        z = np.asarray(z).ravel()
        cx, cy, s, r = float(z[0]), float(z[1]), max(float(z[2]), 1e-6), max(float(z[3]), 1e-6)
        w = float(np.sqrt(s * r))
        h = s / max(w, 1e-6)
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def predict(self) -> tuple[float, float, float, float]:
        if float(self.x[2, 0]) + float(self.x[6, 0]) <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._z_to_box(self.x)

    def update(self, box: tuple[float, float, float, float]) -> None:
        z = self._box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self._z_to_box(self.x)

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.x[4, 0]), float(self.x[5, 0]))
