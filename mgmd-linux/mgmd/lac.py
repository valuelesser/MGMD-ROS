"""mgmd.lac — 局部外观分类器（论文 III-C1，SPEC T07）。

论文：3 层 CNN，输入 32×32，ReLU + MaxPool，softmax 二分类。

**权重可用性已独立验证（PRD §6.0b）**：官方 `Net_best.pth` 正常，
真实目标 6/9 命中（P≈1.0），背景 9/9 拒绝（P≈0.0）。
旧工程"权重失效"的结论已被推翻，不再假设必须重训。

网络结构按官方 `Functions.py` 的 `Net`（非 MyNet）：
    Conv(3,6,5) → MaxPool2 → Conv(6,16,5) → MaxPool2
    → Flatten(16·5·5=400) → Linear(120) → Linear(84) → Linear(2)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mgmd import config as C


def build_net():
    """官方 Net 结构（逐层对齐官方 Functions.py）。"""
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 2)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.reshape(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    return Net()


class LacClassifier:
    """LAC 推理。device 默认自动（有 CUDA 用 CUDA，否则 CPU）。"""

    def __init__(self, weights: Path | None = None, input_size: int = 32,
                 device: str = "auto"):
        import torch
        self.weights = Path(weights) if weights else (C.WEIGHTS_DIR / "Net_best.pth")
        if not self.weights.is_file():
            raise C.MgmdError("E_WEIGHTS_MISSING", f"LAC 权重不存在: {self.weights}",
                              "确认 /home/gaoduan/tracking/data_raw/mgmd_weights/ 存在",
                              C.EXIT_DATA)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.input_size = input_size

        net = build_net()
        state = torch.load(self.weights, map_location=self.device, weights_only=True)
        net.load_state_dict(state)
        self.net = net.to(self.device).eval()
        self._torch = torch

    def mav_prob(self, crop_bgr: np.ndarray) -> float:
        """单张裁剪 → P(MAV) ∈ [0,1]。保持官方 BGR 直喂口径。"""
        import cv2
        torch = self._torch
        s = self.input_size
        img = np.ascontiguousarray(crop_bgr)
        if img.shape[0] != s or img.shape[1] != s:
            img = cv2.resize(img, (s, s))
        # BGR HWC → RGB CHW / 255
        t = torch.from_numpy(img[:, :, ::-1].copy()).float().permute(2, 0, 1) / 255.0
        with torch.no_grad():
            logits = self.net(t.unsqueeze(0).to(self.device))
            prob = torch.softmax(logits, dim=1)
        return float(prob[0, 1].item())

    def mav_prob_batch(self, crops: list[np.ndarray]) -> list[float]:
        """批量推理，同帧多候选时更快。"""
        if not crops:
            return []
        import cv2
        torch = self._torch
        s = self.input_size
        arrs = []
        for c in crops:
            img = np.ascontiguousarray(c)
            if img.shape[0] != s or img.shape[1] != s:
                img = cv2.resize(img, (s, s))
            arrs.append(img[:, :, ::-1].copy())
        t = torch.from_numpy(np.stack(arrs)).float().permute(0, 3, 1, 2) / 255.0
        with torch.no_grad():
            logits = self.net(t.to(self.device))
            probs = torch.softmax(logits, dim=1)
        return [float(p) for p in probs[:, 1].cpu().numpy()]


def crop_box(frame: np.ndarray, box: tuple[float, float, float, float],
             pad: int = 2) -> np.ndarray | None:
    """按框裁图，四周扩 pad 并 clamp 到图像内。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(round(x2)) + pad)
    y2 = min(h, int(round(y2)) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]
