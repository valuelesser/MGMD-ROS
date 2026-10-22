"""mgmd.classify — LAC 局部外观分类器（SPEC §5.5）。

官方 Functions.py 的 ``Net``（注意非 ``MyNet``）：
Conv(3,6,5)→MaxPool2→Conv(6,16,5)→MaxPool2→Flatten(16·5·5)→Linear(120)→Linear(84)→Linear(2)。
输入 crop resize 32×32 + torchvision ToTensor（/255，保持官方 BGR 直喂口径）；
softmax 后取 P(class=1) 为 MAV 概率；prob_thresh 默认 0.5（官方 argmax 等价）。

torch/torchvision 懒加载（FR-11）：缺失仅影响本模块与对应测试。
权重缺失 → E_WEIGHTS_MISSING（退出码 3，指向 fetch_mgmd_weights.sh）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mgmd.config import LacConfig, MgmdError


def _require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise MgmdError("E_DEP_MISSING",
                        f"LAC 依赖缺失: {exc.name}",
                        ".venv/bin/pip install -r env/requirements.txt（torch 组）",
                        4) from exc


def _build_net():
    """官方 Net 结构（Functions.py:422-439，逐层对齐）。"""
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
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    return Net


def resolve_device(device: str) -> str:
    """auto → cuda（可用时）否则 cpu。"""
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class LacClassifier:
    """LAC：Net_best.pth 推理，输出 MAV 概率 ∈[0,1]。"""

    def __init__(self, cfg: LacConfig) -> None:
        _require_torch()
        if not Path(cfg.weights).is_file():
            raise MgmdError("E_WEIGHTS_MISSING",
                            f"LAC 权重不存在: {cfg.weights}",
                            "bash scripts/fetch_mgmd_weights.sh 下载并校验官方权重",
                            3)
        import torch

        self.cfg = cfg
        self.device = torch.device(resolve_device(cfg.device))
        Net = _build_net()
        self._model = Net()
        state = torch.load(cfg.weights, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.to(self.device).eval()

    def mav_prob(self, crop_bgr: np.ndarray) -> float:
        """crop（BGR，任意尺寸）→ P(MAV) ∈ [0,1]。"""
        import torch
        from torchvision import transforms

        size = self.cfg.input_size
        img = np.ascontiguousarray(crop_bgr)
        if img.shape[0] != size or img.shape[1] != size:
            import cv2
            img = cv2.resize(img, (size, size))
        tensor = transforms.ToTensor()(img)          # HWC→CHW，/255（官方口径）
        tensor = torch.unsqueeze(tensor, dim=0).to(self.device)
        with torch.no_grad():
            logits = torch.squeeze(self._model(tensor))
            prob = torch.softmax(logits, dim=0)
        return float(prob[1].item())

    def is_mav(self, crop_bgr: np.ndarray) -> bool:
        return self.mav_prob(crop_bgr) >= self.cfg.prob_thresh
