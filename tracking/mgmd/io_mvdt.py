"""mgmd.io_mvdt — MVDT 帧读取包装（SPEC §2.2/§3.1）。

复用 ``spa2d.io_mvdt.FrameSource`` 常驻 capture 顺序读帧约定（禁止逐帧 open/seek），
附加 ``runtime.downscale`` 降采样（D4 4K 默认 0.5，SPEC §1.3 D7）。
"""
from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

from spa2d.io_mvdt import FrameSource as _FrameSource

from mgmd.config import SequenceConfig, RuntimeConfig


class MgmdFrameSource:
    """FrameSource 包装：顺序读帧 + 可选 downscale（输出坐标仍以处理尺度计）。

    调用方在写记录/评测前需自行 ``÷downscale`` 映射回原图坐标（SPEC §5.8）。
    """

    def __init__(self, seq: SequenceConfig, runtime: RuntimeConfig,
                 use_video: bool = True) -> None:
        if not (0 < runtime.downscale <= 1):
            raise ValueError(f"downscale 必须在 (0,1]，实际 {runtime.downscale}")
        self.downscale = float(runtime.downscale)
        video = seq.video if (use_video and seq.video) else None
        frames_dir = None if video else seq.frames_dir
        self._src = _FrameSource(frames_dir=frames_dir, video=video, glob=seq.frame_glob)
        self.width = int(round(seq.resolution[0] * self.downscale))
        self.height = int(round(seq.resolution[1] * self.downscale))

    def __len__(self) -> int:
        return len(self._src)

    def _scale(self, img: np.ndarray) -> np.ndarray:
        if self.downscale == 1.0:
            return img
        return cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        for i, img in self._src:
            yield i, self._scale(img)

    def read(self, idx: int) -> np.ndarray:
        return self._scale(self._src.read(idx))

    def close(self) -> None:
        """释放底层 capture（视频）或文件句柄（帧目录）。"""
        self._src.close()
