"""spa2d.frame_cache — 主机预处理 LRU + 设备端窗口驻留（SPEC §8.2 / §8.3）。

- 主机 LRU 缓存预处理帧（float32，默认预算 2 GiB，可配置）：重叠窗口零重复预处理。
- 设备端仅驻留当前窗口的 N 帧（越界驱逐）。
统计 hits / misses / evictions / host_bytes / hit_rate 供 run_meta。
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

DEFAULT_MAX_BYTES = 2 * 1024 ** 3


class FrameCache:
    def __init__(self, source, kernel, color_order: str = "BGR",
                 max_bytes: int = DEFAULT_MAX_BYTES, backend=None) -> None:
        self.source = source
        self.kernel = kernel
        self.color_order = color_order
        self.max_bytes = int(max_bytes)
        self.backend = backend
        info = source.probe()
        self.height = int(info.height)
        self.width = int(info.width)
        self._host: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._host_bytes = 0
        self._device: dict[int, object] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def has_preproc(self, idx: int) -> bool:
        return idx in self._host

    def get(self, idx: int) -> np.ndarray:
        arr = self._host[idx]
        self._host.move_to_end(idx)
        self.hits += 1
        return arr

    def put(self, idx: int, arr: np.ndarray) -> None:
        self._host[idx] = arr
        self._host.move_to_end(idx)
        self._host_bytes += int(arr.nbytes)
        self.misses += 1
        self._evict()

    def _evict(self) -> None:
        while self._host_bytes > self.max_bytes and self._host:
            _, arr = self._host.popitem(last=False)
            self._host_bytes -= int(arr.nbytes)
            self.evictions += 1

    def device_window(self, idxs):
        """返回当前窗口的设备端帧堆叠 (N,H,W)；仅驻留窗口内 N 帧。"""
        keep = set(idxs)
        for k in list(self._device):
            if k not in keep:
                del self._device[k]
        be = self.backend
        for idx in idxs:
            if idx not in self._device:
                host = self._host[idx]
                self._device[idx] = be.asarray(host) if be is not None else host
        frames = [self._device[i] for i in idxs]
        if be is not None:
            return be.stack(frames)
        return np.stack(frames)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "host_bytes": self._host_bytes,
            "host_max_bytes": self.max_bytes,
            "host_frames": len(self._host),
            "device_frames": len(self._device),
            "hit_rate": (self.hits / total) if total else 0.0,
        }
