"""spa2d.timing — 分层计时（SPEC §5.9 / §4.2）。

层：T_pre（预处理）、T_search（候选+累加+得分+峰值+置信）、T_state（门控+状态机+日志）、
T_decode（视频解码/磁盘 IO，仅 end-to-end 口径）。

    window_fps          = 1 / mean(T_window)，T_window = T_pre + T_search + T_state
    effective_frame_fps = stride / mean(T_window)

停表前对 JAX 后端做设备同步（effects_barrier）。warmup 默认排除前 20 窗；
统计输出 mean / median / p95（以及 min / max / count）。
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from spa2d.records import TimingStats

LAYERS = ("t_pre", "t_search", "t_state", "t_decode")


class Timer:
    def __init__(self, backend=None, warmup: int = 20) -> None:
        self.backend = backend
        self.warmup = int(warmup)
        self.times: dict[str, list[float]] = {layer: [] for layer in LAYERS}

    def _sync(self) -> None:
        be = self.backend
        if be is None or getattr(be, "name", "") != "jax":
            return
        try:
            be.jax.effects_barrier()
        except Exception:  # noqa: BLE001
            try:
                be.jax.block_until_ready(be.jnp.zeros(()))
            except Exception:  # noqa: BLE001
                pass

    @contextmanager
    def tick(self, layer: str):
        if layer not in self.times:
            raise ValueError(f"未知计时层 {layer!r}")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.times[layer].append((time.perf_counter() - t0) * 1000.0)

    def add(self, layer: str, ms: float = 0.0) -> None:
        self.times[layer].append(float(ms))

    def n_windows(self) -> int:
        return len(self.times["t_pre"])

    def window_totals(self) -> np.ndarray:
        return (np.asarray(self.times["t_pre"]) + np.asarray(self.times["t_search"])
                + np.asarray(self.times["t_state"]))

    def _slice(self, layer: str, warmup: int | None) -> np.ndarray:
        w = self.warmup if warmup is None else warmup
        return np.asarray(self.times[layer], dtype=np.float64)[w:]

    def summary(self, warmup: int | None = None) -> dict[str, TimingStats]:
        out: dict[str, TimingStats] = {}
        for layer in LAYERS:
            vals = self._slice(layer, warmup)
            out[layer] = _stats(layer, vals)
        totals = self.window_totals()[self.warmup if warmup is None else warmup:]
        out["t_window"] = _stats("t_window", totals)
        return out

    def fps(self, stride: int, warmup: int | None = None) -> dict[str, float]:
        w = self.warmup if warmup is None else warmup
        totals = self.window_totals()[w:]
        e2e = (self.window_totals() + np.asarray(self.times["t_decode"]))[w:]
        mean_w = float(totals.mean()) if totals.size else float("nan")
        mean_e = float(e2e.mean()) if e2e.size else float("nan")
        return {
            "stride": int(stride),
            "mean_t_window_ms": mean_w,
            "window_fps": (1000.0 / mean_w) if mean_w > 0 else float("nan"),
            "effective_frame_fps": (1000.0 * stride / mean_w) if mean_w > 0 else float("nan"),
            "mean_e2e_window_ms": mean_e,
            "e2e_window_fps": (1000.0 / mean_e) if mean_e > 0 else float("nan"),
            "e2e_effective_frame_fps": (1000.0 * stride / mean_e) if mean_e > 0 else float("nan"),
        }


def _stats(layer: str, vals: np.ndarray) -> TimingStats:
    if vals.size == 0:
        return TimingStats(layer=layer, count=0, mean=float("nan"), median=float("nan"),
                           p95=float("nan"), minimum=float("nan"), maximum=float("nan"))
    return TimingStats(
        layer=layer, count=int(vals.size), mean=float(vals.mean()),
        median=float(np.median(vals)), p95=float(np.percentile(vals, 95)),
        minimum=float(vals.min()), maximum=float(vals.max()))


def summary(times_ms, warmup: int = 20) -> TimingStats:
    """便捷函数：对单层时间序列计算统计（排除前 warmup）。"""
    return _stats("layer", np.asarray(times_ms, dtype=np.float64)[warmup:])
