"""spa2d.backends — jax / numpy 后端选择与统一接口（SPEC §2.3、§8.2）。

统一函数签名（由 `get_backend(name)` 返回的对象提供）：
    shift_bilinear(img, dx, dy) -> (值, valid 权重)
    accumulate_window(frames, v) -> (A, M)
    score_batch(frames, hyps) -> scores
    block_until_ready(value)   # JAX 计时：停表前强制设备同步
    asarray(array)

JAX 后端对 `accumulate_window` 与 `score_batch` 做 jit 封装（编译时间由调用方单列）。
"""
from __future__ import annotations

import numpy as np

from spa2d import accumulate as A

_ALL = ("jax", "numpy")


def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
        return jax, jnp
    except Exception:  # noqa: BLE001
        return None, None


def jax_available() -> bool:
    jax, _ = _jax_modules()
    return jax is not None


def available_backends() -> tuple[str, ...]:
    return ("jax", "numpy") if jax_available() else ("numpy",)


class NumpyBackend:
    name = "numpy"

    def shift_bilinear(self, img, dx, dy):
        return A.shift_bilinear(img, dx, dy, xp=np)

    def accumulate_window(self, frames, v):
        return A.accumulate_window(frames, v, xp=np)

    def score_batch(self, frames, hyps):
        return A.score_batch(frames, hyps, xp=np)

    def accumulate_window_accel(self, frames, v, a):
        return A.accumulate_window_accel(frames, v, a, xp=np)

    def score_batch_accel(self, frames, hyps):
        return A.score_batch_accel(frames, hyps, xp=np)

    def score_batch_chunked(self, frames, hyps, chunk_size):
        return A.score_batch_chunked(frames, hyps, chunk_size, xp=np)

    def asarray(self, array):
        return np.asarray(array)

    def stack(self, arrays):
        return np.stack(arrays)

    def block_until_ready(self, value):
        return value


class JaxBackend:
    name = "jax"

    def __init__(self, jax, jnp):
        self.jax = jax
        self.jnp = jnp
        self._accum = jax.jit(lambda frames, v: A.accumulate_window(frames, v, xp=jnp))
        self._score = jax.jit(lambda frames, hyps: A.score_batch(frames, hyps, xp=jnp))
        self._accum_a = jax.jit(lambda frames, v, a: A.accumulate_window_accel(frames, v, a, xp=jnp))
        self._score_a = jax.jit(lambda frames, hyps: A.score_batch_accel(frames, hyps, xp=jnp))

    def shift_bilinear(self, img, dx, dy):
        return A.shift_bilinear(img, dx, dy, xp=self.jnp)

    def accumulate_window(self, frames, v):
        return self._accum(frames, v)

    def score_batch(self, frames, hyps):
        return self._score(frames, hyps)

    def accumulate_window_accel(self, frames, v, a):
        return self._accum_a(frames, v, a)

    def score_batch_accel(self, frames, hyps):
        return self._score_a(frames, hyps)

    def score_batch_chunked(self, frames, hyps, chunk_size):
        return A.score_batch_chunked(frames, hyps, chunk_size, xp=self.jnp)

    def asarray(self, array):
        return self.jnp.asarray(array)

    def stack(self, arrays):
        return self.jnp.stack(arrays)

    def block_until_ready(self, value):
        self.jax.block_until_ready(value)
        return value


def get_backend(name: str | None = None):
    """按名称取后端；name=None 时优先 jax（不可用则 numpy）。"""
    if name is not None and name not in _ALL:
        raise ValueError(f"未知后端 {name!r}（可选 {_ALL}）")
    jax, jnp = _jax_modules()
    if name is None:
        name = "jax" if jax is not None else "numpy"
    if name == "jax":
        if jax is None:
            raise RuntimeError("JAX 不可用：请使用 --backend numpy 或安装 jax")
        return JaxBackend(jax, jnp)
    return NumpyBackend()
