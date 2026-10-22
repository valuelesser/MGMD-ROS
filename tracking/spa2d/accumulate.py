"""spa2d.accumulate — 运动补偿累加（SPEC §5.2，对齐论文 Eq.7）。

对候选速度 ``v=(vx,vy)`` 与窗口帧 ``frames[k]=I_{t+k}``：
    A_v(x,y) = Σ_k frames[k](x + k·vx, y + k·vy)    # 双线性，BORDER_CONSTANT=0
    M_v(x,y) = Σ_k ones    (x + k·vx, y + k·vy)    # valid 计数 ∈ [0,N]

符号约定：``v`` 为目标在图像平面的 pixel/frame 速度，累加时把后续帧按 ``-k·v``
搬回首帧坐标系，使**正确速度**（v_best = 目标速度）处相干叠加。这与 SPEC §5.2
的公式文本 ``x − k·v`` 相差一个符号：该文本与 S1（目标以 (0.7,0) 运动必须返回
(0.7,0)）、Eq.9（p_center = p0 + (N-1)/2·v）及连续性模型（p_pred = p_prev + v·Δt）
不自洽（``x−k·v`` 会在 v = −w 处相干）。此处采用物理自洽的 ``x + k·v``，保证
``v_best`` 即目标速度。

双线性采样手写实现（floor + 4 邻居 + 权重 + 越界置零 + valid mask），numpy 与
jax.numpy 逐位同构：所有函数通过 ``xp`` 参数选择数组后端。JAX 的 jit 封装与设备
同步见 ``spa2d/backends.py``。``shift_bilinear`` 本身定义为 ``out(x)=img(x-dx)``
（内容右移 dx），与 cv2.remap 对照测试见 tests/test_accumulate_shift.py。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "shift_bilinear",
    "accumulate_window",
    "score_batch",
    "score_batch_chunked",
]


def _is_jax_np(xp) -> bool:
    return getattr(xp, "__name__", "") == "jax.numpy"


def _shift_impl(img, dx, dy, xp):
    img = xp.asarray(img, dtype=xp.float32)
    h, w = img.shape
    gx = xp.broadcast_to(xp.arange(w, dtype=xp.float32)[None, :], (h, w))
    gy = xp.broadcast_to(xp.arange(h, dtype=xp.float32)[:, None], (h, w))
    sx = gx - xp.float32(dx)
    sy = gy - xp.float32(dy)

    x0 = xp.floor(sx).astype(xp.int32)
    y0 = xp.floor(sy).astype(xp.int32)
    fx = sx - x0.astype(xp.float32)
    fy = sy - y0.astype(xp.float32)

    val = xp.zeros((h, w), dtype=xp.float32)
    msk = xp.zeros((h, w), dtype=xp.float32)
    zero = xp.float32(0.0)
    for ax in (0, 1):
        for ay in (0, 1):
            wx = fx if ax else (1.0 - fx)
            wy = fy if ay else (1.0 - fy)
            weight = wx * wy
            xi = x0 + ax
            yi = y0 + ay
            inb = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            xic = xp.clip(xi, 0, w - 1)
            yic = xp.clip(yi, 0, h - 1)
            sampled = img[yic, xic]
            val = val + xp.where(inb, weight * sampled, zero)
            msk = msk + xp.where(inb, weight, zero)
    return val, msk


def shift_bilinear(img, dx: float, dy: float, xp=np):
    """按 (dx,dy) 平移采样：返回 (采样值, valid 权重图)，float32，常量 0 边界。"""
    return _shift_impl(img, dx, dy, xp)


def accumulate_window(frames, v, xp=np):
    """窗口累加：frames (N,H,W)，v=(vx,vy)，返回 (A, M)，M ∈ [0,N]。"""
    frames = xp.asarray(frames, dtype=xp.float32)
    n, h, w = frames.shape
    acc = xp.zeros((h, w), dtype=xp.float32)
    cnt = xp.zeros((h, w), dtype=xp.float32)
    vx = v[0]
    vy = v[1]
    for k in range(n):
        val, msk = _shift_impl(frames[k], -(xp.float32(k) * vx), -(xp.float32(k) * vy), xp)
        acc = acc + val
        cnt = cnt + msk
    return acc, cnt


def accumulate_window_accel(frames, v, a, xp=np):
    """带加速度的窗口累加：A_{v,a}(x,y)=Σ_k I(x - k·v - ½k²·a)；返回 (A, M)。"""
    frames = xp.asarray(frames, dtype=xp.float32)
    n, h, w = frames.shape
    acc = xp.zeros((h, w), dtype=xp.float32)
    cnt = xp.zeros((h, w), dtype=xp.float32)
    vx, vy = v[0], v[1]
    ax, ay = a[0], a[1]
    for k in range(n):
        kk = xp.float32(k)
        dx = -(kk * vx + xp.float32(0.5) * kk * kk * ax)
        dy = -(kk * vy + xp.float32(0.5) * kk * kk * ay)
        val, msk = _shift_impl(frames[k], dx, dy, xp)
        acc = acc + val
        cnt = cnt + msk
    return acc, cnt


def _score_batch_accel_np(frames, hyps):
    hyps = np.asarray(hyps, dtype=np.float32)
    out = np.empty((hyps.shape[0],), dtype=np.float32)
    for b in range(hyps.shape[0]):
        acc, _ = accumulate_window_accel(frames, hyps[b, :2], hyps[b, 2:], xp=np)
        out[b] = acc.max()
    return out


def _score_batch_accel_jax(frames, hyps):
    import jax
    import jax.numpy as jnp

    n = frames.shape[0]

    def acc_one(h):
        acc = jnp.zeros(frames.shape[1:], dtype=jnp.float32)
        for k in range(n):
            kk = jnp.float32(k)
            dx = -(kk * h[0] + jnp.float32(0.5) * kk * kk * h[2])
            dy = -(kk * h[1] + jnp.float32(0.5) * kk * kk * h[3])
            val, _ = _shift_impl(frames[k], dx, dy, jnp)
            acc = acc + val
        return jnp.max(acc)

    return jax.vmap(acc_one)(hyps)


def score_batch_accel(frames, hyps, xp=np):
    """对 (B,4)=[vx,vy,ax,ay] 假设返回标量得分 (B,)。"""
    if _is_jax_np(xp):
        return _score_batch_accel_jax(frames, hyps)
    return _score_batch_accel_np(frames, hyps)


def _score_batch_np(frames, hyps):
    hyps = np.asarray(hyps, dtype=np.float32)
    out = np.empty((hyps.shape[0],), dtype=np.float32)
    for b in range(hyps.shape[0]):
        acc, _ = accumulate_window(frames, hyps[b], xp=np)
        out[b] = acc.max()
    return out


def _score_batch_jax(frames, hyps):
    import jax
    import jax.numpy as jnp

    n = frames.shape[0]

    def acc_one(v):
        acc = jnp.zeros(frames.shape[1:], dtype=jnp.float32)
        for k in range(n):
            val, _ = _shift_impl(frames[k], -(jnp.float32(k) * v[0]), -(jnp.float32(k) * v[1]), jnp)
            acc = acc + val
        return jnp.max(acc)

    return jax.vmap(acc_one)(hyps)


def score_batch(frames, hyps, xp=np):
    """对 B 个候选速度仅返回标量得分 (B,)（不保留全量累加图）。"""
    if _is_jax_np(xp):
        return _score_batch_jax(frames, hyps)
    return _score_batch_np(frames, hyps)


def score_batch_chunked(frames, hyps, chunk_size: int, xp=np):
    """分块计算得分并拼接；语义与整批一致（SPEC §8.3 显存分块）。"""
    hyps = xp.asarray(hyps, dtype=xp.float32) if _is_jax_np(xp) else np.asarray(hyps, np.float32)
    total = hyps.shape[0]
    if chunk_size < 1:
        raise ValueError("chunk_size 必须 ≥ 1")
    parts = [score_batch(frames, hyps[i:i + chunk_size], xp=xp)
             for i in range(0, total, chunk_size)]
    if _is_jax_np(xp):
        import jax.numpy as jnp

        return jnp.concatenate(parts, axis=0) if parts else jnp.zeros((0,), jnp.float32)
    return np.concatenate(parts, axis=0) if parts else np.zeros((0,), np.float32)
