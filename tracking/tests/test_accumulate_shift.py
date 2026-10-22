"""test_accumulate_shift — 双线性平移/累加（SPEC §5.2/§8.2/§8.3，I-04 验收）。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from spa2d import accumulate as A
from spa2d import backends


def _cv2_remap(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = (gx - np.float32(dx)).astype(np.float32)
    my = (gy - np.float32(dy)).astype(np.float32)
    return cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def _has_jax_gpu() -> bool:
    if not backends.jax_available():
        return False
    import jax

    return any(("gpu" in str(d).lower() or "cuda" in str(d).lower())
               for d in jax.devices())


needs_gpu = pytest.mark.skipif(not _has_jax_gpu(), reason="需要 JAX GPU 设备")


def test_shift_matches_cv2_remap():
    rng = np.random.default_rng(0)
    img = rng.random((40, 50), dtype=np.float32)
    shifts = [(-5.0, -5.0), (5.0, 5.0), (0.0, 0.0), (0.7, 0.0), (0.37, -1.23)]
    shifts += [tuple(v) for v in rng.uniform(-5.0, 5.0, size=(200, 2))]
    worst = 0.0
    for dx, dy in shifts:
        val, _ = A.shift_bilinear(img, dx, dy)
        worst = max(worst, float(np.abs(val - _cv2_remap(img, dx, dy)).max()))
    assert worst <= 1e-4, f"max|Δ|={worst}"


def test_shift_full_grid_step_005():
    rng = np.random.default_rng(2)
    img = rng.random((24, 30), dtype=np.float32)
    grid = np.arange(-5.0, 5.0 + 1e-9, 0.05)
    for dx in grid[::7]:
        for dy in grid[::11]:
            val, _ = A.shift_bilinear(img, float(dx), float(dy))
            assert np.abs(val - _cv2_remap(img, float(dx), float(dy))).max() <= 1e-4


def test_valid_mask_hand_computed():
    h, w = 8, 10
    ones = np.ones((h, w), dtype=np.float32)
    for dx, dy in [(0.0, 0.0), (0.3, 0.0), (-1.7, 0.4), (2.0, -3.0), (0.5, 0.5)]:
        _, msk = A.shift_bilinear(ones, dx, dy)
        ref = np.zeros((h, w), dtype=np.float64)
        for yy in range(h):
            for xx in range(w):
                sx = xx - dx
                sy = yy - dy
                x0 = int(np.floor(sx))
                y0 = int(np.floor(sy))
                fx = sx - x0
                fy = sy - y0
                for ax in (0, 1):
                    for ay in (0, 1):
                        wt = (fx if ax else 1 - fx) * (fy if ay else 1 - fy)
                        if 0 <= x0 + ax < w and 0 <= y0 + ay < h:
                            ref[yy, xx] += wt
        assert np.allclose(msk, ref, atol=1e-6), (dx, dy)


def test_border_constant_zero():
    img = np.ones((10, 10), dtype=np.float32)
    val, msk = A.shift_bilinear(img, dx=25.0, dy=0.0)  # 源完全出界
    assert np.all(val == 0.0) and np.all(msk == 0.0)


def test_accumulate_window_sum_and_mask():
    rng = np.random.default_rng(3)
    frames = rng.random((5, 16, 20), dtype=np.float32)
    v = np.array([0.6, -0.3], dtype=np.float32)
    acc, cnt = A.accumulate_window(frames, v, xp=np)
    ref = np.zeros((16, 20), dtype=np.float32)
    ref_m = np.zeros((16, 20), dtype=np.float32)
    for k in range(5):
        # 符号约定：A_v(x,y)=Σ I(x + k·v) ⇒ shift_bilinear 用 dx=-k·v
        val, msk = A.shift_bilinear(frames[k], float(-k * v[0]), float(-k * v[1]))
        ref += val
        ref_m += msk
    assert np.allclose(acc, ref, atol=1e-6)
    assert np.allclose(cnt, ref_m, atol=1e-6)
    assert cnt.min() >= 0.0 and cnt.max() <= 5.0 + 1e-6


def test_score_batch_coherence_sign():
    # 目标以 w 运动的合成窗口：v=+w 应相干（得分最高），v=-w 与 v=0 更低
    h, w = 48, 60
    p0 = np.array([25.0, 30.0], np.float32)
    w_true = np.array([0.7, -0.3], np.float32)
    yy, xx = np.mgrid[0:h, 0:w]

    def blob(center):
        return np.exp(-(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * 1.5 ** 2))).astype(np.float32)

    frames = np.stack([blob(p0 + k * w_true) for k in range(5)])
    hyps = np.stack([w_true, -w_true, np.zeros(2, np.float32)]).astype(np.float32)
    scores = A.score_batch(frames, hyps, xp=np)
    assert int(np.argmax(scores)) == 0
    assert scores[0] > scores[1] and scores[0] > scores[2]


def test_score_batch_equals_accumulate_max():
    rng = np.random.default_rng(4)
    frames = rng.random((4, 12, 14), dtype=np.float32)
    hyps = rng.uniform(-1, 1, size=(7, 2)).astype(np.float32)
    scores = A.score_batch(frames, hyps, xp=np)
    assert scores.shape == (7,)
    for b in range(7):
        acc, _ = A.accumulate_window(frames, hyps[b], xp=np)
        assert scores[b] == pytest.approx(float(acc.max()), abs=1e-6)


def test_score_batch_chunked_consistency():
    rng = np.random.default_rng(5)
    frames = rng.random((3, 10, 12), dtype=np.float32)
    hyps = rng.uniform(-1, 1, size=(9, 2)).astype(np.float32)
    full = A.score_batch(frames, hyps, xp=np)
    for chunk in (1, 2, 4, 9, 20):
        assert np.allclose(A.score_batch_chunked(frames, hyps, chunk, xp=np), full, atol=1e-6)
    with pytest.raises(ValueError):
        A.score_batch_chunked(frames, hyps, 0, xp=np)


def test_backend_selection():
    assert "numpy" in backends.available_backends()
    nb = backends.get_backend("numpy")
    assert nb.name == "numpy"
    with pytest.raises(ValueError):
        backends.get_backend("torch")
    if backends.jax_available():
        assert backends.get_backend("jax").name == "jax"
    else:
        with pytest.raises(RuntimeError):
            backends.get_backend("jax")


@pytest.mark.gpu
@needs_gpu
def test_jax_matches_numpy():
    rng = np.random.default_rng(6)
    frames = rng.random((5, 32, 40), dtype=np.float32)
    hyps = rng.uniform(-1, 1, size=(16, 2)).astype(np.float32)
    jb = backends.get_backend("jax")
    nb = backends.get_backend("numpy")
    js = jb.block_until_ready(jb.score_batch(frames, hyps))
    ns = nb.score_batch(frames, hyps)
    assert np.allclose(np.asarray(js), ns, atol=1e-4)

    aj, mj = jb.accumulate_window(frames, hyps[3])
    jb.block_until_ready((aj, mj))
    an, mn = nb.accumulate_window(frames, hyps[3])
    assert np.allclose(np.asarray(aj), an, atol=1e-4)
    assert np.allclose(np.asarray(mj), mn, atol=1e-4)


@pytest.mark.gpu
@needs_gpu
def test_jax_shift_matches_cv2():
    rng = np.random.default_rng(7)
    img = rng.random((32, 40), dtype=np.float32)
    jb = backends.get_backend("jax")
    for dx, dy in [(0.37, -1.23), (-3.5, 2.1)]:
        val, msk = jb.shift_bilinear(img, dx, dy)
        jb.block_until_ready((val, msk))
        assert np.abs(np.asarray(val) - _cv2_remap(img, dx, dy)).max() <= 1e-4


def test_accumulate_window_accel_matches_manual():
    rng = np.random.default_rng(11)
    frames = rng.random((5, 16, 20), dtype=np.float32)
    v = np.array([0.6, -0.3], np.float32)
    a = np.array([0.2, 0.15], np.float32)
    acc, cnt = A.accumulate_window_accel(frames, v, a, xp=np)
    ref = np.zeros((16, 20), np.float32)
    for k in range(5):
        dx = -(k * v[0] + 0.5 * k * k * a[0])
        dy = -(k * v[1] + 0.5 * k * k * a[1])
        val, _ = A.shift_bilinear(frames[k], float(dx), float(dy))
        ref += val
    assert np.allclose(acc, ref, atol=1e-6)
    assert cnt.min() >= 0 and cnt.max() <= 5 + 1e-6
