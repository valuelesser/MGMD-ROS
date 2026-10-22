"""spa2d.search — 候选速度集合生成与搜索执行（SPEC §5.3 / §5.7）。

- 全局 coarse-to-fine：coarse 网格 → top-K → fine 邻域并集（独立评分）→ argmax。
- 动态局部搜索：绕上一 accepted 速度的邻域。
- 去重（Chebyshev ≤ 1e-9）与确定性排序（字典序 vx→vy）；top-K 并列按 (|v|, vx, vy)
  稳定取；argmax 并列取排序后第一个。
- 物理分块：``chunk = clamp(floor(gpu_chunk_bytes/(2·H·W·4)), 1, batch_logical)``，
  跨块 max 归约，语义不变。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spa2d import backends
from spa2d.records import SearchResult

__all__ = [
    "coarse_grid",
    "neighborhood_offsets",
    "fine_neighborhoods",
    "local_candidates",
    "dedupe",
    "top_k_indices",
    "run_search",
    "global_search",
    "local_search",
    "brute_force",
    "gpu_chunk_size",
]

_DEDUPE_EPS = 1e-9


def _frame_hw(frames) -> tuple[int, int]:
    """(H, W)，不对 JAX 设备数组做 np.asarray（避免 device→host 拷贝）。"""
    shape = frames.shape if hasattr(frames, "shape") else np.asarray(frames).shape
    return int(shape[-2]), int(shape[-1])


def _axis_values(vmax: float, step: float) -> np.ndarray:
    n = int(round(2.0 * vmax / step))
    return (np.arange(n + 1, dtype=np.float64) * step - vmax).astype(np.float32)


def coarse_grid(vmax: float, delta_c: float) -> np.ndarray:
    """返回 ((2·vmax/δc+1)^2, 2) 的粗网格候选（vx 外层，vy 内层）。"""
    vals = _axis_values(vmax, delta_c)
    vx, vy = np.meshgrid(vals, vals, indexing="ij")
    return np.stack([vx.ravel(), vy.ravel()], axis=1).astype(np.float32)


def neighborhood_offsets(radius: float, step: float) -> np.ndarray:
    """对称方形邻域偏移：{ (a·step, b·step) | |a|,|b| ≤ floor(radius/step) }。"""
    a_max = int(np.floor(radius / step + 1e-9))
    if a_max < 0:
        raise ValueError(f"邻域半径过小：floor({radius}/{step}) < 0")
    rng = np.arange(-a_max, a_max + 1, dtype=np.float64) * step
    ax, ay = np.meshgrid(rng, rng, indexing="ij")
    return np.stack([ax.ravel(), ay.ravel()], axis=1).astype(np.float32)


def fine_neighborhoods(top: np.ndarray, rf: float, delta_f: float) -> np.ndarray:
    """对每个 retained coarse 候选生成 fine 邻域，取并集并去重（SPEC §5.3）。"""
    top = np.asarray(top, dtype=np.float32).reshape(-1, 2)
    offsets = neighborhood_offsets(rf, delta_f)
    parts = [top[i][None, :] + offsets for i in range(top.shape[0])]
    union = np.concatenate(parts, axis=0) if parts else np.zeros((0, 2), np.float32)
    return dedupe(union)


def local_candidates(v_prev, rl: float, delta_l: float) -> np.ndarray:
    """V_local = { v_prev + (a·δl, b·δl) | |aδl|,|bδl| ≤ rl }（SPEC §5.3）。"""
    v = np.asarray(v_prev, dtype=np.float32).reshape(2)
    offsets = neighborhood_offsets(rl, delta_l)
    return dedupe(v[None, :] + offsets)


def dedupe(cands, eps: float = _DEDUPE_EPS) -> np.ndarray:
    """Chebyshev ≤ eps 视为重复；保留字典序最小者并整体按 (vx,vy) 排序。"""
    arr = np.asarray(cands, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] == 0:
        return np.zeros((0, 2), np.float32)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    kept: list[np.ndarray] = []
    for row in arr:
        dup = False
        for k in kept:
            if np.max(np.abs(row - k)) <= eps:
                dup = True
                break
        if not dup:
            kept.append(row)
    return np.asarray(kept, dtype=np.float32).reshape(-1, 2)


def top_k_indices(scores, cands, k: int) -> np.ndarray:
    """返回 top-K 候选的索引；并列按 (|v|, vx, vy) 升序稳定取。"""
    s = np.asarray(scores, dtype=np.float64)
    c = np.asarray(cands, dtype=np.float64).reshape(-1, 2)
    speed = np.max(np.abs(c), axis=1)
    order = sorted(range(len(s)), key=lambda i: (-s[i], speed[i], c[i, 0], c[i, 1]))
    return np.asarray(order[:k], dtype=np.int64)


def gpu_chunk_size(gpu_chunk_bytes: int, height: int, width: int, batch_logical: int) -> int:
    """chunk = clamp(floor(bytes / (2·H·W·4)), 1, batch_logical)（SPEC §5.2/§8.3）。"""
    per = 2 * height * width * 4
    raw = gpu_chunk_bytes // per if per > 0 else batch_logical
    return int(max(1, min(raw, batch_logical)))


def run_search(frames, cands, batch_logical: int = 512, gpu_chunk_bytes: int = 1610612736,
               backend=None, min_speed: float = 0.0) -> SearchResult:
    """对候选集合评分并取 v_best（分块 + 跨块 max 归约，确定性 argmax）。

    min_speed>0 时启用运动相干约束：排除速度 < min_speed 的静态候选
    （惩罚 v≈0 静态峰）；若全部被排除则回退到全集。
    """
    be = backends.get_backend(backend) if not hasattr(backend, "score_batch") else backend
    cands = dedupe(cands)
    if min_speed and min_speed > 0 and cands.shape[0] > 0:
        speed = np.max(np.abs(cands.astype(np.float64)), axis=1)
        keep = cands[speed >= min_speed - 1e-9]
        if keep.shape[0] > 0:
            cands = keep
    h, w = _frame_hw(frames)
    if cands.shape[0] == 0:
        return SearchResult(cands=[], scores=[], v_best=(0.0, 0.0),
                            second_best=None, chunk_count=0)
    chunk = gpu_chunk_size(gpu_chunk_bytes, h, w, batch_logical)
    scores = np.asarray(be.block_until_ready(be.score_batch_chunked(frames, cands, chunk)),
                        dtype=np.float64)
    best = int(np.argmax(scores))
    second = None
    if scores.size > 1:
        others = np.delete(scores, best)
        second = float(others.max())
    return SearchResult(
        cands=[(float(c[0]), float(c[1])) for c in cands],
        scores=[float(s) for s in scores],
        v_best=(float(cands[best, 0]), float(cands[best, 1])),
        second_best=second,
        chunk_count=int(np.ceil(cands.shape[0] / chunk)),
    )


def global_search(frames, vmax: float, delta_c: float, rf: float, delta_f: float,
                  top_k: int = 5, batch_logical: int = 512,
                  gpu_chunk_bytes: int = 1610612736, backend=None,
                  min_speed: float = 0.0) -> SearchResult:
    """全局 coarse-to-fine：fine 并集独立评分（不继承 coarse 分数）。"""
    be = backends.get_backend(backend) if not hasattr(backend, "score_batch") else backend
    coarse = coarse_grid(vmax, delta_c)
    cs = run_search(frames, coarse, batch_logical, gpu_chunk_bytes, backend=be, min_speed=min_speed)
    top = np.asarray(cs.cands, dtype=np.float32)[top_k_indices(cs.scores, cs.cands, top_k)]
    fine = fine_neighborhoods(top, rf, delta_f)
    return run_search(frames, fine, batch_logical, gpu_chunk_bytes, backend=be, min_speed=min_speed)


def local_search(frames, v_prev, rl: float, delta_l: float,
                 batch_logical: int = 512, gpu_chunk_bytes: int = 1610612736,
                 backend=None, min_speed: float = 0.0) -> SearchResult:
    cands = local_candidates(v_prev, rl, delta_l)
    return run_search(frames, cands, batch_logical, gpu_chunk_bytes, backend=backend,
                      min_speed=min_speed)


def brute_force(frames, vmax: float, step: float,
                batch_logical: int = 512, gpu_chunk_bytes: int = 1610612736,
                backend=None, min_speed: float = 0.0) -> SearchResult:
    return run_search(frames, coarse_grid(vmax, step), batch_logical, gpu_chunk_bytes,
                      backend=backend, min_speed=min_speed)


def local_search_accel(frames, v_prev, a_prev, rv: float, dv: float, ra: float, da: float,
                       batch_logical: int = 512, gpu_chunk_bytes: int = 1610612736,
                       backend=None):
    """逐窗 (v,a) 精细搜索：围绕 (v_prev,a_prev) 在半径 (rv,ra) 内步长 (dv,da)。

    返回 (v_best, a_best, scores, cands)；cands 为 (B,4)=[vx,vy,ax,ay]。
    """
    be = backends.get_backend(backend) if not hasattr(backend, "score_batch") else backend
    voff = neighborhood_offsets(rv, dv)
    aoff = neighborhood_offsets(ra, da)
    v = np.asarray(v_prev, dtype=np.float64).reshape(2)
    a = np.asarray(a_prev, dtype=np.float64).reshape(2)
    cands = np.array([[v[0] + o[0], v[1] + o[1], a[0] + q[0], a[1] + q[1]]
                      for o in voff for q in aoff], dtype=np.float32)
    h, w = _frame_hw(frames)
    chunk = gpu_chunk_size(gpu_chunk_bytes, h, w, batch_logical)
    parts = []
    for i in range(0, cands.shape[0], chunk):
        sc = be.block_until_ready(be.score_batch_accel(frames, cands[i:i + chunk]))
        parts.append(np.asarray(sc, dtype=np.float64))
    scores = np.concatenate(parts) if parts else np.zeros((0,), np.float64)
    best = int(np.argmax(scores))
    v_best = (float(cands[best, 0]), float(cands[best, 1]))
    a_best = (float(cands[best, 2]), float(cands[best, 3]))
    return v_best, a_best, scores, cands


def _dedupe4(arr, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1, 4)
    if arr.shape[0] == 0:
        return np.zeros((0, 4), np.float32)
    order = np.lexsort((arr[:, 3], arr[:, 2], arr[:, 1], arr[:, 0]))
    arr = arr[order]
    kept: list[np.ndarray] = []
    for row in arr:
        if not any(np.max(np.abs(row - k)) <= eps for k in kept):
            kept.append(row)
    return np.asarray(kept, dtype=np.float32).reshape(-1, 4)


def accel_grid(vmax: float, delta_c: float, amax: float, delta_a: float) -> np.ndarray:
    """(v,a) 粗网格：v∈[−vmax,vmax]² × a∈[−amax,amax]² 的笛卡尔积（去重）。"""
    vs = coarse_grid(vmax, delta_c)
    av = _axis_values(amax, delta_a)
    ax, ay = np.meshgrid(av, av, indexing="ij")
    aa = np.stack([ax.ravel(), ay.ravel()], axis=1)
    combos = np.array([[v[0], v[1], a[0], a[1]] for v in vs for a in aa], dtype=np.float32)
    return _dedupe4(combos)


def accel_neighborhoods(top, rf: float, delta_f: float, ra: float, delta_a: float) -> np.ndarray:
    voff = neighborhood_offsets(rf, delta_f)
    aoff = neighborhood_offsets(ra, delta_a)
    parts = []
    for t in np.asarray(top, np.float64):
        for o in voff:
            for q in aoff:
                parts.append([t[0] + o[0], t[1] + o[1], t[2] + q[0], t[3] + q[1]])
    return _dedupe4(np.asarray(parts, np.float32))


def _topk4(scores, cands, k: int) -> np.ndarray:
    s = np.asarray(scores, np.float64)
    c = np.asarray(cands, np.float64).reshape(-1, 4)
    sp = np.max(np.abs(c[:, :2]), axis=1)
    ac = np.max(np.abs(c[:, 2:]), axis=1)
    order = sorted(range(len(s)), key=lambda i: (-s[i], sp[i], ac[i], c[i, 0], c[i, 1]))
    return np.asarray(order[:k], dtype=np.int64)


def _score4(be, frames, cands, chunk):
    parts = []
    for i in range(0, cands.shape[0], chunk):
        parts.append(np.asarray(be.block_until_ready(be.score_batch_accel(
            frames, cands[i:i + chunk])), dtype=np.float64))
    return np.concatenate(parts) if parts else np.zeros((0,), np.float64)


def global_search_accel(frames, vmax: float, delta_c: float, amax: float, delta_a: float,
                        rf: float, delta_f: float, ra: float, da: float, top_k: int = 8,
                        batch_logical: int = 512, gpu_chunk_bytes: int = 1610612736,
                        backend=None, min_speed: float = 0.0):
    """4D (v,a) 粗到细全局搜索：coarse(v,a) → top-K → fine(v,a) 邻域并集 → argmax。

    返回 (v_best, a_best, scores, cands(,4))。
    """
    be = backends.get_backend(backend) if not hasattr(backend, "score_batch") else backend
    h, w = _frame_hw(frames)
    chunk = gpu_chunk_size(gpu_chunk_bytes, h, w, batch_logical)
    C = accel_grid(vmax, delta_c, amax, delta_a)
    if min_speed > 0:
        sp = np.max(np.abs(C[:, :2]), axis=1)
        Ck = C[sp >= min_speed - 1e-9]
        if Ck.shape[0] > 0:
            C = Ck
    sc = _score4(be, frames, C, chunk)
    top = C[_topk4(sc, C, top_k)]
    V = accel_neighborhoods(top, rf, delta_f, ra, da)
    sc2 = _score4(be, frames, V, chunk)
    b = int(np.argmax(sc2))
    return ((float(V[b, 0]), float(V[b, 1])), (float(V[b, 2]), float(V[b, 3])),
            sc2, V)
