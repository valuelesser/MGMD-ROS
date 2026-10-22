"""test_search_grid — 速度搜索/去重/分块 + S1/S2（SPEC §5.3/§5.11，I-06 验收）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spa2d import accumulate as A
from spa2d import backends
from spa2d import confidence as C
from spa2d import peak as PK
from spa2d import search as S
from spa2d.config import load_config

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
BENCH_FRAMES = REPO / "data_raw" / "synthetic" / "benchmark" / "frames"


def _has_jax_gpu() -> bool:
    if not backends.jax_available():
        return False
    import jax

    return any(("gpu" in str(d).lower() or "cuda" in str(d).lower()) for d in jax.devices())


needs_gpu = pytest.mark.skipif(not _has_jax_gpu(), reason="需要 JAX GPU 设备")


# ---------------- 候选集合规模（§5.3 表） ----------------

@pytest.mark.parametrize("cfg,coarse,fine_per,local", [
    ("D1_no_overlap", 25, 25, 49),
    ("D1_overlap", 25, 25, 49),
    ("D2_no_overlap", 289, 25, 49),
    ("D2_overlap", 441, 25, 49),
    ("D3_overlap", 256, 25, 25),
    ("D4_overlap", 256, 121, 81),
])
def test_candidate_counts(cfg, coarse, fine_per, local):
    c = load_config(CONFIGS / f"{cfg}.yaml")
    assert S.coarse_grid(c.search.coarse.vmax, c.search.coarse.delta_c).shape[0] == coarse
    assert S.neighborhood_offsets(c.search.fine.rf, c.search.fine.delta_f).shape[0] == fine_per
    assert S.local_candidates((0.0, 0.0), c.search.local.rl,
                              c.search.local.delta_l).shape[0] == local


def test_fine_union_upper_bound():
    # D1：5 个相距很远的种子 → 并集 = 5×25 = 125（上限）
    top = np.array([[100.0 * i, 0.0] for i in range(5)], np.float32)
    assert S.fine_neighborhoods(top, 0.2, 0.1).shape[0] == 125
    top4 = np.array([[100.0 * i, 0.0] for i in range(5)], np.float32)
    assert S.fine_neighborhoods(top4, 1.0, 0.2).shape[0] == 605


# ---------------- 去重 / 排序 / top-k / argmax ----------------

def test_dedupe_chebyshev_and_sort():
    cands = [(0.5, 0.0), (-0.2, 0.1), (0.5, 0.0), (0.5, 1e-10), (0.3, -0.3), (-0.2, 0.1 + 1e-10)]
    out = S.dedupe(cands)
    assert out.shape[1] == 2
    rows = [tuple(r) for r in out]
    assert (0.5, 0.0) in [(round(a, 6), round(b, 6)) for a, b in rows]
    assert len(rows) == 3
    # 字典序 vx→vy
    assert rows == sorted(rows, key=lambda r: (r[0], r[1]))


def test_dedupe_empty_and_eps():
    assert S.dedupe(np.zeros((0, 2), np.float32)).shape == (0, 2)
    out = S.dedupe([(0.0, 0.0), (1e-12, 0.0), (1e-8, 0.0)], eps=1e-9)
    assert out.shape[0] == 2  # 1e-12 与 0 视为重复；1e-8 保留


def test_top_k_tie_break():
    cands = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.5], [2.0, 0.0], [-1.0, 0.0]], np.float32)
    scores = np.array([5.0, 5.0, 5.0, 5.0, 1.0])
    idx = S.top_k_indices(scores, cands, 3)
    picked = [tuple(cands[i]) for i in idx]
    # 并列时按 (|v|, vx, vy)：先 (0,0)，再 (0,0.5)，再 (-1,0) 或 (2,0)?  (|v|: 0,0.5,1,2)
    assert picked[0] == (0.0, 0.0)
    assert picked[1] == (0.0, 0.5)
    assert S.top_k_indices(scores, cands, 10).shape[0] == 5


def test_run_search_argmax_deterministic_on_ties():
    frames = np.zeros((3, 8, 10), np.float32)
    cands = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5]], np.float32)
    res = S.run_search(frames, cands, backend=backends.get_backend("numpy"))
    # 全零 → 得分并列；去重排序后 (vx,vy) 最小为 (-0.5,0.0)
    assert res.v_best == (-0.5, 0.0)
    assert res.second_best == 0.0


# ---------------- 分块 ----------------

def test_gpu_chunk_size_formula_and_clamp():
    assert S.gpu_chunk_size(1610612736, 1080, 1920, 512) == 97
    assert S.gpu_chunk_size(1, 8, 8, 512) == 1
    assert S.gpu_chunk_size(10 ** 15, 8, 8, 512) == 512


def test_run_search_chunk_consistency():
    rng = np.random.default_rng(0)
    frames = rng.random((4, 16, 20), np.float32)
    cands = rng.uniform(-1, 1, size=(20, 2)).astype(np.float32)
    be = backends.get_backend("numpy")
    full = S.run_search(frames, cands, batch_logical=512, gpu_chunk_bytes=10 ** 12, backend=be)
    parts = S.run_search(frames, cands, batch_logical=3, gpu_chunk_bytes=10 ** 12, backend=be)
    assert np.allclose(full.scores, parts.scores, atol=1e-6)
    assert full.v_best == parts.v_best
    assert parts.chunk_count == int(np.ceil(20 / 3))


# ---------------- 搜索流程 ----------------

def _moving_frames(v_true, shape=(128, 160), N=7, width=1.0, amplitude=2.0, seed=0):
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    p0 = np.array([w / 2 - v_true[0] * N / 2, h / 2 - v_true[1] * N / 2], np.float32)
    frames = []
    for k in range(N):
        c = p0 + k * np.array(v_true, np.float32)
        blob = amplitude * np.exp(-(((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * width ** 2)))
        frames.append((blob + rng.normal(0, 0.005, (h, w))).astype(np.float32))
    return np.stack(frames)


def test_global_search_recovers_velocity():
    v_true = (0.7, -0.3)
    frames = _moving_frames(v_true, seed=1)
    res = S.global_search(frames, 1.0, 0.5, 0.2, 0.1, top_k=5,
                          backend=backends.get_backend("numpy"))
    assert len(res.cands) <= 125
    assert abs(res.v_best[0] - v_true[0]) <= 0.1 + 1e-6
    assert abs(res.v_best[1] - v_true[1]) <= 0.1 + 1e-6
    assert res.v_best != (0.0, 0.0)


def test_local_search_near_prior():
    v_true = np.array([1.0, 0.5], np.float32)
    frames = _moving_frames(tuple(v_true), seed=2)
    res = S.local_search(frames, v_prev=v_true + np.array([0.1, -0.1]), rl=0.3, delta_l=0.1,
                         backend=backends.get_backend("numpy"))
    assert res.cands is not None and len(res.cands) == 49
    assert abs(res.v_best[0] - v_true[0]) <= 0.2
    assert abs(res.v_best[1] - v_true[1]) <= 0.2


def test_brute_force_recovers_point_seven():
    frames = _moving_frames((0.7, 0.0), seed=3)
    res = S.brute_force(frames, 1.0, 0.1, backend=backends.get_backend("numpy"))
    assert len(res.cands) == 441
    err = max(abs(res.v_best[0] - 0.7), abs(res.v_best[1] - 0.0))
    assert err <= 0.1 + 1e-6
    assert err <= 1e-6 or err <= 0.1


# ---------------- S1（真实 benchmark 序列） ----------------

@pytest.mark.gpu
@needs_gpu
@pytest.mark.skipif(not BENCH_FRAMES.is_dir(), reason="合成 benchmark 未生成")
def test_S1_real_benchmark():
    import cv2

    from spa2d import preprocess as P

    ker = P.build_kernel(5)
    frames = np.stack([P.preprocess_frame(cv2.imread(str(BENCH_FRAMES / f"{i:06d}.png")), ker)
                       for i in range(5)])
    res = S.brute_force(frames, 1.0, 0.1, batch_logical=512,
                        gpu_chunk_bytes=134217728, backend=backends.get_backend("jax"))
    err = max(abs(res.v_best[0] - 0.7), abs(res.v_best[1] - 0.0))
    assert err <= 1e-6 or err <= 0.1
    assert res.v_best == (pytest.approx(0.7, abs=1e-6), pytest.approx(0.0, abs=1e-6))


# ---------------- S2（PNR 分离度分布） ----------------

def _peak_pnr(frames, v, N, rho):
    acc, msk = A.accumulate_window(frames, np.array(v, np.float32), xp=np)
    peak = PK.extract_peak(acc, msk, N, PK.edge_margin_for(N, v))
    return C.peak_pnr(acc, msk, peak.xy_int, rho), acc, msk


def test_S2_pnr_separation(tmp_path):
    combos = [
        ((0.7, 0.0), 0),
        ((-0.7, 0.0), 1),
        ((0.5, 0.5), 2),
        ((1.0, -0.5), 3),
        ((0.3, -0.7), 4),
        ((-0.5, -0.5), 5),
    ]
    rows = []
    for v_true, seed in combos:
        frames = _moving_frames(v_true, seed=seed)
        N = frames.shape[0]
        rho = 16
        v_true = np.array(v_true, np.float32)
        grid = S.coarse_grid(1.0, 0.1)  # 441
        scores = A.score_batch(frames, grid, xp=np)
        near = np.max(np.abs(grid - v_true), axis=1) <= max(2 * 0.1, 0.2) + 1e-9
        wrong_idx = int(np.argmax(np.where(near, -np.inf, scores)))
        v_wrong = grid[wrong_idx]
        pnr_c, _, _ = _peak_pnr(frames, v_true, N, rho)
        pnr_w, _, _ = _peak_pnr(frames, v_wrong, N, rho)
        rows.append((tuple(np.round(v_true, 3)), tuple(np.round(v_wrong, 3)),
                     pnr_c, pnr_w, pnr_c - pnr_w))
    diffs = np.array([r[4] for r in rows])
    out = tmp_path / "s2_pnr_separation.csv"
    with out.open("w", encoding="utf-8") as fh:
        fh.write("v_true,v_wrong,pnr_correct,pnr_wrong,diff\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")
    assert np.median(diffs) > 0, diffs
    assert (diffs > 0).mean() >= 0.8, diffs
    # 落盘报告（可复现产物）
    (REPO / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)
    (REPO / "outputs" / "metrics" / "s2_pnr_separation.csv").write_text(
        out.read_text(encoding="utf-8"), encoding="utf-8")


def test_min_speed_motion_coherence():
    # 纯静止强斑：无 min_speed 时最佳为 v≈0（静态峰）；min_speed 强制给运动假设
    frames = np.zeros((5, 64, 80), np.float32)
    yy, xx = np.mgrid[0:64, 0:80]
    blob = np.exp(-(((xx - 40) ** 2 + (yy - 32) ** 2) / (2 * 1.2 ** 2))).astype(np.float32)
    frames[:] = blob[None, :, :]
    be = backends.get_backend("numpy")
    r_all = S.brute_force(frames, 1.0, 0.1, backend=be)
    r_mov = S.brute_force(frames, 1.0, 0.1, backend=be, min_speed=0.3)
    assert max(abs(r_all.v_best[0]), abs(r_all.v_best[1])) <= 1e-6      # 静态峰
    assert max(abs(r_mov.v_best[0]), abs(r_mov.v_best[1])) >= 0.3 - 1e-9  # 强制运动


def test_local_search_accel_recovers():
    # 加速暗斑：常数速度失配，逐窗 (v,a) 搜索应恢复真值
    h, w = 96, 128
    p0 = np.array([40.0, 30.0]); v0 = np.array([1.2, -0.8]); a0 = np.array([0.4, 0.25])
    yy, xx = np.mgrid[0:h, 0:w]
    N = 8
    fr = []
    for k in range(N):
        c = p0 + k * v0 + 0.5 * k * k * a0
        blob = 2.0 * np.exp(-(((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * 1.2 ** 2)))
        fr.append(blob.astype(np.float32))
    frames = np.stack(fr)
    be = backends.get_backend("numpy")
    vb, ab, sc, cands = S.local_search_accel(frames, v0 * 0.0, (0.0, 0.0),
                                             rv=2.0, dv=0.4, ra=1.2, da=0.2,
                                             gpu_chunk_bytes=10 ** 12, backend=be)
    assert abs(vb[0] - v0[0]) <= 0.4 + 1e-6 and abs(vb[1] - v0[1]) <= 0.4 + 1e-6
    assert abs(ab[0] - a0[0]) <= 0.2 + 1e-6 and abs(ab[1] - a0[1]) <= 0.2 + 1e-6
