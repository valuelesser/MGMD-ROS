"""test_benchmark_search — S3 四策略（SPEC §5.11，I-07 验收）。"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import benchmark_search as B  # noqa: E402

from spa2d import backends  # noqa: E402
from spa2d import synthetic as S  # noqa: E402


def _moving_frames(v_true, shape=(128, 160), N=7, width=1.0, amp=2.0, seed=0):
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    p0 = np.array([w / 2 - v_true[0] * N / 2, h / 2 - v_true[1] * N / 2], np.float32)
    fr = []
    for k in range(N):
        c = p0 + k * np.array(v_true, np.float32)
        blob = amp * np.exp(-(((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * width ** 2)))
        fr.append((blob + rng.normal(0, 0.005, (h, w))).astype(np.float32))
    return np.stack(fr)


def test_climb_recovers_velocity():
    frames = _moving_frames((0.7, 0.0), seed=1)
    be = backends.get_backend("numpy")
    v_best, score = B.climb(frames, B._starts_grid(), 0.1, be, chunk=10 ** 12)
    err = max(abs(v_best[0] - 0.7), abs(v_best[1] - 0.0))
    assert err <= 0.1 + 1e-6
    assert score > 0


def test_climb_visited_cache_shrinks_work():
    frames = _moving_frames((0.5, 0.5), seed=2)
    be = backends.get_backend("numpy")
    # 相邻起点会爬向同一局部峰：visited 缓存使其不重复计分（仅验证不报错且命中）
    v_best, _ = B.climb(frames, B._starts_grid(), 0.1, be, chunk=10 ** 12)
    assert max(abs(v_best[0] - 0.5), abs(v_best[1] - 0.5)) <= 0.15


def test_run_benchmark_rows(tmp_path):
    out = tmp_path / "bench"
    S.generate_benchmark(out, seed=0, n_frames=6, canvas=(128, 160))
    frames, true_v = B._load_benchmark(out / "benchmark", N=5, kernel_k=5)
    assert true_v == (0.7, 0.0)
    rows, accepted = B.run_benchmark(frames, "numpy", warmup=1, reps=2,
                                     batch=512, chunk=10 ** 12, true_v=true_v)
    names = {r["strategy"] for r in rows}
    assert names == set(B.STRATEGIES)
    for r in rows:
        assert set(r) == {"strategy", "fps", "hit", "abs_err", "jit_compile_ms", "device"}
        assert r["fps"] > 0 and r["device"] == "numpy"
    ctf = [r for r in rows if r["strategy"] == "coarse-to-fine"][0]
    assert ctf["hit"] is True
    assert isinstance(accepted, bool)


def test_cli_numpy_warning_and_csv(tmp_path):
    bench = tmp_path / "synthetic"
    S.generate_benchmark(bench, seed=0, n_frames=6, canvas=(128, 160))
    out = tmp_path / "s3_search.csv"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "benchmark_search.py"),
         "--benchmark", str(bench / "benchmark"), "--out", str(out),
         "--backend", "numpy", "--warmup", "1", "--reps", "2",
         "--chunk-bytes", str(10 ** 12), "--no-assert"],
        cwd=str(REPO), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["strategy"] for r in rows} == set(B.STRATEGIES)
    assert "numpy 后端结果仅用于算法正确性" in proc.stderr


def test_cli_missing_benchmark(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "benchmark_search.py"),
         "--benchmark", str(tmp_path / "nope"), "--out", str(tmp_path / "x.csv"),
         "--backend", "numpy"],
        cwd=str(REPO), capture_output=True, text=True)
    assert proc.returncode == 3
    assert "make_synthetic" in (proc.stdout + proc.stderr)
