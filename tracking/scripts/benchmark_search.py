"""scripts/benchmark_search.py — S3 四策略搜索对照（SPEC §5.11/§4.1）。

在同一合成 benchmark（目标 (0.7,0) px/frame）上比较：
    brute-force                 vx,vy∈[−1,1]，步长 0.1（441 假设）
    multiple hill-climbing      9 起点均匀网格，步长 0.1，邻居爬山，并行，visited 缓存
    coarse-to-fine              δc=0.5 top-5，δf=0.1，rf=0.2（§5.3）
    coarse-seeded fine-hill-climbing  coarse top-5 起点 + 爬山

计时排除 JIT 编译（单列 jit_compile_ms），warmup≥20 后统计；输出
outputs/metrics/s3_search.csv（strategy,fps,hit,abs_err,jit_compile_ms,device）。
自动断言：CTF hit 且 CTF_fps ≥ max(MHC_fps, CS-FHC_fps)。无 GPU 时用 numpy 并显式警告。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spa2d import backends  # noqa: E402
from spa2d import preprocess as P  # noqa: E402
from spa2d import search as S  # noqa: E402

TRUE_V = (0.7, 0.0)
HIT_TOL = 0.1
DEFAULT_CHUNK_BYTES = 268435456  # 256 MiB（1080p 下避免 4 邻域 gather 临时缓冲 OOM）


def _key(v):
    return (round(float(v[0]), 6), round(float(v[1]), 6))


def climb(frames, starts, step: float, backend, chunk: int, max_iters: int = 200):
    """并行爬山：多起点同时向最高分邻居移动，visited 缓存避免重复计分。"""
    cache: dict = {}

    def scores_of(cands):
        pending: dict = {}
        order = []
        for v in cands:
            k = _key(v)
            if k in cache or k in pending:
                continue
            pending[k] = np.asarray(v, np.float32)
            order.append(k)
        if order:
            arr = np.stack([pending[k] for k in order])
            sc = backend.block_until_ready(backend.score_batch_chunked(frames, arr, chunk))
            sc = np.asarray(sc, dtype=np.float64)
            for k, val in zip(order, sc):
                cache[k] = float(val)
        return np.array([cache[_key(v)] for v in cands], dtype=np.float64)

    pos = [np.asarray(s, np.float32) for s in starts]
    scores_of(pos)
    active = list(range(len(pos)))
    it = 0
    while active and it < max_iters:
        it += 1
        neigh, owner = [], []
        for i in active:
            for d in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
                neigh.append(pos[i] + np.asarray(d, np.float32))
                owner.append(i)
        sc = scores_of(neigh)
        moved = []
        for i in active:
            idxs = [j for j, o in enumerate(owner) if o == i]
            bj = max(idxs, key=lambda j: sc[j])
            if sc[bj] > cache[_key(pos[i])] + 1e-12:
                pos[i] = neigh[bj]
                moved.append(i)
        active = moved
    final = scores_of(pos)
    b = int(np.argmax(final))
    return (float(pos[b][0]), float(pos[b][1])), float(final[b])


def _starts_grid():
    vals = (-1.0, 0.0, 1.0)
    return np.array([[x, y] for x in vals for y in vals], np.float32)


def strategy_brute(frames, backend, batch, chunk):
    res = S.brute_force(frames, 1.0, 0.1, batch, chunk, backend=backend)
    return res.v_best


def strategy_mhc(frames, backend, batch, chunk):
    return climb(frames, _starts_grid(), 0.1, backend, chunk)[0]


def strategy_ctf(frames, backend, batch, chunk):
    res = S.global_search(frames, 1.0, 0.5, 0.2, 0.1, top_k=5,
                          batch_logical=batch, gpu_chunk_bytes=chunk, backend=backend)
    return res.v_best


def strategy_csfhc(frames, backend, batch, chunk):
    cs = S.run_search(frames, S.coarse_grid(1.0, 0.5), batch, chunk, backend=backend)
    top = np.asarray(cs.cands, np.float32)[S.top_k_indices(cs.scores, cs.cands, 5)]
    return climb(frames, top, 0.1, backend, chunk)[0]


STRATEGIES = {
    "brute-force": strategy_brute,
    "multiple-hill-climbing": strategy_mhc,
    "coarse-to-fine": strategy_ctf,
    "coarse-seeded-fine-hill-climbing": strategy_csfhc,
}


def _measure(fn, warmup: int, reps: int):
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    mean = float(np.mean(times))
    fps = 1.0 / mean if mean > 0 else float("inf")
    jit_ms = max(0.0, (first - mean) * 1000.0)
    return fps, jit_ms


def device_label(backend_name: str) -> str:
    if backend_name == "numpy":
        return "numpy"
    try:
        import jax

        return str(jax.devices()[0])
    except Exception:  # noqa: BLE001
        return "jax"


def run_benchmark(frames, backend_name: str = "jax", warmup: int = 20, reps: int = 30,
                  batch: int = 512, chunk: int = DEFAULT_CHUNK_BYTES,
                  true_v=TRUE_V):
    be = backends.get_backend(backend_name)
    dev = device_label(backend_name)
    rows = []
    for name, fn in STRATEGIES.items():
        v_best = fn(frames, be, batch, chunk)
        fps, jit_ms = _measure(lambda f=fn: f(frames, be, batch, chunk), warmup, reps)
        err = max(abs(v_best[0] - true_v[0]), abs(v_best[1] - true_v[1]))
        rows.append({
            "strategy": name,
            "fps": round(fps, 4),
            "hit": bool(err <= HIT_TOL),
            "abs_err": round(float(err), 6),
            "jit_compile_ms": round(jit_ms, 4),
            "device": dev,
        })
    by = {r["strategy"]: r for r in rows}
    ctf = by["coarse-to-fine"]
    mhc = by["multiple-hill-climbing"]
    csfhc = by["coarse-seeded-fine-hill-climbing"]
    accepted = bool(ctf["hit"] and ctf["fps"] >= max(mhc["fps"], csfhc["fps"]))
    return rows, accepted


def _load_benchmark(bench_dir: Path, N: int, kernel_k: int):
    meta_path = bench_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"benchmark 不存在: {bench_dir}（先运行 make_synthetic.py）")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames_dir = bench_dir / "frames"
    files = sorted(frames_dir.glob("*.png"))[:N]
    if len(files) < N:
        raise FileNotFoundError(f"benchmark 帧不足 N={N}: {frames_dir}")
    import cv2

    ker = P.build_kernel(kernel_k)
    frames = np.stack([P.preprocess_frame(cv2.imread(str(f)), ker) for f in files])
    true_v = (float(meta["velocity"]["vx"]), float(meta["velocity"]["vy"]))
    return frames, true_v


def build_parser():
    p = argparse.ArgumentParser(prog="benchmark_search.py",
                                description="S3 四策略搜索对照（SPEC §5.11）")
    p.add_argument("--benchmark", default="data_raw/synthetic/benchmark")
    p.add_argument("--out", default="outputs/metrics/s3_search.csv")
    p.add_argument("--backend", choices=["jax", "numpy"], default=None)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--no-assert", action="store_true", help="不因验收断言失败而返回非零")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    backend_name = args.backend or ("jax" if backends.jax_available() else "numpy")
    if backend_name == "numpy":
        print("⚠️  未使用 GPU：numpy 后端结果仅用于算法正确性，FPS 不可用于论文对照",
              file=sys.stderr)
    try:
        frames, true_v = _load_benchmark(Path(args.benchmark), args.window, args.kernel)
    except FileNotFoundError as exc:
        print(f"[spa2d] E_DATA_MISSING: {exc} → 运行 "
              f"python scripts/make_synthetic.py --suite benchmark --out data_raw/synthetic",
              file=sys.stderr)
        return 3

    rows, accepted = run_benchmark(frames, backend_name, args.warmup, args.reps,
                                   args.batch, args.chunk_bytes, true_v)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["strategy", "fps", "hit", "abs_err",
                                                "jit_compile_ms", "device"])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['strategy']:34s} fps={r['fps']:8.3f} hit={r['hit']!s:5s} "
              f"abs_err={r['abs_err']:.4f} jit={r['jit_compile_ms']:.1f}ms")
    print(f"device={rows[0]['device']}  -> {out}")
    if not accepted:
        msg = "❌ S3 验收失败：coarse-to-fine 未命中或 FPS 不高于两种 hill-climbing"
        print(msg, file=sys.stderr)
        return 0 if args.no_assert else 1
    print("✅ S3 验收通过：CTF 命中且 GPU FPS ≥ MHC / CS-FHC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
