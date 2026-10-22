"""scripts/run_sequence.py — SPA2D 单序列端到端运行（SPEC §5.7 / §2.3 / §3.3）。

用法：
    python scripts/run_sequence.py --config configs/D1_no_overlap.yaml \
        --gt data_derived/gt/D1_cam2.csv --out outputs/runs/D1_no_overlap \
        [--limit N] [--repeat 3] [--backend jax|numpy] [--allow-unfrozen]

主循环：prepare → (skip) → device_window → search → accumulate → extract_peak →
continuity → confidence → step → 写 windows.jsonl；并写 run_meta.json / timing.json。
frozen 校验失败退出码 5（--allow-unfrozen 例外并警告）。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from types import SimpleNamespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spa2d import backends, confidence as C, frame_cache as FC, io_mvdt as io
from spa2d import metrics as MET, peak as PK, preprocess as P, records as R
from spa2d import search as S, state_machine as SM, timing as T
from spa2d.config import (ConfigError, EXIT_CONFIG_INVALID, EXIT_DATA, EXIT_FROZEN,
                          EXIT_RUNTIME, FrozenThresholds, effective_thresholds,
                          load_config, load_frozen)

EXIT_OK = 0


# ---------------- thresholds / frozen ----------------

def _default_frozen(cfg) -> FrozenThresholds:
    return FrozenThresholds(
        schema_version=1, sha256="unfrozen", frozen_at="unfrozen",
        kernel_size=cfg.preprocess.kernel_size,
        theta_high_pnr=8.0, theta_low_pnr=5.0, theta_margin=0.02, theta_edge_pnr=10.0,
        v_stat=0.05, m_lost=3, reacq_max=5, gap_prior_windows=2,
        width_ref={cfg.sequence.name: 8.0}, raw={})


def _resolve_thresholds(cfg, allow_unfrozen: bool):
    try:
        frozen = load_frozen(cfg.thresholds_file)
        used_default = False
    except ConfigError as exc:
        if not allow_unfrozen:
            raise
        print(f"⚠️  {exc.format()}", file=sys.stderr)
        print("⚠️  --allow-unfrozen：使用 SPEC §5.5 smoke-test 默认阈值运行", file=sys.stderr)
        frozen = _default_frozen(cfg)
        used_default = True
    th = effective_thresholds(cfg, frozen, width_ref=frozen.width_ref.get(cfg.sequence.name))
    return th, frozen, used_default


# ---------------- environment ----------------

def _env_info(backend_name: str) -> dict:
    os.environ.setdefault("SPA2D_CV_THREADS", "4")
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "backend": backend_name, "cv_threads_env": os.environ["SPA2D_CV_THREADS"]}
    try:
        import cv2

        cv2.setNumThreads(int(os.environ["SPA2D_CV_THREADS"]))
        info["opencv_version"] = cv2.__version__
        info["opencv_threads"] = int(cv2.getNumThreads())
        build = cv2.getBuildInformation()
        info["opencv_build_summary"] = [
            ln.strip() for ln in build.splitlines()
            if "CUDA" in ln or "parallel" in ln.lower()][:8]
    except Exception:  # noqa: BLE001
        pass
    if backend_name == "jax" and backends.jax_available():
        import jax
        import jaxlib

        info["jax"] = jax.__version__
        info["jaxlib"] = jaxlib.__version__
        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
        info["xla_flags"] = os.environ.get("XLA_FLAGS", "")
    return info


def _data_hash(cfg) -> dict:
    seq = cfg.sequence
    import hashlib

    if seq.video:
        vp = Path(seq.video)
        if vp.is_file():
            return {"video": str(vp), "sha256": hashlib.sha256(vp.read_bytes()).hexdigest()}
    frames_dir = Path(seq.frames_dir)
    if frames_dir.is_dir():
        return {"frames_dir": str(frames_dir),
                "frames_sha256": io.sequence_frames_hash(frames_dir)}
    if frames_dir.is_file():
        return {"video": str(frames_dir),
                "sha256": hashlib.sha256(frames_dir.read_bytes()).hexdigest()}
    return {"frames_dir": str(frames_dir), "frames_sha256": None}


# ---------------- one run ----------------

def _prior_from(last):
    if last is None:
        return None
    return C.Prior(c_prev=last.c_prev, p_prev=last.p_prev, v_prev=last.v_prev)


def _run_once(cfg, th, source, backend_name: str, limit: int | None, out_dir: Path,
              chunk_bytes: int | None = None, match_sigma: float | None = None):
    be = backends.get_backend(backend_name)
    info = source.probe()
    F = len(source)
    N, stride = cfg.window.N, cfg.window.stride
    kernel = P.build_kernel(th.kernel_size, cfg.preprocess.kernel_shape)
    cache = FC.FrameCache(source, kernel, color_order=cfg.sequence.color_order,
                          max_bytes=cfg.cache.host_max_bytes, backend=be)
    sm = SM.StateMachine(th, stride, N, image_size=(info.width, info.height))
    timer = T.Timer(backend=be, warmup=cfg.timing.warmup_windows)

    start_frame = int(os.environ.get("SPA2D_START_FRAME", "0") or 0)
    starts = [t for t in range(0, F - N + 1, stride) if t >= start_frame]
    if limit is not None:
        starts = starts[:limit]

    windows: list[R.WindowRecord] = []
    search_ms_list: list[float] = []
    a_prev = (0.0, 0.0)
    for wi, t in enumerate(starts):
        plan = sm.prepare(t)
        if plan.skip_search:
            timer.add("t_decode", 0.0)
            timer.add("t_pre", 0.0)
            timer.add("t_search", 0.0)
            cont_skip = R.Continuity(None, th.r_cont_high, th.r_cont_low, True)
            t0 = time.perf_counter()
            outcome = sm.step(SM.WindowObservation(
                pnr=0.0, margin=0.0, edge_candidate=False,
                continuity=cont_skip,
                p_center=(float("nan"), float("nan")), v_best=(0.0, 0.0)), plan)
            state_ms = (time.perf_counter() - t0) * 1000.0
            timer.add("t_state", state_ms)
            windows.append(_record(wi, t, N, cfg, plan, outcome, None, None, None,
                                   None, cont_skip, 0.0, 0.0, state_ms, 0.0))
            continue

        idxs = list(range(t, t + N))
        decode_ms = pre_ms = 0.0
        env_sigma = float(os.environ.get("SPA2D_MATCH_SIGMA", "0") or 0.0)
        eff_sigma = env_sigma if env_sigma > 0 else float(match_sigma or 0.0)
        multi_sigmas = [float(x) for x in os.environ.get("SPA2D_MATCH_SIGMAS", "").split(",")
                        if x.strip()]
        for idx in idxs:
            if cache.has_preproc(idx):
                cache.get(idx)          # 命中计数；实际窗口由 device_window 取
            else:
                t0 = time.perf_counter()
                bgr = source.read(idx)
                decode_ms += (time.perf_counter() - t0) * 1000.0
                t0 = time.perf_counter()
                if multi_sigmas:
                    arr = P.matched_filter_multiscale(bgr, multi_sigmas,
                                                      color_order=cfg.sequence.color_order)
                elif eff_sigma > 0:
                    arr = P.matched_filter_frame(bgr, eff_sigma,
                                                 color_order=cfg.sequence.color_order)
                else:
                    arr = P.preprocess_frame(bgr, kernel, color_order=cfg.sequence.color_order)
                pre_ms += (time.perf_counter() - t0) * 1000.0
                cache.put(idx, arr)
        timer.add("t_decode", decode_ms)
        timer.add("t_pre", pre_ms)

        frames = cache.device_window(idxs)
        v_prev = tuple(sm.last_accepted.v_prev) if sm.last_accepted is not None else (0.0, 0.0)
        chunk = int(os.environ.get("SPA2D_CHUNK_BYTES")
                    or (chunk_bytes if chunk_bytes is not None else cfg.search.gpu_chunk_bytes))
        min_speed = float(os.environ.get("SPA2D_MIN_SPEED", "0") or 0.0)
        use_accel = os.environ.get("SPA2D_ACCEL", "0") == "1"
        t0 = time.perf_counter()
        a_best = (0.0, 0.0)
        if use_accel:
            a_radius = float(os.environ.get("SPA2D_A_RADIUS", "1.0"))
            a_da = float(os.environ.get("SPA2D_A_DA", "0.25"))
            if plan.search_mode == "local":
                v_best, a_best, asc, acands = S.local_search_accel(
                    frames, v_prev, a_prev, cfg.search.local.rl, cfg.search.local.delta_l,
                    a_radius, a_da, cfg.search.batch_logical, chunk, backend=be)
            else:
                v_best, a_best, asc, acands = S.global_search_accel(
                    frames, cfg.search.coarse.vmax, cfg.search.coarse.delta_c,
                    float(os.environ.get("SPA2D_A_ACC", "1.5")),
                    float(os.environ.get("SPA2D_DA_COARSE", "1.5")),
                    cfg.search.fine.rf, cfg.search.fine.delta_f, a_radius, a_da,
                    int(os.environ.get("SPA2D_A_TOPK", "8")),
                    cfg.search.batch_logical, chunk, backend=be, min_speed=min_speed)
            A, M = be.accumulate_window_accel(frames, np.asarray(v_best, np.float32),
                                              np.asarray(a_best, np.float32))
            margin, single = 1.0, False
            res = SimpleNamespace(v_best=v_best)
        elif plan.search_mode == "local":
            res = S.local_search(frames, v_prev, cfg.search.local.rl, cfg.search.local.delta_l,
                                 cfg.search.batch_logical, chunk, backend=be, min_speed=min_speed)
            v_best = res.v_best
            A, M = be.accumulate_window(frames, np.asarray(v_best, np.float32))
            margin, single = C.velocity_margin(res.scores, res.cands, res.v_best,
                                               cfg.search.local.delta_l)
        else:
            res = S.global_search(frames, cfg.search.coarse.vmax, cfg.search.coarse.delta_c,
                                  cfg.search.fine.rf, cfg.search.fine.delta_f, cfg.search.top_k,
                                  cfg.search.batch_logical, chunk, backend=be, min_speed=min_speed)
            v_best = res.v_best
            A, M = be.accumulate_window(frames, np.asarray(v_best, np.float32))
            margin, single = C.velocity_margin(res.scores, res.cands, res.v_best,
                                               cfg.search.fine.delta_f)
        be.block_until_ready((A, M))
        A = np.asarray(A, np.float64)
        M = np.asarray(M, np.float64)
        pk = PK.extract_peak(A, M, N, PK.edge_margin_for(N, v_best))
        mu, sigma, bg_count = C.background_stats(A, M, pk.xy_int, th.rho)
        pnr_v = 0.0 if bg_count < C.INSUFFICIENT_BG else C.pnr(pk.value, mu, sigma)
        p_center = (PK.center_frame_position_accel(pk.xy_sub, v_best, a_best, N) if use_accel
                    else PK.center_frame_position(pk.xy_sub, v_best, N))
        search_ms = (time.perf_counter() - t0) * 1000.0
        timer.add("t_search", search_ms)
        search_ms_list.append(search_ms)

        t0 = time.perf_counter()
        cont = C.continuity(p_center, plan.t_center, th.width_ref, _prior_from(sm.last_accepted))
        obs = SM.WindowObservation(pnr=pnr_v, margin=margin, edge_candidate=pk.edge_candidate,
                                   continuity=cont, p_center=p_center, v_best=res.v_best,
                                   single_candidate=single, peak_value=pk.value)
        outcome = sm.step(obs, plan)
        if outcome.accepted:
            a_prev = a_best
        state_ms = (time.perf_counter() - t0) * 1000.0
        timer.add("t_state", state_ms)

        windows.append(_record(wi, t, N, cfg, plan, outcome, res, pk, pnr_v, margin, cont,
                               decode_ms, pre_ms, state_ms, search_ms,
                               mu=mu, sigma=sigma, single=single,
                               a_best=a_best if use_accel else None,
                               use_accel=use_accel))

    out_dir.mkdir(parents=True, exist_ok=True)
    R.write_jsonl(windows, out_dir / "windows.jsonl")
    warmup_eff = min(cfg.timing.warmup_windows, max(0, len(windows) - 1))
    timing_summary = timer.summary(warmup_eff)
    fps = timer.fps(stride, warmup_eff)
    first_ms = search_ms_list[0] if search_ms_list else 0.0
    steady = float(np.median(search_ms_list[warmup_eff:])) if \
        len(search_ms_list) > warmup_eff else float(np.median(search_ms_list or [0.0]))
    jit_ms = max(0.0, first_ms - steady)
    timing_payload = {
        "layers": {k: v.to_dict() for k, v in timing_summary.items()},
        "fps": fps,
        "jit_compile_ms": jit_ms,
        "warmup_windows": warmup_eff,
        "requested_warmup_windows": cfg.timing.warmup_windows,
        "n_windows": len(windows),
    }
    (out_dir / "timing.json").write_text(
        json.dumps(timing_payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8")
    n_accepted = sum(1 for w in windows if w.accepted)
    return {"windows": windows, "timer": timer, "cache": cache, "jit_ms": jit_ms,
            "fps": fps, "n_accepted": n_accepted, "n_windows": len(windows),
            "cache_stats": cache.stats()}


def _record(wi, t, N, cfg, plan, outcome, res, pk, pnr_v, margin, cont,
            decode_ms, pre_ms, state_ms, search_ms, mu=None, sigma=None, single=False,
            a_best=None, use_accel=False):
    total = pre_ms + search_ms + state_ms
    est = None
    if outcome.target_present and res is not None:
        if use_accel:
            px, py = PK.center_frame_position_accel(pk.xy_sub, res.v_best,
                                                   a_best if a_best is not None else (0.0, 0.0),
                                                   N)
        else:
            px, py = PK.center_frame_position(pk.xy_sub, res.v_best, N)
        est = R.Estimate(x=float(px), y=float(py),
                         vx=float(res.v_best[0]), vy=float(res.v_best[1]))
    peak = None if pk is None else R.PeakResult(
        value=float(pk.value), xy_int=pk.xy_int, xy_sub=pk.xy_sub,
        subpixel_used=bool(pk.subpixel_used), valid_count=int(pk.valid_count),
        edge_candidate=bool(pk.edge_candidate))
    conf = None if pnr_v is None else R.Confidence(
        pnr=float(pnr_v), mu_bg=float(mu or 0.0), sigma_bg=float(sigma or 0.0),
        margin=float(margin), single_candidate=bool(single))
    return R.WindowRecord(
        window_index=wi, t_start=t, t_center=plan.t_center,
        assoc_frame=MET.associate_frame(t, N, cfg.metrics.association),
        search_mode=plan.search_mode or "skip",
        state_in=outcome.state_in, state_out=outcome.state_out,
        accepted=bool(outcome.accepted), target_present=bool(outcome.target_present),
        reason=outcome.reason, est=est, peak=peak, confidence=conf,
        continuity=cont, flags=outcome.flags, miss_count=outcome.miss_count,
        reacq_count=outcome.reacq_count, skipped_by_period=outcome.skipped_by_period,
        timing_ms=R.TimingMs(t_pre=pre_ms, t_search=search_ms, t_state=state_ms,
                             t_total=total))


# ---------------- repeat ranges ----------------

def _repeat_ranges(runs: list[dict]) -> dict:
    def rng(vals):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return {"min": None, "max": None, "range": None}
        return {"min": float(min(vals)), "max": float(max(vals)),
                "range": float(max(vals) - min(vals))}
    return {
        "n_runs": len(runs),
        "window_fps": rng([r["fps"]["window_fps"] for r in runs]),
        "effective_frame_fps": rng([r["fps"]["effective_frame_fps"] for r in runs]),
        "mean_t_window_ms": rng([r["fps"]["mean_t_window_ms"] for r in runs]),
        "n_accepted": rng([r["n_accepted"] for r in runs]),
    }


# ---------------- CLI ----------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_sequence.py",
                                description="SPA2D 单序列端到端运行（SPEC §5.7）")
    p.add_argument("--config", required=True)
    p.add_argument("--gt", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--backend", choices=["jax", "numpy"], default=None)
    p.add_argument("--allow-unfrozen", action="store_true")
    return p


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg or "resource_exhausted" in msg or "cuda_error_out_of_memory" in msg)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc.format(), file=sys.stderr)
        return exc.exit_code

    backend_name = args.backend or cfg.backend
    if backend_name == "jax" and not backends.jax_available():
        print("⚠️  JAX 不可用，回退 numpy 后端", file=sys.stderr)
        backend_name = "numpy"

    try:
        th, frozen, used_default = _resolve_thresholds(cfg, args.allow_unfrozen)
    except ConfigError as exc:
        print(exc.format(), file=sys.stderr)
        return exc.exit_code

    frames_dir = Path(cfg.sequence.frames_dir)
    video_path = Path(cfg.sequence.video) if cfg.sequence.video else None
    src_kwargs = {}
    if video_path is not None and video_path.is_file():
        src_kwargs["video"] = str(video_path)
    elif frames_dir.is_dir():
        src_kwargs["frames_dir"] = str(frames_dir)
    elif frames_dir.is_file():
        src_kwargs["video"] = str(frames_dir)
    else:
        print(f"[spa2d] E_DATA_MISSING: 找不到帧源（video={video_path} / frames_dir={frames_dir}）"
              f" → 运行 scripts/unpack_mvdt.sh 或 make_synthetic.py", file=sys.stderr)
        return EXIT_DATA
    try:
        source = io.open_sequence(**src_kwargs)
        source.probe()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[spa2d] E_DATA_MISSING: {exc} → 检查 configs 的 frames_dir/video 或运行 "
              f"make_synthetic.py / unpack_mvdt.sh", file=sys.stderr)
        return EXIT_DATA

    out_dir = Path(args.out)
    env = _env_info(backend_name)
    nrep = max(1, args.repeat)
    limit = args.limit
    if limit is None and os.environ.get("SPA2D_LIMIT"):
        limit = int(os.environ["SPA2D_LIMIT"])
    chunk_bytes = cfg.search.gpu_chunk_bytes
    runs = []
    while True:
        try:
            runs = []
            for r in range(nrep):
                run_dir = out_dir if r == 0 else out_dir / f"repeat_{r}"
                result = _run_once(cfg, th, source, backend_name, limit, run_dir,
                                   chunk_bytes, match_sigma=frozen.matched_sigma)
                runs.append({"dir": str(run_dir), **{k: result[k] for k in
                             ("jit_ms", "fps", "n_accepted", "n_windows", "cache_stats")}})
            break
        except ConfigError as exc:
            print(exc.format(), file=sys.stderr)
            return exc.exit_code
        except (FileNotFoundError, ValueError, IndexError) as exc:
            print(f"[spa2d] E_RUNTIME: {exc}", file=sys.stderr)
            return EXIT_RUNTIME
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc) and chunk_bytes > (1 << 20):
                chunk_bytes = max(chunk_bytes // 2, 1 << 20)
                print(f"⚠️  GPU OOM（E_OOM_CHUNK）：降低 gpu_chunk_bytes 至 {chunk_bytes} 重试",
                      file=sys.stderr)
                continue
            traceback.print_exc()
            print(f"[spa2d] E_RUNTIME: {exc}", file=sys.stderr)
            return EXIT_RUNTIME

    first = runs[0]
    meta = {
        "config": {"path": str(args.config), "name": cfg.config_name,
                   "snapshot": _config_snapshot(args.config)},
        "frozen": {"path": str(cfg.thresholds_file), "sha256": frozen.sha256,
                   "frozen_at": frozen.frozen_at, "used_default": used_default,
                   "snapshot": frozen.raw},
        "environment": env,
        "data": _data_hash(cfg),
        "seed": cfg.seed,
        "argv": sys.argv,
        "backend": backend_name,
        "matched_sigma": float(os.environ.get("SPA2D_MATCH_SIGMA", "0") or 0.0) or float(frozen.matched_sigma or 0.0),
        "repeat": max(1, args.repeat),
        "gt": args.gt,
        "jit_compile_ms": first["jit_ms"],
        "cache_stats": first["cache_stats"],
        "chunk_bytes_used": chunk_bytes,
        "fps": first["fps"],
        "n_windows": first["n_windows"],
        "n_accepted": first["n_accepted"],
    }
    # cache stats captured during run 0 are attached by re-reading? store directly:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    if args.repeat > 1:
        (out_dir / "repeat_ranges.json").write_text(
            json.dumps(_repeat_ranges(runs), indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8")

    print(f"{cfg.config_name}: windows={first['n_windows']} accepted={first['n_accepted']} "
          f"window_fps={first['fps']['window_fps']:.3f} "
          f"eff_fps={first['fps']['effective_frame_fps']:.3f} backend={backend_name} "
          f"-> {out_dir}")
    return EXIT_OK


def _config_snapshot(path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
