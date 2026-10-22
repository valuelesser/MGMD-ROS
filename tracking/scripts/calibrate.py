"""scripts/calibrate.py — 七步标定协议 + 冻结阈值（SPEC §5.12 / §3.5 / PRD §9）。

用法：
    python scripts/calibrate.py --synthetic data_raw/synthetic \
        --validation synthetic --configs configs/D1_no_overlap.yaml \
        --out configs/thresholds_frozen.yaml [--force]

标定输入白名单：{D1/cam2, D3/cam3, synthetic}（代码级断言，禁止测试序列 D2/D4 参与选择）。
输出 thresholds_frozen.yaml（含输入哈希）与 outputs/metrics/calibration_trace.json（逐步记录）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spa2d import backends, confidence as C, peak as PK, preprocess as P  # noqa: E402
from spa2d import io_mvdt as io, records as R, search as S  # noqa: E402
from spa2d.config import load_config  # noqa: E402

WHITELIST = ("D1/cam2", "D3/cam3", "synthetic")
KERNELS = (3, 5, 7, 9, 11)
VALIDATION_VMAX = {"D1/cam2": 1.0, "D3/cam3": 15.0}
CAL_CHUNK_BYTES = 134217728  # 128MiB -> 1080p chunk≈8，避免 OOM


class CalibrationError(Exception):
    def __init__(self, code: str, message: str, exit_code: int = 4):
        self.code, self.message, self.exit_code = code, message, exit_code
        super().__init__(message)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------- synthetic loading ----------------

def _load_synthetic(root: Path):
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    seqs = []
    for s in man["sequences"]:
        if s.get("kind") != "sanity":
            continue
        d = root / s["dir"]
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        seqs.append({"name": s["seq_id"], "dir": d, "meta": meta,
                     "gt": R.read_gt_csv(d / "gt.csv"),
                     "frames_dir": d / "frames"})
    return seqs


def _preprocessed(frames_dir: Path, kernel, k: int, max_frames: int, color="BGR"):
    import cv2

    files = sorted(frames_dir.glob("*.png"))[:max_frames] or sorted(frames_dir.glob("*.jpg"))[:max_frames]
    return np.stack([P.preprocess_frame(cv2.imread(str(f)), kernel, color_order=color)
                     for f in files])


def _window_observations(frames, gt, N, stride, search_params, kernel, backend, rho, width_ref):
    """在 (vmax,δc,rf,δf,top_k) 下逐窗收集 (pnr, margin, r_cont, exempt, edge, assoc, correct, vbest)。"""
    vmax, dc, rf, df, top_k = search_params
    obs = []
    last = None
    for t in range(0, len(frames) - N + 1, stride):
        win = frames[t:t + N]
        res = S.global_search(win, vmax, dc, rf, df, top_k, 512, CAL_CHUNK_BYTES, backend=backend)
        A, M = backend.accumulate_window(win, np.asarray(res.v_best, np.float32))
        backend.block_until_ready((A, M))
        A = np.asarray(A, np.float64)
        M = np.asarray(M, np.float64)
        pk = PK.extract_peak(A, M, N, PK.edge_margin_for(N, res.v_best))
        mu, sigma, cnt = C.background_stats(A, M, pk.xy_int, rho)
        pnr = 0.0 if cnt < C.INSUFFICIENT_BG else C.pnr(pk.value, mu, sigma)
        margin, single = C.velocity_margin(res.scores, res.cands, res.v_best, df)
        prior = None if last is None else C.Prior(last["c"], last["p"], last["v"])
        p_center = PK.center_frame_position(pk.xy_sub, res.v_best, N)
        cont = C.continuity(p_center, t + (N - 1) / 2.0, width_ref, prior)
        f = t + (N - 1) // 2
        g = gt[f] if 0 <= f < len(gt) else None
        correct = bool(g is not None and g.visible == 1 and g.cx is not None
                       and math.hypot(p_center[0] - g.cx, p_center[1] - g.cy) <= 100)
        obs.append({"pnr": pnr, "margin": margin, "r_cont": cont.r_cont,
                    "exempt": cont.exempt, "edge": pk.edge_candidate,
                    "correct": correct, "visible": bool(g is not None and g.visible == 1),
                    "vbest": tuple(res.v_best), "scores": list(res.scores),
                    "cands": [tuple(c) for c in res.cands]})
        last = {"c": t + (N - 1) / 2.0, "p": np.asarray(p_center), "v": np.asarray(res.v_best)}
    return obs


def _f1(obs, theta_high, theta_margin, radius=100):
    tp = fp = 0
    visible = sum(1 for o in obs if o["visible"])
    for o in obs:
        accepted = o["pnr"] >= theta_high and o["margin"] >= theta_margin
        if accepted and o["correct"]:
            tp += 1
        elif accepted and not o["correct"]:
            fp += 1
    recall = tp / visible if visible else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * recall * prec / (recall + prec) if (recall + prec) else 0.0
    return f1, prec, recall


def _collect_source_obs(source, gt, F, N, stride, search_params, kernel, backend, rho,
                        width_ref, color_order):
    """对真实视频/帧目录逐窗收集观测（用于真实验证片标定）。"""
    obs = []
    last = None
    vmax, dc, rf, df, top_k = search_params
    t = 0
    while t + N <= F:
        frames = np.stack([P.preprocess_frame(source.read(t + k), kernel, color_order=color_order)
                           for k in range(N)])
        res = S.global_search(frames, vmax, dc, rf, df, top_k, 512, CAL_CHUNK_BYTES, backend=backend)
        A, M = backend.accumulate_window(frames, np.asarray(res.v_best, np.float32))
        backend.block_until_ready((A, M))
        A = np.asarray(A, np.float64)
        M = np.asarray(M, np.float64)
        pk = PK.extract_peak(A, M, N, PK.edge_margin_for(N, res.v_best))
        mu, sigma, cnt = C.background_stats(A, M, pk.xy_int, rho)
        pnr = 0.0 if cnt < C.INSUFFICIENT_BG else C.pnr(pk.value, mu, sigma)
        margin, _ = C.velocity_margin(res.scores, res.cands, res.v_best, df)
        prior = None if last is None else C.Prior(last["c"], last["p"], last["v"])
        p_center = PK.center_frame_position(pk.xy_sub, res.v_best, N)
        cont = C.continuity(p_center, t + (N - 1) / 2.0, width_ref, prior)
        f = t + (N - 1) // 2
        g = gt[f] if 0 <= f < len(gt) else None
        correct = bool(g is not None and g.visible == 1 and g.cx is not None
                       and math.hypot(p_center[0] - g.cx, p_center[1] - g.cy) <= 100)
        obs.append({"pnr": pnr, "margin": margin, "r_cont": cont.r_cont,
                    "exempt": cont.exempt, "edge": pk.edge_candidate, "correct": correct,
                    "visible": bool(g is not None and g.visible == 1)})
        last = {"c": t + (N - 1) / 2.0, "p": np.asarray(p_center), "v": np.asarray(res.v_best)}
        t += stride
    return obs


def calibrate(synthetic_root: Path, validation_specs, config_path: Path,
              backend_name: str, trace: dict, gt_dir: Path | None = None,
              val_frames: int = 500) -> dict:
    be = backends.get_backend(backend_name)
    validation = [n for n, _ in validation_specs]
    cfg = load_config(config_path)
    N, stride = cfg.window.N, cfg.window.stride
    seqs = _load_synthetic(synthetic_root)
    if not seqs:
        raise CalibrationError("E_DATA_MISSING", "合成集无 sanity 序列", 3)
    syn_vmax = max((max(abs(sq["meta"]["velocity"]["vx"]), abs(sq["meta"]["velocity"]["vy"]))
                    for sq in seqs), default=1.0)
    search_params = (float(syn_vmax) * 1.1 + 0.5, cfg.search.coarse.delta_c,
                     cfg.search.fine.rf, cfg.search.fine.delta_f, cfg.search.top_k)
    width_ref = 8.0
    rho = C.rho_for(width_ref)
    dev = be.name

    # step1: kernel selection
    step1 = {}
    for k in KERNELS:
        seps = []
        for sq in seqs[:8]:
            frames = _preprocessed(sq["frames_dir"], P.build_kernel(k), k, 40, cfg.sequence.color_order)
            obs = _window_observations(frames, sq["gt"], N, stride, search_params,
                                       P.build_kernel(k), be, rho, width_ref)
            vtrue = np.array([sq["meta"]["velocity"]["vx"], sq["meta"]["velocity"]["vy"]])
            corr = [o["pnr"] for o in obs if max(abs(o["vbest"][0] - vtrue[0]),
                                                 abs(o["vbest"][1] - vtrue[1])) <= max(2 * search_params[3], 0.2)]
            wrong = []
            for o in obs:
                far = [i for i, c in enumerate(o["cands"])
                       if max(abs(c[0] - vtrue[0]), abs(c[1] - vtrue[1])) > max(2 * search_params[3], 0.2)]
                if far:
                    bi = max(far, key=lambda i: o["scores"][i])
                    wrong.append(max(o["scores"][bi] - min(o["scores"]), 0.0))
            if corr and wrong:
                seps.append(float(np.median(corr) - np.median(wrong)))
        step1[k] = float(np.median(seps)) if seps else -1e9
    best_k = max(KERNELS, key=lambda k: (step1[k], -k))
    trace["step1_kernel"] = {"separation": step1, "chosen": best_k}

    # validation observations: real clips (D1/cam2, D3/cam3) if paths given, else synthetic
    val_obs = []
    real_used = []
    for name, path in validation_specs:
        if not path:
            continue
        p = Path(path)
        try:
            source = io.open_sequence(frames_dir=str(p) if p.is_dir() else None,
                                      video=str(p) if p.is_file() else None)
        except (FileNotFoundError, ValueError):
            continue
        gt_path = (gt_dir or Path("data_derived/gt")) / f"{name.replace('/', '_')}.csv"
        if not gt_path.is_file():
            continue
        gt = R.read_gt_csv(gt_path)
        F = min(len(source), int(val_frames))
        vmax = VALIDATION_VMAX.get(name, cfg.search.coarse.vmax)
        sp = (vmax, cfg.search.coarse.delta_c, cfg.search.fine.rf, cfg.search.fine.delta_f,
              cfg.search.top_k)
        val_obs += _collect_source_obs(source, gt, F, N, stride, sp, P.build_kernel(best_k),
                                       be, rho, width_ref, cfg.sequence.color_order)
        real_used.append({"name": name, "path": str(p), "frames": F})
    if not val_obs:
        for sq in seqs[:8]:
            frames = _preprocessed(sq["frames_dir"], P.build_kernel(best_k), best_k, 60,
                                   cfg.sequence.color_order)
            val_obs += _window_observations(frames, sq["gt"], N, stride, search_params,
                                            P.build_kernel(best_k), be, rho, width_ref)
    trace["validation_used"] = real_used or [{"name": "synthetic", "frames": "synthetic"}]

    # step2: theta_high
    scan2 = []
    t = 1.0
    while t <= 20.0 + 1e-9:
        f1, prec, rec = _f1(val_obs, t, 0.0)
        scan2.append({"theta_high_pnr": round(t, 2), "f1": f1, "precision": prec * 100, "recall": rec * 100})
        t += 0.5
    best_f1 = max(s["f1"] for s in scan2)
    cand = [s for s in scan2 if s["f1"] >= best_f1 - 0.002]
    theta_high = max(cand, key=lambda s: (s["precision"], s["theta_high_pnr"]))["theta_high_pnr"]
    trace["step2_theta_high"] = {"scan": scan2, "chosen": theta_high, "f1": best_f1}

    # step3: theta_low
    base_prec = _f1(val_obs, theta_high, 0.0)[1]
    scan3 = []
    t = 1.0
    while t <= theta_high + 1e-9:
        vo = _valid_outcome(val_obs, theta_high, t)
        prec = _f1(val_obs, theta_high, 0.0)[1]
        scan3.append({"theta_low_pnr": round(t, 2), "valid_outcome": vo, "precision": prec * 100})
        t += 0.5
    feasible = [s for s in scan3 if s["precision"] >= base_prec * 100 - 0.5]
    theta_low = max(feasible, key=lambda s: s["valid_outcome"])["theta_low_pnr"] if feasible else theta_high
    trace["step3_theta_low"] = {"scan": scan3, "chosen": theta_low}

    # step4: theta_margin
    scan4 = []
    for m in (0.0, 0.01, 0.02, 0.05, 0.1):
        n_bad = sum(1 for o in val_obs if o["pnr"] >= theta_high and o["margin"] >= m
                    and (not o["correct"]) and (o["r_cont"] is not None)
                    and o["r_cont"] > C.cont_radii(width_ref)[1])
        scan4.append({"theta_margin": m, "n_bad": n_bad})
    zero = [s for s in scan4 if s["n_bad"] == 0]
    theta_margin = min(zero, key=lambda s: s["theta_margin"])["theta_margin"] if zero else 0.1
    trace["step4_theta_margin"] = {"scan": scan4, "chosen": theta_margin}

    # step5: edge
    theta_edge = theta_high + 2.0
    trace["step5_theta_edge"] = {"theta_edge_pnr": theta_edge}

    # step6: width_ref per sequence (synthetic seqs + validation names)
    width_by_seq = {}
    for sq in seqs:
        ws = [g.width_px for g in sq["gt"][:500] if g.visible == 1 and g.width_px]
        width_by_seq[sq["name"]] = float(np.median(ws)) if ws else None
    for name, _path in validation_specs:
        if name in width_by_seq:
            continue
        gp = (gt_dir or Path("data_derived/gt")) / f"{name.replace('/', '_')}.csv"
        if gp.is_file():
            gt = R.read_gt_csv(gp)
            ws = [g.width_px for g in gt[:500] if g.visible == 1 and g.width_px]
            width_by_seq[name] = float(np.median(ws)) if ws else None
        else:
            width_by_seq[name] = None
    trace["step6_width_ref"] = width_by_seq

    # step7: held-out tail check (synthetic: second half of each seq)
    tail = {}
    for sq in seqs[:8]:
        frames = _preprocessed(sq["frames_dir"], P.build_kernel(best_k), best_k, 60,
                               cfg.sequence.color_order)
        for label, sub in (("all", range(len(frames))), ("tail", range(len(frames) // 2, len(frames)))):
            idx = list(sub)
            f2 = frames[min(idx):max(idx) + 1] if idx else frames[:0]
            if len(f2) < N:
                continue
            o = _window_observations(f2, sq["gt"], N, stride, search_params,
                                     P.build_kernel(best_k), be, rho, width_ref)
            f1, _, _ = _f1(o, theta_high, theta_margin)
            tail.setdefault(sq["name"], {})[label] = f1
    trace["step7_tail_check"] = tail

    return {
        "kernel_size": best_k,
        "theta_high_pnr": float(theta_high),
        "theta_low_pnr": float(min(theta_low, theta_high)),
        "theta_margin": float(theta_margin),
        "theta_edge_pnr": float(theta_edge),
        "width_ref": width_by_seq,
    }


def _valid_outcome(obs, theta_high, theta_low):
    vo = 0
    for o in obs:
        accepted = (o["pnr"] >= theta_high) or (theta_low <= o["pnr"] < theta_high and o["margin"] >= 0)
        if accepted and o["correct"]:
            vo += 1
        elif (not accepted) and (not o["visible"]):
            vo += 1
    return vo / len(obs) if obs else 0.0


def _code_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="calibrate.py", description="七步标定 + 冻结阈值")
    p.add_argument("--synthetic", default="data_raw/synthetic")
    p.add_argument("--validation", default="synthetic",
                   help="逗号列表 name[:path]；白名单 {D1/cam2,D3/cam3,synthetic}；"
                        "给出 path（视频/帧目录）则在该真实验证片上标定")
    p.add_argument("--gt-dir", default="data_derived/gt")
    p.add_argument("--validation-frames", type=int, default=500)
    p.add_argument("--configs", nargs="+", default=["configs/D1_no_overlap.yaml"])
    p.add_argument("--out", default="configs/thresholds_frozen.yaml")
    p.add_argument("--backend", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    validation_specs = []
    for entry in args.validation.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, _, path = entry.partition(":")
        if name not in WHITELIST:
            print(f"[spa2d] E_CONFIG_INVALID: 标定白名单禁止 {name}（仅 {WHITELIST}）", file=sys.stderr)
            return 4
        validation_specs.append((name, path or None))
    validation = [n for n, _ in validation_specs]

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"[spa2d] E_FROZEN_TAMPER: {out} 已存在（--force 覆盖会记录已重标定）", file=sys.stderr)
        return 5

    backend_name = args.backend or ("jax" if backends.jax_available() else "numpy")
    trace = {"protocol_version": "prd-v1.0", "whitelist": list(WHITELIST),
             "validation": validation, "backend": backend_name,
             "generated_at": datetime.now(timezone.utc).isoformat()}
    try:
        result = calibrate(Path(args.synthetic), validation_specs, Path(args.configs[0]),
                           backend_name, trace, gt_dir=Path(args.gt_dir),
                           val_frames=args.validation_frames)
    except CalibrationError as exc:
        print(f"[spa2d] {exc.code}: {exc.message}", file=sys.stderr)
        return exc.exit_code

    import yaml

    out.parent.mkdir(parents=True, exist_ok=True)
    frozen = {
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "calibration": {
            "synthetic_manifest_sha256": _sha256_file(Path(args.synthetic) / "manifest.json"),
            "validation": validation,
            "configs_sha256": {Path(c).stem: _sha256_file(Path(c)) for c in args.configs},
            "code_commit": _code_commit(),
            "protocol_version": "prd-v1.0",
            "scan_trace": "outputs/metrics/calibration_trace.json",
        },
        "thresholds": {
            "theta_high_pnr": result["theta_high_pnr"],
            "theta_low_pnr": result["theta_low_pnr"],
            "theta_margin": result["theta_margin"],
            "theta_edge_pnr": result["theta_edge_pnr"],
            "v_stat": 0.05, "m_lost": 3, "reacq_max": 5, "gap_prior_windows": 2,
            "rho_formula": "max(8, ceil(2*width_ref))",
            "r_cont_high_formula": "max(25, 3*width_ref)",
            "r_cont_low_formula": "max(10, 1.5*width_ref)",
        },
        "width_ref": result["width_ref"],
        "preprocess": {"kernel_size": result["kernel_size"],
                       "matched_sigma": float(min(max(8.0 / 3.0, 1.2), 4.0))},
    }
    out.write_text(yaml.safe_dump(frozen, sort_keys=True, allow_unicode=True), encoding="utf-8")
    trace_path = Path("outputs/metrics/calibration_trace.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")
    print(f"frozen: {out} (k={result['kernel_size']}, theta_high={result['theta_high_pnr']}, "
          f"theta_low={result['theta_low_pnr']}, theta_margin={result['theta_margin']})")
    print(f"trace: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
