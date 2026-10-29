"""scripts/bench_laclad.py — LAC/LAD 单模块性能测试（两个环境通用）。

在 3.11+torch 与 3.8+onnxruntime 分别运行，参数一致即可对比。

    # 3.11 + torch
    source scripts/env_setup.sh
    $PY scripts/bench_laclad.py --backend torch --device cuda --out /tmp/b_torch.json

    # 3.8 + onnxruntime
    .venv38/bin/python scripts/bench_laclad.py --backend onnx --device cuda --out /tmp/b_onnx.json

规则（SPEC-ROS V2）：必须先预热再计时，否则首帧含 CUDA context 初始化。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mgmd import config as C
from mgmd.lac import crop_box

N_WARMUP = 10
N_REPEAT = 30


def timeit(fn, warmup=N_WARMUP, repeat=N_REPEAT) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(ts)
    return dict(mean=round(statistics.mean(s), 2),
                p50=round(s[len(s) // 2], 2),
                p95=round(s[int(len(s) * 0.95)], 2),
                min=round(s[0], 2), max=round(s[-1], 2))


def load_samples(n_frames: int):
    """从 phantom30 取真实帧、真实 LAC 裁剪、真实 LAD 中心。"""
    from mgmd.annotations import load_video_gt
    from mgmd.io_video import VideoReader
    from mgmd.mfe import MFEConfig, enhance
    from mgmd.tf import iou

    cfg = MFEConfig()
    cfg.apply_recommended()
    cfg.validate()
    gt = load_video_gt("phantom30")
    buf = {}
    with VideoReader(C.ARD_VIDEOS / "phantom30") as vr:
        for i, f in vr.iter_frames(start=1, limit=n_frames + 3):
            buf[i] = f

    frames, lac_src, centers = [], [], []
    for t in range(2, n_frames + 1):
        if t - 1 not in buf or t + 1 not in buf:
            break
        res = enhance(buf[t - 1], buf[t], buf[t + 1], cfg, t)
        cands = [(float(x), float(y), float(x + w), float(y + h))
                 for (x, y, w, h) in res.boxes]
        if not cands:
            continue
        o = gt[t].objects[0]
        gb = (o.x1, o.y1, o.x2, o.y2)
        cov = [c for c in cands if iou(c, gb) >= 0.25]
        if cov:
            c0 = cov[0]
            centers.append(((c0[0] + c0[2]) / 2, (c0[1] + c0[3]) / 2))
            frames.append(buf[t])
        # 存 (帧, 框)，推理时再裁 —— crop_box 返回尺寸可变，不能直接堆叠
        lac_src.append((buf[t], cands[0]))

    n = min(len(centers), len(frames), len(lac_src))
    return lac_src[:n], frames[:n], centers[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "onnx"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lac_src, frames, centers = load_samples(args.frames)
    n = len(centers)
    res = dict(backend=args.backend, device=args.device,
               python=sys.version.split()[0], n_samples=n)

    if args.backend == "torch":
        import torch
        from mgmd.lac import LacClassifier
        from mgmd.lad import LadDetector
        res["torch"] = torch.__version__
        dev = args.device if torch.cuda.is_available() else "cpu"
        lac = LacClassifier(device=dev)
        lad = LadDetector(conf_thresh=0.5, device=dev)

        def run_lac():
            crops = [crop_box(f, b) for f, b in lac_src]
            return lac.mav_prob_batch([c for c in crops if c is not None])

        def run_lad():
            out = [lad.refine(frames[i], centers[i]) for i in range(n)]
            return [o for o in out if o is not None]

        # 数值基准
        p_torch = np.array(run_lac(), dtype=np.float64)
        res["lac_n_valid"] = int(len(p_torch))
        boxes_torch = []
        for i in range(n):
            rb = lad.refine(frames[i], centers[i])
            boxes_torch.append(np.array(rb.box) if rb else None)
        np.save("/tmp/ref_lac.npy", p_torch)
        np.save("/tmp/ref_lad.npy", np.array(
            [b if b is not None else np.full(4, np.nan) for b in boxes_torch]))
        res["lac_probs_sample"] = [round(float(v), 6) for v in p_torch[:5]]

    else:
        import onnxruntime as ort
        res["ort"] = ort.__version__
        provs = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.device == "cuda" else ["CPUExecutionProvider"])
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        lac_s = ort.InferenceSession(str(C.WEIGHTS_DIR / "lac.onnx"), so, providers=provs)
        lad_s = ort.InferenceSession(str(C.WEIGHTS_DIR / "yolov5s_GLAD-crop.onnx"),
                                     so, providers=provs)
        res["providers"] = lac_s.get_providers()
        import cv2
        from mgmd.lad import crop_region

        def run_lac():
            crops = [crop_box(f, b) for f, b in lac_src]
            batch = np.stack([cv2.resize(c, (32, 32))[:, :, ::-1].copy()
                              for c in crops if c is not None])
            # 必须连续化：负步长视图直接 astype 会读错数据
            x = np.ascontiguousarray(
                batch.transpose(0, 3, 1, 2)).astype(np.float32) / 255.0
            return lac_s.run(None, {"input": x})[0]

        def run_lad():
            for i in range(n):
                h, w = frames[i].shape[:2]
                x0, y0, cw, ch = crop_region((h, w), centers[i], 320)
                crop = frames[i][y0:y0 + ch, x0:x0 + cw]
                im = cv2.resize(crop, (320, 320))[:, :, ::-1].copy()
                x = np.ascontiguousarray(
                    im.transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
                lad_s.run(None, {"images": x})

        logits = run_lac()
        res["lac_n_valid"] = int(len(logits))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = (e / e.sum(axis=1, keepdims=True))[:, 1]
        res["lac_probs_sample"] = [round(float(v), 6) for v in probs[:5]]
        # 与 torch 基准对比
        ref = Path("/tmp/ref_lac.npy")
        if ref.exists():
            r = np.load(ref)
            m = min(len(r), len(probs))
            res["lac_compare_n"] = int(m)
            res["torch_n"] = int(len(r))
            res["onnx_n"] = int(len(probs))
            if m > 0:
                res["lac_max_abs_diff_vs_torch"] = float(np.abs(r[:m] - probs[:m]).max())

    res["lac_ms"] = timeit(run_lac)
    res["lad_ms"] = timeit(run_lad)
    res["lad_per_box_ms"] = round(res["lad_ms"]["p50"] / max(1, n), 3)

    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
