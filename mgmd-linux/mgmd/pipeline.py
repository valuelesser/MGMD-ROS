"""mgmd.pipeline — 逐帧串联 MFE → TF → LAC（SPEC §3/T06）。

模块可任意子集（SPEC F1，Table 3 需 6 种组合）：
    mfe / mfe,tf / mfe,lac / mfe,tf,lac / mfe,lac,lad / mfe,tf,lac,lad
未启用的模块为恒等直通。

输出每帧多框记录（多目标支持，SPEC §5.4），写入 JSONL。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mgmd import config as C
from mgmd.annotations import load_video_gt
from mgmd.io_video import VideoReader
from mgmd.mfe import MFEConfig, enhance
from mgmd.tf import SortTracker


@dataclass
class FrameOutput:
    frame: int
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    n_cand: int = 0
    timing_ms: float = 0.0


def parse_modules(spec: str) -> frozenset[str]:
    allowed = {"mfe", "tf", "lac", "lad"}
    parts = {p.strip().lower() for p in spec.split(",") if p.strip()}
    bad = parts - allowed
    if bad:
        raise C.MgmdError("E_CONFIG", f"未知模块: {sorted(bad)}",
                          f"可选: {sorted(allowed)}", C.EXIT_CONFIG)
    if "mfe" not in parts:
        raise C.MgmdError("E_CONFIG", "mfe 是必需模块（候选来源）",
                          "至少包含 mfe", C.EXIT_CONFIG)
    return frozenset(parts)


def run(video: str, mfe_cfg: MFEConfig, modules: frozenset[str],
        start: int = 1, limit: int | None = None,
        d_mode: str = "enclosing", d_min: float = 1.0,
        lac_thresh: float = 0.5, iou_thresh: float = 0.3,
        max_age: int = 3, span: int = 3, lad_conf: float = 0.5,
        device: str = "auto", lad_batch: int = 1,
        progress: bool = False) -> list[FrameOutput]:
    """跑一段视频，返回逐帧输出。"""
    gt = load_video_gt(video)          # 仅用于确定帧范围与校验
    use_tf = "tf" in modules
    use_lac = "lac" in modules
    use_lad = "lad" in modules

    tracker = None
    if use_tf:
        tracker = SortTracker(iou_thresh=iou_thresh, max_age=max_age, min_hits=1,
                              span=span, d_mode=d_mode, d_min=d_min)

    clf = None
    if use_lac:
        from mgmd.lac import LacClassifier
        clf = LacClassifier(device=device)

    lad = None
    if use_lad:
        from mgmd.lad import LadDetector
        lad = LadDetector(conf_thresh=lad_conf, device=device)

    outs: list[FrameOutput] = []
    k = mfe_cfg.k
    end = (start + limit - 1) if limit else max(gt)
    read_end = end + k

    with VideoReader(video) as vr:
        buf: dict[int, np.ndarray] = {}
        for idx, fr in vr.iter_frames(start=start, limit=max(0, read_end - start + 1)):
            buf[idx] = fr
            t = idx - k
            if t < start or (t - k) not in buf:
                continue
            if mfe_cfg.needs_next and (t + k) not in buf:
                continue

            t0 = time.perf_counter()
            res = enhance(buf[t - k], buf[t],
                          buf.get(t + k) if mfe_cfg.needs_next else None, mfe_cfg, t)
            cand_boxes = [(float(x), float(y), float(x + w), float(y + h)) for (x, y, w, h) in res.boxes]
            cand_scores = [float(w * h) for (_x, _y, w, h) in res.boxes]

            pool = list(zip(cand_boxes, cand_scores))

            if tracker is not None:
                tracks = tracker.update([b for b, _ in pool], t,
                                        scores=[s for _, s in pool])
                pool = [(tr.box, tr.score) for tr in tracks]

            if clf is not None and pool:
                from mgmd.lac import crop_box
                frame = buf[t]
                crops = []
                keep = []
                for b, s in pool:
                    c = crop_box(frame, b)
                    if c is not None:
                        crops.append(c)
                        keep.append((b, s))
                if crops:
                    probs = clf.mav_prob_batch(crops)
                    pool = [(b, p) for (b, _s), p in zip(keep, probs) if p >= lac_thresh]
                else:
                    pool = []

            if lad is not None and pool:
                frame = buf[t]
                centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b, _s in pool]
                if lad_batch > 1:
                    rbs = lad.refine_batch(frame, centers, batch_size=lad_batch)
                else:
                    rbs = [lad.refine(frame, c) for c in centers]
                refined = [(rb.box, rb.conf) for rb in rbs
                           if rb is not None and rb.conf >= lad_conf]
                pool = refined

            dt = (time.perf_counter() - t0) * 1000.0
            outs.append(FrameOutput(
                frame=t,
                boxes=[b for b, _ in pool],
                scores=[s for _, s in pool],
                n_cand=len(res.boxes),
                timing_ms=dt,
            ))
            if progress and len(outs) % 50 == 0:
                print(f"  ... {len(outs)} 帧, 当前 {t}")
            for kk in [b for b in buf if b < t - 2 * k - 2]:
                del buf[kk]

    return outs


def write_jsonl(outs: list[FrameOutput], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for o in outs:
            f.write(json.dumps({
                "frame": o.frame, "n_cand": o.n_cand,
                "boxes": [[round(v, 2) for v in b] for b in o.boxes],
                "scores": [round(s, 6) for s in o.scores],
                "timing_ms": round(o.timing_ms, 2),
            }, ensure_ascii=False) + "\n")
