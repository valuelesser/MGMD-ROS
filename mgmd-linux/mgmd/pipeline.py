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
        min_hits: int = 1,
        device: str = "auto", lad_batch: int = 1, lad_topk: int = 0,
        prefilter_lac: float = 0.0, lad_trt_engine: str | None = None,
        progress: bool = False) -> list[FrameOutput]:
    """跑一段视频，返回逐帧输出。"""
    gt = load_video_gt(video)          # 仅用于确定帧范围与校验
    use_tf = "tf" in modules
    use_lac = "lac" in modules
    use_lad = "lad" in modules

    tracker = None
    if use_tf:
        tracker = SortTracker(iou_thresh=iou_thresh, max_age=max_age, min_hits=min_hits,
                              span=span, d_mode=d_mode, d_min=d_min)

    clf = None
    if use_lac:
        from mgmd.lac import LacClassifier
        clf = LacClassifier(device=device)

    lad = None
    if use_lad:
        from mgmd.lad import LadDetector
        lad = LadDetector(conf_thresh=lad_conf, device=device,
                          trt_engine=lad_trt_engine)

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

            # 前置预筛（SPEC §8.18）：先 LAC 再 TF，避免 SORT 轨迹池爆炸
            # 论文原序是 MFE→TF→LAC；此处为性能优化，报告中必须声明
            if prefilter_lac > 0 and pool and use_lac:
                from mgmd.lac import crop_box as _cb
                nonlocal_clf = clf
                if nonlocal_clf is None:
                    from mgmd.lac import LacClassifier
                    nonlocal_clf = LacClassifier(device=device)
                    clf = nonlocal_clf
                _frame = buf[t]
                _crops = [_cb(_frame, b) for b, _ in pool]
                _pairs = [(b, c) for (b, _s), c in zip(pool, _crops) if c is not None]
                if _pairs:
                    _probs = nonlocal_clf.mav_prob_batch([c for _, c in _pairs])
                    pool = [(b, p) for (b, _c), p in zip(_pairs, _probs)
                            if p >= prefilter_lac]
                else:
                    pool = []

            if tracker is not None:
                # step_H 对齐（SPEC §8 缺口1 / SPEC-ROS §22）：
                # MFE 的 prev_H 是 t-k -> t 的单应，按 key=帧号 存入轨迹状态，
                # 供 Algorithm 1 step3 把 s_{N-3} 变换到 s_N 坐标系后再算 d。
                # 此前从未传递 → compose_H 恒返回 None → 对齐从未生效。
                # k!=1 时单帧 prev_H 跨 k 帧，语义与 key=f 的"相邻步"不符，
                # 故仅在 k==1 时传递（k>1 保持原行为，避免错误级联）。
                _step_H = None
                if k == 1:
                    _step_H = res.info.get("prev_H")
                tracks = tracker.update([b for b, _ in pool], t,
                                        scores=[s for _, s in pool],
                                        step_H=_step_H)
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
                # lad_topk>0 时只精修 score 最高的前 k 个候选（SPEC §8.14）
                # 依据：实测 GT 有 83% 概率落在 LAC 排序 top-3 内
                if lad_topk and len(pool) > lad_topk:
                    pool = sorted(pool, key=lambda x: -x[1])[:lad_topk]
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
