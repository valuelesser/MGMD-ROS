"""mgmd.pipeline — 逐帧编排 + WindowRecord 适配（SPEC §4.1/§5.6/§5.7）。

模块子集开关（--modules mfe,tf,lac,lad）：关闭的模块恒等直通；任一级输出为空 →
当帧 reason=ABSENT。每帧至多 1 个最终框（SPEC §5.6 选择规则）。输出逐帧 N=1 伪窗口
WindowRecord（SPEC §3.2 映射），写 windows.jsonl / run_meta.json / timing.json /
metrics.json，保证 make_tables.py --runs outputs/runs_mgmd 直接出表（FR-9）。

坐标约定：检测在 downscale 后的处理尺度进行；写记录前一律 ÷downscale 映射回原图
（SPEC §5.8，A4：MVDT GT 为全分辨率坐标系）。
"""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from mgmd.config import (EXIT_DATA, EXIT_OK, EXIT_RUNTIME, Config, MgmdError)
from mgmd.io_mvdt import MgmdFrameSource

MODULES_ALL = ("mfe", "tf", "lac", "lad")
WARMUP_WINDOWS = 20


class _VizStop(Exception):
    """可视化请求提前结束（q/关窗）；带上最后处理的帧号。"""

    def __init__(self, frame: int) -> None:
        super().__init__(f"viz stop at frame {frame}")
        self.frame = int(frame)


class _VizError(Exception):
    """可视化回调异常；保留原始异常供 E_RUNTIME 映射。"""

    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


def parse_modules(spec: str) -> frozenset[str]:
    """解析 --modules 逗号列表；非法模块名 → E_CONFIG_INVALID（退出码 2）。"""
    from mgmd.config import ALLOWED_MODULES
    mods = frozenset(m.strip() for m in spec.split(",") if m.strip())
    bad = sorted(mods - set(ALLOWED_MODULES))
    if bad:
        raise MgmdError("E_CONFIG_INVALID",
                        f"--modules 含未知模块: {bad}（可选 {ALLOWED_MODULES}）",
                        "修正 --modules，例如 mfe,tf,lac,lad")
    if not mods:
        raise MgmdError("E_CONFIG_INVALID", "--modules 不能为空",
                        "至少启用 mfe，例如 --modules mfe")
    if "mfe" not in mods:
        raise MgmdError("E_CONFIG_INVALID", f"模块组合必须包含 mfe（候选来源），实际 {sorted(mods)}",
                        "--modules 以 mfe 开头，例如 mfe,tf")
    return mods


# ---------------- 候选池与选择规则（SPEC §5.6） ----------------

class _Pick:
    """待选框：rect 为处理尺度 (x,y,w,h)；key 越大越优（并列取面积小者）。"""

    def __init__(self, rect, key: float, velocity=(0.0, 0.0), tracked=False):
        self.rect = tuple(float(v) for v in rect)
        self.key = float(key)
        self.velocity = velocity
        self.tracked = tracked

    @property
    def area(self) -> float:
        return self.rect[2] * self.rect[3]


def _select(pool: list[_Pick]) -> _Pick | None:
    if not pool:
        return None
    return max(pool, key=lambda p: (p.key, -p.area))


def _crop(frame: np.ndarray, rect) -> np.ndarray:
    """按 (x,y,w,h) 裁剪（边界 clamp，空裁剪返回 None 由调用方处理）。"""
    h, w = frame.shape[:2]
    x, y, rw, rh = (int(round(v)) for v in rect)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    x2 = max(x + 1, min(x + max(int(round(rw)), 1), w))
    y2 = max(y + 1, min(y + max(int(round(rh)), 1), h))
    return frame[y:y2, x:x2]


# ---------------- WindowRecord 适配（SPEC §3.2） ----------------

def _history_alignment_homography(buf, t: int, gap: int, cur_gray, cfg, res_m):
    """上一帧→当前帧单应（供 SORT 历史位置对齐，SPEC §5.4）。

    `align()` 返回 H = findHomography(当前帧→上一帧)（配合 WARP_INVERSE_MAP 补偿）；
    轨迹历史需要上一帧→当前帧，故取逆。MFE 在 gap>1 时对齐跨 k 帧，而轨迹历史
    必须逐帧对齐：此处单独估计 t−1→t。回退（无有效 H）返回 None，不搬历史。
    """
    from mgmd import align as A

    res = res_m if gap == 1 else (
        A.align(buf[t - 1][2], cur_gray, cfg.mfe) if (t - 1) in buf else None)
    if res is None or res.fallback:
        return None
    try:
        return np.linalg.inv(res.homography)
    except np.linalg.LinAlgError:
        return None


def _frame_record(idx: int, pick: _Pick | None, prev_state: str, downscale: float,
                  t_pre: float, t_search: float, t_state: float, pool_size: int = 0):
    from spa2d import records as R

    est = None
    conf = None
    if pick is not None:
        s = 1.0 / downscale
        cx = (pick.rect[0] + pick.rect[2] / 2.0) * s
        cy = (pick.rect[1] + pick.rect[3] / 2.0) * s
        est = R.Estimate(x=cx, y=cy, vx=pick.velocity[0] * s, vy=pick.velocity[1] * s)
        # pnr/margin 承载选择得分（依模块档：motion score / LAC prob / LAD conf），
        # 供半径匹配口径 AP（SPEC §9.3 stretch）排序使用；mu_bg/sigma_bg 占位 0。
        conf = R.Confidence(pnr=pick.key, mu_bg=0.0, sigma_bg=0.0, margin=pick.key,
                            single_candidate=(pool_size == 1))
    state_out = ("TRACKED" if pick.tracked else "INIT") if pick is not None else "ABSENT"
    return R.WindowRecord(
        window_index=idx, t_start=idx, t_center=float(idx), assoc_frame=idx,
        search_mode="global", state_in=prev_state, state_out=state_out,
        accepted=pick is not None, target_present=pick is not None,
        reason="OK" if pick is not None else "ABSENT",
        est=est, peak=None, confidence=conf, continuity=None,
        timing_ms=R.TimingMs(t_pre=t_pre, t_search=t_search, t_state=t_state,
                             t_total=t_pre + t_search + t_state),
    ), state_out


# ---------------- 主流程 ----------------

def _open_source(cfg: Config) -> MgmdFrameSource:
    """优先视频，其次帧目录；均缺失 → E_DATA_MISSING（退出码 3）。"""
    seq, rt = cfg.sequence, cfg.runtime
    if seq.video and Path(seq.video).is_file():
        return MgmdFrameSource(seq, rt, use_video=True)
    if Path(seq.frames_dir).is_dir():
        return MgmdFrameSource(seq, rt, use_video=False)
    raise MgmdError("E_DATA_MISSING",
                    f"序列数据缺失: video={seq.video} frames_dir={seq.frames_dir}",
                    "bash scripts/fetch_mvdt_videos.sh && bash scripts/unpack_mvdt.sh",
                    EXIT_DATA)


def frame_count(cfg: Config) -> int:
    """返回序列总帧数（只探测帧源，不启动流水线）。"""
    source = _open_source(cfg)
    try:
        return len(source)
    finally:
        source.close()


def _sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cfg: Config, modules: frozenset[str], gt_path: Path, out_dir: Path,
        limit: int | None, backend: str, start: int = 0,
        on_frame: Callable[..., bool] | None = None) -> int:
    """端到端运行 MGMD 流水线（SPEC §4.1）。返回退出码。

    start：跳过前 start 帧（仅解码不产出），记录使用绝对帧号（GT 关联不受影响）。
    on_frame：可选回调 ``(index, frame_bgr, record, pick_rect) -> bool``；返回 False
    请求提前结束（保留已处理 records/metrics，返回 0）。
    """
    from spa2d import metrics as MET
    from spa2d import records as R
    from spa2d import timing as T

    from mgmd import align as A, diff as D

    tracker = clf = lad = None
    if "tf" in modules:
        from mgmd.track import SortTracker
        tracker = SortTracker(cfg.tf)
    if "lac" in modules:
        from mgmd.classify import LacClassifier
        clf = LacClassifier(cfg.lac)
    if "lad" in modules:
        from mgmd.detect import LadDetector
        lad = LadDetector(cfg.lad)

    source = _open_source(cfg)
    n_total = len(source)
    start = max(0, min(start, n_total - 1))
    n_frames = min(limit, n_total - start) if limit else (n_total - start)
    downscale = cfg.runtime.downscale

    timer = T.Timer(backend=None, warmup=WARMUP_WINDOWS)
    records: list = []
    prev_state = "ABSENT"
    n_fallback = 0
    gap = cfg.mfe.frame_gap
    or_mode = cfg.mfe.or_aggregation
    t_run0 = time.perf_counter()

    def _notify(rec, frame, rect) -> None:
        """调用可视化回调；False → _VizStop，异常 → _VizError。"""
        if on_frame is None:
            return
        try:
            cont = on_frame(rec.window_index, frame, rec, rect)
        except Exception as exc:  # noqa: BLE001 - 统一映射为 E_RUNTIME
            raise _VizError(exc) from exc
        if cont is False:
            raise _VizStop(rec.window_index)

    def _absent(idx: int, frame=None) -> None:
        nonlocal prev_state
        rec, prev_state = _frame_record(idx, None, prev_state, downscale, 0.0, 0.0, 0.0)
        records.append(rec)
        timer.add("t_pre", 0.0)
        timer.add("t_search", 0.0)
        timer.add("t_state", 0.0)
        _notify(rec, frame, None)

    def _process(buf: dict[int, tuple], t: int) -> None:
        """处理帧 t：buf 需含 t（以及 t−k；OR 模式还需 t+k）。"""
        nonlocal prev_state, n_fallback
        _i, frame, gray_cur, _dec = buf[t]
        t0 = time.perf_counter()
        res_m = A.align(buf[t - gap][2], gray_cur, cfg.mfe)
        pairs = [(res_m.compensated, res_m.mask)]
        if res_m.fallback:
            n_fallback += 1
        if or_mode:
            res_p = A.align(buf[t + gap][2], gray_cur, cfg.mfe)   # Î_{t+k}→t（D12）
            pairs.append((res_p.compensated, res_p.mask))
            if res_p.fallback:
                n_fallback += 1
        cands = D.candidates_or(frame, pairs, cfg.mfe)
        t_pre = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        pool: list[_Pick] = []
        if tracker is not None:
            hist_H = _history_alignment_homography(buf, t, gap, gray_cur, cfg, res_m)
            tracks = tracker.update([c.rect for c in cands], t, align_H=hist_H,
                                    scores=[c.score for c in cands])
            # 选择键：运动得分（跨帧可比），并列取小面积（SPEC §5.6）
            pool = [_Pick(tr.rect, key=float(tr.score), velocity=tr.velocity, tracked=True)
                    for tr in tracks if tr.confirmed]
        else:
            pool = [_Pick(c.rect, key=c.score) for c in cands]

        if clf is not None and pool:
            # prob_thresh=0 为 MVDT 域失效直通档（SPEC D15）：此时保持上游排序键
            # （LAC 概率域外无判别力），仅执行概率计算与阈值过滤
            rank_by_prob = cfg.lac.prob_thresh > 0
            passed = []
            for p in pool:
                prob = clf.mav_prob(_crop(frame, p.rect))
                if prob >= cfg.lac.prob_thresh:
                    passed.append(_Pick(p.rect, key=(prob if rank_by_prob else p.key),
                                        velocity=p.velocity, tracked=p.tracked))
            pool = passed

        if lad is not None and pool:
            refined = []
            for p in pool:
                cx = p.rect[0] + p.rect[2] / 2.0
                cy = p.rect[1] + p.rect[3] / 2.0
                rb = lad.refine(frame, (cx, cy), downscale=1.0)
                if rb is not None and rb.conf >= cfg.lad.conf_thresh:
                    refined.append(_Pick(rb.rect, key=rb.conf, velocity=p.velocity,
                                         tracked=p.tracked))
            pool = refined

        pick = _select(pool)
        t_search = (time.perf_counter() - t1) * 1000.0
        t2 = time.perf_counter()
        rec, prev_state = _frame_record(t, pick, prev_state, downscale,
                                        t_pre, t_search, 0.0, pool_size=len(pool))
        records.append(rec)
        t_state = (time.perf_counter() - t2) * 1000.0
        timer.add("t_pre", t_pre)
        timer.add("t_search", t_search)
        timer.add("t_state", t_state)
        _notify(rec, frame, pick.rect if pick is not None else None)

    lookahead = gap if or_mode else 0
    limit_records = min(n_frames, n_total - start)
    reads_needed = start + min(n_frames + lookahead, n_total - start)

    def _read_next(it) -> tuple[int, tuple] | None:
        td0 = time.perf_counter()
        try:
            idx, frame = next(it)
        except StopIteration:
            return None
        dec_ms = (time.perf_counter() - td0) * 1000.0
        return idx, (idx, frame, D.preprocess_gray(frame, cfg.mfe), dec_ms)

    aborted = False
    abort_frame: int | None = None
    viz_error: Exception | None = None
    caught: list = []
    it = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            buf: dict[int, tuple] = {}
            it = iter(source)
            n_read = 0
            end_t = start + limit_records              # 记录窗口 [start, end_t)
            # 1) 预读至 start+gap+lookahead（start 之前的帧仅解码丢弃）
            while n_read < min(start + gap + lookahead, reads_needed):
                got = _read_next(it)
                if got is None:
                    break
                idx, entry = got
                n_read += 1
                if idx >= start:
                    buf[idx] = entry
            # 2) 窗口前 k 帧无完整差分对 → ABSENT（SPEC §5.8/D11/D12）
            for t in range(start, start + min(gap, limit_records)):
                entry = buf.get(t)
                dec = entry[3] if entry is not None else 0.0
                timer.times["t_decode"].append(dec)
                _absent(t, entry[1] if entry is not None else None)
            # 3) 主循环：读至 reads_needed；帧 t 在 t+lookahead 到达后处理
            while n_read < reads_needed:
                got = _read_next(it)
                if got is None:
                    break
                idx, entry = got
                n_read += 1
                if idx < start:
                    continue
                buf[idx] = entry
                t = idx - lookahead
                if not (max(gap, start) <= t < end_t):
                    continue
                if t - gap not in buf:
                    continue
                dec = buf.get(t, (None, None, None, 0.0))[3]
                timer.times["t_decode"].append(dec)
                _process(buf, t)
                # 修剪：只保留 [t−k, t+lookahead] 窗口
                keep_from = max(start, t - gap)
                for old in [k0 for k0 in buf if k0 < keep_from]:
                    del buf[old]
            # 4) 尾部：源耗尽或 OR 模式无未来帧 → ABSENT 补齐
            while len(records) < limit_records:
                timer.times["t_decode"].append(0.0)
                _absent(start + len(records), None)

    except _VizStop as exc:
        aborted = True
        abort_frame = exc.frame
    except _VizError as exc:
        aborted = True
        viz_error = exc.original
    finally:
        closer = getattr(it, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
        try:
            source.close()
        except Exception:  # noqa: BLE001
            pass
    n_warn = sum(1 for w in caught if issubclass(w.category, UserWarning))

    out_dir.mkdir(parents=True, exist_ok=True)
    R.write_jsonl(records, out_dir / "windows.jsonl")

    warmup_eff = min(WARMUP_WINDOWS, max(0, len(records) - 1))
    timing_summary = timer.summary(warmup_eff)
    fps = timer.fps(1, warmup_eff)
    timing_payload = {
        "layers": {k: v.to_dict() for k, v in timing_summary.items()},
        "fps": fps,
        "jit_compile_ms": 0.0,
        "warmup_windows": warmup_eff,
        "n_windows": len(records),
    }
    (out_dir / "timing.json").write_text(
        json.dumps(timing_payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8")

    # 半径匹配指标（FR-10：bbox 中心 → MVDT 点 GT；复用 spa2d.metrics 零改动）
    try:
        gt = R.read_gt_csv(gt_path)
    except (ValueError, OSError) as exc:
        raise MgmdError("E_DATA_MISSING",
                        f"GT 文件不可读或列不匹配: {gt_path}: {exc}",
                        "先运行 scripts/make_derived_gt.py 生成 data_derived/gt/*.csv",
                        EXIT_DATA) from exc
    metrics = MET.evaluate_windows(records, gt, radii=cfg.metrics.radii_px,
                                   assoc_mode=cfg.metrics.association,
                                   config_name=cfg.config_name,
                                   primary_radius=cfg.metrics.primary_radius_px)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8")

    snapshot = {}
    if cfg.source_path and Path(cfg.source_path).is_file():
        import yaml
        snapshot = yaml.safe_load(Path(cfg.source_path).read_text(encoding="utf-8"))
    meta = {
        "config": {"path": cfg.source_path, "name": cfg.config_name, "snapshot": snapshot},
        "mgmd": {
            "modules": sorted(modules),
            "downscale": downscale,
            "align_fallback_frames": n_fallback,
            "user_warnings": n_warn,
            "weights": {
                "lac": {"path": cfg.lac.weights,
                        "sha256": _sha256(cfg.lac.weights) if "lac" in modules else None},
                "lad": {"path": cfg.lad.weights,
                        "sha256": _sha256(cfg.lad.weights) if "lad" in modules else None},
            },
            "official_repo": "WindyLab/Global-Local-MAV-Detection@main",
            "spec": "tasks/mgmd_mvdt_reproduction_spec.md",
        },
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "seed": cfg.seed,
        "argv": sys.argv,
        "backend": backend,
        "gt": str(gt_path),
        "fps": fps,
        "n_windows": len(records),
        "n_accepted": sum(1 for r in records if r.accepted),
        "wall_time_s": time.perf_counter() - t_run0,
    }
    if on_frame is not None:
        viz_meta: dict = {}
        owner = getattr(on_frame, "__self__", None)
        summary = getattr(owner, "summary", None) or getattr(on_frame, "summary", None)
        if callable(summary):
            try:
                viz_meta = dict(summary())
            except Exception:  # noqa: BLE001 - summary 不阻塞主流程
                viz_meta = {}
        viz_meta.setdefault("enabled", True)
        viz_meta.setdefault("viewed", len(records))
        viz_meta.setdefault("jumps", 0)
        viz_meta.setdefault("fp", 0)
        viz_meta.setdefault("miss", 0)
        viz_meta["aborted"] = bool(aborted)
        viz_meta["abort_frame"] = abort_frame
        if viz_error is not None:
            viz_meta["error"] = f"{type(viz_error).__name__}: {viz_error}"
        meta["viz"] = viz_meta
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")

    if viz_error is not None:
        print(f"[mgmd] warning: viz callback error at frame {len(records) - 1}: "
              f"{type(viz_error).__name__}: {viz_error}；部分指标已写出",
              file=sys.stderr)
        raise MgmdError("E_RUNTIME",
                        f"可视化回调异常: {type(viz_error).__name__}: {viz_error}",
                        "检查 --viz 绘制/窗口环境；可用不带 --viz 重跑",
                        EXIT_RUNTIME)
    if aborted:
        print(f"[mgmd] warning: visualization aborted at frame {abort_frame}; "
              f"metrics computed over {len(records)} frames", file=sys.stderr)

    print(f"{cfg.config_name}: frames={len(records)} accepted={meta['n_accepted']} "
          f"recall@100={metrics.recall.get(100, float('nan')):.4f} "
          f"window_fps={fps['window_fps']:.3f} modules={','.join(sorted(modules))} "
          f"-> {out_dir}")
    return EXIT_OK
