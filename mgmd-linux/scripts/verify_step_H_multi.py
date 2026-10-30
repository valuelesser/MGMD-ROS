#!/usr/bin/env python3
"""多视频 × 双口径 step_H 验证（SPEC §10.13）。

对每个视频，并行跑「传 step_H」与「不传 step_H」两份 SORT 输出对比，
统计 compose_H 成功率、d 值变化、判定翻转率。

用法：
    python3 scripts/verify_step_H_multi.py --videos phantom30 phantom05 ... \
        --frames 300 --w 960 --h 720 --jobs 5

**注意**：`reason` 字段被 classify 末尾覆盖，不可用来自查对齐状态
（见 tests/test_tf.py::test_align_missing_reason_is_overwritten）。
本脚本一律用 compose_H 的返回值判断。
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _resolve(name: str) -> str:
    p = Path(name)
    if p.is_file():
        return str(p)
    try:
        from mgmd import config as C
        cand = Path(C.ARD_VIDEOS) / f"{name}.mp4"
        if cand.is_file():
            return str(cand)
    except Exception:                                   # noqa: BLE001
        pass
    raise SystemExit(f"找不到视频: {name}")


def run_one(args):
    """单个视频：返回统计字典。在子进程里跑，故自包含导入。"""
    name, video, n_frames, w, h, d_min, span, d_mode, k = args
    try:
        from mgmd.mfe import MFEConfig, enhance
        from mgmd.tf import SortTracker

        cfg = MFEConfig()
        cfg.apply_recommended()
        cfg.validate()
        cfg.k = k

        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        while len(frames) < n_frames + 2 * k + 1:
            ok, f = cap.read()
            if not ok:
                break
            if w and h:
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(f)
        cap.release()
        if len(frames) < 3:
            return dict(video=name, error="帧数不足")

        mk = lambda: SortTracker(span=span, d_mode=d_mode, d_min=d_min)
        tr_a, tr_p = mk(), mk()

        d_a, d_p, flips = [], [], 0
        compose_ok = compose_tot = h_ok = h_tot = n_cmp = 0
        conf_a = conf_p = 0

        for t in range(k, len(frames) - k):
            res = enhance(frames[t - k], frames[t], frames[t + k], cfg, t)
            if not res.boxes:
                continue
            dets = [(float(x), float(y), float(x + ww), float(y + hh))
                    for (x, y, ww, hh) in res.boxes]
            scores = [float(ww * hh) for (_x, _y, ww, hh) in res.boxes]

            step = res.info.get("prev_H") if k == 1 else None
            h_tot += 1
            if step is not None:
                h_ok += 1

            tr_a.update(dets, t, scores=scores, step_H=step)
            tr_p.update(dets, t, scores=scores, step_H=None)

            for tid, s in tr_a.states.items():
                if s.reason == "lifetime_lt_3":
                    continue
                pair = s.window(span)
                if pair is None:
                    continue
                p = tr_p.states.get(tid)
                if p is None:
                    continue
                (f0, _), (f1, _) = pair
                compose_tot += 1
                H = s.compose_H(f0, f1)
                if H is not None:
                    compose_ok += 1
                n_cmp += 1
                d_a.append(s.d_value)
                d_p.append(p.d_value)
                if (s.d_value >= d_min) != (p.d_value >= d_min):
                    flips += 1
                conf_a += s.d_value >= d_min
                conf_p += p.d_value >= d_min

        n = max(1, n_cmp)
        return dict(
            video=name, frames_read=len(frames), src_frames=total,
            n_cmp=n_cmp,
            h_ok=h_ok, h_tot=h_tot,
            compose_ok=compose_ok, compose_tot=compose_tot,
            d_a_p50=st.median(d_a) if d_a else 0.0,
            d_p_p50=st.median(d_p) if d_p else 0.0,
            d_a_mean=st.mean(d_a) if d_a else 0.0,
            d_p_mean=st.mean(d_p) if d_p else 0.0,
            conf_a=conf_a, conf_p=conf_p, flips=flips,
            flip_pct=100.0 * flips / n,
            conf_a_pct=100.0 * conf_a / n,
            conf_p_pct=100.0 * conf_p / n,
        )
    except Exception as exc:                            # noqa: BLE001
        return dict(video=name, error=f"{exc}\n{traceback.format_exc()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--d-min", type=float, default=0.02)
    ap.add_argument("--span", type=int, default=3)
    ap.add_argument("--d-mode", default="self")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    jobs = []
    for v in a.videos:
        try:
            p = _resolve(v)
        except SystemExit as e:
            print(f"[跳过] {e}")
            continue
        jobs.append((v, p, a.frames, a.w, a.h, a.d_min, a.span,
                     a.d_mode, a.k))
    if not jobs:
        raise SystemExit("没有可跑的视频")

    print(f"并行 {min(a.jobs, len(jobs))} 进程，共 {len(jobs)} 个视频，"
          f"每视频 {a.frames} 帧 @ {a.w}x{a.h}\n")
    results = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, j): j[0] for j in jobs}
        for fu in as_completed(futs):
            r = fu.result()
            results.append(r)
            if "error" in r:
                print(f"❌ {r['video']}: {r['error'].splitlines()[0]}")
            else:
                print(f"✅ {r['video']}: compose {r['compose_ok']}/{r['compose_tot']} "
                      f"翻转 {r['flips']}({r['flip_pct']:.1f}%)")

    ok = [r for r in results if "error" not in r]
    if not ok:
        raise SystemExit("全部失败")

    print("\n" + "=" * 96)
    print(f"{'视频':<14}{'可比较':>8}{'compose_H':>12}{'stepH':>8}"
          f"{'d均值(对齐)':>12}{'d均值(原)':>11}{'翻转':>8}{'通过率(对齐)':>13}{'通过率(原)':>11}")
    print("-" * 96)
    for r in sorted(ok, key=lambda x: x["video"]):
        print(f"{r['video']:<14}{r['n_cmp']:>8}"
              f"{r['compose_ok']:>7}/{r['compose_tot']:<5}"
              f"{100*r['h_ok']/max(1,r['h_tot']):>7.0f}%"
              f"{r['d_a_mean']:>12.4f}{r['d_p_mean']:>11.4f}"
              f"{r['flips']:>5}({r['flip_pct']:.1f}%)"
              f"{r['conf_a_pct']:>12.1f}%{r['conf_p_pct']:>10.1f}%")
    print("-" * 96)

    tot_cmp = sum(r["n_cmp"] for r in ok)
    tot_flip = sum(r["flips"] for r in ok)
    tot_co = sum(r["compose_ok"] for r in ok)
    tot_ct = sum(r["compose_tot"] for r in ok)
    print(f"{'合计':<14}{tot_cmp:>8}{tot_co:>7}/{tot_ct:<5}"
          f"{'':>8}{'':>12}{'':>11}{tot_flip:>5}({100*tot_flip/max(1,tot_cmp):.1f}%)")
    print("=" * 96)

    if a.out:
        Path(a.out).write_text(json.dumps(ok, indent=2, ensure_ascii=False))
        print(f"明细已写 {a.out}")


if __name__ == "__main__":
    main()
