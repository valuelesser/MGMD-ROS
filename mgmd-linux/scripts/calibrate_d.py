"""scripts/calibrate_d.py — D 阈值标定（只用训练集，SPEC §5 / D-02）。

原则（SPEC D-11）：
    1. 阈值只能从 45 个训练视频的 D 分布标定
    2. 测试集完全不参与
    3. 只产出一个全局阈值，不是每序列一个

用法：
    python3 scripts/calibrate_d.py                      # 默认 P75
    python3 scripts/calibrate_d.py --percentiles 50 75 90 95
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mgmd import config as C
from mgmd.annotations import load_video_gt
from mgmd.metrics import box_iou
from mgmd.tf import distance_metric


def collect_d_values(videos: list[int], span: int = 3,
                     d_mode: str = "enclosing", max_videos: int | None = None) -> dict:
    """从 GT 轨迹收集 D 值（替代真实检测轨迹，用于阈值先验）。

    这是**上和下界**的参考：GT 轨迹本身是真实运动目标，其 D 分布给出
    "真实目标应有的 D 量级"，比用含噪检测轨迹更干净。
    """
    vals = []
    per_video = {}
    use = videos if max_videos is None else videos[:max_videos]

    for vid in use:
        name = C.video_id_to_name(vid)
        try:
            gt = load_video_gt(name)
        except C.MgmdError:
            continue
        frames = sorted(gt)
        vs = []
        for i in range(len(frames) - span):
            fa, fb = frames[i], frames[i + span]
            ga, gb = gt[fa], gt[fb]
            if ga.n_obj != 1 or gb.n_obj != 1:
                continue
            vs.append(distance_metric(
                (ga.objects[0].x1, ga.objects[0].y1, ga.objects[0].x2, ga.objects[0].y2),
                (gb.objects[0].x1, gb.objects[0].y1, gb.objects[0].x2, gb.objects[0].y2),
                d_mode))
        if vs:
            per_video[name] = {"n": len(vs), "max": float(np.max(vs)),
                               "median": float(np.median(vs))}
            vals.extend(vs)

    return {"values": vals, "per_video": per_video, "n": len(vals)}


def calibrate(percentiles=(50, 75, 90, 95), span: int = 3, d_mode: str = "enclosing") -> dict:
    split = C.load_split()
    train_ids = split["train_ids"]

    print(f"训练视频 {len(train_ids)} 个（测试集 {len(split['test_ids'])} 个不参与）")
    col = collect_d_values(train_ids, span=span, d_mode=d_mode)
    vals = sorted(col["values"])
    n = len(vals)
    if n == 0:
        raise C.MgmdError("E_DATA_MISSING", "训练集未收集到 D 值",
                          "检查帧号与 GT 解析（SPEC T02）", C.EXIT_DATA)

    print(f"训练集 D 样本数: {n}（d_mode={d_mode}, span={span}）")
    print(f"  min={vals[0]:.4f}  max={vals[-1]:.4f}  median={vals[n//2]:.4f}")

    table = {}
    for p in percentiles:
        idx = int(round(p / 100.0 * (n - 1)))
        thr = vals[idx]
        keep = sum(1 for v in vals if v >= thr)
        table[f"P{p}"] = {"d_min": round(thr, 6), "keep_ratio": round(keep / n, 4)}
        print(f"  P{p:<3} = {thr:.4f}  (训练集保留 {100*keep/n:.2f}%)")

    return {
        "spec_version": "0.1.0",
        "source": "train-only calibration (SPEC D-11)",
        "d_mode": d_mode,
        "span": span,
        "n_samples": n,
        "train_videos": len(train_ids),
        "test_videos_excluded": len(split["test_ids"]),
        "percentile_table": table,
        "distribution": {
            "min": round(vals[0], 6), "max": round(vals[-1], 6),
            "median": round(vals[n // 2], 6),
            "p25": round(vals[int(0.25 * (n - 1))], 6),
            "p75": round(vals[int(0.75 * (n - 1))], 6),
            "p90": round(vals[int(0.90 * (n - 1))], 6),
        },
        "per_video": col["per_video"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentiles", type=int, nargs="+", default=[50, 75, 90, 95])
    ap.add_argument("--span", type=int, default=3)
    ap.add_argument("--d-mode", default="enclosing", choices=["enclosing", "self"])
    ap.add_argument("--out", default=str(C.PROJECT_ROOT / "configs" / "d_calibration.json"))
    args = ap.parse_args(argv)

    try:
        res = calibrate(tuple(args.percentiles), args.span, args.d_mode)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\n已写入 {args.out}")
        recommended = res["percentile_table"].get("P75", {}).get("d_min")
        if recommended:
            print(f"推荐 d_min（P75）= {recommended}  ← 训练集标定，测试集未参与")
        return C.EXIT_OK
    except C.MgmdError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
