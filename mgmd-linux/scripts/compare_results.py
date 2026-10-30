#!/usr/bin/env python3
"""对比两个结果目录的 P/R/F1/AP（SPEC §10.13 影响评估）。

用法：
    python3 scripts/compare_results.py --base final_d002 --new with_stepH \
        --videos phantom05 phantom08 phantom30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs" / "testset"


def load(run: str, video: str):
    p = LOGS / run / f"{video}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    a = ap.parse_args()

    keys = [("precision", "P"), ("recall", "R"), ("f1", "F1"),
            ("ap_11point", "AP")]
    rows = []
    print(f"{'视频':<12}" + "".join(f"{n:>9}{n+'(新)':>9}{'Δ':>8}"
                                    for _, n in keys))
    print("-" * 84)

    sum_b = {k: 0.0 for k, _ in keys}
    sum_n = {k: 0.0 for k, _ in keys}
    cnt = 0
    for v in a.videos:
        b, n = load(a.base, v), load(a.new, v)
        if b is None or n is None:
            miss = a.base if b is None else a.new
            print(f"{v:<12} 缺失: {miss}")
            continue
        mb, mn = b["metrics"], n["metrics"]
        line = f"{v:<12}"
        for k, _name in keys:
            vb, vn = mb[k], mn[k]
            sum_b[k] += vb
            sum_n[k] += vn
            line += f"{vb:>9.4f}{vn:>9.4f}{vn-vb:>+8.4f}"
        print(line)
        rows.append((v, mb, mn))
        cnt += 1

    if not cnt:
        raise SystemExit("没有可对比的数据")

    print("-" * 84)
    line = f"{'宏平均':<12}"
    for k, _name in keys:
        vb, vn = sum_b[k] / cnt, sum_n[k] / cnt
        line += f"{vb:>9.4f}{vn:>9.4f}{vn-vb:>+8.4f}"
    print(line)
    print("=" * 84)
    print(f"口径: {a.base} (基准) vs {a.new} (带 step_H)   n={cnt} 视频")


if __name__ == "__main__":
    main()
