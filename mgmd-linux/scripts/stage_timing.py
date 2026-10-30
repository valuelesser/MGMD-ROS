#!/usr/bin/env python3
"""订阅 /mav/detections，统计各阶段耗时（SPEC-ROS §20）。

字段来源：DetectionArray.mfe_ms / tf_ms / lac_ms / lad_ms
（均由 detector_node 在各阶段前后 time.perf_counter() 实测）。

另算"单帧总耗时" wall_ms = 相邻两次发布的时间差 —— 这是流水线真实吞吐，
它 = MFE + TF + LAC + LAD + 未计入部分（crop、resize、消息构造、等待、调度）。

用法：
    python3 scripts/stage_timing.py [--duration 30] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time

import rospy

from mav_tracker_msgs.msg import DetectionArray


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class Collector:
    def __init__(self):
        self.rows = []
        self.t_prev = None

    def cb(self, msg):
        now = time.perf_counter()
        wall = (now - self.t_prev) * 1000.0 if self.t_prev else float("nan")
        self.t_prev = now
        mfe, tf = msg.mfe_ms, msg.tf_ms
        lac, lad = msg.lac_ms, msg.lad_ms
        self.rows.append(dict(
            frame=int(msg.header.seq),
            n_cand=int(msg.n_candidates),
            n_det=len(msg.detections),
            mfe=mfe, tf=tf, lac=lac, lad=lad,
            sum4=mfe + tf + lac + lad,
            wall=wall,
            unc=(wall - (mfe + tf + lac + lad)) if wall == wall else float("nan"),
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    rospy.init_node("stage_timing", anonymous=True)
    c = Collector()
    rospy.Subscriber("/mav/detections", DetectionArray, c.cb, queue_size=50)
    print(f"采样 {a.duration:.0f}s ...", flush=True)
    time.sleep(a.duration)
    rows = c.rows
    if not rows:
        print("❌ 没收到任何 /mav/detections 消息")
        return 1

    # 丢弃前 3 帧（MFE 预热 + 首次发布毛刺）
    rows = rows[3:] if len(rows) > 6 else rows
    n = len(rows)

    def col(k):
        return [r[k] for r in rows if r[k] == r[k]]

    print("\n" + "=" * 74)
    print(f"各阶段耗时  n={n} 帧")
    print("=" * 74)
    print(f"{'阶段':<22}{'p50':>9}{'p95':>9}{'mean':>9}{'max':>9}   {'占总耗时':>9}")
    print("-" * 74)
    total = st.mean(col("sum4")) if col("sum4") else 0.0
    wall_mean = st.mean(col("wall")) if col("wall") else 0.0
    for key, name in [("mfe", "MFE  帧差对齐+OR"),
                      ("tf", "TF   SORT"),
                      ("lac", "LAC  MAV 分类"),
                      ("lad", "LAD  YOLO 精修")]:
        v = col(key)
        if not v:
            continue
        share = st.mean(v) / wall_mean * 100 if wall_mean else 0
        print(f"{name:<22}{pct(v,.5):>8.2f} {pct(v,.95):>8.2f} "
              f"{st.mean(v):>8.2f} {max(v):>8.2f}   {share:>8.1f}%")
    print("-" * 74)
    s4 = col("sum4")
    print(f"{'四阶段合计':<22}{pct(s4,.5):>8.2f} {pct(s4,.95):>8.2f} "
          f"{st.mean(s4):>8.2f} {max(s4):>8.2f}   "
          f"{st.mean(s4)/wall_mean*100 if wall_mean else 0:>8.1f}%")
    w = col("wall")
    print(f"{'实测单帧总耗时':<20}{pct(w,.5):>8.2f} {pct(w,.95):>8.2f} "
          f"{st.mean(w):>8.2f} {max(w):>8.2f}   100.0%")
    u = col("unc")
    print(f"{'未计入(裁剪/等待)':<19}{pct(u,.5):>8.2f} {pct(u,.95):>8.2f} "
          f"{st.mean(u):>8.2f} {max(u):>8.2f}   "
          f"{st.mean(u)/wall_mean*100 if wall_mean else 0:>8.1f}%")
    print("-" * 74)
    print(f"吞吐 ≈ {1000.0/wall_mean:.2f} Hz" if wall_mean else "")
    print(f"候选 p50={pct(col('n_cand'),.5):.0f}  "
          f"输出框 p50={pct(col('n_det'),.5):.0f}  "
          f"有框帧占比 {100.0*sum(1 for r in rows if r['n_det']>0)/n:.0f}%")
    print("=" * 74)

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"明细已写 {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
