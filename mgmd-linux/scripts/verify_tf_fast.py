"""scripts/verify_tf_fast.py — S3-c 验收：FastSortTracker 与 SortTracker 逐帧一致。

    source scripts/env_setup_ros38.sh
    $PY38 scripts/verify_tf_fast.py --video phantom30

判据：所有帧的 track_id / box / score 完全一致（容差 0）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mgmd import config as C
from mgmd.io_video import VideoReader
from mgmd.mfe import MFEConfig, enhance
from mgmd.tf import SortTracker
from mgmd_onnx.tf_fast import FastSortTracker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom30")
    ap.add_argument("--frames", type=int, default=1500)
    args = ap.parse_args()

    cfg = MFEConfig()
    cfg.apply_recommended()
    cfg.validate()
    k = cfg.k

    kw = dict(iou_thresh=0.3, max_age=3, min_hits=1, span=3,
              d_mode="self", d_min=0.02)
    ref = SortTracker(**kw)
    fast = FastSortTracker(**kw)

    buf = {}
    n = 0
    bad = []
    max_box_diff = 0.0
    with VideoReader(args.video) as vr:
        for idx, fr in vr.iter_frames(start=1, limit=args.frames + 2 * k + 2):
            buf[idx] = fr
            t = idx - k
            if t < 1 or (t - k) not in buf or (t + k) not in buf:
                continue
            res = enhance(buf[t - k], buf[t], buf[t + k], cfg, t)
            boxes = [(float(x), float(y), float(x + w), float(y + h))
                     for (x, y, w, h) in res.boxes]
            scores = [float(w * h) for (_x, _y, w, h) in res.boxes]

            a = ref.update(list(boxes), t, scores=list(scores))
            b = fast.update(list(boxes), t, scores=list(scores))
            n += 1

            ka = [(tr.track_id, tr.box) for tr in a]
            kb = [(tr.track_id, tr.box) for tr in b]
            if len(ka) != len(kb):
                bad.append((t, "count", len(ka), len(kb)))
                continue
            for (ia, ba), (ib, bb) in zip(ka, kb):
                if ia != ib:
                    bad.append((t, "id", ia, ib))
                    break
                d = float(np.abs(np.array(ba) - np.array(bb)).max())
                max_box_diff = max(max_box_diff, d)

    print(f"视频 {args.video}，对比 {n} 帧")
    print(f"不一致帧: {len(bad)}")
    if bad:
        for x in bad[:10]:
            print("  ", x)
    print(f"框坐标最大差: {max_box_diff:.6e}")
    ok = not bad and max_box_diff == 0.0
    print("\nS3-c 判定:", "✅ 通过（逐帧完全一致）" if ok else "❌ 未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
