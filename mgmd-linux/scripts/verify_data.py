"""scripts/verify_data.py — 数据契约校验 + 划分固化（SPEC T02 / G0）。

用法：
    python3 scripts/verify_data.py            # 校验并写划分
    python3 scripts/verify_data.py --force    # 覆盖已有划分
    python3 scripts/verify_data.py --stats    # 只打印统计，不写文件
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mgmd import config as C
from mgmd.annotations import video_stats
from mgmd.split import build_split, verify_split, write_split


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ARD-MAV 数据校验与划分固化")
    ap.add_argument("--force", action="store_true", help="覆盖已有 split 文件")
    ap.add_argument("--stats", action="store_true", help="只统计，不写文件")
    ap.add_argument("--out", default=str(C.LOGS / "data_stats.json"))
    args = ap.parse_args(argv)

    try:
        C.ensure_dirs()
        videos = C.list_annotated_videos()
        print(f"标注视频数: {len(videos)}")

        rows = []
        for v in videos:
            rows.append(video_stats(v))

        tot_frames = sum(r["frames_annotated"] for r in rows)
        tot_obj = sum(r["objects"] for r in rows)
        gaps = [r for r in rows if r["frame_gaps"] > 0]
        multi = sum(r["multi_obj_frames"] for r in rows)
        empty = sum(r["empty_frames"] for r in rows)

        med_areas = sorted(r["median_area"] for r in rows)
        med_w = sorted(r["median_w"] for r in rows)
        med_h = sorted(r["median_h"] for r in rows)

        print(f"总帧数(有标注): {tot_frames}")
        print(f"总目标数:       {tot_obj}")
        print(f"空标注帧:       {empty}")
        print(f"多目标帧:       {multi}")
        print(f"帧号有缺口的视频: {len(gaps)}")
        for r in gaps:
            print(f"  {r['video']}: {r['frames_annotated']} XML, span {r['frame_span']}, 缺 {r['frame_gaps']}")
        print(f"中位目标尺寸(各视频中位数再取中位): {med_w[len(med_w)//2]:.0f} x {med_h[len(med_h)//2]:.0f} px")
        print(f"中位目标面积: {med_areas[len(med_areas)//2]:.0f} px²")

        report = {
            "videos": len(videos),
            "frames_annotated": tot_frames,
            "objects": tot_obj,
            "empty_frames": empty,
            "multi_obj_frames": multi,
            "videos_with_frame_gaps": [r["video"] for r in gaps],
            "per_video": rows,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"统计已写入 {args.out}")

        if args.stats:
            return C.EXIT_OK

        split = build_split(seed=0)
        problems = verify_split(split)
        if problems:
            print("划分校验失败:")
            for p in problems:
                print("  -", p)
            return C.EXIT_CHECK

        print(f"\n划分: train={split['counts']['train']} "
              f"val={split['counts']['val']} test={split['counts']['test']} "
              f"total={split['counts']['total']}")
        print(f"测试集: {split['test_ids']}")
        write_split(split, force=args.force)
        print(f"已写入 {C.SPLIT_FILE}")
        return C.EXIT_OK

    except C.MgmdError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
