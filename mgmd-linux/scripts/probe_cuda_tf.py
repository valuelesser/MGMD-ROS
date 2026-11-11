"""Replay the same MFE candidates through ROS FastSortTracker and device TF."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
import torch
from scipy import __version__ as scipy_version
from scipy.optimize import linear_sum_assignment

from mgmd.mfe import MFEConfig, enhance
from mgmd_onnx.cuda_tf import CudaTf
from mgmd_onnx.tf_fast import FastSortTracker


def reference_tracker():
    return FastSortTracker(iou_thresh=.3, max_age=3, min_hits=1, span=3, d_mode="self",
                           d_min=.02, frame_delta_predict=True, association_radius_factor=2.0)


def reference_step(tracker, boxes, scores, frame):
    calls = []
    ids = list(tracker.states)
    next_id = tracker._next_id

    def record(cost):
        rows, cols = linear_sum_assignment(cost)
        calls.append((cost, rows, cols))
        return rows, cols

    with patch("scipy.optimize.linear_sum_assignment", side_effect=record):
        tracks = tracker.update(boxes, frame, scores=scores)
    assigned = {}
    if calls:
        cost, rows, cols = calls[0]
        used_rows = set()
        for r, c in zip(rows, cols):
            if cost[r, c] <= -.3:
                assigned[int(c)] = ids[r]
                used_rows.add(r)
        if len(calls) == 2:
            rest_rows = [r for r in range(len(ids)) if r not in used_rows]
            rest_cols = [c for c in range(len(boxes)) if c not in assigned]
            cost, rows, cols = calls[1]
            for r, c in zip(rows, cols):
                if cost[r, c] <= 1:
                    assigned[rest_cols[c]] = ids[rest_rows[r]]
    for c in range(len(boxes)):
        if c not in assigned:
            assigned[c] = next_id
            next_id += 1
    return tracks, [assigned[c] for c in range(len(boxes))]


def extract(video, limit):
    cap = cv2.VideoCapture(str(video))
    cfg = MFEConfig().apply_recommended()
    cfg.validate()
    window = []
    rows = []
    try:
        for seq in range(limit):
            ok, frame = cap.read()
            assert ok and frame.shape == (1080, 1920, 3), (video, seq)
            window.append(frame)
            if len(window) == 3:
                frame_id = seq - 1
                boxes = enhance(*window, cfg, frame_id).boxes
                rows.append((frame_id, [(float(x), float(y), float(x + w), float(y + h))
                                         for x, y, w, h in boxes],
                             [float(w * h) for _, _, w, h in boxes]))
                window.pop(0)
    finally:
        cap.release()
    return rows


def to_device(boxes, scores):
    return (torch.tensor(boxes, dtype=torch.float64, device="cuda").reshape(-1, 4),
            torch.tensor(scores, dtype=torch.float64, device="cuda"))


def compare(gpu, rows, log, name):
    tracker = reference_tracker()
    timing_tracker = reference_tracker()
    gpu.reset()
    differences = Counter()
    counts = Counter()
    peak = Counter()
    times = {"cpu": [], "cuda": []}
    for frame, boxes, scores in rows:
        before = perf_counter()
        timing_tracker.update(boxes, frame, scores=scores)
        times["cpu"].append((perf_counter() - before) * 1000)
        expected, assigned = reference_step(tracker, boxes, scores, frame)
        device_boxes, device_scores = to_device(boxes, scores)
        count = torch.full((1,), len(boxes), dtype=torch.int32, device="cuda")
        gpu.stream.synchronize()
        before = perf_counter()
        result = gpu.step(device_boxes, device_scores, frame, count)
        gpu.stream.synchronize()
        times["cuda"].append((perf_counter() - before) * 1000)
        assert result.status.item() == 0, (name, frame, result.status.item())
        count_out = result.count.item()
        ids = result.ids[:count_out].tolist()
        wanted = [t.track_id for t in expected]
        bad = []
        if result.assigned_ids.tolist() != assigned:
            bad.append("assignment")
        if ids != wanted:
            bad.append("output_ids")
        else:
            wanted_boxes = np.asarray([t.box for t in expected]).reshape(-1, 4)
            wanted_details = np.asarray([(t.score, t.d_value, *t.velocity) for t in expected]).reshape(-1, 4)
            wanted_life = np.asarray([(t.hits, t.miss_count, t.age) for t in expected]).reshape(-1, 3)
            for field, actual, ref in (
                    ("boxes", result.boxes[:count_out], wanted_boxes),
                    ("details", result.details[:count_out], wanted_details),
                    ("lifecycle", result.lifecycle[:count_out], wanted_life)):
                if not np.allclose(actual.cpu().numpy(), ref, atol=1e-8, rtol=1e-9):
                    bad.append(field)
        differences.update(bad)
        counts.update(frames=1, candidates=len(boxes), tf_output=count_out,
                      reference_output=len(expected), different_frames=bool(bad))
        peak["candidates"] = max(peak["candidates"], len(boxes))
        peak["active_tracks"] = max(peak["active_tracks"], len(tracker.states))
        log.write(json.dumps({"sequence": name, "frame": frame, "input": len(boxes),
                              "output": count_out, "difference": bad,
                              "expected_ids": wanted, "device_ids": ids}) + "\n")
    return {"sequence": name, "counts": dict(counts), "peak": dict(peak),
            "differences": dict(differences),
            "synchronized_wall_ms": {key: {"p50": float(np.percentile(samples, 50)),
                                           "p95": float(np.percentile(samples, 95))}
                                     for key, samples in times.items()}}


def stress_cases():
    # Identical costs, tall/wide matrices, misses, exact/large gaps, and track creation.
    box = (0., 0., 12., 12.)
    moved = (3., 2., 15., 14.)
    detections = [[box, box], [moved], [moved, moved, moved], [], [], [], [],
                  [box], [(0., 0., 0., 0.)], [moved]]
    return [(frame, boxes, [144.] * len(boxes))
            for frame, boxes in zip((0, 1, 4, 5, 6, 8, 10, 50, 51, 60), detections)]


def check_overflow(library):
    tracker = CudaTf(library, track_capacity=1, det_capacity=2)
    boxes, scores = to_device([(0, 0, 1, 1), (3, 3, 4, 4)], [1, 1])
    result = tracker.step(boxes, scores, 0)
    assert result.status.item() == 1 and result.count.item() == 0
    tracker.reset()
    result = tracker.step(boxes[:1], scores[:1], 0)
    assert result.status.item() == 0 and result.assigned_ids.item() == 1
    # Padded input supplies its valid prefix via a device scalar.
    count = torch.ones(1, dtype=torch.int32, device="cuda")
    result = tracker.step(boxes, scores, 1, count)
    assert result.status.item() == 0 and result.assigned_ids[0].item() == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    if args.frames < 6:
        parser.error("--frames must be at least 6")
    args.out.mkdir()
    gpu = CudaTf(args.library)
    results = []
    check_overflow(args.library)
    with (args.out / "frames.jsonl").open("w") as log:
        results.append(compare(gpu, stress_cases(), log, "synthetic"))
        for video in args.videos:
            rows = extract(video, args.frames)
            for name, selected in (("continuous", rows), ("gaps", rows[::3])):
                sequence = f"{video.stem}.{name}"
                results.append(compare(gpu, selected, log, sequence))
                print(sequence, results[-1]["counts"], flush=True)
    report = json.dumps({"opencv": cv2.__version__, "scipy": scipy_version,
                         "sequences": results, "overflow_and_reset": "pass"}, indent=2) + "\n"
    (args.out / "summary.json").write_text(report)
    return int(any(row["differences"] for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
