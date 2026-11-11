"""Compare the CUDA assignment library on the matrices produced by ROS TF."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch_linear_assignment.assignment import backend, batch_linear_assignment

from probe_cuda_tf import extract, reference_tracker, stress_cases


def solve_double(cost):
    if cost.shape[1] > cost.shape[2]:
        return backend.batch_linear_assignment(cost.transpose(1, 2).contiguous())[1]
    return backend.batch_linear_assignment(cost)[0]


def evaluate(rows):
    tracker = reference_tracker()
    matrices = []
    frame_number = 0

    def capture(cost):
        start = perf_counter()
        row, col = linear_sum_assignment(cost)
        elapsed = (perf_counter() - start) * 1000
        matrices.append((frame_number, cost, row, col, elapsed))
        return row, col

    with patch("scipy.optimize.linear_sum_assignment", side_effect=capture):
        for frame_number, boxes, scores in rows:
            tracker.update(boxes, frame_number, scores=scores)

    counts = Counter()
    times = {"scipy": [], "public_cuda": [], "native_float64_cuda": []}
    differing_frames = {name: set() for name in times if name != "scipy"}
    previous_frame = None
    for frame, cost, row, col, elapsed in matrices:
        second_round = frame == previous_frame
        previous_frame = frame
        threshold = 1.0 if second_round else -.3
        expected = {(int(r), int(c)) for r, c in zip(row, col) if cost[r, c] <= threshold}
        times["scipy"].append(elapsed)
        counts["distance_matrices" if second_round else "iou_matrices"] += 1
        counts["max_rows"] = max(counts["max_rows"], cost.shape[0])
        counts["max_cols"] = max(counts["max_cols"], cost.shape[1])
        device = torch.as_tensor(cost, device="cuda").unsqueeze(0)
        torch.cuda.synchronize()
        for name, solve in (("public_cuda", batch_linear_assignment),
                            ("native_float64_cuda", solve_double)):
            start = perf_counter()
            output = solve(device)
            torch.cuda.synchronize()
            times[name].append((perf_counter() - start) * 1000)
            assert output.is_cuda
            assignment = output[0].cpu().numpy()
            assigned_rows = np.flatnonzero(assignment >= 0)
            assigned_cols = assignment[assigned_rows]
            if (len(assigned_rows) != len(row) or len(set(assigned_cols)) != len(row)
                    or not np.isclose(cost[assigned_rows, assigned_cols].sum(),
                                      cost[row, col].sum(), atol=1e-9, rtol=1e-12)):
                counts[name + "_objective_difference"] += 1
            accepted = {(int(r), int(c)) for r, c in zip(assigned_rows, assigned_cols)
                        if cost[r, c] <= threshold}
            if accepted != expected:
                counts[name + "_gated_difference"] += 1
                differing_frames[name].add(frame)
    return {"frames": len(rows), "counts": dict(counts),
            "gated_difference_frames": {k: sorted(v) for k, v in differing_frames.items()},
            "synchronized_solver_ms": {
                name: dict(zip(("p50", "p95"), np.percentile(samples, (50, 95)).tolist()))
                for name, samples in times.items() if samples}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    if not backend.has_cuda():
        raise RuntimeError("CUDA assignment backend is unavailable")
    cost = torch.zeros((1, 2, 3), dtype=torch.float64, device="cuda")
    batch_linear_assignment(cost)
    solve_double(cost)
    torch.cuda.synchronize()
    results = {"synthetic": evaluate(stress_cases())}
    for video in args.videos:
        rows = extract(video, args.frames)
        results[video.stem + ".continuous"] = evaluate(rows)
        results[video.stem + ".gaps"] = evaluate(rows[::3])
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
