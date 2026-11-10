"""Time the CUDA model chain using recorded candidate metadata."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from mgmd_onnx import LacOnnx, LadOnnx
from mgmd_onnx.cuda_frames import CudaDecoder
from mgmd_onnx.cuda_models import CudaLac, CudaLad


def measure(library, video, rows, lac, lad):
    stages = defaultdict(list)
    metadata_bytes = detections = 0

    def timed(stage, function, *args):
        start = perf_counter()
        result = function(*args)
        torch.cuda.current_stream().synchronize()
        stages[stage].append((perf_counter() - start) * 1000)
        return result

    torch.cuda.reset_peak_memory_stats()
    with CudaDecoder(library, video) as decoder, torch.inference_mode():
        for row in rows:
            start = perf_counter()
            frame = timed("decode_and_bgr", decoder.read)
            while frame is not None and frame.seq < row["frame"]:
                frame = timed("decode_and_bgr", decoder.read)
            assert frame is not None and frame.seq == row["frame"] and frame.pts == row["pts"]
            boxes = row["candidates"]
            indices, inputs = timed("lac_preprocess", lac.prepare, frame.image, boxes)
            probs = timed("lac_inference_and_softmax", lac.probabilities, inputs)
            selected = (probs >= .5).nonzero().flatten()
            metadata_bytes += selected.numel() * selected.element_size()
            for index in selected.tolist():
                x1, y1, x2, y2 = boxes[indices[index]]
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                tensor, region = timed("lad_preprocess", lad.prepare, frame.image, center)
                raw = timed("lad_inference", lad.model.run, tensor)[0]
                result = timed("lad_nms_and_select", lad.select, raw, region)
                detections += result.shape[0]
            stages["per_frame"].append((perf_counter() - start) * 1000)
    return {
        "video": video.name, "frames": len(rows), "detections": detections,
        "explicit_frame_uploads": 0, "explicit_frame_downloads": 0,
        "dispatch_metadata_d2h_bytes": metadata_bytes,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "synchronized_wall_ms": {name: {
            "samples": len(values), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        } for name, values in stages.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("rows", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("videos", type=Path, nargs="+")
    args = parser.parse_args()
    args.out.mkdir()
    by_video = defaultdict(list)
    with args.rows.open() as rows:
        for line in rows:
            row = json.loads(line)
            by_video[row["video"]].append(row)
    lac = CudaLac(LacOnnx(device="cuda"))
    lad = CudaLad(LadOnnx(device="cuda"))
    results = []
    for video in args.videos:
        assert by_video[video.stem], video
        results.append(measure(args.library, video, by_video[video.stem], lac, lad))
    report = json.dumps({"videos": results}, indent=2, allow_nan=False) + "\n"
    (args.out / "timing.json").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
