"""Replay fixed NVDEC windows with and without two-stream scheduling."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
from pathlib import Path
from time import perf_counter

import torch

from mgmd_onnx.cuda_frames import CudaDecoder
from mgmd_onnx.cuda_pipeline import CudaPipeline


def mark():
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


@torch.inference_mode()
def extract(pipeline, stream, window):
    with torch.cuda.stream(stream):
        start = mark()
        boxes, scores = pipeline.mfe.candidates(*(frame.image for frame in window))
        return boxes, scores, start, mark()


def run(pipeline, executor, mfe_stream, post_stream, frames, overlap):
    windows = list(zip(frames, frames[1:], frames[2:]))
    torch.cuda.synchronize()
    origin = mark()
    mfe_stream.wait_event(origin)
    post_stream.wait_event(origin)
    started = perf_counter()
    results, intervals = [], []
    with torch.cuda.stream(post_stream):
        pending = executor.submit(extract, pipeline, mfe_stream, windows[0])
        for i, (_, center, _) in enumerate(windows):
            boxes, scores, mfe_start, ready = pending.result()
            post_stream.wait_event(ready)
            boxes.record_stream(post_stream)
            scores.record_stream(post_stream)
            if overlap and i + 1 < len(windows):
                pending = executor.submit(extract, pipeline, mfe_stream, windows[i + 1])
            post_start = mark()
            output, status, count, _ = pipeline.finish(center.image, boxes, scores, center.seq)
            intervals.append((mfe_start, ready, post_start, mark()))
            results.append((output, status, count))
            if not overlap and i + 1 < len(windows):
                intervals[-1][-1].synchronize()
                pending = executor.submit(extract, pipeline, mfe_stream, windows[i + 1])
    post_stream.synchronize()
    elapsed = perf_counter() - started
    timeline = [[origin.elapsed_time(event) for event in row] for row in intervals]
    intersect_ms = sum(max(0., min(a[3], b[1]) - max(a[2], b[0]))
                       for a, b in zip(timeline, timeline[1:]))
    return results, elapsed, intersect_ms


def same_outputs(a, b):
    different = 0
    for (da, sa, ca), (db, sb, cb) in zip(a, b):
        if sa.item() or sb.item():
            raise RuntimeError("TF reported a failure")
        if (ca != cb or da.shape != db.shape or not torch.equal(da[:, 5], db[:, 5])
                or not torch.allclose(da[:, :5], db[:, :5], atol=1e-6, rtol=0)):
            different += 1
    return different


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decoder", type=Path)
    parser.add_argument("tf", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=150)
    args = parser.parse_args()
    if args.frames < 15:
        parser.error("--frames must be at least 15")
    mfe_stream, post_stream = torch.cuda.Stream(), torch.cuda.Stream()
    totals = {"serial": {"seconds": 0., "interval_overlap_ms": 0.},
              "overlap": {"seconds": 0., "interval_overlap_ms": 0.}}
    count = differing = 0
    with CudaDecoder(args.decoder, args.video) as decoder, ThreadPoolExecutor(1) as executor:
        frames = list(islice(iter(decoder.read, None), 12))
        if len(frames) < 12:
            raise ValueError("Video must have at least 12 frames")
        height, width = frames[0].image.shape[:2]
        with torch.cuda.stream(post_stream):
            pipe = CudaPipeline(args.tf, width, height, warp_backend="opencv",
                                mfe_stream=mfe_stream, model_stream=post_stream)
        run(pipe, executor, mfe_stream, post_stream, frames, True)
        frames = frames[-2:]
        torch.cuda.reset_peak_memory_stats()
        for start in range(12, args.frames, 12):
            frames += list(islice(iter(decoder.read, None), min(12, args.frames - start)))
            if len(frames) < 3:
                break
            results = {}
            order = ("serial", "overlap") if start % 24 else ("overlap", "serial")
            for name in order:
                with torch.cuda.stream(post_stream):
                    pipe.tf.reset()
                output, seconds, intersect_ms = run(
                    pipe, executor, mfe_stream, post_stream, frames, name == "overlap")
                results[name] = output
                totals[name]["seconds"] += seconds
                totals[name]["interval_overlap_ms"] += intersect_ms
            differing += same_outputs(results["serial"], results["overlap"])
            count += len(frames) - 2
            frames = frames[-2:]
    for result in totals.values():
        result["fps"] = count / result["seconds"]
    report = {"windows": count, "different_windows": differing, "modes": totals,
              "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(differing != 0)


if __name__ == "__main__":
    raise SystemExit(main())
