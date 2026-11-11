"""Compare libav and OpenCV GPU BGR delivery on one sequential video."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from mgmd_onnx.cuda_frames import CudaDecoder


def opencv_frames(video):
    reader = cv2.cudacodec.createVideoReader(str(video))
    if not reader.set(colorFormat=cv2.cudacodec.BGR):
        raise RuntimeError("OpenCV could not select BGR output")
    while True:
        ok, frame = reader.nextFrame()
        if not ok:
            return
        if frame.type() != cv2.CV_8UC3 or not frame.cudaPtr():
            raise RuntimeError("Expected a GPU BGR frame")
        yield frame


def libav_frames(library, video):
    with CudaDecoder(library, video) as decoder:
        for frame in iter(decoder.read, None):
            yield frame.image


def torch_view(frame):
    width, height = frame.size()

    class View:
        __cuda_array_interface__ = {
            "version": 2, "shape": (height, width, 3), "strides": (frame.step, 3, 1),
            "typestr": "|u1", "data": (frame.cudaPtr(), False),
        }

    tensor = torch.as_tensor(View(), device="cuda")
    if tensor.data_ptr() != frame.cudaPtr():
        raise RuntimeError("OpenCV frame import did not share device memory")
    return tensor


def benchmark(frames):
    samples = []
    resolution = None
    started = perf_counter()
    while True:
        begin = perf_counter()
        frame = next(frames, None)
        if frame is None:
            break
        if resolution is None:
            resolution = frame.size() if isinstance(frame, cv2.cuda.GpuMat) else frame.shape[1::-1]
        torch.cuda.current_stream().synchronize()
        samples.append((perf_counter() - begin) * 1000)
    elapsed = perf_counter() - started
    if not samples:
        raise RuntimeError("No decoded frames")
    return {"resolution": list(resolution), "frames": len(samples),
            "seconds": elapsed, "fps": len(samples) / elapsed,
            "ready_ms": dict(zip(("p50", "p95"), np.percentile(samples, (50, 95)).tolist()))}


def compare(library, video, leading_frames):
    checked = different = max_difference = total = absolute_error = 0
    with CudaDecoder(library, video) as decoder:
        frames = opencv_frames(video)
        try:
            for _ in range(leading_frames):
                if next(frames, None) is None:
                    raise RuntimeError("OpenCV ended before the requested frame offset")
            for index, frame in enumerate(frames):
                reference = decoder.read()
                if reference is None:
                    break
                if index % 30:
                    continue
                actual = torch_view(frame)
                if actual.shape != reference.image.shape:
                    raise RuntimeError("Decoder sizes differ")
                error = (actual.short() - reference.image.short()).abs()
                different += error.count_nonzero().item()
                absolute_error += error.sum().item()
                max_difference = max(max_difference, error.max().item())
                total += error.numel()
                checked += 1
        finally:
            frames.close()
    if not checked:
        raise RuntimeError("No corresponding frames to compare")
    return {"opencv_leading_frames_skipped_for_comparison": leading_frames,
            "sampled_frames": checked, "different_channels": different,
            "sampled_channels": total, "mean_abs_channel_difference": absolute_error / total,
            "max_channel_difference": max_difference}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--opencv-leading-frames", type=int, default=0)
    args = parser.parse_args()
    if args.opencv_leading_frames < 0:
        parser.error("--opencv-leading-frames must be nonnegative")
    torch.empty(1, device="cuda")
    report = {
        "opencv_version": cv2.__version__,
        "opencv_module": cv2.__file__,
        "video": str(args.video),
        "libav": benchmark(libav_frames(args.library, args.video)),
        "opencv": benchmark(opencv_frames(args.video)),
        "color": compare(args.library, args.video, args.opencv_leading_frames),
    }
    result = json.dumps(report, indent=2) + "\n"
    args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
