"""Audit NVDEC frame identity, lifetime, timestamps, EOF and reset."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mgmd_onnx.cuda_frames import CudaDecoder


def verify(library, path, count):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {path}")
    try:
        with CudaDecoder(library, path) as decoder, torch.inference_mode():
            first = decoder.read()
            assert first is not None and first.pts == 0
            saved = first.image.clone()
            prev_pts = -1
            max_delta = different = pixels = total_delta = 0
            for seq in range(count):
                frame = first if seq == 0 else decoder.read()
                assert frame is not None and frame.seq == seq and frame.pts > prev_pts
                assert frame.image.is_cuda and frame.image.shape == (1080, 1920, 3)
                prev_pts = frame.pts
                if seq < 120:
                    ok, reference = capture.read()
                    assert ok
                    delta = np.abs(frame.image.cpu().numpy().astype(np.int16) - reference)
                    max_delta = max(max_delta, int(delta.max()))
                    different += int(np.count_nonzero(delta))
                    total_delta += int(delta.sum())
                    pixels += delta.size
            assert decoder.read() is None and decoder.read() is None
            assert torch.equal(first.image, saved)
            decoder.reset()
            replay = decoder.read()
            assert replay.seq == 0 and replay.pts == first.pts
            assert torch.equal(replay.image, saved)
            result = {
                "video": path.name, "frames": count, "time_base": first.time_base,
                "first_pts": first.pts, "last_pts": prev_pts, "strides": first.strides,
                "eof_and_reset": "pass", "first_120_bgr_max_abs": max_delta,
                "first_120_bgr_mean_abs": total_delta / pixels,
                "first_120_bgr_different_fraction": different / pixels,
            }
        assert not decoder.handle
        return result
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("videos", type=Path, nargs="+")
    args = parser.parse_args()
    args.out.mkdir()
    counts = {"phantom30": 1498, "phantom70": 1349}
    summary = [verify(args.library, video, counts[video.stem]) for video in args.videos]
    report = json.dumps({"videos": summary}, indent=2) + "\n"
    (args.out / "summary.json").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
