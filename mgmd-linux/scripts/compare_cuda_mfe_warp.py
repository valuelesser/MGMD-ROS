"""Compare OpenCV remap and Torch warp on identical NVDEC frames and homographies."""

import argparse
from collections import Counter, deque
import json
from pathlib import Path

import torch

from mgmd.mfe import MFEConfig
from mgmd_onnx.cuda_frames import CudaDecoder
from mgmd_onnx.cuda_mfe import CudaMfe, warp_inverse


class ComparedMfe(CudaMfe):
    def __init__(self, cfg, width, height):
        super().__init__(cfg, width, height, warp_backend="opencv")
        self.counts = Counter()

    def _warp(self, source, matrix):
        actual, valid = super()._warp(source, matrix)
        expected, expected_valid = warp_inverse(source, matrix, return_valid=True)
        difference = (actual.short() - expected.short()).abs()
        self.counts["directions"] += 1
        self.counts["pixels"] += difference.numel()
        self.counts["pixel_differences"] += difference.count_nonzero().item()
        self.counts["valid_differences"] += (valid != expected_valid).count_nonzero().item()
        self.counts["max_gray_difference"] = max(
            self.counts["max_gray_difference"], difference.max().item())
        return actual, valid


@torch.inference_mode()
def compare(library, video, limit):
    frames = deque(maxlen=3)
    counts = Counter()
    with CudaDecoder(library, video) as decoder:
        first = decoder.read()
        if first is None:
            raise RuntimeError(f"Empty video: {video}")
        height, width = first.image.shape[:2]
        cfg = MFEConfig().apply_recommended()
        reference = CudaMfe(cfg, width, height)
        candidate = ComparedMfe(cfg, width, height)
        frames.append(first.image)
        for _ in range(1, limit):
            frame = decoder.read()
            if frame is None:
                break
            frames.append(frame.image)
            if len(frames) < 3:
                continue
            a = [tuple(box) for box in reference.candidates(*frames)[0].tolist()]
            b = [tuple(box) for box in candidate.candidates(*frames)[0].tolist()]
            counts["center_frames"] += 1
            counts["torch_boxes"] += len(a)
            counts["opencv_boxes"] += len(b)
            counts["ordered_frames_changed"] += a != b
            counts["box_set_frames_changed"] += set(a) != set(b)
            counts["boxes_added"] += len(set(b) - set(a))
            counts["boxes_removed"] += len(set(a) - set(b))
    return {"resolution": [width, height], "candidates": dict(counts),
            "warp": dict(candidate.counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decoder_library", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    result = {video.stem: compare(args.decoder_library, video, args.frames)
              for video in args.videos}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
