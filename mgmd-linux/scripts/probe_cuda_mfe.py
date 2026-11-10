"""Compare fixed-homography CUDA MFE image stages with the existing implementation."""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mgmd import mfe
from mgmd_onnx import cuda_mfe


def pixels(actual, expected):
    actual = actual.cpu().numpy()
    if actual.dtype == np.bool_:
        expected = expected != 0
    assert actual.shape == expected.shape
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    return int(np.count_nonzero(delta)), int(delta.max(initial=0))


def compare_window(host, device, cfg):
    previous, center, following = host
    prev_gpu, center_gpu, next_gpu = device
    details = {}
    directions = []
    isolated = []
    for name, gray, gpu in (("previous", previous, prev_gpu), ("following", following, next_gpu)):
        aligned, valid, transform = mfe.align_to_reference(gray, center, cfg)
        warp = (gpu if transform is None else
                cuda_mfe.warp_inverse(gpu, torch.from_numpy(transform).cuda()))
        mask, threshold = cuda_mfe.threshold_direction(
            center_gpu, warp, torch.from_numpy(valid).cuda(), cfg)
        delta = cv2.absdiff(center, aligned)
        ref_mask = mfe._binarize(delta, cfg.threshold, valid, cfg)
        details[f"{name}_warp"] = pixels(warp, aligned)
        details[f"{name}_threshold"] = (
            int(int(threshold) != cfg.threshold + int(delta.mean())), 0)
        details[f"{name}_mask"] = pixels(mask, ref_mask)
        isolated_mask, isolated_threshold = cuda_mfe.threshold_direction(
            center_gpu, torch.from_numpy(aligned).cuda(), torch.from_numpy(valid).cuda(), cfg)
        assert int(isolated_threshold) == cfg.threshold + int(delta.mean())
        details[f"{name}_mask_fixed_pixels"] = pixels(isolated_mask, ref_mask)
        isolated.append(isolated_mask)
        directions.append((mask, ref_mask))
    motion = cuda_mfe.combine_motion(directions[0][0], directions[1][0], cfg.mode)
    ref_motion = cv2.bitwise_or(directions[0][1], directions[1][1])
    result = cuda_mfe.morphology(motion, cfg.morph_size, cfg.open_iters, cfg.close_iters)
    ref_result, ref_boxes = mfe.postprocess(ref_motion, cfg)
    details["motion"] = pixels(motion, ref_motion)
    details["morphology"] = pixels(result, ref_result)
    fixed_motion = cuda_mfe.combine_motion(isolated[0], isolated[1], cfg.mode)
    fixed_result = cuda_mfe.morphology(fixed_motion, cfg.morph_size, cfg.open_iters, cfg.close_iters)
    details["morphology_fixed_pixels"] = pixels(fixed_result, ref_result)
    # CPU CCL is only the oracle for this fixed-alignment step.
    _, boxes = mfe.postprocess(result.cpu().numpy().astype(np.uint8) * 255,
                               replace(cfg, open_iters=0, close_iters=0))
    details["candidate_list"] = (int(boxes != ref_boxes), 0)
    return details, len(ref_boxes)


def compare_video(video, frames, log):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    cfg = mfe.MFEConfig()
    cfg.apply_recommended()
    cfg.validate()
    host, device = [], []
    mismatches, max_abs = Counter(), Counter()
    candidates = 0
    try:
        for seq in range(frames):
            ok, frame = capture.read()
            if not ok or frame.shape != (1080, 1920, 3):
                raise ValueError(f"Missing 1080p frame {seq}: {video}")
            gray = mfe.preprocess(frame, cfg)
            gray_gpu = cuda_mfe.bgr_to_gray(torch.from_numpy(frame).cuda())
            count, delta = pixels(gray_gpu, gray)
            mismatches["gray"] += count
            max_abs["gray"] = max(max_abs["gray"], delta)
            host.append(gray)
            device.append(gray_gpu)
            if len(host) < 3:
                continue
            details, total = compare_window(host, device, cfg)
            candidates += total
            for stage, (count, delta) in details.items():
                mismatches[stage] += count
                max_abs[stage] = max(max_abs[stage], delta)
            log.write(json.dumps({"video": video.stem, "frame": seq - 1,
                                  "candidates": total, "differences": details}) + "\n")
            host.pop(0)
            device.pop(0)
    finally:
        capture.release()
    return {"video": video.name, "center_frames": frames - 2, "reference_candidates": candidates,
            "mismatch_counts": dict(mismatches), "max_abs": dict(max_abs)}


def synthetic_warp():
    image = np.random.default_rng(7).integers(0, 256, (64, 96), dtype=np.uint8)
    source = torch.from_numpy(image).cuda()
    matrices = [np.eye(3), np.array([[1., 0., -1.25], [0., 1., .625], [0., 0., 1.]])]
    for matrix in matrices:
        expected = cv2.warpPerspective(image, matrix, image.shape[::-1],
                                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        count, delta = pixels(cuda_mfe.warp_inverse(source, torch.from_numpy(matrix).cuda()), expected)
        assert delta <= 1
        if np.array_equal(matrix, np.eye(3)):
            assert count == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    args.out.mkdir()
    with torch.inference_mode(), (args.out / "frames.jsonl").open("w") as log:
        synthetic_warp()
        videos = [compare_video(video, args.frames, log) for video in args.videos]
    report = {"opencv": cv2.__version__, "torch": torch.__version__,
              "synthetic_warp": "pass", "videos": videos}
    summary = json.dumps(report, indent=2) + "\n"
    (args.out / "summary.json").write_text(summary)
    print(summary, end="")
    isolated_stages = ("gray", "previous_mask_fixed_pixels", "following_mask_fixed_pixels",
                       "morphology_fixed_pixels")
    return int(any(row["mismatch_counts"][key] for row in videos for key in isolated_stages))


if __name__ == "__main__":
    raise SystemExit(main())
