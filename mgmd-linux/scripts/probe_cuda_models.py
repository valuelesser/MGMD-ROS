"""Replay existing videos with the same MFE/TF candidates on both model paths."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mgmd.lac import crop_box
from mgmd.lad import crop_region
from mgmd.mfe import MFEConfig, enhance
from mgmd_onnx import LadOnnx
from mgmd_onnx.cuda_models import CudaLac, CudaLad


ROOT = Path(__file__).resolve().parents[1]


def reference_pipeline():
    path = ROOT / "ros1_ws/src/mav_tracker/mav_tracker_nodes/scripts/detector_node.py"
    spec = importlib.util.spec_from_file_location("reference_detector", path)
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    cfg = MFEConfig()
    cfg.apply_recommended()
    cfg.validate()
    return node.Pipeline("cuda", cfg, 0.02, 0.5, 0.5, True)


def difference(actual, expected):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    return float(np.max(np.abs(actual - expected), initial=0))


def compare_video(video, limit, pipe, lac, lad, independent_lad, log):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video}")
    tracker = pipe._new_tracker()
    counts = Counter()
    maxima = Counter()
    with torch.inference_mode():
        try:
            window = []
            for seq in range(limit):
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"{video} ended before frame {seq}")
                assert image.shape == (1080, 1920, 3)
                window.append(image)
                if len(window) < 3:
                    continue
                prev, frame, following = window
                window.pop(0)
                frame_id = seq - 1
                boxes = enhance(prev, frame, following, pipe.cfg, frame_id).boxes
                tracks = tracker.update([(float(x), float(y), float(x + w), float(y + h))
                                         for x, y, w, h in boxes], frame_id,
                                        scores=[float(w * h) for _, _, w, h in boxes])
                boxes = [track.box for track in tracks]
                gpu = torch.from_numpy(frame).cuda()
                crops = [crop_box(frame, box) for box in boxes]
                indices, inputs = lac.prepare(gpu, boxes)
                assert indices == [i for i, crop in enumerate(crops) if crop is not None]
                counts["frames"] += 1
                counts["lac_candidates"] += len(indices)
                row = {"video": video.stem, "frame": frame_id,
                       "candidates": [list(map(float, box)) for box in boxes],
                       "track_ids": [int(track.track_id) for track in tracks], "lad": []}
                if indices:
                    host_crops = [crops[i] for i in indices]
                    expected = np.asarray(pipe.lac.mav_prob_batch(host_crops))
                    actual = lac.probabilities(inputs).cpu().numpy()
                    host_inputs = pipe.lac._prep(host_crops)
                    maxima["lac_input"] = max(maxima["lac_input"], difference(inputs.cpu().numpy(), host_inputs))
                    host_logits = pipe.lac.session.run(None, {lac.model.input.name: host_inputs})[0]
                    maxima["lac_logits"] = max(maxima["lac_logits"],
                                               difference(lac.model.run(inputs)[0].cpu().numpy(), host_logits))
                    maxima["lac_probability"] = max(maxima["lac_probability"], difference(actual, expected))
                    keep = expected >= pipe.lac_thresh
                    gpu_keep = actual >= pipe.lac_thresh
                    counts["lac_threshold_flips"] += int(np.count_nonzero(keep != gpu_keep))
                    row["lac_reference_keep"] = [i for i, flag in zip(indices, keep) if flag]
                    row["lac_gpu_keep"] = [i for i, flag in zip(indices, gpu_keep) if flag]
                    for i, ref_keep, dev_keep in zip(indices, keep, gpu_keep):
                        if not (ref_keep or dev_keep):
                            continue
                        x1, y1, x2, y2 = boxes[i]
                        center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        tensor, region = lad.prepare(gpu, center)
                        x, y, w, h = crop_region(frame.shape[:2], center, lad.size)
                        assert region == (x, y, w, h)
                        crop = frame[y:y + h, x:x + w]
                        maxima["lad_input"] = max(maxima["lad_input"],
                                                 difference(tensor.cpu().numpy(), pipe.lad._preprocess(crop)[None]))
                        ref_raw = pipe.lad.infer(crop)
                        raw = lad.model.run(tensor)[0]
                        maxima["lad_raw"] = max(maxima["lad_raw"], difference(raw.cpu().numpy(), ref_raw))
                        ref = pipe.lad.refine(frame, center)
                        expected_box = list(ref.box) + [ref.conf] if ref is not None else None
                        other_raw = independent_lad.infer(crop)
                        maxima["lad_independent_raw"] = max(maxima["lad_independent_raw"],
                                                             difference(other_raw, ref_raw))
                        other = independent_lad.refine(frame, center)
                        other_box = list(other.box) + [other.conf] if other is not None else None
                        counts["lad_independent_presence_flips"] += int(
                            (expected_box is None) != (other_box is None))
                        if expected_box is not None and other_box is not None:
                            counts["lad_independent_coordinate_changes"] += int(other_box[:4] != expected_box[:4])
                        actual_boxes = lad.select(raw, region).cpu().numpy()
                        actual_box = actual_boxes[0].tolist() if len(actual_boxes) else None
                        counts["lad_candidates"] += 1
                        counts["lad_reference_boxes"] += int(expected_box is not None)
                        counts["lad_gpu_boxes"] += int(actual_box is not None)
                        counts["lad_presence_flips"] += int((expected_box is None) != (actual_box is None))
                        if expected_box is not None and actual_box is not None:
                            delta = difference(np.asarray(actual_box), np.asarray(expected_box))
                            maxima["lad_box_or_confidence"] = max(maxima["lad_box_or_confidence"], delta)
                            counts["lad_coordinate_changes"] += int(actual_box[:4] != expected_box[:4])
                        row["lad"].append({"candidate": i, "reference": expected_box, "gpu": actual_box,
                                           "independent_reference": other_box})
                log.write(json.dumps(row, allow_nan=False) + "\n")
        finally:
            cap.release()
    return {"video": video.name, "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "counts": dict(counts), "max_abs": dict(maxima)}


def edge_cases(lac, lad):
    with torch.inference_mode():
        image = torch.zeros((1080, 1920, 3), dtype=torch.uint8, device="cuda")
        indices, empty = lac.prepare(image, [])
        assert not indices and lac.probabilities(empty).numel() == 0
        indices, edge = lac.prepare(image, [(-2, -2, 3, 4), (-30, -30, -20, -20)])
        assert indices == [0] and tuple(edge.shape) == (1, 3, 32, 32)
        sample, region = lad.prepare(image, (0, 0))
        assert tuple(sample.shape) == (1, 3, 320, 320) and region == (0, 0, 320, 320)
        # Equal confidence, non-overlapping boxes with different distances to crop center.
        raw = torch.tensor([[[100, 100, 10, 10, .9, .9],
                             [160, 160, 10, 10, .9, .9]]], device="cuda")
        assert lad.select(raw, region)[0, :4].tolist() == [155, 155, 165, 165]
        assert lad.select(torch.zeros_like(raw), region).numel() == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    args.out.mkdir()
    pipe = reference_pipeline()
    lac, lad = CudaLac(pipe.lac), CudaLad(pipe.lad)
    independent = LadOnnx(device="cuda", conf_thresh=pipe.lad_conf)
    edge_cases(lac, lad)
    with (args.out / "frames.jsonl").open("w") as log:
        results = [compare_video(video, args.frames, pipe, lac, lad, independent, log)
                   for video in args.videos]
    report = json.dumps({"edge_cases": "pass", "videos": results}, indent=2, allow_nan=False) + "\n"
    (args.out / "summary.json").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
