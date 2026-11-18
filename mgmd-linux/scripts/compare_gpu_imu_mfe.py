#!/usr/bin/env python3
"""Replay identical ROS bag windows through both single-stream CUDA pipelines."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import rosbag
import rospy
import torch
import yaml
from sensor_msgs.msg import Image

from mgmd_onnx.cuda_buf_ccl import BufCcl
from mgmd_onnx.cuda_imu import camera_warps
from mgmd_onnx.cuda_pipeline import CudaPipeline
from mgmd_onnx.cuda_preview import CudaPreview


def read_segment(path, camera, offset, duration):
    image_topic = f"/usb_cam{camera}/image_raw"
    imu_topic = "/mavros/imu/data"
    frames, orientations = {}, {}
    with rosbag.Bag(path) as bag:
        start = rospy.Time.from_sec(bag.get_start_time() + offset)
        end = start + rospy.Duration(duration)
        pad = rospy.Duration(.35)
        for topic, msg, _ in bag.read_messages(topics=[image_topic, imu_topic],
                                               start_time=start-pad, end_time=end+pad):
            stamp = msg.header.stamp.to_nsec()
            if topic == imu_topic:
                q = msg.orientation
                orientations[stamp] = (q.x, q.y, q.z, q.w)
            elif start.to_nsec() <= stamp < end.to_nsec() and stamp not in frames:
                if msg.encoding != "bgr8":
                    raise ValueError(f"Expected bgr8, got {msg.encoding}")
                data = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
                frames[stamp] = data[:, :msg.width*3].reshape(msg.height, msg.width, 3).copy()
    if len(frames) < 3 or len(orientations) < 2:
        raise ValueError(f"Insufficient bag data at offset {offset}")
    return sorted(frames.items()), sorted(orientations.items())


def paired_indices(frames, coverage):
    gaps = np.diff([stamp for stamp, _ in frames])
    good = (gaps >= 15_000_000) & (gaps <= 85_000_000)
    return np.flatnonzero(coverage & good[:-1] & good[1:]) + 1


def detect(pipe, stream, preview, images, warps, indices):
    pipe.tf.reset()
    pipe.mfe.clear_gray_cache()
    stream.synchronize()
    output, stages = {}, []
    begin = perf_counter()
    for i in indices:
        ids = (int(i)-1, int(i), int(i)+1)
        args = (images[j] for j in ids)
        boxes, status, _, marks = pipe.step(*args, int(i), frame_ids=ids,
                                            alignment=None if warps is None else warps[i-1])
        marks[-1].synchronize()
        if status.item():
            raise RuntimeError(f"CUDA TF status {status.item()} at frame {i}")
        output[int(i)] = boxes.tolist()
        times = [a.elapsed_time(b) for a, b in zip(marks, marks[1:])]
        stages.append(times)
        preview.draw(images[i], boxes, [f"Frame {i}", "  ".join(
            f"{name} {t:.1f} ms" for name, t in zip(("MFE", "TF", "LAC", "LAD"), times))])
        frame = preview.download()
        # Construct the same ROS image payload without involving a subscriber or sink.
        Image(height=frame.shape[0], width=frame.shape[1], encoding="bgr8",
              step=frame.shape[1]*3, data=frame.tobytes())
    return output, stages, perf_counter()-begin


def match_outputs(lk, imu):
    counts = {"lk_boxes": 0, "imu_boxes": 0, "matched": 0, "both_empty": 0,
              "lk_only_frames": 0, "imu_only_frames": 0}
    ious, centers, scores = [], [], []
    for i, v in lk.items():
        w = imu[i]
        counts["lk_boxes"] += len(v)
        counts["imu_boxes"] += len(w)
        counts["both_empty"] += not v and not w
        counts["lk_only_frames"] += bool(v) and not w
        counts["imu_only_frames"] += bool(w) and not v
        if not v or not w:
            continue
        a, b = np.asarray(v), np.asarray(w)
        intersection = (np.minimum(a[:, None, 2:4], b[None, :, 2:4]) -
                        np.maximum(a[:, None, :2], b[None, :, :2])).clip(0).prod(axis=2)
        union = ((a[:, 2:4]-a[:, :2]).prod(axis=1)[:, None] +
                 (b[:, 2:4]-b[:, :2]).prod(axis=1)[None, :] - intersection)
        iou = intersection / np.maximum(union, 1e-9)
        while iou.max() >= .5:
            j, k = np.unravel_index(iou.argmax(), iou.shape)
            ious.append(float(iou[j, k]))
            centers.append(float(np.linalg.norm((a[j, :2]+a[j, 2:4]-b[k, :2]-b[k, 2:4])*.5)))
            scores.append(float(abs(a[j, 4]-b[k, 4])))
            iou[j, :] = 0
            iou[:, k] = 0
    counts["matched"] = len(ious)
    counts["lk_unmatched"] = counts["lk_boxes"] - len(ious)
    counts["imu_unmatched"] = counts["imu_boxes"] - len(ious)
    return counts, ious, centers, scores


def quantiles(values):
    return np.percentile(values, [50, 95]).tolist() if len(values) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration")
    parser.add_argument("output", type=Path)
    parser.add_argument("--bag", nargs=2, action="append", metavar=("CAMERA_NUMBER", "PATH"), required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=[20, 55, 90])
    parser.add_argument("--duration", type=float, default=4)
    parser.add_argument("--tf-library", default="/tmp/mgmd-tf-g3.so")
    parser.add_argument("--buf-library", default="/tmp/mgmd-buf-ccl.so")
    parser.add_argument("--preview-library", default="/tmp/mgmd-cuda-preview.so")
    args = parser.parse_args()
    with open(args.calibration, encoding="utf-8") as source:
        cal = yaml.safe_load(source)
    width, height = int(cal["image_width"]), int(cal["image_height"])
    stream = torch.cuda.Stream()
    report = {"resolution": [width, height], "duration_s": args.duration, "cameras": {}}
    with torch.inference_mode(), torch.cuda.stream(stream):
        pipe = CudaPipeline(args.tf_library, width, height, warp_backend="opencv", mfe_stream=stream,
                            model_stream=stream, async_lad=True)
        pipe.mfe.buf_ccl = BufCcl(args.buf_library, width, height, stream)
        preview = CudaPreview(args.preview_library, torch.empty((height, width, 3), device="cuda", dtype=torch.uint8))
        for camera, path in args.bag:
            camera = int(camera)
            intrinsic = torch.tensor(cal[f"camera_matrix_{camera:02d}"], device="cuda", dtype=torch.float64).reshape(3, 3)
            rotation = torch.tensor(cal["uavs"][camera-1]["T_bc"], device="cuda", dtype=torch.float64).reshape(4, 4)[:3, :3]
            total_frames, total_prep = 0, 0.
            totals, all_ious, all_centers, all_scores = {}, [], [], []
            time_s = {"lk": 0., "imu": 0.}
            stage_ms = {"lk": [], "imu": []}
            segments = []
            for clip, offset in enumerate(args.offsets):
                frames, imu = read_segment(path, camera, offset, args.duration)
                images = torch.as_tensor(np.stack([img for _, img in frames]), device="cuda")
                if images.shape[1:] != (height, width, 3):
                    raise ValueError("Calibration does not match the bag resolution")
                frame_ns = torch.tensor([t for t, _ in frames], device="cuda", dtype=torch.int64)
                imu_ns = torch.tensor([t for t, _ in imu], device="cuda", dtype=torch.int64)
                imu_q = torch.tensor([q for _, q in imu], device="cuda", dtype=torch.float64)
                if clip == 0:
                    camera_warps(frame_ns, imu_ns, imu_q, intrinsic, rotation)
                stream.synchronize()
                start = perf_counter()
                warps, covered = camera_warps(frame_ns, imu_ns, imu_q, intrinsic, rotation)
                coverage = covered.cpu().numpy()
                total_prep += perf_counter() - start
                indices = paired_indices(frames, coverage)
                if not len(indices):
                    raise ValueError(f"No synchronized windows at {offset}s")
                total_frames += len(indices)
                measures = {}
                for mode in (("lk", "imu") if clip % 2 == 0 else ("imu", "lk")):
                    h = warps if mode == "imu" else None
                    detect(pipe, stream, preview, images, h, indices[:12])
                    out, stage, wall = detect(pipe, stream, preview, images, h, indices)
                    measures[mode] = out
                    stage_ms[mode].extend(stage)
                    time_s[mode] += wall
                counts, ious, centers, scores = match_outputs(measures["lk"], measures["imu"])
                for key, value in counts.items(): totals[key] = totals.get(key, 0) + value
                all_ious.extend(ious)
                all_centers.extend(centers)
                all_scores.extend(scores)
                segments.append({"offset_s": offset, "frames": len(frames), "paired_windows": len(indices), **counts})
                print(f"camera {camera}, {offset}s: {len(indices)} windows, {counts}", flush=True)
            result = {"bag": Path(path).name, "windows": total_frames, "segments": segments,
                      "imu_precompute_s": total_prep, "agreement": totals,
                      "iou_p50_p05": np.percentile(all_ious, [50, 5]).tolist() if all_ious else None,
                      "center_delta_px_p50_p95": quantiles(all_centers), "confidence_delta_p50_p95": quantiles(all_scores)}
            for mode in time_s:
                result[mode] = {"wall_s": time_s[mode], "fps": total_frames/(time_s[mode]+(total_prep if mode == "imu" else 0)),
                                "stage_mean_ms": dict(zip(("MFE", "TF", "LAC", "LAD"), np.mean(stage_ms[mode], axis=0).tolist()))}
            report["cameras"][str(camera)] = result
            print(result, flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
