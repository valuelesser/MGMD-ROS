#!/usr/bin/env python3
"""Publish detections and an optional preview from the GPU file pipeline."""

from collections import defaultdict, deque
from contextlib import ExitStack
import json
from statistics import mean
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import torch

from detector_node import build_detection_array
from mav_tracker_msgs.msg import DetectionArray
from mgmd_onnx.cuda_frames import CudaDecoder
from mgmd_onnx.cuda_pipeline import CudaPipeline
from mgmd_onnx.cuda_parity import ParityPipeline


STAGES = ("MFE", "TF", "LAC", "LAD")
MFE_HUD_GROUPS = (
    (("gray", "Gray"), ("LK", "LK"), ("RANSAC", "RANSAC"), ("warp", "Warp")),
    (("threshold_median", "Diff/threshold/median"), ("morphology", "Morph"),
     ("CCL", "CCL"), ("region_stats", "Region stats")),
)


def ready_windows(decoder, first, limit, started, rate, realtime, profile, counts, wall_bins, prefetched):
    frames = deque((first,), maxlen=3)
    while not rospy.is_shutdown() and (limit == 0 or counts["read"] < limit):
        if prefetched:
            frame = prefetched.popleft()
        else:
            frame = decoder.read(wall_bins)
        if frame is None:
            break
        counts["read"] += 1
        frames.append(frame)
        if len(frames) < 3:
            continue
        if realtime:
            now = time.perf_counter()
            if frame.seq < int((now - started) * rate):
                counts["skipped"] += 1
                continue
            delay = started + frame.seq / rate - now
            if delay > 0:
                time.sleep(delay)
                if wall_bins is not None:
                    wall_bins["source_sleep"] += time.perf_counter() - now
        marks = [] if profile else None
        yield tuple(frames), None, marks


def preview_image(frame, boxes):
    image = frame.clone()
    height, width = frame.shape[:2]
    x = torch.arange(width, device=frame.device)[None, :]
    y = torch.arange(height, device=frame.device)[:, None]
    color = torch.tensor((0, 255, 0), device=frame.device, dtype=frame.dtype)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        within_x = (x >= x1) & (x <= x2)
        within_y = (y >= y1) & (y <= y2)
        border = (within_x & (((y - y1).abs() <= 1) | ((y - y2).abs() <= 1))
                  | within_y & (((x - x1).abs() <= 1) | ((x - x2).abs() <= 1)))
        image = torch.where(border[:, :, None], color, image)
    return image


def hud_lines(playback, processed, skipped, recent, now):
    fps = (len(recent) - 1) / max(now - recent[0][0], 1e-6)
    stage = [mean(samples) for samples in zip(*(row[1] for row in recent))]
    slowest = int(np.argmax(stage))
    text = [
        f"Processing {fps:.1f} FPS   {playback}   Processed {processed}   Skipped {skipped}",
        "Mean (~2 s): " + "   ".join(f"{name} {ms:.1f} ms" for name, ms in zip(STAGES, stage))
        + f"   |   Slowest: {STAGES[slowest]}",
    ]
    if recent[-1][2]:
        for group in MFE_HUD_GROUPS:
            text.append("   ".join(
                f"{label} {mean(row[2][key] for row in recent if key in row[2]):.1f} ms"
                for key, label in group if key in recent[-1][2]))
    return text


def add_hud(image, text):
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, .68
    for index, line in enumerate(text):
        origin = (14, 30 + index * 28)
        cv2.putText(image, line, origin, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, origin, font, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    rospy.init_node("cuda_pipeline")
    if not torch.cuda.is_available():
        raise RuntimeError("track_gpu.launch requires a CUDA device")
    get = rospy.get_param
    video = get("~video")
    source_rate = float(get("~rate", 30))
    if source_rate <= 0:
        raise ValueError("Playback rate must be positive")
    realtime = get("~realtime", True)
    profile_mfe = get("~profile_mfe", False)
    wall_bins = defaultdict(float) if get("~profile_wall", False) else None
    warp_backend = get("~warp_backend", "torch")
    mode = get("~pipeline_mode", "legacy")
    warmup_frames = get("~warmup_frames", 0)
    if warmup_frames < 0 or (warmup_frames and mode not in ("legacy", "parity")):
        raise ValueError("warmup_frames requires legacy or parity and a nonnegative count")
    reuse_gray = get("~reuse_gray", True)
    buf_library = get("~buf_ccl_library", "/tmp/mgmd-buf-ccl.so")
    if mode not in ("legacy", "parity"):
        raise ValueError("Expected legacy or parity")
    confirm_frames = get("~confirm_frames", 3 if mode == "parity" else 0)
    compute_lanes = get("~compute_lanes", 2)
    if compute_lanes not in (1, 2) or (compute_lanes == 1 and mode != "parity"):
        raise ValueError("compute_lanes=1 requires parity mode; parity supports one or two lanes")
    if confirm_frames < 0 or (confirm_frames and (mode != "parity" or confirm_frames < 2)):
        raise ValueError("confirm_frames must be 0 (disabled) or at least 2 in parity mode")
    tracked_total = reused_total = reused_windows = 0
    playback = f"Source {source_rate:g} Hz" if realtime else "Full-frame playback"
    limit = get("~limit", 0)
    decoder_backend = get("~decoder_backend", "libav")
    if decoder_backend not in ("libav", "opencv"):
        raise ValueError("decoder_backend must be libav or opencv")
    detections = rospy.Publisher("/mav/detections", DetectionArray, queue_size=10)
    debug = (rospy.Publisher("/mav/debug_image", Image, queue_size=1)
             if get("~preview", True) else None)
    preview_library = get("~preview_library", "")
    counts = {"read": 0, "skipped": 0}
    processed = detected = 0
    recent = deque()
    stages = []
    latency = []
    mfe_samples = defaultdict(list)
    elapsed = 0.
    if decoder_backend == "opencv":
        from mgmd_onnx.cuda_opencv_decoder import OpenCvCudaDecoder
        source = OpenCvCudaDecoder(video, get("~decoder_leading_frames", 0))
    else:
        source = CudaDecoder(get("~decoder_library"), video)
    with torch.inference_mode(), source as decoder, ExitStack() as stack:
        first = decoder.read()
        if first is None:
            raise RuntimeError("Video has no decodable frames")
        counts["read"] = 1
        height, width = first.image.shape[:2]
        gpu_preview = None
        if debug is not None and preview_library:
            from mgmd_onnx.cuda_preview import CudaPreview
            gpu_preview = CudaPreview(preview_library, first.image)
            rospy.loginfo("GPU preview compositor: %dx%d with CUDA boxes and HUD", width, height)
        if min(height, width) < 320:
            raise ValueError("GPU LAD requires both dimensions to be at least 320")
        library = get("~tf_library")
        parity = None
        pipe = None
        if mode == "parity":
            parity = ParityPipeline(library, width, height, warp_backend, reuse_gray, stack,
                                    buf_library=buf_library, confirm_frames=confirm_frames,
                                    compute_lanes=compute_lanes)
            rospy.loginfo("GPU compute topology: %d detection lane(s), TF %s",
                          compute_lanes, "on the lane stream" if compute_lanes == 1 else "on a shared stream")
        else:
            pipe = CudaPipeline(library, width, height, warp_backend=warp_backend)
        torch.cuda.synchronize()
        rospy.loginfo("GPU path initialized: NVDEC, MFE, TF, LAC, LAD (%dx%d)", width, height)
        rospy.loginfo("MFE warp backend: %s; pipeline mode: %s", warp_backend, mode)
        rospy.loginfo("GPU decoder backend: %s", decoder_backend)
        prefetched = deque()
        if warmup_frames:
            warm_started = time.perf_counter()
            frames = [first]
            target = warmup_frames + 2
            for _ in range(1, min(target, limit) if limit else target):
                frame = decoder.read()
                if frame is None:
                    break
                frames.append(frame)
            prefetched.extend(frames[1:])
            samples = ((tuple(frames[i:i + 3]), None, None)
                       for i in range(len(frames) - 2))
            warmed = 0
            if parity:
                for _, job, _ in parity.windows(samples):
                    _, status, _, _, _ = job.result()
                    if status.item():
                        raise RuntimeError("CUDA TF failed during warmup")
                    warmed += 1
                parity.reset_after_warmup()
            else:
                for (a, b, c), _, _ in samples:
                    _, status, _, _ = pipe.step(a.image, b.image, c.image, b.seq)
                    if status.item():
                        raise RuntimeError("CUDA TF failed during warmup")
                    warmed += 1
                pipe.tf.reset()
            torch.cuda.synchronize()
            rospy.loginfo("GPU warmup: %d windows in %.3fs", warmed,
                          time.perf_counter() - warm_started)
        started = time.perf_counter()
        startup_processed = 0
        first_publish_ms = None
        dispatch_idle = parity is not None and get("~dispatch_idle", True)
        dispatch_stats = defaultdict(float)
        source_bins = (defaultdict(float) if wall_bins is not None else None) if dispatch_idle else wall_bins
        windows = ready_windows(decoder, first, limit, started, source_rate, realtime,
                                profile_mfe, counts, source_bins, prefetched)
        if dispatch_idle:
            windows = parity.available_windows(windows, dispatch_stats)
            stack.callback(windows.close)
        elif parity:
            windows = parity.windows(windows)
        for (previous, center, following), job, mfe_timing in windows:
            begin = time.perf_counter()
            reuse_count = 0
            if parity:
                boxes, status, candidates, stage_ms, (tracked, reuse_count) = job.result()
                tracked_total += tracked
            else:
                frame_ids = (previous.seq, center.seq, following.seq) if reuse_gray else None
                boxes, status, candidates, timing = pipe.step(
                    previous.image, center.image, following.image, center.seq, mfe_timing,
                    frame_ids=frame_ids)
            compute_done = time.perf_counter()
            if wall_bins is not None:
                wall_bins["wait_parity" if parity else "legacy_compute"] += compute_done - begin
            # Detection metadata and the optional display image cross the ROS boundary here.
            error = status.item()
            if error:
                raise RuntimeError("CUDA TF failed with status {}".format(error))
            data = boxes.tolist()
            verified_count = len(data) - reuse_count
            if parity and parity.confirmed is not None:
                parity.confirmed.observe(center.seq, data[:verified_count], data[verified_count:])
                reused_total += reuse_count
                reused_windows += bool(reuse_count)
            if wall_bins is not None:
                wall_bins["detection_readback"] += time.perf_counter() - compute_done
            if not parity:
                timing[-1].synchronize()
                stage_ms = [a.elapsed_time(b) for a, b in zip(timing, timing[1:])]
            subtotals = defaultdict(float)
            if mfe_timing:
                for (_, start), (name, end) in zip(mfe_timing, mfe_timing[1:]):
                    if name != "start":
                        subtotals[name] += start.elapsed_time(end)
                for name, total in subtotals.items():
                    mfe_samples[name].append(total)
            now = time.perf_counter()
            elapsed += now - begin
            stages.append(stage_ms)
            recent.append((now, stage_ms, subtotals))
            while len(recent) > 1 and recent[1][0] < now - 2:
                recent.popleft()
            header = Header(seq=center.seq, stamp=rospy.Time.now(), frame_id="camera")
            publish_started = time.perf_counter() if wall_bins is not None else None
            detections.publish(build_detection_array(
                header, width, height, [row[:4] for row in data], [row[4] for row in data],
                ["lad"] * verified_count + ["tf"] * reuse_count, candidates, *stage_ms,
                track_ids=[int(row[5]) for row in data]))
            if wall_bins is not None:
                preview_started = time.perf_counter()
                wall_bins["detection_publish"] += preview_started - publish_started
            if debug is not None:
                text = hud_lines(playback, processed + 1, counts["skipped"], recent, now)
                if gpu_preview is None:
                    preview = preview_image(center.image, boxes)
                else:
                    gpu_preview.draw(center.image, boxes, text)
                if wall_bins is not None:
                    download_started = time.perf_counter()
                    wall_bins["preview_overlay_enqueue"] += download_started - preview_started
                image = preview.cpu().numpy() if gpu_preview is None else gpu_preview.download()
                if wall_bins is not None:
                    hud_started = time.perf_counter()
                    wall_bins["preview_download_and_gpu_wait"] += hud_started - download_started
                if gpu_preview is None:
                    add_hud(image, text)
                payload = image.tobytes()
                if wall_bins is not None:
                    image_publish_started = time.perf_counter()
                    wall_bins["preview_hud_and_pack"] += image_publish_started - hud_started
                debug.publish(Image(header=header, height=height, width=width, encoding="bgr8",
                                    step=width * 3, data=payload))
                if wall_bins is not None:
                    wall_bins["preview_publish"] += time.perf_counter() - image_publish_started
            processed += 1
            since_start = time.perf_counter() - started
            if first_publish_ms is None:
                first_publish_ms = 1000 * since_start
            startup_processed += since_start <= 3
            detected += bool(data)
            if realtime:
                latency.append(1000 * (time.perf_counter() - started - center.seq / source_rate))
            if processed % 50 == 0:
                rospy.loginfo("GPU progress: %d frames, %.2f fps", processed,
                              processed / (time.perf_counter() - started))
        wall = time.perf_counter() - started
        if parity:
            rospy.loginfo("GPU track confirmation: frames=%d tf_track_rows=%d reused_rows=%d windows_with_reuse=%d",
                          confirm_frames, tracked_total, reused_total, reused_windows)
        if first_publish_ms is not None:
            rospy.loginfo("GPU playback startup: first_publish=%.2fms windows_first_3s=%d",
                          first_publish_ms, startup_processed)
        if dispatch_idle and processed:
            rospy.loginfo("GPU idle dispatch: queued_peak=%d wait_for_lane_ms_per_window=%.3f "
                          "source_ms_per_window=%s", dispatch_stats["queued_peak"],
                          1000 * dispatch_stats["wait_for_lane"] / processed,
                          json.dumps({name: round(1000 * value / processed, 3)
                                      for name, value in (source_bins or {}).items()}))
        if wall_bins is not None and processed:
            wall_bins["other"] = wall - sum(wall_bins.values())
            rospy.loginfo("GPU wall ledger: %s", json.dumps({
                "mode": mode, "read": counts["read"], "processed": processed,
                "skipped": counts["skipped"], "wall_s": wall,
                "ms_per_window": {key: round(seconds * 1000 / processed, 3)
                                  for key, seconds in wall_bins.items()},
            }, sort_keys=True))
        rospy.loginfo("GPU summary: read=%d processed=%d skipped=%d detection_frames=%d wall=%.3fs "
                      "fps=%.2f compute_mean=%.2fms",
                      counts["read"], processed, counts["skipped"], detected, wall,
                      processed / max(wall, 1e-6),
                      1000 * elapsed / max(processed, 1))
        if latency:
            rospy.loginfo("GPU source-to-publish latency (ms p50/p95): %.2f/%.2f",
                          *np.percentile(latency, (50, 95)))
        if stages:
            average = np.mean(stages, axis=0)
            p95 = np.percentile(stages, 95, axis=0)
            rospy.loginfo("GPU stages (ms mean/p95): %s; slowest=%s",
                          ", ".join(f"{name}={avg:.2f}/{tail:.2f}"
                                    for name, avg, tail in zip(STAGES, average, p95)),
                          STAGES[int(np.argmax(average))])
        for name, samples in sorted(mfe_samples.items(), key=lambda row: mean(row[1]), reverse=True):
            rospy.loginfo("MFE profile: %s n=%d mean=%.3f p50=%.3f p95=%.3f ms",
                          name, len(samples), mean(samples),
                          np.percentile(samples, 50), np.percentile(samples, 95))


if __name__ == "__main__":
    main()
