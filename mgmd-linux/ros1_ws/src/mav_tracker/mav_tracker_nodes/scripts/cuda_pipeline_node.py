#!/usr/bin/env python3
"""Publish detections and an optional preview from the GPU file pipeline."""

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
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
from mgmd_onnx.cuda_four_stream import FourStream, four_stage_overlap
from mgmd_onnx.cuda_pipeline import CudaPipeline


STAGES = ("MFE", "TF", "LAC", "LAD")


@torch.inference_mode()
def extract_mfe(pipe, stream, window, profile, reuse_gray):
    with torch.cuda.stream(stream):
        start, ready = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for frame in window:
            frame.image.record_stream(stream)
        frame_ids = tuple(frame.seq for frame in window) if reuse_gray else None
        boxes, scores = pipe.mfe.candidates(*(frame.image for frame in window), profile,
                                          frame_ids=frame_ids)
        ready.record()
        return boxes, scores, start, ready


def ready_windows(decoder, first, limit, started, rate, realtime, mode, pipe, stream,
                  executor, profile, counts, wall_bins, reuse_gray, decode_lock=None):
    frames = deque((first,), maxlen=3)
    pending = None
    while not rospy.is_shutdown() and (limit == 0 or counts["read"] < limit):
        with decode_lock or nullcontext():
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
        window = tuple(frames)
        if mode in ("legacy", "four_stream"):
            yield window, None, marks
            continue
        job = executor.submit(extract_mfe, pipe, stream, window, marks, reuse_gray)
        if mode == "serial":
            yield window, job, marks
        else:
            if pending is not None:
                yield pending
            pending = window, job, marks
    if pending is not None and not rospy.is_shutdown():
        yield pending


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


def add_hud(image, playback, processed, skipped, recent, now):
    fps = (len(recent) - 1) / max(now - recent[0][0], 1e-6)
    stage = [mean(samples) for samples in zip(*(row[1] for row in recent))]
    slowest = int(np.argmax(stage))
    text = (
        f"Processing {fps:.1f} FPS   {playback}   Processed {processed}   Skipped {skipped}",
        "   ".join(f"{name} {ms:.1f} ms" for name, ms in zip(STAGES, stage))
        + f"   |   Slowest: {STAGES[slowest]}",
    )
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, .85, 2
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in text)
    cv2.rectangle(image, (16, 16), (min(image.shape[1] - 1, width + 40), 95), (18, 18, 18), -1)
    for y, line in zip((48, 80), text):
        cv2.putText(image, line, (28, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


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
    four_stream_depth = get("~four_stream_depth", 4)
    reuse_gray = get("~reuse_gray", False)
    if mode not in ("legacy", "serial", "overlap", "four_stream"):
        raise ValueError("pipeline_mode must be legacy, serial, overlap or four_stream")
    if reuse_gray and mode == "legacy":
        raise ValueError("reuse_gray requires a non-legacy pipeline mode")
    if mode == "four_stream" and four_stream_depth not in (2, 3, 4):
        raise ValueError("four_stream_depth must be 2, 3 or 4")
    playback = f"Source {source_rate:g} Hz" if realtime else "Full-frame playback"
    limit = get("~limit", 0)
    decoder_backend = get("~decoder_backend", "libav")
    if decoder_backend not in ("libav", "opencv"):
        raise ValueError("decoder_backend must be libav or opencv")
    detections = rospy.Publisher("/mav/detections", DetectionArray, queue_size=10)
    debug = (rospy.Publisher("/mav/debug_image", Image, queue_size=1)
             if get("~preview", True) else None)
    counts = {"read": 0, "skipped": 0}
    processed = detected = 0
    recent = deque()
    stages = []
    latency = []
    mfe_samples = defaultdict(list)
    split_samples = []
    elapsed = 0.
    if decoder_backend == "opencv":
        from mgmd_onnx.cuda_opencv_decoder import OpenCvCudaDecoder
        source = OpenCvCudaDecoder(video, get("~decoder_leading_frames", 0))
    else:
        source = CudaDecoder(get("~decoder_library"), video)
    with torch.inference_mode(), source as decoder, ExitStack() as stack:
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        first = decoder.read()
        if first is None:
            raise RuntimeError("Video has no decodable frames")
        counts["read"] = 1
        height, width = first.image.shape[:2]
        if min(height, width) < 320:
            raise ValueError("GPU LAD requires both dimensions to be at least 320")
        mfe_stream = torch.cuda.Stream() if mode != "legacy" else None
        post_stream = torch.cuda.Stream() if mode != "legacy" else None
        lad_stream = torch.cuda.Stream() if mode == "four_stream" else None
        with torch.cuda.stream(post_stream):
            pipe = CudaPipeline(get("~tf_library"), width, height, warp_backend=warp_backend,
                                mfe_stream=mfe_stream, model_stream=post_stream, lad_stream=lad_stream)
        four = (FourStream(pipe, (mfe_stream, torch.cuda.Stream(), post_stream, lad_stream),
                           stack, reuse_gray, profile=wall_bins is not None)
                if lad_stream is not None else None)
        torch.cuda.synchronize()
        rospy.loginfo("GPU path initialized: NVDEC, MFE, TF, LAC, LAD (%dx%d)", width, height)
        rospy.loginfo("MFE warp backend: %s; pipeline mode: %s", warp_backend, mode)
        rospy.loginfo("GPU decoder backend: %s", decoder_backend)
        started = time.perf_counter()
        windows = ready_windows(
                decoder, first, limit, started, source_rate, realtime, mode, pipe,
                mfe_stream, executor, profile_mfe, counts, wall_bins, reuse_gray,
                pipe.mfe.default_stream_lock if four else None)
        if four:
            windows = four.windows(windows, four_stream_depth)
        for (previous, center, following), job, mfe_timing in windows:
            begin = time.perf_counter()
            if four:
                boxes, status, candidates, stage_ms, split = job.result()
                split_samples.append(split)
            elif job is None:
                boxes, status, candidates, timing = pipe.step(
                    previous.image, center.image, following.image, center.seq, mfe_timing)
            else:
                candidates_gpu, scores, mfe_start, ready = job.result()
                mfe_done = time.perf_counter()
                if wall_bins is not None:
                    wall_bins["wait_mfe_worker"] += mfe_done - begin
                post_stream.wait_event(ready)
                candidates_gpu.record_stream(post_stream)
                scores.record_stream(post_stream)
                center.image.record_stream(post_stream)
                with torch.cuda.stream(post_stream):
                    timing = [mfe_start] + [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    timing[1].record()
                    boxes, status, candidates, _ = pipe.finish(
                        center.image, candidates_gpu, scores, center.seq, timing)
                # Complete the post stream before the ROS output boundary reads its buffers.
                timing[-1].synchronize()
            compute_done = time.perf_counter()
            if wall_bins is not None:
                if four:
                    wall_bins["wait_four_stream"] += compute_done - begin
                elif job is None:
                    wall_bins["legacy_compute"] += compute_done - begin
                else:
                    wall_bins["tf_lac_lad_and_gpu_wait"] += compute_done - mfe_done
            # Detection metadata and the optional display image cross the ROS boundary here.
            error = status.item()
            if error:
                raise RuntimeError("CUDA TF failed with status {}".format(error))
            data = boxes.tolist()
            if wall_bins is not None:
                wall_bins["detection_readback"] += time.perf_counter() - compute_done
            if not four:
                stage_ms = [a.elapsed_time(b) for a, b in zip(timing, timing[1:])]
                if job is not None:
                    stage_ms[0] = mfe_start.elapsed_time(ready)
            if mfe_timing:
                subtotals = defaultdict(float)
                for (_, start), (name, end) in zip(mfe_timing, mfe_timing[1:]):
                    if name != "start":
                        subtotals[name] += start.elapsed_time(end)
                for name, total in subtotals.items():
                    mfe_samples[name].append(total)
            now = time.perf_counter()
            elapsed += now - begin
            stages.append(stage_ms)
            recent.append((now, stage_ms))
            while len(recent) > 1 and recent[1][0] < now - 2:
                recent.popleft()
            header = Header(seq=center.seq, stamp=rospy.Time.now(), frame_id="camera")
            publish_started = time.perf_counter() if wall_bins is not None else None
            detections.publish(build_detection_array(
                header, width, height, [row[:4] for row in data], [row[4] for row in data],
                ["lad"] * len(data), candidates, *stage_ms,
                track_ids=[int(row[5]) for row in data]))
            if wall_bins is not None:
                preview_started = time.perf_counter()
                wall_bins["detection_publish"] += preview_started - publish_started
            if debug is not None:
                preview = preview_image(center.image, boxes)
                if wall_bins is not None:
                    download_started = time.perf_counter()
                    wall_bins["preview_overlay_enqueue"] += download_started - preview_started
                image = preview.cpu().numpy()
                if wall_bins is not None:
                    hud_started = time.perf_counter()
                    wall_bins["preview_download_and_gpu_wait"] += hud_started - download_started
                add_hud(image, playback, processed + 1, counts["skipped"], recent, now)
                payload = image.tobytes()
                if wall_bins is not None:
                    image_publish_started = time.perf_counter()
                    wall_bins["preview_hud_and_pack"] += image_publish_started - hud_started
                debug.publish(Image(header=header, height=height, width=width, encoding="bgr8",
                                    step=width * 3, data=payload))
                if wall_bins is not None:
                    wall_bins["preview_publish"] += time.perf_counter() - image_publish_started
            processed += 1
            detected += bool(data)
            if realtime:
                latency.append(1000 * (time.perf_counter() - started - center.seq / source_rate))
            if processed % 50 == 0:
                rospy.loginfo("GPU progress: %d frames, %.2f fps", processed,
                              processed / (time.perf_counter() - started))
        wall = time.perf_counter() - started
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
        if split_samples:
            rospy.loginfo("GPU MFE split mean/p95 ms: align=%.2f/%.2f candidates=%.2f/%.2f",
                          *np.stack((np.mean(split_samples, axis=0),
                                     np.percentile(split_samples, 95, axis=0)), axis=1).flatten())
        if four and four.spans:
            rospy.loginfo("GPU four-stage event overlap: %.2f ms", four_stage_overlap(four.spans))
        for name, samples in sorted(mfe_samples.items(), key=lambda row: mean(row[1]), reverse=True):
            rospy.loginfo("MFE profile: %s n=%d mean=%.3f p50=%.3f p95=%.3f ms",
                          name, len(samples), mean(samples),
                          np.percentile(samples, 50), np.percentile(samples, 95))


if __name__ == "__main__":
    main()
