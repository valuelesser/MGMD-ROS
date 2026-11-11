#!/usr/bin/env python3
"""Publish detections and an optional preview from the GPU file pipeline."""

from collections import deque
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


STAGES = ("MFE", "TF", "LAC", "LAD")


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
    playback = f"Source {source_rate:g} Hz" if realtime else "Full-frame playback"
    limit = get("~limit", 0)
    detections = rospy.Publisher("/mav/detections", DetectionArray, queue_size=10)
    debug = (rospy.Publisher("/mav/debug_image", Image, queue_size=1)
             if get("~preview", True) else None)
    frames = deque(maxlen=3)
    read = processed = detected = skipped = 0
    recent = deque()
    stages = []
    elapsed = 0.
    with torch.inference_mode(), CudaDecoder(get("~decoder_library"), video) as decoder:
        pipe = None
        started = time.perf_counter()
        while not rospy.is_shutdown() and (limit == 0 or read < limit):
            frame = decoder.read()
            if frame is None:
                break
            read += 1
            frames.append(frame)
            if pipe is None:
                height, width = frame.image.shape[:2]
                if min(height, width) < 320:
                    raise ValueError("GPU LAD requires both dimensions to be at least 320")
                pipe = CudaPipeline(get("~tf_library"), width, height)
                rospy.loginfo("GPU path initialized: NVDEC, MFE, TF, LAC, LAD (%dx%d)", width, height)
                started = time.perf_counter()
            if len(frames) < 3:
                continue
            if realtime:
                target = int((time.perf_counter() - started) * source_rate)
                if frame.seq < target:
                    skipped += 1
                    continue
                delay = started + frame.seq / source_rate - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            previous, center, following = frames
            begin = time.perf_counter()
            boxes, status, candidates, timing = pipe.step(
                previous.image, center.image, following.image, center.seq)
            # Detection metadata and the optional display image cross the ROS boundary here.
            error = status.item()
            if error:
                raise RuntimeError("CUDA TF failed with status {}".format(error))
            data = boxes.tolist()
            stage_ms = [a.elapsed_time(b) for a, b in zip(timing, timing[1:])]
            now = time.perf_counter()
            elapsed += now - begin
            stages.append(stage_ms)
            recent.append((now, stage_ms))
            while len(recent) > 1 and recent[1][0] < now - 2:
                recent.popleft()
            header = Header(seq=center.seq, stamp=rospy.Time.now(), frame_id="camera")
            detections.publish(build_detection_array(
                header, width, height, [row[:4] for row in data], [row[4] for row in data],
                ["lad"] * len(data), candidates, *stage_ms,
                track_ids=[int(row[5]) for row in data]))
            if debug is not None:
                image = preview_image(center.image, boxes).cpu().numpy()
                add_hud(image, playback, processed + 1, skipped, recent, now)
                debug.publish(Image(header=header, height=height, width=width, encoding="bgr8",
                                    step=width * 3, data=image.tobytes()))
            processed += 1
            detected += bool(data)
            if processed % 50 == 0:
                rospy.loginfo("GPU progress: %d frames, %.2f fps", processed,
                              processed / (time.perf_counter() - started))
        wall = time.perf_counter() - started
        rospy.loginfo("GPU summary: read=%d processed=%d skipped=%d detection_frames=%d wall=%.3fs "
                      "fps=%.2f compute_mean=%.2fms",
                      read, processed, skipped, detected, wall, processed / max(wall, 1e-6),
                      1000 * elapsed / max(processed, 1))
        if stages:
            average = np.mean(stages, axis=0)
            p95 = np.percentile(stages, 95, axis=0)
            rospy.loginfo("GPU stages (ms mean/p95): %s; slowest=%s",
                          ", ".join(f"{name}={avg:.2f}/{tail:.2f}"
                                    for name, avg, tail in zip(STAGES, average, p95)),
                          STAGES[int(np.argmax(average))])


if __name__ == "__main__":
    main()
