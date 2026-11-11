#!/usr/bin/env python3
"""Publish detections and an optional preview from the GPU file pipeline."""

from collections import deque
import time

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import torch

from detector_node import build_detection_array
from mav_tracker_msgs.msg import DetectionArray
from mgmd_onnx.cuda_frames import CudaDecoder
from mgmd_onnx.cuda_pipeline import CudaPipeline


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


def main():
    rospy.init_node("cuda_pipeline")
    if not torch.cuda.is_available():
        raise RuntimeError("track_gpu.launch requires a CUDA device")
    get = rospy.get_param
    video = get("~video")
    rate = rospy.Rate(get("~rate", 30))
    limit = get("~limit", 0)
    detections = rospy.Publisher("/mav/detections", DetectionArray, queue_size=10)
    debug = (rospy.Publisher("/mav/debug_image", Image, queue_size=1)
             if get("~preview", True) else None)
    frames = deque(maxlen=3)
    read = processed = detected = 0
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
            elapsed += time.perf_counter() - begin
            header = Header(seq=center.seq, stamp=rospy.Time.now(), frame_id="camera")
            detections.publish(build_detection_array(
                header, width, height, [row[:4] for row in data], [row[4] for row in data],
                ["lad"] * len(data), candidates, *stage_ms,
                track_ids=[int(row[5]) for row in data]))
            if debug is not None:
                image = preview_image(center.image, boxes)
                debug.publish(Image(header=header, height=height, width=width, encoding="bgr8",
                                    step=width * 3, data=image.cpu().numpy().tobytes()))
            processed += 1
            detected += bool(data)
            if processed % 50 == 0:
                rospy.loginfo("GPU progress: %d frames, %.2f fps", processed,
                              processed / (time.perf_counter() - started))
            rate.sleep()
        wall = time.perf_counter() - started
        rospy.loginfo("GPU summary: read=%d processed=%d detection_frames=%d wall=%.3fs "
                      "fps=%.2f compute_mean=%.2fms",
                      read, processed, detected, wall, processed / max(wall, 1e-6),
                      1000 * elapsed / max(processed, 1))


if __name__ == "__main__":
    main()
