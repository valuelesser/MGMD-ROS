#!/usr/bin/env python3
"""Run single-stream GPU IMU detection on timestamped ROS image and IMU topics."""

from collections import deque
from threading import Condition
from time import perf_counter

import numpy as np
import rospy
import torch
import yaml
from sensor_msgs.msg import Image, Imu

from detector_node import build_detection_array
from mav_tracker_msgs.msg import DetectionArray
from mgmd_onnx.cuda_buf_ccl import BufCcl
from mgmd_onnx.cuda_imu import camera_warps
from mgmd_onnx.cuda_pipeline import CudaPipeline
from mgmd_onnx.cuda_preview import CudaPreview


class TopicWindows:
    def __init__(self):
        self.condition = Condition()
        self.images = deque(maxlen=120)
        self.imu = deque(maxlen=512)
        self.dropped = 0
        self.sequence = 0

    def image(self, msg):
        with self.condition:
            self.dropped += len(self.images) == self.images.maxlen
            if not self.images or msg.header.stamp > self.images[-1][1].header.stamp:
                self.images.append((self.sequence, msg))
                self.sequence += 1
            self.condition.notify()

    def attitude(self, msg):
        with self.condition:
            t = msg.header.stamp.to_nsec()
            if not self.imu or t > self.imu[-1][0]:
                q = msg.orientation
                self.imu.append((t, (q.x, q.y, q.z, q.w)))
            self.condition.notify()

    def next(self):
        with self.condition:
            while not rospy.is_shutdown():
                if len(self.images) < 3 or len(self.imu) < 2:
                    self.condition.wait(.1)
                    continue
                frames = tuple(self.images[i] for i in range(3))
                first, last = (frames[i][1].header.stamp.to_nsec() for i in (0, 2))
                if self.imu[0][0] > first:
                    self.images.popleft()
                    self.dropped += 1
                    continue
                if self.imu[-1][0] <= last:
                    self.condition.wait(.1)
                    continue
                self.images.popleft()
                while len(self.imu) > 2 and self.imu[1][0] <= first:
                    self.imu.popleft()
                return frames, tuple(self.imu)
            return None


def as_cuda_image(msg, width, height):
    if (msg.width, msg.height, msg.encoding) != (width, height, "bgr8"):
        raise ValueError("ROS image must be bgr8 and match the calibration dimensions")
    data = np.frombuffer(msg.data, np.uint8).reshape(height, msg.step)
    return torch.as_tensor(data[:, :width*3].reshape(height, width, 3).copy(), device="cuda")


def warmup(pipe, preview, intrinsic, rotation, width, height):
    image = torch.zeros((height, width, 3), device="cuda", dtype=torch.uint8)
    t = torch.arange(4, device="cuda", dtype=torch.int64) * 10_000_000
    q = torch.tensor([[0., 0., 0., 1.]] * 4, device="cuda", dtype=torch.float64)
    warp, _ = camera_warps(t[:3], t, q, intrinsic, rotation)
    pipe.lac.run(torch.zeros((1, 3, 32, 32), device="cuda"))
    pipe.lad.run(torch.zeros((1, 3, 320, 320), device="cuda"))
    boxes, _, _, _ = pipe.step(image, image, image, 1, frame_ids=(0, 1, 2), alignment=warp[0])
    preview.draw(image, boxes, ["GPU IMU"])
    preview.download()
    pipe.tf.reset()
    pipe.mfe.clear_gray_cache()


@torch.inference_mode()
def main():
    rospy.init_node("imu_gpu")
    get = rospy.get_param
    camera = get("~camera", 1)
    with open(get("~calibration"), encoding="utf-8") as handle:
        cal = yaml.safe_load(handle)
    width, height = int(cal["image_width"]), int(cal["image_height"])
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        pipe = CudaPipeline(get("~tf_library", "/tmp/mgmd-tf-g3.so"), width, height,
                            warp_backend="opencv", mfe_stream=stream, model_stream=stream,
                            async_lad=True)
        pipe.mfe.buf_ccl = BufCcl(get("~buf_library", "/tmp/mgmd-buf-ccl.so"), width, height, stream)
        preview = CudaPreview(get("~preview_library", "/tmp/mgmd-cuda-preview.so"),
                              torch.empty((height, width, 3), device="cuda", dtype=torch.uint8))
        intrinsic = torch.tensor(cal[f"camera_matrix_{camera:02d}"], device="cuda", dtype=torch.float64).reshape(3, 3)
        rotation = torch.tensor(cal["uavs"][camera-1]["T_bc"], device="cuda", dtype=torch.float64).reshape(4, 4)[:3, :3]
        warmup(pipe, preview, intrinsic, rotation, width, height)
        stream.synchronize()
        input_queue = TopicWindows()
        detections = rospy.Publisher("/mav/detections", DetectionArray, queue_size=10)
        debug = rospy.Publisher("/mav/debug_image", Image, queue_size=1)
        subscribers = (
            rospy.Subscriber(get("~image_topic", f"/usb_cam{camera}/image_raw"), Image,
                             input_queue.image, queue_size=30, buff_size=2**24),
            rospy.Subscriber(get("~imu_topic", "/mavros/imu/data"), Imu,
                             input_queue.attitude, queue_size=250),
        )
        images, recent = {}, deque()
        processed = positive = rejected = 0
        rospy.loginfo("IMU GPU ready camera=%d %dx%d", camera, width, height)
        while not rospy.is_shutdown():
            item = input_queue.next()
            if item is None:
                break
            frames, imu = item
            ids, messages = zip(*frames)
            ts = [msg.header.stamp.to_nsec() for msg in messages]
            if not all(15_000_000 <= b-a <= 85_000_000 for a, b in zip(ts, ts[1:])):
                rejected += 1
                continue
            warps, covered = camera_warps(
                torch.tensor(ts, device="cuda"), torch.tensor([t for t, _ in imu], device="cuda"),
                torch.tensor([q for _, q in imu], device="cuda", dtype=torch.float64), intrinsic, rotation)
            if not covered.item():
                rejected += 1
                continue
            images = {seq: images[seq] if seq in images else as_cuda_image(msg, width, height)
                      for seq, msg in frames}
            boxes, status, candidates, marks = pipe.step(*(images[seq] for seq in ids), ids[1],
                                                         frame_ids=ids, alignment=warps[0])
            marks[-1].synchronize()
            if status.item():
                raise RuntimeError(f"CUDA TF failed: {status.item()}")
            times = [a.elapsed_time(b) for a, b in zip(marks, marks[1:])]
            rows = boxes.tolist()
            header = messages[1].header
            detections.publish(build_detection_array(header, width, height, [row[:4] for row in rows],
                                                      [row[4] for row in rows], ["lad"] * len(rows),
                                                      candidates, *times, track_ids=[int(row[5]) for row in rows]))
            now = perf_counter()
            recent.append(now)
            while len(recent) > 2 and now - recent[0] > 2:
                recent.popleft()
            fps = (len(recent)-1) / max(now-recent[0], 1e-6)
            preview.draw(images[ids[1]], boxes, [f"GPU IMU  {fps:.1f} FPS  #{ids[1]}  Boxes {len(rows)}",
                         "ms: " + "  ".join(f"{name} {value:.1f}" for name, value in zip(("MFE", "TF", "LAC", "LAD"), times))])
            image = preview.download()
            debug.publish(Image(header=header, height=height, width=width, encoding="bgr8",
                                step=width*3, data=image.tobytes()))
            processed += 1
            positive += bool(rows)
            if processed % 100 == 0:
                rospy.loginfo("IMU ROS processed=%d positive=%d queue_dropped=%d rejected=%d fps=%.2f",
                              processed, positive, input_queue.dropped, rejected, fps)
        for subscriber in subscribers:
            subscriber.unregister()


if __name__ == "__main__":
    main()
