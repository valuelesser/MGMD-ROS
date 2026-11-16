"""OpenCV NVDEC reader for verified constant-rate source ranges."""

from time import perf_counter

import cv2
import torch

from mgmd_onnx.cuda_frames import CudaFrame


class BgrDeviceView:
    def __init__(self, frame):
        self.frame = frame
        width, height = frame.size()
        self.__cuda_array_interface__ = {
            "version": 2, "shape": (height, width, 3), "strides": (frame.step, 3, 1),
            "typestr": "|u1", "data": (frame.cudaPtr(), False),
        }


class OpenCvCudaDecoder:
    def __init__(self, video, leading_frames):
        if leading_frames < 0:
            raise ValueError("OpenCV leading frame count cannot be negative")
        if not hasattr(cv2, "cudacodec"):
            raise RuntimeError("OpenCV must be built with cudacodec")
        torch.empty(0, device="cuda")
        self.reader = cv2.cudacodec.createVideoReader(str(video))
        if not self.reader.set(colorFormat=cv2.cudacodec.BGR):
            raise RuntimeError("Could not select OpenCV GPU BGR output")
        self.seq = 0
        for _ in range(leading_frames):
            ok, _ = self.reader.nextFrame()
            if not ok:
                raise RuntimeError("OpenCV ended before the requested source range")
        torch.cuda.current_stream().synchronize()

    def read(self, profile=None):
        started = perf_counter() if profile is not None else None
        ok, frame = self.reader.nextFrame()
        if not ok:
            if profile is not None:
                profile["decode_opencv_bgr"] += perf_counter() - started
            return None
        if frame.type() != cv2.CV_8UC3 or not frame.cudaPtr():
            raise RuntimeError("Expected a GPU BGR frame")
        image = torch.as_tensor(BgrDeviceView(frame), device="cuda")
        if image.data_ptr() != frame.cudaPtr():
            raise RuntimeError("OpenCV decoder frame copied during Torch import")
        torch.cuda.current_stream().synchronize()
        if profile is not None:
            profile["decode_opencv_bgr"] += perf_counter() - started
        decoded = CudaFrame(image, self.seq, None, None, (frame.step,), frame)
        self.seq += 1
        return decoded

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.reader = None
