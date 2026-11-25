"""Device-resident TF backend for the ROS recommended FastSort configuration."""

import ctypes as C
from dataclasses import dataclass
import os

import torch


_WORK_FIELDS = ("ctl", "tracks", "predictions", "cost", "u", "v", "dist",
                "row_map", "col_map", "match", "owner", "path", "row_to_col",
                "col_to_row", "remaining", "seen_rows", "seen_cols")


class _Workspace(C.Structure):
    _fields_ = ([(name, C.c_void_p) for name in _WORK_FIELDS] +
                [("track_capacity", C.c_int), ("det_capacity", C.c_int)])


_RESULT_FIELDS = ("boxes", "ids", "details", "lifecycle", "assigned_ids", "count", "status")


class _Result(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in _RESULT_FIELDS]


@dataclass
class TfResult:
    boxes: torch.Tensor
    ids: torch.Tensor
    details: torch.Tensor
    lifecycle: torch.Tensor
    assigned_ids: torch.Tensor
    count: torch.Tensor
    status: torch.Tensor


class CudaTf:
    """Run the recommended TF profile on one CUDA device and stream.

    The output is a device buffer and valid count. Status is sticky until reset:
    1 means capacity overflow and 2 means the exact assignment failed.
    """

    def __init__(self, library, track_capacity=1024, det_capacity=512):
        if min(track_capacity, det_capacity) < 1 or track_capacity * det_capacity > 2**30:
            raise ValueError("Invalid TF capacities")
        self.lib = C.CDLL(os.fspath(library))
        self.lib.mgmd_tf_track_size.restype = C.c_size_t
        self.lib.mgmd_tf_control_size.restype = C.c_size_t
        self.lib.mgmd_tf_reset.argtypes = [_Workspace, C.c_void_p]
        self.lib.mgmd_tf_reset.restype = C.c_int
        self.lib.mgmd_tf_step.argtypes = [_Workspace, C.c_void_p, C.c_void_p, C.c_void_p,
                                         C.c_int, C.c_int64, _Result, C.c_void_p]
        self.lib.mgmd_tf_step.restype = C.c_int
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.stream = torch.cuda.current_stream(self.device)
        self.track_capacity, self.det_capacity = track_capacity, det_capacity
        width = max(track_capacity, det_capacity)
        d, i, b = torch.float64, torch.int32, torch.uint8
        self._buffers = {
            "ctl": self._empty(self.lib.mgmd_tf_control_size(), b),
            "tracks": self._empty(track_capacity * self.lib.mgmd_tf_track_size(), b),
            "predictions": self._empty((track_capacity, 4), d),
            "cost": self._empty((track_capacity, det_capacity), d),
            "row_map": self._empty(track_capacity, i),
            "match": self._empty(track_capacity, i),
            "col_map": self._empty(det_capacity, i),
            "owner": self._empty(det_capacity, i),
        }
        for names, dtype in ((("u", "v", "dist"), d),
                             (("path", "row_to_col", "col_to_row", "remaining"), i),
                             (("seen_rows", "seen_cols"), b)):
            for name in names:
                self._buffers[name] = self._empty(width, dtype)
        self._workspace = _Workspace(
            *(self._buffers[name].data_ptr() for name in _WORK_FIELDS),
            track_capacity, det_capacity)
        self.reset()

    def _empty(self, shape, dtype):
        if isinstance(shape, int):
            shape = (shape,)
        return torch.empty(shape, dtype=dtype, device=self.device)

    def _stream_ptr(self):
        current = torch.cuda.current_stream(self.device)
        if current.cuda_stream != self.stream.cuda_stream:
            raise RuntimeError("TF must be called on its creation stream")
        return current.cuda_stream

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"CUDA TF launch failed with CUDA error {code}")

    def reset(self):
        self._check(self.lib.mgmd_tf_reset(self._workspace, self._stream_ptr()))

    def step(self, boxes, scores, frame, count=None):
        if (boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != boxes.shape[:1]
                or boxes.shape[0] > self.det_capacity):
            raise ValueError("Expected bounded [N,4] boxes and [N] scores")
        for tensor in (boxes, scores):
            if tensor.device != self.device or tensor.dtype != torch.float64 or not tensor.is_contiguous():
                raise ValueError("Expected contiguous CUDA float64 boxes and scores on the tracker device")
        if count is not None and (count.device != self.device or count.dtype != torch.int32
                                  or count.numel() != 1 or not count.is_contiguous()):
            raise ValueError("Expected a CUDA int32 valid count on the tracker device")
        n = self.track_capacity
        result = TfResult(self._empty((n, 4), torch.float64), self._empty(n, torch.int64),
                          self._empty((n, 4), torch.float64), self._empty((n, 3), torch.int32),
                          self._empty(boxes.shape[0], torch.int64),
                          self._empty(1, torch.int32), self._empty(1, torch.int32))
        pointers = _Result(*(getattr(result, name).data_ptr() for name in _RESULT_FIELDS))
        self._check(self.lib.mgmd_tf_step(self._workspace, boxes.data_ptr(), scores.data_ptr(),
                                         count.data_ptr() if count is not None else None,
                                         boxes.shape[0], frame, pointers, self._stream_ptr()))
        return result
