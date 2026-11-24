"""Attribute interposed CUDA runtime calls to the innermost active stage."""
import ctypes
from functools import wraps
from threading import local

from mgmd_onnx.cuda_mfe import CudaMfe
from mgmd_onnx.cuda_opencv_decoder import OpenCvCudaDecoder
from mgmd_onnx.cuda_parity import ParityPipeline
from mgmd_onnx.cuda_pipeline import CudaPipeline
from mgmd_onnx.cuda_preview import CudaPreview


def install():
    state = local()
    scope = ctypes.CDLL(None).audit_copy_scope
    scope.argtypes = [ctypes.c_char_p]

    def instrument(cls, method, enabled):
        original = getattr(cls, method)

        @wraps(original)
        def wrapped(self, *args, **kwargs):
            previous = getattr(state, "name", None)
            path = "resident" if enabled(self) else "base"
            state.name = f"{path}/{method}".encode()
            scope(state.name)
            try:
                return original(self, *args, **kwargs)
            finally:
                state.name = previous
                scope(previous)

        setattr(cls, method, wrapped)

    instrument(OpenCvCudaDecoder, "read", lambda c: c.decode_stream is not None)
    for name in ("_boxes", "_estimate", "_homography", "_warp"):
        instrument(CudaMfe, name, lambda m: m.resident_regions is not None)
    for name in ("track", "select_lac", "_refine", "refine_tracks"):
        instrument(CudaPipeline, name, lambda p: p.resident_ops is not None)
    instrument(ParityPipeline, "detect", lambda p: p.pipes[0].resident_ops is not None)
    instrument(CudaPreview, "draw", lambda p: p.text is None)
