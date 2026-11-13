"""NPP 8-connected labeling on an application-owned CUDA stream."""

import ctypes as C

import torch


class _Size(C.Structure):
    _fields_ = [("width", C.c_int), ("height", C.c_int)]


class _Context(C.Structure):
    _fields_ = [("stream", C.c_void_p), ("device", C.c_int), ("sm_count", C.c_int),
                ("threads_per_sm", C.c_int), ("threads_per_block", C.c_int),
                ("shared_memory", C.c_size_t), ("major", C.c_int), ("minor", C.c_int),
                ("flags", C.c_uint), ("reserved", C.c_int)]


def _check(status):
    if status != 0:
        raise RuntimeError(f"NPP connected components failed with status {status}")


class NppCcl:
    def __init__(self, width, height, stream):
        core = C.CDLL("libnppc.so.12")
        image = C.CDLL("libnppif.so.12")
        core.nppGetStreamContext.argtypes = [C.POINTER(_Context)]
        self.context = _Context()
        _check(core.nppGetStreamContext(C.byref(self.context)))
        self.context.stream = stream.cuda_stream
        # torch.cuda.Stream creates a non-blocking stream.
        if stream.cuda_stream:
            self.context.flags = 1
        self.stream = stream
        self.size = _Size(width, height)
        size = C.c_int()
        get_size = image.nppiLabelMarkersUFGetBufferSize_32u_C1R
        get_size.argtypes = [_Size, C.POINTER(C.c_int)]
        _check(get_size(self.size, C.byref(size)))
        self.scratch = torch.empty(size.value, dtype=torch.uint8, device=stream.device)
        self.label = image.nppiLabelMarkersUF_8u32u_C1R_Ctx
        self.label.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int, _Size,
                               C.c_int, C.c_void_p, _Context]

    def __call__(self, pixels):
        if torch.cuda.current_stream() != self.stream:
            raise RuntimeError("NPP labeling must run on its creation stream")
        pixels = pixels.contiguous()
        labels = torch.empty_like(pixels, dtype=torch.int32)
        _check(self.label(pixels.data_ptr(), pixels.stride(0), labels.data_ptr(),
                          labels.stride(0) * labels.element_size(), self.size,
                          0, self.scratch.data_ptr(), self.context))
        return labels
