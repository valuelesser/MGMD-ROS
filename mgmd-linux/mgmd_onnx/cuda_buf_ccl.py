"""Device-buffer interface to the standalone eight-connected labeler."""

import ctypes as C

import torch


class BufCcl:
    def __init__(self, path, width, height, stream):
        self.library = C.CDLL(path)
        self.label = self.library.mgmd_buf_ccl
        self.label.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p,
                               C.c_void_p, C.c_void_p]
        self.label.restype = C.c_int
        self.stream = stream
        self.labels = torch.empty((height, width), device=stream.device, dtype=torch.int32)
        self.parents = torch.empty(((height + 1) // 2, (width + 1) // 2),
                                   device=stream.device, dtype=torch.int32)

    def __call__(self, mask):
        if torch.cuda.current_stream() != self.stream:
            raise RuntimeError("BUF labeling must run on its creation stream")
        if (mask.shape != self.labels.shape or mask.dtype != torch.uint8
                or mask.device != self.labels.device or not mask.is_contiguous()):
            raise ValueError("BUF requires a contiguous CUDA uint8 mask of the configured size")
        h, w = mask.shape
        status = self.label(mask.data_ptr(), w, h, self.parents.data_ptr(),
                            self.labels.data_ptr(), self.stream.cuda_stream)
        if status:
            raise RuntimeError(f"BUF connected components failed with CUDA status {status}")
        return self.labels
