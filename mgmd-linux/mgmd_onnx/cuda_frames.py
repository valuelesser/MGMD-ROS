"""Import borrowed libav NV12 surfaces into Torch and return owned CUDA BGR frames."""

import ctypes as C
from dataclasses import dataclass
import os

import torch


class FrameView(C.Structure):
    _fields_ = (
        [("y", C.c_size_t), ("uv", C.c_size_t)]
        + [(name, C.c_int) for name in ("width", "height", "y_stride", "uv_stride",
                                       "color_space", "full_range", "time_num", "time_den")]
        + [("pts", C.c_int64)]
    )


class DevicePlane:
    def __init__(self, pointer, shape, stride):
        self.__cuda_array_interface__ = {
            "version": 2, "shape": shape, "strides": (stride, 1),
            "typestr": "|u1", "data": (pointer, False),
        }


@dataclass
class CudaFrame:
    image: torch.Tensor
    seq: int
    pts: int
    time_base: tuple
    strides: tuple


def _to_bgr(y, uv, color_space, full_range):
    # libswscale defaults to BT.601 when the source has no matrix metadata.
    if color_space == 1:
        kr, kb = .2126, .0722
    elif color_space in (2, 5, 6):
        kr, kb = .299, .114
    else:
        raise ValueError(f"Unsupported NV12 color matrix: {color_space}")
    kg = 1 - kr - kb
    u = uv[:, ::2].repeat_interleave(2, 0).repeat_interleave(2, 1).float() - 128
    v = uv[:, 1::2].repeat_interleave(2, 0).repeat_interleave(2, 1).float() - 128
    luma = y.float()
    if not full_range:
        luma = (luma - 16) * (255 / 219)
        u *= 255 / 224
        v *= 255 / 224
    b = luma + 2 * (1 - kb) * u
    g = luma - 2 * (kb * (1 - kb) * u + kr * (1 - kr) * v) / kg
    r = luma + 2 * (1 - kr) * v
    return torch.stack((b, g, r), -1).round_().clamp_(0, 255).to(torch.uint8)


class CudaDecoder:
    def __init__(self, library, video):
        self.lib = C.CDLL(os.fspath(library))
        self.lib.mgmd_decoder_open.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p)]
        self.lib.mgmd_decoder_open.restype = C.c_int
        self.lib.mgmd_decoder_next.argtypes = [C.c_void_p, C.POINTER(FrameView)]
        self.lib.mgmd_decoder_next.restype = C.c_int
        self.lib.mgmd_decoder_close.argtypes = [C.c_void_p]
        self.lib.mgmd_decoder_close.restype = None
        self.lib.mgmd_decoder_error.argtypes = [C.c_int, C.POINTER(C.c_char), C.c_size_t]
        self.lib.mgmd_decoder_error.restype = None
        self.video = video
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.handle = C.c_void_p()
        self.reset()

    def _check(self, result):
        if result < 0:
            message = C.create_string_buffer(128)
            self.lib.mgmd_decoder_error(result, message, len(message))
            raise RuntimeError(message.value.decode())
        return result

    def read(self):
        if not self.handle:
            raise RuntimeError("CUDA decoder is closed")
        view = FrameView()
        if self._check(self.lib.mgmd_decoder_next(self.handle, C.byref(view))) == 0:
            return None
        y = torch.as_tensor(DevicePlane(view.y, (view.height, view.width), view.y_stride),
                            device=self.device)
        uv = torch.as_tensor(DevicePlane(view.uv, (view.height // 2, view.width), view.uv_stride),
                             device=self.device)
        if y.data_ptr() != view.y or uv.data_ptr() != view.uv:
            raise RuntimeError("NV12 plane import copied the decoder surface")
        try:
            image = _to_bgr(y, uv, view.color_space, view.full_range)
        finally:
            # The next libav read may reuse the borrowed NV12 surface.
            torch.cuda.current_stream(self.device).synchronize()
        result = CudaFrame(image, self.seq, view.pts, (view.time_num, view.time_den),
                           (view.y_stride, view.uv_stride))
        self.seq += 1
        return result

    def reset(self):
        self.close()
        device = str(self.device.index).encode()
        with torch.cuda.device(self.device):
            # Ensure the calling thread has Torch's CUDA context before libav borrows it.
            torch.empty(1, device=self.device)
            self._check(self.lib.mgmd_decoder_open(os.fsencode(self.video), device,
                                                   C.byref(self.handle)))
        self.seq = 0

    def close(self):
        if self.handle:
            self.lib.mgmd_decoder_close(self.handle)
            self.handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
