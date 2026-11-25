"""Experimental CUDA interfaces for bounded MFE output and LAD postprocessing."""
import ctypes as C

import torch


class ResidentOps:
    def __init__(self, library, split_head=False):
        self.split_head = split_head
        self.lib = C.CDLL(library)
        ptr = C.c_void_p
        self.lib.mgmd_host_pointer.argtypes = [C.POINTER(ptr), ptr]
        self.lib.mgmd_regions.argtypes = [ptr, C.c_int, C.c_int, ptr, C.c_int, ptr]
        self.lib.mgmd_select_box.argtypes = [ptr, ptr, C.c_int, C.c_float, ptr, ptr]

    def mapped_text(self, tensor):
        pointer = C.c_void_p()
        if not tensor.is_pinned() or self.lib.mgmd_host_pointer(C.byref(pointer), tensor.data_ptr()):
            raise RuntimeError("HUD layout needs mapped pinned host memory")
        return pointer.value

    def regions(self, labels, cfg, capacity):
        if (labels.ndim != 2 or not labels.is_cuda or labels.dtype != torch.int32
                or not labels.is_contiguous() or capacity < 1):
            raise ValueError("Expected packed CUDA BUF labels and positive capacity")
        h, w = labels.shape
        size = (h + 1) // 2 * ((w + 1) // 2) + 1
        stats = labels.new_empty((size, 6))
        stream = torch.cuda.current_stream().cuda_stream
        if self.lib.mgmd_regions(labels.data_ptr(), w, h, stats.data_ptr(), size, stream):
            raise RuntimeError("CUDA region extraction failed")
        area, first, left, top, right, bottom = stats.unbind(1)
        keep = area >= max(1, cfg.official_area_min or cfg.min_area)
        if cfg.max_area:
            keep &= area <= cfg.max_area
        if cfg.aspect_filter:
            width, height = right - left + 1, bottom - top + 1
            keep &= (width >= cfg.min_aspect * height) & (width <= cfg.max_aspect * height)
        count = keep.sum(dtype=torch.int32).reshape(1)
        _, ids = first.masked_fill(~keep, 2147483647).topk(min(capacity, size), largest=False)
        boxes = torch.stack((left[ids], top[ids], right[ids] + 1, bottom[ids] + 1), 1).double()
        scores = ((right[ids] - left[ids] + 1) * (bottom[ids] - top[ids] + 1)).double()
        return boxes, scores, count

    def select_box(self, raw, threshold):
        if (raw.shape != (1, 6300, 6) or raw.dtype != torch.float32
                or not raw.is_cuda or not raw.is_contiguous()):
            raise ValueError("Expected the 320px single-class LAD output")
        scores = raw[0, :, 4] * raw[0, :, 5]
        order = scores.argsort(descending=True, stable=True)
        out = raw.new_zeros((1, 5))
        if self.lib.mgmd_select_box(raw.data_ptr(), order.data_ptr(), len(order), threshold,
                                    out.data_ptr(), torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("CUDA NMS failed")
        return out
