"""CUDA BGR preview compositor; only the ROS transport needs host pixels."""

import ctypes as C

import cv2
import numpy as np
import torch


class CudaPreview:
    glyph_width, line_height, text_columns, max_lines = 24, 28, 128, 4

    def __init__(self, library, sample, resident_ops=None):
        self.library = C.CDLL(library)
        self.render = self.library.mgmd_render_preview_hud
        self.render.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_int, C.c_int,
                                C.c_void_p, C.c_int, C.c_void_p, C.c_void_p,
                                C.c_int, C.c_int, C.c_void_p]
        self.render.restype = C.c_int
        self.stream = torch.cuda.current_stream(sample.device)
        atlas, self.advances = self.build_atlas()
        self.atlas = torch.from_numpy(atlas).to(sample.device)
        self.host_text = torch.empty((self.max_lines, 2 * self.text_columns + 1), dtype=torch.int16,
                                     pin_memory=True)
        self.text_view = self.host_text.numpy()
        self.text = None if resident_ops else torch.empty_like(self.host_text, device=sample.device)
        self.layout_ptr = resident_ops.mapped_text(self.host_text) if resident_ops else self.text.data_ptr()
        self.image = torch.empty(sample.shape, dtype=torch.uint8, device=sample.device)
        self.host_image = torch.empty(sample.shape, dtype=torch.uint8, pin_memory=True)

    def build_atlas(self):
        # Supersample once at startup. Each glyph stores white ink and black outline coverage.
        factor = 4
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = .68 * factor
        size = self.glyph_width, self.line_height
        atlas = np.zeros((128, self.line_height, self.glyph_width, 2), dtype=np.uint8)
        advances = np.zeros(128, dtype=np.int16)
        for char in range(32, 127):
            glyph = chr(char)
            width = cv2.getTextSize(glyph, font, scale, 0)[0][0]
            advances[char] = (width + factor - 1) // factor + 3
            for channel, thickness in enumerate((6, 13)):
                mask = np.zeros((size[1] * factor, size[0] * factor), dtype=np.uint8)
                cv2.putText(mask, glyph, (6, 20 * factor), font, scale, 255, thickness)
                atlas[char, :, :, channel] = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
        return atlas, advances

    def draw(self, frame, boxes, lines):
        if (frame.shape != self.image.shape or frame.dtype != torch.uint8
                or frame.device != self.image.device or frame.stride()[1:] != (3, 1)):
            raise ValueError("CUDA preview requires a packed BGR frame with optional row padding")
        if boxes.ndim != 2 or boxes.shape[1] != 6 or boxes.device != frame.device:
            raise ValueError("CUDA preview requires CUDA detection rows with six values")
        if not 1 <= len(lines) <= self.max_lines:
            raise ValueError("CUDA preview requires one to four HUD lines")
        self.text_view.fill(0)
        width = 0
        for row, line in enumerate(lines):
            chars = np.frombuffer(line.encode("ascii", "replace")[:self.text_columns], dtype=np.uint8)
            size = len(chars)
            self.text_view[row, 0] = size
            self.text_view[row, 1:1 + size] = chars
            ends = self.text_view[row, 1 + self.text_columns:1 + self.text_columns + size]
            np.cumsum(self.advances[chars], out=ends)
            if size:
                width = max(width, int(ends[-1]))
        with torch.cuda.stream(self.stream):
            if self.text is not None:
                self.text.copy_(self.host_text, non_blocking=True)
            # LAD already rounds pixel coordinates; do not convert confidence or track IDs.
            # Keep the packed coordinates alive until the matching download synchronizes.
            self.boxes = boxes[:, :4].to(torch.float32).contiguous()
            status = self.render(frame.data_ptr(), frame.stride(0), self.image.data_ptr(),
                                 frame.shape[1], frame.shape[0], self.boxes.data_ptr(), len(boxes),
                                 self.atlas.data_ptr(), self.layout_ptr, len(lines), width,
                                 self.stream.cuda_stream)
        if status:
            raise RuntimeError(f"CUDA preview compositor failed with CUDA status {status}")

    def download(self):
        with torch.cuda.stream(self.stream):
            self.host_image.copy_(self.image, non_blocking=True)
        self.stream.synchronize()
        self.boxes = None
        return self.host_image.numpy()
