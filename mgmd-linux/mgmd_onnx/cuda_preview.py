"""CUDA BGR preview compositor; only the ROS transport needs host pixels."""

import ctypes as C

import cv2
import numpy as np
import torch


class CudaPreview:
    cell_width, line_height, text_columns, max_lines = 17, 32, 128, 4

    def __init__(self, library, sample):
        self.library = C.CDLL(library)
        self.render = self.library.mgmd_render_preview
        self.render.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_int, C.c_int,
                                C.c_void_p, C.c_int, C.c_void_p, C.c_void_p,
                                C.c_int, C.c_int, C.c_void_p]
        self.render.restype = C.c_int
        self.stream = torch.cuda.current_stream(sample.device)
        # Rasterize the printable alphabet once. Per-frame HUD composition is on CUDA.
        atlas = np.zeros((128, self.line_height, self.cell_width), dtype=np.uint8)
        for char in range(32, 127):
            font = cv2.FONT_HERSHEY_SIMPLEX
            width = cv2.getTextSize(chr(char), font, .67, 1)[0][0]
            cv2.putText(atlas[char], chr(char), (max(0, (self.cell_width - width) // 2), 23),
                        font, .67, 255, 1, cv2.LINE_AA)
        self.atlas = torch.from_numpy(atlas).to(sample.device)
        self.host_text = torch.empty((self.max_lines, self.text_columns), dtype=torch.uint8,
                                     pin_memory=True)
        self.text_view = self.host_text.numpy()
        self.text = torch.empty_like(self.host_text, device=sample.device)
        self.image = torch.empty(sample.shape, dtype=torch.uint8, device=sample.device)
        self.host_image = torch.empty(sample.shape, dtype=torch.uint8, pin_memory=True)

    def draw(self, frame, boxes, lines):
        if (frame.shape != self.image.shape or frame.dtype != torch.uint8
                or frame.device != self.image.device or frame.stride()[1:] != (3, 1)):
            raise ValueError("CUDA preview requires a packed BGR frame with optional row padding")
        if boxes.ndim != 2 or boxes.shape[1] != 6 or boxes.device != frame.device:
            raise ValueError("CUDA preview requires CUDA detection rows with six values")
        if not 1 <= len(lines) <= self.max_lines:
            raise ValueError("CUDA preview requires one to four HUD lines")
        self.text_view.fill(0)
        columns = 0
        for row, line in enumerate(lines):
            chars = np.frombuffer(line.encode("ascii", "replace")[:self.text_columns], dtype=np.uint8)
            self.text_view[row, :len(chars)] = chars
            columns = max(columns, len(chars))
        with torch.cuda.stream(self.stream):
            self.text.copy_(self.host_text, non_blocking=True)
            # LAD already rounds pixel coordinates; do not convert confidence or track IDs.
            # Keep the packed coordinates alive until the matching download synchronizes.
            self.boxes = boxes[:, :4].to(torch.float32).contiguous()
            status = self.render(frame.data_ptr(), frame.stride(0), self.image.data_ptr(),
                                 frame.shape[1], frame.shape[0], self.boxes.data_ptr(), len(boxes),
                                 self.atlas.data_ptr(), self.text.data_ptr(), len(lines), columns,
                                 self.stream.cuda_stream)
        if status:
            raise RuntimeError(f"CUDA preview compositor failed with CUDA status {status}")

    def download(self):
        with torch.cuda.stream(self.stream):
            self.host_image.copy_(self.image, non_blocking=True)
        self.stream.synchronize()
        self.boxes = None
        return self.host_image.numpy()
