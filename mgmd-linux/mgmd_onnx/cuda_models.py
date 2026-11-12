"""CUDA model stages for the GPU backend, using the existing ONNX sessions."""

from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.functional import pad

from mgmd.lac import crop_box
from mgmd.lad import crop_region


class CudaModel:
    def __init__(self, session, default_stream_lock=None, enqueue_options=None):
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise ValueError("CUDAExecutionProvider is required")
        self.session = session
        self.default_stream_lock = default_stream_lock
        self.enqueue_options = enqueue_options
        self.input, = session.get_inputs()
        self.outputs = session.get_outputs()

    def run(self, value):
        if not value.is_cuda or value.dtype != torch.float32 or not value.is_contiguous():
            raise ValueError("Expected a contiguous CUDA float32 input")
        device = value.device.index
        binding = self.session.io_binding()
        binding.bind_input(self.input.name, "cuda", device, np.float32,
                           tuple(value.shape), value.data_ptr())
        outputs = []
        for meta in self.outputs:
            if meta.type != "tensor(float)":
                raise ValueError(f"Unsupported output type: {meta.type}")
            shape = tuple(value.shape[0] if i == 0 and dim == self.input.shape[0]
                          else dim for i, dim in enumerate(meta.shape))
            if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                raise ValueError(f"Unsupported output shape: {meta.shape}")
            output = torch.empty(shape, dtype=torch.float32, device=value.device)
            binding.bind_output(meta.name, "cuda", device, np.float32, shape, output.data_ptr())
            outputs.append(output)
        stream = torch.cuda.current_stream(value.device)
        if self.enqueue_options is not None:
            # Device I/O Binding and the ORT user stream keep input and output ordered.
            # Release the capture guard before waiting for the submitted GPU work.
            with self.default_stream_lock:
                self.session.run_with_iobinding(binding, self.enqueue_options)
            stream.synchronize()
            return outputs
        stream.synchronize()
        # ORT can capture CUDA work during initialization of a new input shape.
        with self.default_stream_lock or nullcontext():
            self.session.run_with_iobinding(binding)
            binding.synchronize_outputs()
        return outputs


def resize_u8(image, size):
    if image.shape[:2] == size:
        return image
    height, width = image.shape[:2]
    if (height, width) == (size[0] * 2, size[1] * 2):
        pixels = image.to(torch.int32)
        return ((pixels[::2, ::2] + pixels[1::2, ::2] +
                 pixels[::2, 1::2] + pixels[1::2, 1::2] + 2) >> 2).to(torch.uint8)

    def axis(src, dst, horizontal=False):
        position = ((torch.arange(dst, dtype=torch.float64, device=image.device) + .5)
                    * (src / dst) - .5).float()
        left = position.floor().long()
        fraction = position - left
        if horizontal:
            fraction = torch.where((left < 0) | (left >= src - 1), 0, fraction)
        weights = ((1 - fraction) * 2048).round().int(), (fraction * 2048).round().int()
        return left.clamp(0, src - 1), (left + 1).clamp(0, src - 1), weights

    # Match OpenCV's uint8 INTER_LINEAR fixed-point rounding, including border rows.
    x0, x1, (a0, a1) = axis(width, size[1], horizontal=True)
    y0, y1, (b0, b1) = axis(height, size[0])
    rows = image[:, x0].int() * a0[:, None] + image[:, x1].int() * a1[:, None]
    upper = ((rows[y0] >> 4) * b0[:, None, None]) >> 16
    lower = ((rows[y1] >> 4) * b1[:, None, None]) >> 16
    return ((upper + lower + 2) >> 2).to(torch.uint8)


def normalization_table():
    return torch.from_numpy(np.arange(256, dtype=np.float32) / 255).cuda()


def rgb_chw(image, table):
    return table[image.permute(2, 0, 1).flip(0).long()].contiguous()


class CudaLac:
    def __init__(self, reference):
        self.model = CudaModel(reference.session)
        self.size = reference.input_size
        self.table = normalization_table()

    def prepare(self, frame, boxes):
        crops = [crop_box(frame, box) for box in boxes]
        indices = [i for i, crop in enumerate(crops) if crop is not None]
        if not indices:
            return indices, torch.empty((0, 3, self.size, self.size), device=frame.device)
        inputs = [rgb_chw(resize_u8(crops[i], (self.size, self.size)), self.table) for i in indices]
        return indices, torch.stack(inputs)

    def probabilities(self, inputs):
        if inputs.shape[0] == 0:
            return torch.empty(0, dtype=torch.float64, device=inputs.device)
        return self.model.run(inputs)[0].double().softmax(dim=1)[:, 1]


class CudaLad:
    def __init__(self, reference):
        self.model = CudaModel(reference.session)
        self.size = reference.crop_size
        self.threshold = reference.conf_thresh
        self.table = normalization_table()

    def prepare(self, frame, center):
        x, y, width, height = crop_region(frame.shape[:2], center, self.size)
        crop = frame[y:y + height, x:x + width]
        ratio = min(self.size / height, self.size / width)
        h, w = round(height * ratio), round(width * ratio)
        dh, dw = ((self.size - h) % 32) / 2, ((self.size - w) % 32) / 2
        pixels = rgb_chw(resize_u8(crop, (h, w)), self.table)
        if dh or dw:
            pixels = pad(pixels, (round(dw - .1), round(dw + .1),
                                  round(dh - .1), round(dh + .1)), value=114 / 255)
        return pixels.unsqueeze(0), (x, y, width, height)

    def select(self, raw, region):
        from utils.general import non_max_suppression, scale_coords

        x, y, width, height = region
        det = non_max_suppression(raw, self.threshold, 0.45)[0]
        if det.shape[0] == 0:
            return det.new_empty((0, 5))
        det[:, :4] = scale_coords((self.size, self.size), det[:, :4], (height, width)).round()
        centers = (det[:, :2].double() + det[:, 2:4].double()) / 2
        distance = (centers[:, 0] - width / 2).square() + (centers[:, 1] - height / 2).square()
        index = distance.argmin().reshape(1)
        selected = det.index_select(0, index)[:, :5]
        selected[:, :4] += selected.new_tensor((x, y, x, y))
        return selected
