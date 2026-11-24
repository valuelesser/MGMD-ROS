"""Test device counts through TF, padded LAC and fixed LAD slots before output."""
import argparse
import ctypes
import json
from collections import deque
from contextlib import nullcontext

import torch

from mgmd import config
from mgmd_onnx.cuda_buf_ccl import BufCcl
from mgmd_onnx.cuda_opencv_decoder import OpenCvCudaDecoder
from mgmd_onnx.cuda_pipeline import CudaPipeline, cuda_model
from mgmd_onnx.resident_lad import combine_heads, split_heads
from mgmd_onnx.resident_ops import ResidentOps
from probe_resident_batches import batch_model
from verify_resident import controlled_sessions


@torch.inference_mode()
def main(batch, windows):
    if windows < 3:
        raise ValueError("Need at least three windows to measure after warmup")
    scope = ctypes.CDLL(None).audit_copy_scope
    scope.argtypes = [ctypes.c_char_p]
    ops = ResidentOps("/tmp/mgmd-all-in-gpu.so", True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), controlled_sessions(), OpenCvCudaDecoder("/data/videos/phantom30.mp4", 25, True) as dec:
        pipe = CudaPipeline("/tmp/mgmd-tf-g3.so", 1920, 1080, warp_backend="opencv",
                            mfe_stream=stream, model_stream=stream, shared_lock=nullcontext(),
                            async_lad=True, resident_ops=ops)
        pipe.mfe.buf_ccl = BufCcl("/tmp/mgmd-buf-ccl.so", 1920, 1080, stream)
        pipe.mfe.resident_regions = ops, pipe.tf.det_capacity
        path = config.WEIGHTS_DIR / "yolov5s_GLAD-crop.onnx"
        model = cuda_model(split_heads(batch_model(path, batch), batch), stream, nullcontext(), True)
        model.enqueue_stream = stream
        rows = torch.arange(pipe.tf.track_capacity, device="cuda")
        window = deque(maxlen=3)
        max_count = overflow = differing = 0
        errors, durations = [], []
        for seq in range(windows + 2):
            frame = dec.read()
            if frame is None:
                raise ValueError("Requested more windows than the video contains")
            stream.wait_event(frame.ready)
            window.append(frame)
            if len(window) < 3:
                continue
            center = window[1]
            scope(b"fixed-MFE-TF-LAC-LAD" if seq > 3 else None)
            try:
                start = stream.record_event(torch.cuda.Event(enable_timing=True))
                gray = pipe.mfe.gray_frames([f.image for f in window], [f.seq for f in window])
                boxes, scores, n = pipe.mfe.candidates_from_gray(gray, lambda _: None)
                tf = pipe.tf.step(boxes, scores, center.seq, n)
                valid = rows < tf.count
                safe = torch.where(valid[:, None], tf.boxes, 0)
                image = pipe.model_image(center.image)
                keep = valid & (pipe._classify(image, safe) >= pipe.lac_threshold)
                count = keep.sum()
                slots = rows.masked_fill(~keep, len(rows)).topk(batch, largest=False).values
                idx = slots.clamp_max(len(rows) - 1)
                roi = safe.index_select(0, idx)
                origin = ((roi[:, :2] + roi[:, 2:]) * .5 - 160).round().long()
                x = origin[:, 0].clamp(0, pipe.width - 320)
                y = origin[:, 1].clamp(0, pipe.height - 320)
                axis = pipe.lad_axis
                crops = image[0, :, y[:, None, None] + axis[None, :, None],
                              x[:, None, None] + axis[None, None, :]].permute(1, 0, 2, 3)
                raw = combine_heads(model.run(crops.div(255).contiguous()))
                selected = torch.cat([ops.select_box(part, pipe.lad_threshold)
                                      for part in raw.split(1)])
                selected[:, :4] += torch.stack((x, y, x, y), dim=1)
                selected[:, 4].masked_fill_(slots >= len(rows), 0)
                result = torch.column_stack((selected, tf.ids.index_select(0, idx)))
                meta = result.new_zeros((1, 6))
                meta[0, :2] = torch.stack((count, tf.status.reshape(())))
                packet = torch.cat((meta, result))
                end = stream.record_event(torch.cuda.Event(enable_timing=True))
            finally:
                scope(None)
            metadata, *padded = packet.tolist()
            actual, status = map(int, metadata[:2])
            assert not status
            max_count = max(max_count, actual)
            overflow += actual > batch
            if seq > 3:
                durations.append(start.elapsed_time(end))
            if actual > batch:
                continue
            # Same TF output and frame, dynamic original-sized processing.
            tracks, ids, _ = pipe.select_lac(image, safe[valid], tf.ids[valid])
            expected = pipe.refine_tracks(image, tracks, ids).tolist()
            got = [r for r in padded if r[4] > 0]
            if len(ids) != actual or len(expected) != len(got):
                differing += 1
                continue
            if any(a[:4] != b[:4] or a[5] != b[5] for a, b in zip(got, expected)):
                differing += 1
            errors.extend(abs(a[4] - b[4]) for a, b in zip(got, expected))
        print(json.dumps(dict(slots=batch, windows=windows, measured=len(durations), max_lad=max_count,
                              overflow_windows=overflow, differing_windows_within_capacity=differing,
                              max_conf_delta=max(errors, default=0),
                              mean_ms=sum(durations)/len(durations))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, choices=(4, 8), default=4)
    parser.add_argument("--windows", type=int, default=64)
    args = parser.parse_args()
    main(args.slots, args.windows)
