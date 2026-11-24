"""Compare baseline and resident paths in lockstep, including ordered confirmation."""
import argparse
import ctypes
from collections import deque
from contextlib import ExitStack, nullcontext
import json
from unittest.mock import patch

import onnxruntime as ort

import torch

from mgmd import config
from mgmd.mfe import MFEConfig
from mgmd_onnx.cuda_buf_ccl import BufCcl
from mgmd_onnx.cuda_opencv_decoder import OpenCvCudaDecoder
from mgmd_onnx.cuda_parity import ParityPipeline
from mgmd_onnx.cuda_pipeline import cuda_model
from mgmd_onnx.cuda_preview import CudaPreview
from mgmd_onnx.resident_lad import combine_heads, split_heads
from mgmd_onnx.resident_ops import ResidentOps


def controlled_sessions():
    original = ort.InferenceSession

    def session(*args, providers, **kwargs):
        name, options = providers[0]
        options = dict(options, cudnn_conv_algo_search="DEFAULT", use_tf32="0")
        return original(*args, providers=[(name, options)], **kwargs)

    return patch.object(ort, "InferenceSession", session)


@torch.inference_mode()
def verify(args):
    ops = ResidentOps(args.library, args.split_head)
    with ExitStack() as stack:
        if args.controlled_conv:
            stack.enter_context(controlled_sessions())
        decoders = [stack.enter_context(OpenCvCudaDecoder(args.video, 25, flag)) for flag in (False, True)]
        pipes = [ParityPipeline("/tmp/mgmd-tf-g3.so", 1920, 1080, "opencv", True, stack,
                                "/tmp/mgmd-buf-ccl.so", confirm_frames=3, compute_lanes=1, resident_ops=mode)
                 for mode in (None, None if args.baseline_control else ops)]
        windows = [deque(maxlen=3), deque(maxlen=3)]
        checked = detections = reused = mismatch = 0
        max_conf_error = 0.
        for seq in range(args.windows + 2):
            for decoder, window in zip(decoders, windows):
                window.append(decoder.read())
            newest = windows[1][-1]
            torch.cuda.current_stream().wait_event(newest.ready)
            assert torch.equal(windows[0][-1].image, newest.image), seq
            if seq < 2:
                continue
            results = []
            for pipe, window in zip(pipes, windows):
                _, job, _ = pipe.submit(0, tuple(window), None)
                boxes, status, count, _, (tracked, reuse) = job.result()
                assert not status.item(), seq
                data = boxes.tolist()
                fresh = len(data) - reuse
                if pipe is pipes[0] or args.baseline_control:
                    pipe.confirmed.observe(seq - 1, data[:fresh], data[fresh:])
                else:
                    pipe.confirmed.observe(seq - 1, boxes, reuse)
                results.append((int(count), tracked, reuse, data))
            a, b = results
            checked += 1
            detections += len(a[3])
            reused += a[2]
            if a[:3] != b[:3] or len(a[3]) != len(b[3]):
                mismatch += 1
                continue
            for old, new in zip(a[3], b[3]):
                max_conf_error = max(max_conf_error, abs(old[4] - new[4]))
                if old[:4] != new[:4] or old[5] != new[5]:
                    mismatch += 1
                    break
        previews = [CudaPreview("/tmp/mgmd-cuda-preview.so", newest.image, value) for value in (None, ops)]
        for preview in previews:
            preview.draw(newest.image, boxes, ["Resident HUD 123", "MFE TF LAC LAD"])
        assert (previews[0].download() == previews[1].download()).all()
        print(json.dumps(dict(windows=checked, decoded=seq+1, detections=detections, reused=reused,
                              mismatched_windows=mismatch, max_conf_error=max_conf_error,
                              hud_equal=True)), flush=True)
        assert mismatch == 0 and max_conf_error <= 1e-5


@torch.inference_mode()
def transfers(args):
    hook = ctypes.CDLL(None).audit_copy_scope
    hook.argtypes = [ctypes.c_char_p]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        path = config.WEIGHTS_DIR / "yolov5s_GLAD-crop.onnx"
        normal, resident = [cuda_model(p, stream, nullcontext(), True) for p in (path, split_heads(path))]
        resident.enqueue_stream = stream
        torch.manual_seed(7)
        crop = torch.rand(1, 3, 320, 320, device="cuda")
        ops = ResidentOps(args.library)
        for iteration in range(3):
            outputs = []
            for model in (normal, resident):
                label = b"base-head" if model is normal else b"resident-head"
                hook(label if iteration else None)
                try:
                    values = model.run(crop)
                    outputs.append(values[0] if model is normal else combine_heads(values))
                finally:
                    hook(None)
            delta = (outputs[0] - outputs[1]).abs().amax(dim=(0, 1))
            print("head max_abs_delta [x,y,w,h,obj,class]", delta.tolist(), flush=True)
            assert torch.isfinite(delta).all()
            hook(b"resident-nms" if iteration else None)
            try:
                ops.select_box(outputs[-1], .5)
            finally:
                hook(None)
            stream.synchronize()
        # Exercise nonempty NMS, overlapping candidates and an exactly-threshold row.
        raw = crop.new_zeros((1, 6300, 6))
        raw[0, :4] = raw.new_tensor([[100, 100, 20, 20, 1, .9],
                                     [101, 101, 20, 20, 1, .8],
                                     [160, 160, 10, 20, 1, .7],
                                     [160, 160, 5, 5, 1, .5]])
        hook(b"resident-nms-nonempty")
        try:
            selected = ops.select_box(raw, .5)
        finally:
            hook(None)
        torch.testing.assert_close(selected, raw.new_tensor([[155, 150, 165, 170, .7]]))
        cfg = MFEConfig().apply_recommended()
        mask = torch.zeros((257, 257), dtype=torch.uint8, device="cuda")
        ccl = BufCcl("/tmp/mgmd-buf-ccl.so", 257, 257, stream)
        _, _, count = ops.regions(ccl(mask), cfg, 512)
        assert int(count) == 0
        axis = torch.arange(257, device=mask.device)
        mask.copy_((axis[:, None] % 8 < 4) & (axis[None, :] % 8 < 4))
        labels = ccl(mask)
        hook(b"resident-regions-overflow")
        try:
            boxes, scores, count = ops.regions(labels, cfg, 512)
        finally:
            hook(None)
        assert int(count) == 1024 and boxes.shape == (512, 4)
        assert boxes[511].tolist() == [248, 120, 252, 124] and scores[511].item() == 16
        print("Nonempty NMS, empty mask and bounded overflow checks passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="/tmp/mgmd-all-in-gpu.so")
    parser.add_argument("--video", default="/data/videos/phantom30.mp4")
    parser.add_argument("--windows", type=int, default=300)
    parser.add_argument("--audit-stages", action="store_true")
    parser.add_argument("--controlled-conv", action="store_true")
    parser.add_argument("--baseline-control", action="store_true")
    parser.add_argument("--split-head", action="store_true")
    parser.add_argument("--transfers", action="store_true")
    args = parser.parse_args()
    if args.audit_stages:
        from trace_resident_stages import install
        install()
    (transfers if args.transfers else verify)(args)
