"""Two compute lanes with an ordered shared tracker."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Lock

import torch

from mgmd_onnx.cuda_four_stream import event, handoff
from mgmd_onnx.cuda_buf_ccl import BufCcl
from mgmd_onnx.cuda_pipeline import CudaPipeline
from mgmd_onnx.cuda_npp_ccl import NppCcl
from mgmd_onnx.cuda_tf import CudaTf


class ParityPipeline:
    def __init__(self, library, width, height, warp, reuse_gray, stack, npp=False,
                 unlocked=False, buf_library=""):
        self.reuse_gray = reuse_gray
        self.streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        self.tracker_stream = torch.cuda.Stream()
        self.lock = nullcontext() if unlocked else Lock()
        with torch.cuda.stream(self.tracker_stream):
            tracker = CudaTf(library)
        self.pipes = []
        for stream in self.streams:
            with torch.cuda.stream(stream):
                self.pipes.append(CudaPipeline(library, width, height, warp_backend=warp,
                                               mfe_stream=stream, model_stream=stream,
                                               tracker=tracker, shared_lock=self.lock,
                                               async_lad=unlocked))
                if npp:
                    self.pipes[-1].mfe.npp_ccl = NppCcl(width, height, stream)
                if buf_library:
                    self.pipes[-1].mfe.buf_ccl = BufCcl(buf_library, width, height, stream)
        self.workers = [stack.enter_context(ThreadPoolExecutor(max_workers=1))
                        for _ in self.streams]
        self.tracker_worker = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        # Initialize Torch's lazy CUDA linear algebra backend before either lane uses it.
        with torch.cuda.stream(self.streams[0]):
            matrix = torch.eye(8, dtype=torch.float64, device="cuda")
            torch.linalg.solve_ex(matrix.expand(256, -1, -1).contiguous(),
                                  matrix[:1].T.expand(256, -1, -1).contiguous())

    def windows(self, source):
        pending = deque()
        for accepted, (window, _, marks) in enumerate(source):
            lane = accepted % len(self.streams)
            with self.lock:
                start = event()
                gray = self.pipes[0].mfe.gray_frames(
                    tuple(frame.image for frame in window),
                    tuple(frame.seq for frame in window) if self.reuse_gray else None)
                ready = event()
            if marks is not None:
                marks.extend((("start", start), ("gray", ready)))
            worker = self.workers[lane]
            candidates = worker.submit(self.mfe, lane, gray, ready, marks)
            tracks = self.tracker_worker.submit(self.track, lane, candidates, window[1].seq)
            result = worker.submit(self.detect, lane, window[1], tracks, start, ready)
            pending.append((window, result, marks))
            if len(pending) == 2:
                yield pending.popleft()
        yield from pending

    @torch.inference_mode()
    def mfe(self, lane, gray, ready, marks):
        with torch.cuda.stream(self.streams[lane]):
            handoff(ready, *gray)
            start = event()
            if marks is not None:
                marks.append(("start", start))

            def stamp(name):
                if marks is not None:
                    marks.append((name, event()))

            boxes, scores = self.pipes[lane].mfe.candidates_from_gray(gray, stamp)
            return boxes, scores, start, event()

    @torch.inference_mode()
    def track(self, lane, previous, seq):
        boxes, scores, mfe_start, ready = previous.result()
        with torch.cuda.stream(self.tracker_stream):
            handoff(ready, boxes, scores)
            start = event()
            tracks, ids, status = self.pipes[lane].track(boxes, scores, seq)
            return tracks, ids, status, len(boxes), mfe_start, ready, start, event()

    @torch.inference_mode()
    def detect(self, lane, center, previous, gray_start, gray_ready):
        tracks, ids, status, count, mfe_start, mfe_end, tf_start, tf_end = previous.result()
        with torch.cuda.stream(self.streams[lane]):
            handoff(tf_end, center.image, tracks, ids, status)
            start = event()
            pipe = self.pipes[lane]
            image, tracks, ids = pipe.classify_tracks(center.image, tracks, ids)
            classified = event()
            output = pipe.refine_tracks(image, tracks, ids)
            end = event()
            end.synchronize()
        stages = [gray_start.elapsed_time(gray_ready) + mfe_start.elapsed_time(mfe_end),
                  tf_start.elapsed_time(tf_end), start.elapsed_time(classified),
                  classified.elapsed_time(end)]
        return output, status, count, stages, None
