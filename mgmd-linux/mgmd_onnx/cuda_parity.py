"""Compute lanes with an ordered shared tracker."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from queue import Empty, Queue
from threading import Event, Lock
from time import perf_counter

import torch

from mgmd_onnx.cuda_four_stream import event, handoff
from mgmd_onnx.cuda_buf_ccl import BufCcl
from mgmd_onnx.cuda_pipeline import CudaPipeline
from mgmd_onnx.cuda_npp_ccl import NppCcl
from mgmd_onnx.cuda_tf import CudaTf


class ParityPipeline:
    def __init__(self, library, width, height, warp, reuse_gray, stack, npp=False,
                 unlocked=False, buf_library="", lanes=2, defer_lad_sync=False):
        self.defer_lad_sync = defer_lad_sync
        self.reuse_gray = reuse_gray
        self.wait_trace = None
        self.streams = [torch.cuda.Stream() for _ in range(lanes)]
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
            pending.append(self.submit(lane, window, marks))
            if len(pending) == len(self.streams):
                yield pending.popleft()
        yield from pending

    def submit(self, lane, window, marks):
        with self.lock:
            start = event()
            gray = self.pipes[0].mfe.gray_frames(
                tuple(frame.image for frame in window),
                tuple(frame.seq for frame in window) if self.reuse_gray else None)
            ready = event()
        if marks is not None:
            marks.extend((("start", start), ("gray", ready)))
        worker = self.workers[lane]
        seq = window[1].seq
        trace = None
        if self.wait_trace is not None:
            trace = {"submitted": perf_counter()}
            self.wait_trace[seq] = trace
        candidates = worker.submit(self.mfe, lane, gray, ready, marks, trace)
        tracks = self.tracker_worker.submit(self.track, lane, candidates, seq, trace)
        result = worker.submit(self.detect, lane, window[1], tracks, start, ready, trace)
        return window, result, marks

    def available_windows(self, source, dispatch_stats):
        """Refill finished lanes independently of the ordered publisher."""
        idle, ordered = Queue(), Queue()
        stop = Event()
        eof = object()
        for lane in range(len(self.streams)):
            idle.put(lane)

        @torch.inference_mode()
        def dispatch():
            try:
                while not stop.is_set():
                    started = perf_counter()
                    try:
                        lane = idle.get(timeout=.05)
                    except Empty:
                        dispatch_stats["wait_for_lane"] += perf_counter() - started
                        continue
                    dispatch_stats["wait_for_lane"] += perf_counter() - started
                    if stop.is_set():
                        return
                    item = next(source, None)
                    if item is None:
                        return
                    window, _, marks = item
                    entry = self.submit(lane, window, marks)
                    entry[1].add_done_callback(lambda _, lane=lane: idle.put(lane))
                    ordered.put(entry)
                    dispatch_stats["queued_peak"] = max(dispatch_stats["queued_peak"],
                                                         ordered.qsize())
            finally:
                ordered.put(eof)

        with ThreadPoolExecutor(max_workers=1) as thread:
            producer = thread.submit(dispatch)
            try:
                while True:
                    entry = ordered.get()
                    if entry is eof:
                        producer.result()
                        return
                    yield entry
            finally:
                stop.set()

    def reset_after_warmup(self):
        with torch.cuda.stream(self.tracker_stream):
            self.pipes[0].tf.reset()
        self.pipes[0].mfe.clear_gray_cache()

    @torch.inference_mode()
    def mfe(self, lane, gray, ready, marks, trace=None):
        with torch.cuda.stream(self.streams[lane]):
            handoff(ready, *gray)
            start = event()
            if marks is not None:
                marks.append(("start", start))

            def stamp(name):
                if marks is not None:
                    marks.append((name, event()))

            boxes, scores = self.pipes[lane].mfe.candidates_from_gray(gray, stamp)
            if trace is not None:
                trace["mfe_submitted"] = perf_counter()
            return boxes, scores, start, event()

    @torch.inference_mode()
    def track(self, lane, previous, seq, trace=None):
        boxes, scores, mfe_start, ready = previous.result()
        with torch.cuda.stream(self.tracker_stream):
            handoff(ready, boxes, scores)
            start = event()
            tracks, ids, status = self.pipes[lane].track(boxes, scores, seq)
            if trace is not None:
                trace["tf_submitted"] = perf_counter()
            return tracks, ids, status, len(boxes), mfe_start, ready, start, event()

    @torch.inference_mode()
    def detect(self, lane, center, previous, gray_start, gray_ready, trace=None):
        tracks, ids, status, count, mfe_start, mfe_end, tf_start, tf_end = previous.result()
        with torch.cuda.stream(self.streams[lane]):
            handoff(tf_end, center.image, tracks, ids, status)
            start = event()
            pipe = self.pipes[lane]
            image, tracks, ids = pipe.classify_tracks(center.image, tracks, ids)
            classified = event()
            if trace is not None:
                trace["lac_submitted"] = perf_counter()
            pending = [] if self.defer_lad_sync else None
            try:
                output = pipe.refine_tracks(image, tracks, ids, pending=pending)
                if trace is not None:
                    trace["lad_submitted"] = perf_counter()
                end = event()
                end.synchronize()
                if trace is not None:
                    trace["gpu_done"] = perf_counter()
            except Exception:
                if pending:
                    self.streams[lane].synchronize()
                raise
        stages = [gray_start.elapsed_time(gray_ready) + mfe_start.elapsed_time(mfe_end),
                  tf_start.elapsed_time(tf_end), start.elapsed_time(classified),
                  classified.elapsed_time(end)]
        return output, status, count, stages, None
