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
from mgmd_onnx.cuda_mfe import CudaMfe
from mgmd_onnx.cuda_pipeline import CudaPipeline
from mgmd_onnx.cuda_npp_ccl import NppCcl
from mgmd_onnx.cuda_tf import CudaTf


class ParityPipeline:
    def __init__(self, library, width, height, warp, reuse_gray, stack, npp=False,
                 unlocked=False, buf_library="", lanes=2, defer_lad_sync=False,
                 extra_mfe_lane=False, lad_lanes=0):
        self.defer_lad_sync = defer_lad_sync
        self.reuse_gray = reuse_gray
        self.extra_mfe_lane = extra_mfe_lane
        self.wait_trace = None
        self.streams = [torch.cuda.Stream() for _ in range(lanes)]
        self.lad_streams = [torch.cuda.Stream() for _ in range(lad_lanes)]
        self.mfe_streams = ([torch.cuda.Stream() for _ in range(lanes + 1)]
                            if extra_mfe_lane else self.streams)
        self.tracker_stream = torch.cuda.Stream()
        self.lock = nullcontext() if unlocked else Lock()
        with torch.cuda.stream(self.tracker_stream):
            tracker = CudaTf(library)
        self.pipes = []
        for lane, (stream, mfe_stream) in enumerate(zip(self.streams, self.mfe_streams)):
            with torch.cuda.stream(stream):
                self.pipes.append(CudaPipeline(library, width, height, warp_backend=warp,
                                               mfe_stream=mfe_stream, model_stream=stream,
                                               lad_stream=self.lad_streams[lane] if lad_lanes else None,
                                               second_lad_stream=(self.lad_streams[-1]
                                                                  if lad_lanes > lanes and lane == 0 else None),
                                               tracker=tracker, shared_lock=self.lock,
                                               async_lad=unlocked))
            with torch.cuda.stream(mfe_stream):
                if npp:
                    self.pipes[-1].mfe.npp_ccl = NppCcl(width, height, mfe_stream)
                if buf_library:
                    self.pipes[-1].mfe.buf_ccl = BufCcl(buf_library, width, height, mfe_stream)
        self.extractors = [pipe.mfe for pipe in self.pipes]
        if lad_lanes:
            self.lad_models = [pipe.lad for pipe in self.pipes]
            if lad_lanes > lanes:
                self.lad_models.append(self.pipes[0].second_lad)
            self.lad_pool = stack.enter_context(ThreadPoolExecutor(max_workers=lad_lanes))
            self.idle_lad = Queue()
            self.lad_counts = [0] * lad_lanes
            for lane in range(lad_lanes):
                self.idle_lad.put(lane)
        if extra_mfe_lane:
            stream = self.mfe_streams[-1]
            with torch.cuda.stream(stream):
                extractor = CudaMfe(self.extractors[0].cfg, width, height, warp_backend=warp)
                extractor.buf_ccl = BufCcl(buf_library, width, height, stream)
                self.extractors.append(extractor)
            self.post_pool = stack.enter_context(ThreadPoolExecutor(max_workers=lanes))
            self.idle_post = Queue()
            for lane in range(lanes):
                self.idle_post.put(lane)
        self.workers = [stack.enter_context(ThreadPoolExecutor(max_workers=1))
                        for _ in self.mfe_streams]
        self.tracker_worker = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        # Initialize Torch's lazy CUDA linear algebra backend before either lane uses it.
        with torch.cuda.stream(self.mfe_streams[0]):
            matrix = torch.eye(8, dtype=torch.float64, device="cuda")
            torch.linalg.solve_ex(matrix.expand(256, -1, -1).contiguous(),
                                  matrix[:1].T.expand(256, -1, -1).contiguous())

    def windows(self, source):
        pending = deque()
        for accepted, (window, _, marks) in enumerate(source):
            lane = accepted % len(self.streams)
            entry, _ = self.submit(lane, window, marks)
            pending.append(entry)
            if len(pending) == len(self.streams):
                yield pending.popleft()
        yield from pending

    def submit(self, lane, window, marks):
        with self.lock:
            start = event()
            gray = self.extractors[0].gray_frames(
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
        tracks = self.tracker_worker.submit(self.track, candidates, seq, trace)
        args = window[1], tracks, start, ready, trace
        if self.lad_streams:
            classified = worker.submit(self.classify, lane, window[1], tracks, trace)
            result = self.lad_pool.submit(self.detect_on_idle_lad, classified, start, ready, trace)
            return (window, result, marks), classified
        if self.extra_mfe_lane:
            result = self.post_pool.submit(self.detect_on_idle_lane, *args)
            return (window, result, marks), candidates
        result = worker.submit(self.detect, lane, *args)
        return (window, result, marks), result

    def detect_on_idle_lane(self, *args):
        # The pool has exactly one worker per model lane. Each call owns its lane until done.
        lane = self.idle_post.get_nowait()
        try:
            return self.detect(lane, *args)
        finally:
            self.idle_post.put(lane)

    def available_windows(self, source, dispatch_stats):
        """Refill finished lanes independently of the ordered publisher."""
        idle, ordered = Queue(), Queue()
        stop = Event()
        eof = object()
        for lane in range(len(self.mfe_streams)):
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
                    entry, reusable = self.submit(lane, window, marks)
                    reusable.add_done_callback(lambda _, lane=lane: idle.put(lane))
                    dispatch_stats[f"lane_{lane}_windows"] += 1
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
        self.extractors[0].clear_gray_cache()
        if self.lad_streams:
            self.lad_counts[:] = [0] * len(self.lad_streams)

    @torch.inference_mode()
    def mfe(self, lane, gray, ready, marks, trace=None):
        with torch.cuda.stream(self.mfe_streams[lane]):
            handoff(ready, *gray)
            start = event()
            if marks is not None:
                marks.append(("start", start))

            def stamp(name):
                if marks is not None:
                    marks.append((name, event()))

            boxes, scores = self.extractors[lane].candidates_from_gray(gray, stamp)
            if trace is not None:
                trace["mfe_submitted"] = perf_counter()
            return boxes, scores, start, event()

    @torch.inference_mode()
    def track(self, previous, seq, trace=None):
        boxes, scores, mfe_start, ready = previous.result()
        with torch.cuda.stream(self.tracker_stream):
            handoff(ready, boxes, scores)
            start = event()
            tracks, ids, status = self.pipes[0].track(boxes, scores, seq)
            if trace is not None:
                trace["tf_submitted"] = perf_counter()
            return tracks, ids, status, len(boxes), mfe_start, ready, start, event()

    @torch.inference_mode()
    def classify(self, lane, center, previous, trace=None):
        tracks, ids, status, count, mfe_start, mfe_end, tf_start, tf_end = previous.result()
        if trace is not None and self.extra_mfe_lane:
            trace["post_started"] = perf_counter()
        with torch.cuda.stream(self.streams[lane]):
            handoff(tf_end, center.image, tracks, ids, status)
            start = event()
            pipe = self.pipes[lane]
            image, tracks, ids = pipe.classify_tracks(center.image, tracks, ids)
            classified = event()
            if trace is not None:
                trace["lac_submitted"] = perf_counter()
        return image, tracks, ids, status, count, (mfe_start, mfe_end, tf_start, tf_end, start, classified)

    @torch.inference_mode()
    def detect(self, lane, center, previous, gray_start, gray_ready, trace=None):
        classified = self.classify(lane, center, previous, trace)
        with torch.cuda.stream(self.streams[lane]):
            return self.finish_lad(classified, gray_start, gray_ready, trace, self.pipes[lane])

    @torch.inference_mode()
    def detect_on_idle_lad(self, future, gray_start, gray_ready, trace):
        classified = future.result()
        lane = self.idle_lad.get_nowait()
        try:
            with torch.cuda.stream(self.lad_streams[lane]):
                handoff(classified[-1][-1], *classified[:4])
                if trace is not None:
                    trace["lad_started"] = perf_counter()
                result = self.finish_lad(classified, gray_start, gray_ready, trace,
                                         self.pipes[0], self.lad_models[lane], event())
                self.lad_counts[lane] += 1
                return result
        finally:
            self.idle_lad.put(lane)

    def finish_lad(self, result, gray_start, gray_ready, trace, pipe, model=None, lad_start=None):
        image, tracks, ids, status, count, marks = result
        mfe_start, mfe_end, tf_start, tf_end, lac_start, classified = marks
        pending = [] if self.defer_lad_sync else None
        try:
            output = pipe.refine_tracks(image, tracks, ids, model=model, pending=pending)
            if trace is not None:
                trace["lad_submitted"] = perf_counter()
            end = event()
            end.synchronize()
            if trace is not None:
                trace["gpu_done"] = perf_counter()
        except Exception:
            if pending:
                torch.cuda.current_stream().synchronize()
            raise
        stages = [gray_start.elapsed_time(gray_ready) + mfe_start.elapsed_time(mfe_end),
                  tf_start.elapsed_time(tf_end), lac_start.elapsed_time(classified),
                  (lad_start or classified).elapsed_time(end)]
        return output, status, count, stages, None
