"""Compute lanes with an ordered shared tracker."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from queue import Empty, Queue
from threading import Event
from time import perf_counter

import torch

from mgmd_onnx.confirmed_tracks import ConfirmedTracks
from mgmd_onnx.cuda_buf_ccl import BufCcl
from mgmd_onnx.cuda_pipeline import CudaPipeline
from mgmd_onnx.device_confirmation import DeviceConfirmation
from mgmd_onnx.cuda_tf import CudaTf


def event():
    mark = torch.cuda.Event(enable_timing=True)
    mark.record()
    return mark


def handoff(ready, *tensors):
    stream = torch.cuda.current_stream()
    stream.wait_event(ready)
    for tensor in tensors:
        tensor.record_stream(stream)


class ParityPipeline:
    def __init__(self, library, width, height, warp, reuse_gray, stack, buf_library,
                 confirm_frames=0, compute_lanes=2, enable_lac=True, enable_lad=True, resident_ops=None):
        self.reuse_gray = reuse_gray
        confirmation_type = DeviceConfirmation if resident_ops else ConfirmedTracks
        self.confirmed = confirmation_type(confirm_frames) if confirm_frames else None
        self.streams = [torch.cuda.Stream() for _ in range(compute_lanes)]
        self.tracker_stream = self.streams[0] if compute_lanes == 1 else torch.cuda.Stream()
        self.lock = nullcontext()
        with torch.cuda.stream(self.tracker_stream):
            tracker = CudaTf(library)
        self.pipes = []
        for stream in self.streams:
            with torch.cuda.stream(stream):
                self.pipes.append(CudaPipeline(library, width, height, warp_backend=warp,
                                               mfe_stream=stream, model_stream=stream,
                                               tracker=tracker, shared_lock=self.lock,
                                               async_lad=True, enable_lac=enable_lac,
                                               enable_lad=enable_lad, resident_ops=resident_ops))
                if resident_ops:
                    self.pipes[-1].mfe.resident_regions = resident_ops, tracker.det_capacity
                self.pipes[-1].mfe.buf_ccl = BufCcl(buf_library, width, height, stream)
        self.workers = [stack.enter_context(ThreadPoolExecutor(max_workers=1))
                        for _ in self.streams]
        self.tracker_worker = (self.workers[0] if compute_lanes == 1 else
                               stack.enter_context(ThreadPoolExecutor(max_workers=1)))
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
        gray_stream = torch.cuda.stream(self.streams[0]) if len(self.streams) == 1 else nullcontext()
        with gray_stream:
            for frame in window:
                if frame.ready is not None:
                    torch.cuda.current_stream().wait_event(frame.ready)
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
        return window, result, marks

    def available_windows(self, source, stats):
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
                        stats["wait_for_lane"] += perf_counter() - started
                        continue
                    stats["wait_for_lane"] += perf_counter() - started
                    if stop.is_set():
                        return
                    item = next(source, None)
                    if item is None:
                        return
                    window, _, marks = item
                    entry = self.submit(lane, window, marks)
                    ordered.put(entry)
                    stats["queued_peak"] = max(stats["queued_peak"], ordered.qsize())
                    entry[1].add_done_callback(lambda _, lane=lane: idle.put(lane))
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
    def mfe(self, lane, gray, ready, marks):
        with torch.cuda.stream(self.streams[lane]):
            handoff(ready, *gray)
            start = event()
            if marks is not None:
                marks.append(("start", start))

            def stamp(name):
                if marks is not None:
                    marks.append((name, event()))

            candidates = self.pipes[lane].mfe.candidates_from_gray(gray, stamp)
            return candidates, start, event()

    @torch.inference_mode()
    def track(self, lane, previous, seq):
        candidates, mfe_start, ready = previous.result()
        boxes, scores = candidates[:2]
        count = candidates[2] if len(candidates) == 3 else None
        with torch.cuda.stream(self.tracker_stream):
            handoff(ready, *candidates)
            start = event()
            tracks, ids, status = self.pipes[lane].track(boxes, scores, seq, count)
            return tracks, ids, status, (len(boxes) if count is None else count), mfe_start, ready, start, event()

    @torch.inference_mode()
    def detect(self, lane, center, previous, gray_start, gray_ready):
        tracks, ids, status, count, mfe_start, mfe_end, tf_start, tf_end = previous.result()
        track_count = len(ids)
        with torch.cuda.stream(self.streams[lane]):
            handoff(tf_end, center.image, tracks, ids, status)
            start = event()
            pipe = self.pipes[lane]
            reused = None
            known = self.confirmed.snapshot if self.confirmed is not None else None
            if pipe.resident_ops and known is not None and len(ids):
                reuse, known_scores, _ = DeviceConfirmation.match(known, ids)
                take = reuse.nonzero().flatten()
                fresh = (~reuse).nonzero().flatten()
                reused = torch.column_stack((tracks.index_select(0, take), known_scores.index_select(0, take),
                                             ids.index_select(0, take)))
                tracks, ids = tracks.index_select(0, fresh), ids.index_select(0, fresh)
            elif known and len(ids):
                keys = ids.new_tensor(tuple(known))
                matches = ids[:, None] == keys
                reuse, indices = matches.max(dim=1)
                scores = tracks.new_tensor(tuple(known.values()))[indices[reuse]]
                reused = torch.column_stack((tracks[reuse], scores, ids[reuse]))
                tracks, ids = tracks[~reuse], ids[~reuse]
            image = None
            scores = None
            if pipe.lac is not None and (len(ids) or self.confirmed is None):
                image = pipe.model_image(center.image)
                tracks, ids, scores = pipe.select_lac(image, tracks, ids)
            classified = event()
            if not len(ids):
                output = tracks.new_empty((0, 6))
            elif pipe.lad is not None:
                if image is None:
                    image = pipe.model_image(center.image)
                output = pipe.refine_tracks(image, tracks, ids)
            else:
                if scores is None:
                    scores = tracks.new_zeros(len(ids))
                output = torch.column_stack((tracks, scores, ids))
            if reused is not None:
                output = torch.cat((output, reused))
            end = event()
            end.synchronize()
        stages = [gray_start.elapsed_time(gray_ready) + mfe_start.elapsed_time(mfe_end),
                  tf_start.elapsed_time(tf_end), start.elapsed_time(classified) if pipe.lac else 0.,
                  classified.elapsed_time(end) if pipe.lad else 0.]
        return output, status, count, stages, (track_count, 0 if reused is None else len(reused))
