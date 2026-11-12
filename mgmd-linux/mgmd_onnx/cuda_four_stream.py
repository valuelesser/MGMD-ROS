"""Bounded, ordered scheduling for the four-stream ROS experiment."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch


def event():
    mark = torch.cuda.Event(enable_timing=True)
    mark.record()
    return mark


def handoff(ready, *tensors):
    stream = torch.cuda.current_stream()
    stream.wait_event(ready)
    for tensor in tensors:
        tensor.record_stream(stream)


def four_stage_overlap(spans):
    """Milliseconds with distinct windows inside all four stage event intervals."""
    edges = sorted((time, delta, stage)
                   for window in spans
                   for stage, (start, end) in enumerate(window)
                   for time, delta in ((start, 1), (end, -1)))
    active = [0] * 4
    previous = overlap = 0.
    for time, delta, stage in edges:
        if all(active):
            overlap += time - previous
        active[stage] += delta
        previous = time
    return overlap


class FourStream:
    def __init__(self, pipe, streams, stack, reuse_gray, profile=False):
        self.pipe = pipe
        self.streams = streams
        self.reuse_gray = reuse_gray
        self.origin = event() if profile else None
        self.spans = []
        self.workers = [stack.enter_context(ThreadPoolExecutor(max_workers=1))
                        for _ in streams]
        self.pipe.mfe.candidate_cv_stream = self.pipe.mfe.cv.cuda.wrapStream(
            streams[1].cuda_stream)

    @torch.inference_mode()
    def _run(self, index, stage, *args):
        with torch.cuda.stream(self.streams[index]):
            return stage(*args)

    def windows(self, source, depth):
        pending = deque()
        for window, _, marks in source:
            args = (window, marks)
            for index, stage in enumerate((self.align, self.candidates,
                                           self.track, self.refine)):
                args = (self.workers[index].submit(self._run, index, stage, *args),)
            pending.append((window, args[0], marks))
            if len(pending) == depth:
                yield pending.popleft()
        yield from pending

    @staticmethod
    def _stamp(marks, name):
        if marks is not None:
            marks.append((name, event()))

    def align(self, window, marks):
        mfe = self.pipe.mfe
        start = event()
        self._stamp(marks, "start")
        images = tuple(frame.image for frame in window)
        for image in images:
            image.record_stream(self.streams[0])
        frame_ids = tuple(frame.seq for frame in window) if self.reuse_gray else None
        gray = mfe.gray_frames(images, frame_ids)
        stamp = lambda name: self._stamp(marks, name)
        stamp("gray")
        estimates = [mfe._estimate(gray[i], gray[1], stamp)
                     for i in ((0, 2) if mfe.cfg.needs_next else (0,))]
        # Keep the LK GpuMat owners in the future result until the solver is done.
        return window[1], gray, estimates, start, event(), marks

    def candidates(self, previous):
        center, gray, estimates, start, aligned, marks = previous.result()
        handoff(aligned, *gray, *(estimate[0] for estimate in estimates))
        begin = event()
        self._stamp(marks, "start")
        stamp = lambda name: self._stamp(marks, name)
        masks = [self.pipe.mfe._motion(gray[i], gray[1], estimate[0], stamp)
                 for i, estimate in zip((0, 2), estimates)]
        boxes, scores = self.pipe.mfe.combine_candidates(
            masks[0], masks[1] if len(masks) == 2 else None, stamp)
        return center, boxes, scores, (start, aligned, begin, event())

    def track(self, previous):
        center, boxes, scores, mfe = previous.result()
        handoff(mfe[-1], center.image, boxes, scores)
        timing = [None, event(), torch.cuda.Event(enable_timing=True),
                  torch.cuda.Event(enable_timing=True)]
        image, tracks, ids, status = self.pipe.track_classify(
            center.image, boxes, scores, center.seq, timing)
        return image, tracks, ids, status, len(boxes), mfe, timing

    def refine(self, previous):
        image, tracks, ids, status, count, mfe, timing = previous.result()
        handoff(timing[3], image, tracks, ids, status)
        begin = event()
        output = self.pipe.refine_tracks(image, tracks, ids)
        end = event()
        end.synchronize()
        if self.origin is not None:
            boundaries = ((mfe[0], mfe[1]), (mfe[2], mfe[3]),
                          (timing[1], timing[3]), (begin, end))
            self.spans.append([(self.origin.elapsed_time(a), self.origin.elapsed_time(b))
                               for a, b in boundaries])
        first = mfe[0].elapsed_time(mfe[1])
        second = mfe[2].elapsed_time(mfe[3])
        return (output, status, count,
                [first + second, timing[1].elapsed_time(timing[2]),
                 timing[2].elapsed_time(timing[3]), begin.elapsed_time(end)],
                [first, second])
