"""Bounded, ordered scheduling with one or two LAD streams."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, Full, Queue
from threading import Event

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


def lad_overlap(spans):
    edges = sorted((time, delta) for window in spans
                   for time, delta in ((window[3][0], 1), (window[3][1], -1)))
    active = 0
    previous = overlap = 0.
    for time, delta in edges:
        if active > 1:
            overlap += time - previous
        active += delta
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
        for number, (window, _, marks) in enumerate(source):
            args = (window, marks)
            for index, stage in enumerate((self.align, self.candidates, self.track)):
                args = (self.workers[index].submit(self._run, index, stage, *args),)
            lad_index = number % (len(self.workers) - 3)
            worker = lad_index + 3
            model = self.pipe.lad if lad_index == 0 else self.pipe.second_lad
            result = self.workers[worker].submit(self._run, worker, self.refine, args[0], model)
            pending.append((window, result, marks))
            if len(pending) == depth:
                yield pending.popleft()
        yield from pending

    def realtime_windows(self, source, dropped):
        """Feed each stage independently, keeping the newest waiting window."""
        queues = [Queue(maxsize=1) for _ in range(5)]
        stop = Event()
        done = object()
        failures = Queue()

        def put(index, item):
            queue = queues[index]
            if item is not done:
                try:
                    queue.put_nowait(item)
                    return
                except Full:
                    try:
                        _, _, _, ready = queue.get_nowait()
                    except Empty:
                        pass
                    else:
                        dropped[index] += 1
                        if ready is not None:
                            ready.synchronize()
                queue.put_nowait(item)
                return
            while not stop.is_set():
                try:
                    queue.put(done, timeout=.05)
                    return
                except Full:
                    continue

        def guarded(work, *args):
            try:
                work(*args)
            except Exception as error:
                failures.put(error)
                stop.set()

        def feed():
            for window, _, marks in source:
                if stop.is_set():
                    break
                put(0, (window, marks, None, None))
            put(0, done)

        def consume(index, stage):
            while not stop.is_set():
                try:
                    item = queues[index].get(timeout=.05)
                except Empty:
                    continue
                if item is done:
                    put(index + 1, done)
                    return
                window, marks, previous, _ = item
                result = Future()
                if index == 0:
                    value = self._run(index, stage, window, marks)
                    ready = value[4]
                elif index == 3:
                    value = self._run(index, stage, previous, self.pipe.lad)
                    ready = None
                else:
                    value = self._run(index, stage, previous)
                    ready = value[-1][-1]
                result.set_result(value)
                put(index + 1, (window, marks, result, ready))

        with ThreadPoolExecutor(max_workers=1) as intake:
            producer = intake.submit(guarded, feed)
            workers = [worker.submit(guarded, consume, index, stage)
                       for index, (worker, stage) in enumerate(zip(
                           self.workers, (self.align, self.candidates, self.track, self.refine)))]
            try:
                while not stop.is_set():
                    try:
                        item = queues[-1].get(timeout=.05)
                    except Empty:
                        continue
                    if item is done:
                        break
                    window, marks, result, _ = item
                    yield window, result, marks
                if not failures.empty():
                    raise failures.get_nowait()
            finally:
                stop.set()
                producer.result()
                for worker in workers:
                    worker.result()
                torch.cuda.synchronize()

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

    def refine(self, previous, model):
        image, tracks, ids, status, count, mfe, timing = previous.result()
        handoff(timing[3], image, tracks, ids, status)
        begin = event()
        output = self.pipe.refine_tracks(image, tracks, ids, model)
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
