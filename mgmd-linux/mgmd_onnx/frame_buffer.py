"""ROS detector frame window selection, independent of ROS and model loading."""

from collections import deque


class FrameWindow:
    def __init__(self, k: int, max_gap: int = 0):
        if k < 1 or max_gap < 0:
            raise ValueError("k must be positive and max_gap nonnegative")
        self.k = k
        self.configured_max_gap = max_gap
        self.gaps = deque(maxlen=32)
        self.last_push = None

    def push(self, frame_id: int) -> bool:
        """Record arrival; return true if the video sequence rewound."""
        rewind = self.last_push is not None and frame_id <= self.last_push
        if rewind:
            self.gaps.clear()
        elif self.last_push is not None:
            self.gaps.append(frame_id - self.last_push)
        self.last_push = frame_id
        return rewind

    def max_gap(self) -> int:
        if self.configured_max_gap:
            return self.configured_max_gap
        if not self.gaps:
            return self.k
        ordered = sorted(self.gaps)
        return max(self.k, ordered[len(ordered) // 2])

    def neighbors(self, frames, t: int):
        """Choose nearest available frames at least k away on each side."""
        if t not in frames:
            return None
        limit = self.max_gap()
        prev = next_ = None
        for distance in range(self.k, limit + 1):
            if prev is None and t - distance in frames:
                prev = t - distance
            if next_ is None and t + distance in frames:
                next_ = t + distance
            if prev is not None and next_ is not None:
                return prev, next_
        return None

    def ready(self, frames, last_done: int):
        if len(frames) < 3:
            return None
        for t in frames:
            if t > last_done:
                pair = self.neighbors(frames, t)
                if pair is not None:
                    return t, pair[0], pair[1]
        return None
