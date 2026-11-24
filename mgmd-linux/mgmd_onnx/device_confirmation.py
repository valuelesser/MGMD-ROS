"""Publish an immutable CUDA confirmation snapshot from ordered ROS results."""
import torch


class DeviceConfirmation:
    def __init__(self, frames):
        self.frames = frames
        self.snapshot = None

    @staticmethod
    def match(snapshot, ids):
        _, event, keys, scores, streaks, confirmed = snapshot
        stream = torch.cuda.current_stream()
        stream.wait_event(event)
        for tensor in (keys, scores, streaks, confirmed):
            tensor.record_stream(stream)
        # A sentinel makes the reduction well-defined after an empty frame.
        present, index = (ids[:, None] == keys).max(dim=1)
        return present & confirmed[index], scores[index], torch.where(present, streaks[index], 0)

    def observe(self, seq, boxes, reuse_count):
        stream = torch.cuda.current_stream()
        boxes.record_stream(stream)
        ids = boxes[:, 5].long()
        scores = boxes[:, 4]
        verified = torch.arange(len(ids), device=ids.device) < len(ids) - reuse_count
        streak = torch.ones_like(ids)
        confirmed = torch.zeros_like(verified)
        previous = self.snapshot
        if previous is not None:
            confirmed, old_scores, previous_streak = self.match(previous, ids)
            if seq == previous[0] + 1:
                streak += previous_streak
            scores = torch.where(verified, scores, old_scores)
        confirmed = confirmed | (verified & (streak >= self.frames))
        streak = torch.where(verified & ~confirmed, streak, 0)
        state = (torch.cat((ids, ids.new_full((1,), -1))),
                 torch.cat((scores, scores.new_zeros(1))),
                 torch.cat((streak, streak.new_zeros(1))),
                 torch.cat((confirmed, confirmed.new_zeros(1))))
        self.snapshot = (seq, stream.record_event(), *state)
