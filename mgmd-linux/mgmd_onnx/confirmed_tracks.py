"""Confirmation from ordered detector results, shared by the two compute lanes."""


class ConfirmedTracks:
    def __init__(self, frames):
        self.frames = frames
        self.streaks = {}
        self.snapshot = {}

    def observe(self, seq, verified, reused):
        streaks, confirmed = {}, {}
        for row in verified:
            track_id = int(row[5])
            last_seq, count = self.streaks.get(track_id, (-2, 0))
            count = count + 1 if seq == last_seq + 1 else 1
            if track_id in self.snapshot or count >= self.frames:
                confirmed[track_id] = row[4]
            else:
                streaks[track_id] = seq, count
        for row in reused:
            track_id = int(row[5])
            if track_id in self.snapshot:
                confirmed[track_id] = self.snapshot[track_id]
        self.streaks = streaks
        # Publish a fresh snapshot. Workers may still hold the previous dictionary.
        self.snapshot = confirmed
