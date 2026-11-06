"""Frame windows used by the ROS detector with continuous and dropped input."""

from mgmd_onnx.frame_buffer import FrameWindow


def _processed(frame_ids, k=1):
    selector = FrameWindow(k)
    frames = {}
    last_done = -1
    output = []
    for frame_id in frame_ids:
        if selector.push(frame_id):
            frames.clear()
            last_done = -1
        frames[frame_id] = True
        while len(frames) > max(6, 2 * k + 1):
            del frames[next(iter(frames))]
        while (window := selector.ready(frames, last_done)) is not None:
            t, prev, next_ = window
            output.append(window)
            last_done = t
            for old in [key for key in frames if key <= prev]:
                del frames[old]
    return output


def test_continuous_input_keeps_exact_neighbors():
    out = _processed(range(20), k=1)
    assert out == [(t, t - 1, t + 1) for t in range(1, 19)]


def test_stride_four_input_keeps_processing():
    out = _processed(range(0, 200, 4), k=1)
    assert len(out) >= 45
    assert all(prev == t - 4 and next_ == t + 4 for t, prev, next_ in out)


def test_k_two_does_not_shrink_to_adjacent_frames():
    out = _processed(range(20), k=2)
    assert out == [(t, t - 2, t + 2) for t in range(2, 18)]


def test_rewind_allows_second_loop():
    out = _processed(list(range(8)) + list(range(8)))
    assert [t for t, _, _ in out] == list(range(1, 7)) * 2


def test_single_oversized_gap_does_not_expand_limit():
    selector = FrameWindow(1)
    for frame_id in (0, 1, 2, 8, 9):
        selector.push(frame_id)
    assert selector.max_gap() == 1
    assert selector.neighbors({2: True, 8: True, 9: True}, 8) is None
