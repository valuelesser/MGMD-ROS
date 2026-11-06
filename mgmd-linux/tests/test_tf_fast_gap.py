"""ROS 跟踪在跳帧和大位移时的关联行为。"""
from mgmd_onnx.tf_fast import FastSortTracker


def test_fast_sort_keeps_id_across_large_motion_and_frame_gaps():
    box = lambda x: (x, 10, x + 10, 20)
    sequence = [(1, 0), (2, 8), (5, 32), (8, 56)]

    baseline = FastSortTracker(d_min=0.02, d_mode="self")
    improved = FastSortTracker(d_min=0.02, d_mode="self",
                               frame_delta_predict=True,
                               association_radius_factor=2.0)
    for frame, x in sequence:
        baseline.update([box(x)], frame)
        improved.update([box(x)], frame)

    assert baseline._next_id == 5
    assert improved._next_id == 2
    assert improved.states[1].boxes[-1][0] == 8
    assert improved._predicted_at[1] == 8
