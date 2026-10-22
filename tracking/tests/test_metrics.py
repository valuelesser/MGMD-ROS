"""test_metrics — evaluate_windows golden 手算对照（SPEC §5.8，I-13 验收）。"""
from __future__ import annotations

import math

import pytest

from spa2d import metrics as M
from spa2d.records import Estimate, Flags, GTFrame, MetricsResult, WindowRecord

N = 5


def _gt():
    rows = []
    for f in range(12):
        if f == 7:
            rows.append(GTFrame(frame=f, timestamp_s=f * 0.02, visible=0, out_of_frame=1,
                                cx=7.0, cy=0.0))
        elif f == 9:
            rows.append(GTFrame(frame=f, timestamp_s=f * 0.02, visible=0, out_of_frame=0))
        else:
            rows.append(GTFrame(frame=f, timestamp_s=f * 0.02, visible=1, out_of_frame=0,
                                cx=float(f), cy=0.0))
    return rows


def _rec(i, est, present, reason="OK", search_mode="local", state_in="TRACKED", edge=False,
         wi=None):
    return WindowRecord(
        window_index=i if wi is None else wi, t_start=i, t_center=i + 2.0,
        assoc_frame=i + 2, search_mode=search_mode, state_in=state_in, state_out="TRACKED",
        accepted=present, target_present=present, reason=reason,
        est=est, flags=Flags(edge=edge),
        continuity=None, timing_ms=None)


def _records():
    # i: f=i+2
    recs = []
    for i in range(5):                       # f=2..6, correct (i=4 offset 30)
        cx = 6.0 + 30.0 if i == 4 else float(i + 2)
        recs.append(_rec(i, Estimate(cx, 0.0, 1.0, 0.0), True))
    recs.append(_rec(5, Estimate(7.0, 0.0, 1.0, 0.0), True))     # f=7 oof → fp
    recs.append(_rec(6, Estimate(8.0, 0.0, 1.0, 0.0), True))     # f=8 correct
    recs.append(_rec(7, None, False, reason="LOW_PNR"))          # f=9 absent → valid_absent
    recs.append(_rec(8, None, False, reason="LOW_MARGIN"))       # f=10 visible → miss
    recs.append(_rec(9, Estimate(11.0, 0.0, 1.0, 0.0), True))    # f=11 correct
    return recs


def test_golden_metrics():
    res = M.evaluate_windows(_records(), _gt(), radii=(10, 20, 50, 100),
                             assoc_mode="round_half_up", config_name="golden")
    assert res.visible_den == 8
    assert res.correct == {10: 6, 20: 6, 50: 7, 100: 7}
    assert res.fp == {10: 2, 20: 2, 50: 1, 100: 1}
    assert res.recall[10] == pytest.approx(6 / 8)
    assert res.recall[50] == pytest.approx(7 / 8)
    assert res.precision[10] == pytest.approx(6 / 8)
    assert res.precision[50] == pytest.approx(7 / 8)
    assert res.valid_absent == 1
    assert res.valid_outcome == pytest.approx((7 + 1) / 10)
    assert res.rmse_p == pytest.approx(math.sqrt(900 / 7), abs=1e-9)
    assert res.rmse_v == pytest.approx(0.0, abs=1e-9)
    assert res.diagnostic_rmse_all_visible == pytest.approx(math.sqrt(900 / 7))
    assert res.reason_counts == {"OK": 8, "LOW_PNR": 1, "LOW_MARGIN": 1}
    assert res.fp_sources == {"unknown": 1}


def test_assoc_modes_half_frame():
    assert M.associate_frame(0, 8, "round_half_up") == 4
    assert M.associate_frame(0, 8, "floor") == 3
    assert M.associate_frame(0, 5, "round_half_up") == 2


def test_both_modes_and_labels():
    res = M.evaluate_both_modes(_records(), _gt(), config_name="x")
    assert set(res) == {"round_half_up", "floor"}
    assert res["floor"].association == "floor"


def test_visible_den_zero_warns():
    gt = [GTFrame(frame=f, timestamp_s=0.0, visible=0, out_of_frame=0) for f in range(12)]
    with pytest.warns(UserWarning):
        res = M.evaluate_windows(_records(), gt, config_name="x")
    assert math.isnan(res.recall[100])
    assert res.valid_absent == 2   # 2 个 target_present False 且 GT 不可见
    assert res.fp[100] == 8        # 8 个 accepted 因 GT 不可见计为 fp


def test_distance_distribution():
    res = M.distance_distribution(_records(), _gt(), width_ref=8.0)
    assert res["n"] == 7  # accepted 且 GT 可见（i=5 的 oof 不计）
    assert res["p50"] is not None and res["p95"] >= res["p50"]


def test_sensitivity_rows_and_json(tmp_path):
    res = M.evaluate_windows(_records(), _gt(), config_name="x")
    rows = M.sensitivity_rows(res)
    assert [r["radius_px"] for r in rows] == [10, 20, 50, 100]
    p = tmp_path / "metrics.json"
    M.write_metrics_json(res, p)
    back = M.load_metrics_json(p)
    assert back == res


def test_repeat_ranges():
    r1 = M.evaluate_windows(_records(), _gt(), config_name="x")
    r2 = M.evaluate_windows(_records(), _gt(), config_name="x")
    rr = M.repeat_ranges([r1, r2])
    assert rr["n_runs"] == 2
    assert rr["recall@100"]["range"] == 0.0
    assert rr["valid_outcome"]["range"] == 0.0


def test_fp_source_overlay():
    overlay = {5: "bird"}
    res = M.evaluate_windows(_records(), _gt(), fp_sources=overlay, config_name="x")
    assert res.fp_sources == {"bird": 1}


def test_gt_velocity_linear():
    vel = M.gt_velocity(_gt())
    assert vel[2] == pytest.approx([1.0, 0.0])
    assert vel[0] == pytest.approx([1.0, 0.0])
    assert vel[11] == pytest.approx([1.0, 0.0])
    assert math.isnan(vel[7, 0])  # oof
