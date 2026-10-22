"""test_records — 数据结构与 JSONL 往返（SPEC §3.3/§3.2，I-01 验收）。"""
from __future__ import annotations

import json

import pytest

from spa2d import records as R


def _full_record() -> R.WindowRecord:
    return R.WindowRecord(
        window_index=12,
        t_start=60,
        t_center=62.0,
        assoc_frame=62,
        search_mode="local",
        state_in="TRACKED",
        state_out="TRACKED",
        accepted=True,
        target_present=True,
        reason="OK",
        est=R.Estimate(x=956.31, y=512.72, vx=0.7, vy=-0.2),
        peak=R.PeakResult(value=3.42, xy_int=(956, 512), xy_sub=(956.31, 512.72),
                          subpixel_used=True, valid_count=5, edge_candidate=False),
        confidence=R.Confidence(pnr=12.7, mu_bg=0.31, sigma_bg=0.24, margin=0.08),
        continuity=R.Continuity(r_cont=3.21, r_cont_high=25.0, r_cont_low=10.0, exempt=False),
        flags=R.Flags(stat=False, edge=False),
        miss_count=0,
        reacq_count=0,
        skipped_by_period=False,
        timing_ms=R.TimingMs(t_pre=8.1, t_search=25.3, t_state=0.2, t_total=33.6),
    )


def _minimal_record() -> R.WindowRecord:
    return R.WindowRecord(
        window_index=0,
        t_start=0,
        t_center=2.0,
        assoc_frame=2,
        search_mode="global",
        state_in="INIT",
        state_out="INIT",
        accepted=False,
        target_present=False,
        reason="NO_PRIOR_GLOBAL",
        est=None,
        peak=None,
        confidence=None,
        continuity=R.Continuity(r_cont=None, r_cont_high=25.0, r_cont_low=10.0, exempt=True),
        flags=R.Flags(),
        miss_count=1,
        reacq_count=0,
        skipped_by_period=False,
        timing_ms=R.TimingMs(t_pre=1.0, t_search=2.0, t_state=0.1, t_total=3.1),
    )


def test_window_record_dict_keys():
    d = _full_record().to_dict()
    assert set(d) == {
        "window_index", "t_start", "t_center", "assoc_frame", "search_mode",
        "state_in", "state_out", "accepted", "target_present", "reason",
        "est", "peak", "confidence", "continuity", "flags",
        "miss_count", "reacq_count", "skipped_by_period", "timing_ms",
    }
    assert d["peak"]["xy_int"] == [956, 512]
    assert d["peak"]["xy_sub"] == [956.31, 512.72]
    assert d["est"] == {"x": 956.31, "y": 512.72, "vx": 0.7, "vy": -0.2}


@pytest.mark.parametrize("rec", [_full_record(), _minimal_record()], ids=["full", "minimal"])
def test_window_record_dict_roundtrip(rec):
    restored = R.WindowRecord.from_dict(rec.to_dict())
    assert restored == rec


@pytest.mark.parametrize("rec", [_full_record(), _minimal_record()], ids=["full", "minimal"])
def test_window_record_json_roundtrip(rec):
    restored = R.WindowRecord.from_json(rec.to_json())
    assert restored == rec
    json.loads(rec.to_json())


def test_window_record_tuple_normalization():
    d = _full_record().to_dict()
    d["peak"]["xy_int"] = [1, 2]
    d["peak"]["xy_sub"] = [1.5, 2.5]
    rec = R.WindowRecord.from_dict(d)
    assert rec.peak.xy_int == (1, 2)
    assert rec.peak.xy_sub == (1.5, 2.5)


def test_jsonl_write_read(tmp_path):
    recs = [_full_record(), _minimal_record()]
    path = tmp_path / "sub" / "windows.jsonl"
    n = R.write_jsonl(recs, path)
    assert n == 2
    back = R.read_jsonl(path)
    assert back == recs
    assert list(R.iter_jsonl(path)) == recs


def test_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "windows.jsonl"
    path.write_text(_full_record().to_json() + "\n\n", encoding="utf-8")
    assert R.read_jsonl(path) == [_full_record()]


@pytest.mark.parametrize("obj", [
    R.Estimate(1.0, 2.0, 3.0, 4.0),
    R.PeakResult(1.5, (3, 4), (3.2, 4.1), True, 5, True),
    R.PeakResult(1.5, (3, 4), None, False, 0, True),
    R.Confidence(9.0, 0.1, 0.2, 0.3),
    R.Confidence(9.0, 0.1, 0.2, 0.3, single_candidate=True),
    R.Continuity(None, 25.0, 10.0, True),
    R.Continuity(3.0, 25.0, 10.0, False),
    R.Flags(stat=True, edge=True),
    R.TimingMs(1.0, 2.0, 3.0, 6.0),
], ids=lambda o: type(o).__name__ + ("_none" if getattr(o, "r_cont", 0) is None else ""))
def test_subrecord_roundtrips(obj):
    assert type(obj).from_dict(obj.to_dict()) == obj


def test_search_result_roundtrip():
    sr = R.SearchResult(cands=[(0.0, 0.5), (1.0, -1.0)], scores=[12.0, 10.0],
                        v_best=(0.0, 0.5), second_best=10.0, chunk_count=3)
    assert R.SearchResult.from_dict(sr.to_dict()) == sr


def test_timing_stats_roundtrip():
    ts = R.TimingStats(layer="t_search", count=100, mean=25.3, median=24.0,
                       p95=40.0, minimum=10.0, maximum=60.0)
    assert R.TimingStats.from_dict(ts.to_dict()) == ts


def test_metrics_result_roundtrip():
    mr = R.MetricsResult(
        config_name="D1_no_overlap",
        association="round_half_up",
        radii_px=(10, 20, 50, 100),
        recall={10: 0.5, 100: 1.0},
        precision={10: 0.4, 100: 0.99},
        correct={100: 1600},
        fp={100: 12},
        valid_outcome=0.995,
        rmse_p=4.43,
        rmse_v=0.26,
        visible_den=1603,
        valid_absent=20,
        total_window_outputs=1603,
        diagnostic_rmse_all_visible=5.1,
        reason_counts={"OK": 1600, "LOW_PNR": 3},
    )
    d = mr.to_dict()
    assert d["recall"] == {"10": 0.5, "100": 1.0}
    assert R.MetricsResult.from_dict(d) == mr


def test_gt_frame_roundtrip_and_columns():
    gt = R.GTFrame(frame=7, timestamp_s=0.14, visible=1, out_of_frame=0,
                   cx=100.5, cy=200.25, bbox_w=12.0, width_px=12.0,
                   width_source="bbox_w", gt_source="published2d", projection_rmse_px=1.3)
    d = gt.to_dict()
    assert tuple(d) == R.GT_COLUMNS
    assert R.GTFrame.from_dict(d) == gt

    missing = R.GTFrame(frame=8, timestamp_s=0.16, visible=0, out_of_frame=1,
                        cx=None, cy=None, projection_rmse_px=None)
    assert R.GTFrame.from_dict(missing.to_dict()) == missing


def test_gt_frame_from_csv_style_empty_strings():
    row = {c: "" for c in R.GT_COLUMNS}
    row.update({"frame": "3", "timestamp_s": "0.06", "visible": "1", "out_of_frame": "0",
                "cx": "10.0", "cy": "20.0"})
    gt = R.GTFrame.from_dict(row)
    assert gt.frame == 3 and gt.cx == 10.0 and gt.cy == 20.0
    assert gt.bbox_w is None and gt.width_source is None


def test_state_and_reason_constants():
    assert "TRACKED_LOW" in R.STATES
    assert "OUT_OF_FRAME" in R.STATES
    assert "NO_PRIOR_GLOBAL" in R.REASONS
    assert set(R.SEARCH_MODES) == {"local", "global"}
