"""test_run_sequence — 端到端主循环/计时/run_meta/帧缓存（SPEC §5.7，I-11 验收）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_sequence as RUN  # noqa: E402

from spa2d import metrics as MET  # noqa: E402
from spa2d import records as R  # noqa: E402
from spa2d import synthetic as S  # noqa: E402


def _frozen_doc(seq="D1/cam2"):
    return {
        "schema_version": 1, "frozen_at": "2026-10-10T00:00:00Z",
        "calibration": {},
        "thresholds": {"theta_high_pnr": 8.0, "theta_low_pnr": 5.0, "theta_margin": 0.02,
                       "theta_edge_pnr": 10.0, "v_stat": 0.05, "m_lost": 3,
                       "reacq_max": 5, "gap_prior_windows": 2},
        "width_ref": {seq: 8.0},
        "preprocess": {"kernel_size": 5},
    }


def _setup(tmp_path, frozen=True, stride=2, n_frames=12):
    syn = tmp_path / "syn"
    S.generate_sanity_set(syn, n_seqs=1, seed=0, n_frames=n_frames, canvas=(96, 128))
    frames_dir = syn / "seq_000" / "frames"

    frozen_path = tmp_path / "thresholds_frozen.yaml"
    if frozen:
        frozen_path.write_text(yaml.safe_dump(_frozen_doc()), encoding="utf-8")

    cfg = yaml.safe_load((REPO / "configs" / "D1_no_overlap.yaml").read_text(encoding="utf-8"))
    cfg["sequence"]["frames_dir"] = str(frames_dir)
    cfg["sequence"]["video"] = None
    cfg["sequence"]["resolution"] = [128, 96]
    cfg["sequence"]["fps"] = 25.0
    cfg["window"] = {"N": 5, "stride": stride}
    cfg["backend"] = "numpy"
    cfg["thresholds_file"] = str(frozen_path)
    cfg["cache"]["host_max_bytes"] = 1 << 20
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path, frames_dir


def test_run_sequence_end_to_end_numpy(tmp_path):
    cfg_path, frames_dir = _setup(tmp_path)
    out = tmp_path / "run"
    rc = RUN.main(["--config", str(cfg_path), "--out", str(out),
                   "--backend", "numpy", "--limit", "4"])
    assert rc == 0

    assert (out / "windows.jsonl").exists()
    records = R.read_jsonl(out / "windows.jsonl")
    assert len(records) == 4
    for i, rec in enumerate(records):
        assert rec.window_index == i
        assert rec.assoc_frame == MET.associate_frame(rec.t_start, 5, "round_half_up")
        assert set(rec.to_dict()) == {
            "window_index", "t_start", "t_center", "assoc_frame", "search_mode",
            "state_in", "state_out", "accepted", "target_present", "reason",
            "est", "peak", "confidence", "continuity", "flags",
            "miss_count", "reacq_count", "skipped_by_period", "timing_ms"}
        assert rec.timing_ms is not None
        assert rec.timing_ms.t_total >= 0.0
        assert rec.reason in R.REASONS and rec.state_out in __import__(
            "spa2d.state_machine", fromlist=["STATES"]).STATES

    timing = json.loads((out / "timing.json").read_text())
    for layer in ("t_pre", "t_search", "t_state", "t_decode", "t_window"):
        assert layer in timing["layers"]
        assert timing["layers"][layer]["layer"] == layer
    assert "window_fps" in timing["fps"] and timing["fps"]["stride"] == 2

    meta = json.loads((out / "run_meta.json").read_text())
    for key in ("config", "frozen", "environment", "data", "seed", "argv",
                "backend", "jit_compile_ms", "cache_stats", "fps"):
        assert key in meta
    assert meta["backend"] == "numpy"
    assert len(meta["frozen"]["sha256"]) == 64
    assert "hit_rate" in meta["cache_stats"]
    assert meta["cache_stats"]["hits"] >= 0


def test_run_sequence_limit_truncates(tmp_path):
    cfg_path, _ = _setup(tmp_path)
    out = tmp_path / "run"
    RUN.main(["--config", str(cfg_path), "--out", str(out), "--backend", "numpy",
              "--limit", "2"])
    assert len(R.read_jsonl(out / "windows.jsonl")) == 2


def test_run_sequence_repeat_ranges(tmp_path):
    cfg_path, _ = _setup(tmp_path)
    out = tmp_path / "run"
    rc = RUN.main(["--config", str(cfg_path), "--out", str(out),
                   "--backend", "numpy", "--limit", "4", "--repeat", "3"])
    assert rc == 0
    for r in (1, 2):
        assert (out / f"repeat_{r}" / "windows.jsonl").exists()
    ranges = json.loads((out / "repeat_ranges.json").read_text())
    assert ranges["n_runs"] == 3
    assert ranges["window_fps"]["range"] is not None
    assert ranges["n_accepted"]["range"] is not None


def test_frozen_missing_exit_5_and_allow_unfrozen(tmp_path):
    cfg_path, _ = _setup(tmp_path, frozen=False)
    out = tmp_path / "run"
    rc = RUN.main(["--config", str(cfg_path), "--out", str(out), "--backend", "numpy",
                   "--limit", "2"])
    assert rc == 5
    rc2 = RUN.main(["--config", str(cfg_path), "--out", str(out), "--backend", "numpy",
                    "--limit", "2", "--allow-unfrozen"])
    assert rc2 == 0
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["frozen"]["used_default"] is True


def test_config_invalid_exit_2(tmp_path):
    cfg_path, _ = _setup(tmp_path)
    doc = yaml.safe_load(cfg_path.read_text())
    doc["preprocess"]["kernel_size"] = 4  # 非法
    cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    rc = RUN.main(["--config", str(cfg_path), "--out", str(tmp_path / "o"),
                   "--backend", "numpy"])
    assert rc == 2


def test_data_missing_exit_3(tmp_path):
    cfg_path, _ = _setup(tmp_path)
    doc = yaml.safe_load(cfg_path.read_text())
    doc["sequence"]["frames_dir"] = str(tmp_path / "nope")
    cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    rc = RUN.main(["--config", str(cfg_path), "--out", str(tmp_path / "o"),
                   "--backend", "numpy"])
    assert rc == 3


def test_cache_reuses_overlapping_preproc(tmp_path):
    cfg_path, _ = _setup(tmp_path, stride=1, n_frames=8)
    out = tmp_path / "run"
    RUN.main(["--config", str(cfg_path), "--out", str(out), "--backend", "numpy"])
    meta = json.loads((out / "run_meta.json").read_text())
    stats = meta["cache_stats"]
    # stride=1、N=5：窗口大量重叠 → 命中率应显著 > 0（零重复预处理）
    assert stats["hit_rate"] > 0.5


def test_record_accel_estimate_includes_acceleration():
    """回归：accel 档 est 必须用 center_frame_position_accel（含 ½f²a）。"""
    from types import SimpleNamespace
    import pytest

    pk = R.PeakResult(value=1.0, xy_int=(10, 10), xy_sub=(10.0, 10.0),
                      subpixel_used=False, valid_count=5, edge_candidate=False)
    outcome = SimpleNamespace(target_present=True, accepted=True, state_in="TRACKED",
                              state_out="TRACKED", reason="OK", flags=R.Flags(),
                              miss_count=0, reacq_count=0, skipped_by_period=False)
    plan = SimpleNamespace(t_center=11.0, search_mode="local")
    cfg = SimpleNamespace(metrics=SimpleNamespace(association="round_half_up"))
    res = SimpleNamespace(v_best=(2.0, 1.0))
    cont = R.Continuity(None, 25.0, 10.0, True)

    rec = RUN._record(0, 8, 8, cfg, plan, outcome, res, pk, 1.0, 0.1, cont,
                      0.0, 0.0, 0.0, 0.0, a_best=(0.5, -0.25), use_accel=True)
    f = (8 - 1) / 2.0
    assert rec.est.x == pytest.approx(10.0 + f * 2.0 + 0.5 * f * f * 0.5)
    assert rec.est.y == pytest.approx(10.0 + f * 1.0 + 0.5 * f * f * -0.25)

    rec2 = RUN._record(0, 8, 8, cfg, plan, outcome, res, pk, 1.0, 0.1, cont,
                       0.0, 0.0, 0.0, 0.0, a_best=(0.5, -0.25), use_accel=False)
    assert rec2.est.x == pytest.approx(10.0 + f * 2.0)
