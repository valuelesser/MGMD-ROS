"""test_calibration — calibrate 七步协议/冻结/白名单（SPEC §5.12，I-12 验收）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import calibrate as CAL  # noqa: E402

from spa2d import synthetic as S  # noqa: E402


def _tiny_synth(tmp_path):
    syn = tmp_path / "syn"
    S.generate_sanity_set(syn, n_seqs=1, seed=0, n_frames=15, canvas=(96, 128))
    return syn


def test_calibrate_end_to_end(tmp_path, monkeypatch):
    syn = _tiny_synth(tmp_path)
    out = tmp_path / "thresholds_frozen.yaml"
    trace_dir = tmp_path / "trace"
    monkeypatch.chdir(tmp_path)  # outputs/metrics 写到 tmp
    rc = CAL.main(["--synthetic", str(syn), "--validation", "synthetic",
                   "--configs", str(REPO / "configs" / "D1_no_overlap.yaml"),
                   "--out", str(out), "--backend", "numpy"])
    assert rc == 0, "calibrate failed"
    doc = yaml.safe_load(out.read_text())
    assert doc["schema_version"] == 1
    th = doc["thresholds"]
    assert th["theta_edge_pnr"] == th["theta_high_pnr"] + 2.0
    assert th["theta_low_pnr"] <= th["theta_high_pnr"]
    assert doc["preprocess"]["kernel_size"] in CAL.KERNELS
    assert isinstance(doc["width_ref"], dict) and doc["width_ref"]
    assert len(doc["calibration"]["synthetic_manifest_sha256"]) == 64
    trace = json.loads((tmp_path / "outputs" / "metrics" / "calibration_trace.json").read_text())
    for key in ("step1_kernel", "step2_theta_high", "step3_theta_low", "step4_theta_margin",
                "step5_theta_edge", "step6_width_ref", "step7_tail_check"):
        assert key in trace


def test_calibrate_whitelist_rejects(tmp_path, monkeypatch):
    syn = _tiny_synth(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = CAL.main(["--synthetic", str(syn), "--validation", "D2/cam1",
                   "--configs", str(REPO / "configs" / "D1_no_overlap.yaml"),
                   "--out", str(tmp_path / "f.yaml"), "--backend", "numpy"])
    assert rc == 4


def test_calibrate_force_guard(tmp_path, monkeypatch):
    syn = _tiny_synth(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "frozen.yaml"
    common = ["--synthetic", str(syn), "--validation", "synthetic",
              "--configs", str(REPO / "configs" / "D1_no_overlap.yaml"),
              "--out", str(out), "--backend", "numpy"]
    assert CAL.main(common) == 0
    assert CAL.main(common) == 5           # 已存在且未 --force
    assert CAL.main(common + ["--force"]) == 0
