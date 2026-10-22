"""test_tables_figures — make_tables / make_figures（SPEC §4.1，I-14 验收）。"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import make_figures as MF  # noqa: E402
import make_tables as MT  # noqa: E402
import run_sequence as RUN  # noqa: E402

from spa2d import synthetic as S  # noqa: E402


def _frozen_doc(seq="D1/cam2"):
    return {"schema_version": 1, "frozen_at": "2026-10-10T00:00:00Z", "calibration": {},
            "thresholds": {"theta_high_pnr": 8.0, "theta_low_pnr": 5.0, "theta_margin": 0.02,
                           "theta_edge_pnr": 10.0, "v_stat": 0.05, "m_lost": 3,
                           "reacq_max": 5, "gap_prior_windows": 2},
            "width_ref": {seq: 8.0}, "preprocess": {"kernel_size": 5}}


def _make_run(tmp_path):
    syn = tmp_path / "syn"
    S.generate_sanity_set(syn, n_seqs=1, seed=0, n_frames=12, canvas=(96, 128))
    seq_dir = syn / "seq_000"
    frozen = tmp_path / "thresholds_frozen.yaml"
    frozen.write_text(yaml.safe_dump(_frozen_doc()), encoding="utf-8")
    cfg = yaml.safe_load((REPO / "configs" / "D1_no_overlap.yaml").read_text())
    cfg["sequence"]["frames_dir"] = str(seq_dir / "frames")
    cfg["sequence"]["video"] = None
    cfg["sequence"]["resolution"] = [128, 96]
    cfg["sequence"]["fps"] = 25.0
    cfg["window"] = {"N": 5, "stride": 2}
    cfg["backend"] = "numpy"
    cfg["thresholds_file"] = str(frozen)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    runs = tmp_path / "runs"
    rc = RUN.main(["--config", str(cfg_path), "--out", str(runs / "D1_no_overlap"),
                   "--backend", "numpy"])
    assert rc == 0
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    shutil.copy(seq_dir / "gt.csv", gt_dir / "D1_cam2.csv")
    return runs, gt_dir


def test_make_tables(tmp_path):
    runs, gt_dir = _make_run(tmp_path)
    out = tmp_path / "metrics"
    rc = MT.main(["--runs", str(runs), "--gt", str(gt_dir), "--out", str(out)])
    assert rc == 0
    for f in ("summary.csv", "summary.md", "paper_vs_ours.md", "sensitivity.csv", "stratified.csv"):
        assert (out / f).exists(), f
    text = (out / "summary.csv").read_text()
    assert "D1_no_overlap" in text
    assert "recall@100" in text
    pv = (out / "paper_vs_ours.md").read_text()
    assert "Tier" in pv and "N/A" in pv
    sens = (out / "sensitivity.csv").read_text()
    assert "radius_px" in sens
    import csv

    rows = list(csv.DictReader((out / "sensitivity.csv").open()))
    assert {r["radius_px"] for r in rows} == {"10", "20", "50", "100"}
    assert {r["association"] for r in rows} == {"round_half_up", "floor"}


def test_make_figures_with_and_without_images(tmp_path):
    runs, gt_dir = _make_run(tmp_path)
    out = tmp_path / "figs"
    rc = MF.main(["--runs", str(runs), "--gt", str(gt_dir), "--out", str(out), "--failures", "3"])
    assert rc == 0
    assert (out / "fig7_recall_fps.png").exists()
    assert (out / "fig8_trajectories.png").exists()

    out2 = tmp_path / "figs_noimg"
    rc2 = MF.main(["--runs", str(runs), "--gt", str(gt_dir), "--out", str(out2),
                   "--no-images"])
    assert rc2 == 0
    assert (out2 / "fig7_recall_fps.png").exists()
    assert not (out2 / "failures_fa.png").exists()
    assert not (out2 / "failures_miss.png").exists()


def test_tables_empty_runs_ok(tmp_path):
    out = tmp_path / "metrics"
    rc = MT.main(["--runs", str(tmp_path / "none"), "--gt", str(tmp_path / "gt"),
                  "--out", str(out)])
    assert rc == 0
    assert (out / "summary.csv").exists()
