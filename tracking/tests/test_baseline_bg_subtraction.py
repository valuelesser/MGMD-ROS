"""test_baseline_bg_subtraction — 背景建模基线（SPEC §5.13，I-15 验收）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import baseline_bg_subtraction as BG  # noqa: E402

from spa2d import synthetic as S  # noqa: E402


def _setup(tmp_path):
    bench = tmp_path / "bench"
    S.generate_benchmark(bench, seed=0, n_frames=24, canvas=(96, 128))
    cfg = yaml.safe_load((REPO / "configs" / "D1_no_overlap.yaml").read_text())
    cfg["sequence"]["frames_dir"] = str(bench / "benchmark" / "frames")
    cfg["sequence"]["video"] = None
    cfg["sequence"]["resolution"] = [128, 96]
    cfg["window"] = {"N": 5, "stride": 5}
    cfg["backend"] = "numpy"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    gt = bench / "benchmark" / "gt.csv"
    return cfg_path, gt


def test_baseline_nn(tmp_path):
    cfg_path, gt = _setup(tmp_path)
    out = tmp_path / "bg"
    rc = BG.main(["--config", str(cfg_path), "--gt", str(gt), "--out", str(out), "--tracker", "nn"])
    assert rc == 0
    for f in ("windows.jsonl", "metrics.json", "metrics_fullframe.json", "run_meta.json"):
        assert (out / f).exists(), f
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["baseline"] == "median_bg" and meta["tracker"] == "nn"
    metrics = json.loads((out / "metrics.json").read_text())
    assert "recall" in metrics and "100" in metrics["recall"]
    # 暗点目标在平坦背景上应被检出（部分窗口 accepted）
    assert sum(1 for line in (out / "windows.jsonl").read_text().splitlines()
               if '"target_present": true' in line) > 0


def test_baseline_kalman(tmp_path):
    cfg_path, gt = _setup(tmp_path)
    out = tmp_path / "bgk"
    rc = BG.main(["--config", str(cfg_path), "--gt", str(gt), "--out", str(out),
                  "--tracker", "kalman"])
    assert rc == 0
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["tracker"] == "kalman"


def test_baseline_missing_gt(tmp_path):
    cfg_path, _ = _setup(tmp_path)
    rc = BG.main(["--config", str(cfg_path), "--gt", str(tmp_path / "nope.csv"),
                  "--out", str(tmp_path / "o")])
    assert rc == 3
