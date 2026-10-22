"""tests/test_mgmd_report.py — 指标汇总 + 消融表 + AP（SPEC §5.7/§9.3，US-007）。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from spa2d import metrics as MET
from spa2d import records as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_report_mgmd as MRM  # noqa: E402


def _rec(idx: int, cx: float | None, cy: float | None, score: float = 0.5) -> R.WindowRecord:
    """构造逐帧记录：cx/cy=None 表示无输出。"""
    est = None if cx is None else R.Estimate(x=cx, y=cy, vx=0.0, vy=0.0)
    conf = None if cx is None else R.Confidence(pnr=score, mu_bg=0.0, sigma_bg=0.0,
                                                margin=score, single_candidate=True)
    present = cx is not None
    return R.WindowRecord(
        window_index=idx, t_start=idx, t_center=float(idx), assoc_frame=idx,
        search_mode="global", state_in="ABSENT", state_out="INIT" if present else "ABSENT",
        accepted=present, target_present=present, reason="OK" if present else "ABSENT",
        est=est, confidence=conf)


def _gt(n: int = 4, cx: float = 100.0, cy: float = 100.0) -> list[R.GTFrame]:
    return [R.GTFrame(frame=i, timestamp_s=i / 50.0, visible=1, out_of_frame=0,
                      cx=cx, cy=cy) for i in range(n)]


def _write_run(root: Path, name: str, seq: str, modules: list[str],
               recs: list[R.WindowRecord], gt: list[R.GTFrame],
               metrics_consistent: bool = True) -> Path:
    run = root / name
    run.mkdir(parents=True)
    R.write_jsonl(recs, run / "windows.jsonl")
    meta = {"config": {"name": f"mgmd_{seq.split('/')[0]}", "path": None,
                       "snapshot": {"sequence": {"name": seq}}},
            "mgmd": {"modules": modules}, "backend": "numpy"}
    (run / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    res = MET.evaluate_windows(recs, gt, config_name=name)
    d = res.to_dict()
    if not metrics_consistent:
        d["recall"]["100"] = 0.123  # 破坏一致性，触发复算告警
    (run / "metrics.json").write_text(json.dumps(d), encoding="utf-8")
    return run


@pytest.fixture
def two_runs(tmp_path: Path):
    """run_full：TP(d=5,s=0.9) / FP(d=200,s=0.8) / 无输出 / TP(d=50,s=0.7)；
    run_mfe：仅 rec0 TP。"""
    gt = _gt(4)
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    R.write_gt_csv(gt, gt_dir / "D1_cam9.csv")
    runs = tmp_path / "runs"
    recs_full = [_rec(0, 105.0, 100.0, 0.9), _rec(1, 300.0, 300.0, 0.8),
                 _rec(2, None, None), _rec(3, 150.0, 100.0, 0.7)]
    _write_run(runs, "D1_mfe-tf-lac-lad", "D1/cam9", ["mfe", "tf", "lac", "lad"],
               recs_full, gt)
    _write_run(runs, "D1_mfe", "D1/cam9", ["mfe"], [_rec(0, 105.0, 100.0, 0.9)] +
               [_rec(i, None, None) for i in range(1, 4)], gt)
    return runs, gt_dir


def test_summary_csv_exact_values(two_runs, tmp_path):
    runs, gt_dir = two_runs
    out, mout = tmp_path / "r.md", tmp_path / "metrics"
    assert MRM.main(["--runs", str(runs), "--gt", str(gt_dir), "--out", str(out),
                     "--metrics-out", str(mout)]) == 0
    rows = {r["run"]: r for r in _read_csv(mout / "mgmd_summary.csv")
            if r["radius_px"] == "100"}
    full = rows["D1_mfe-tf-lac-lad"]
    # recall=2/4=0.5, precision=2/3, F1=0.5714（手算）
    assert float(full["recall"]) == pytest.approx(0.5)
    assert float(full["precision"]) == pytest.approx(2 / 3)
    assert float(full["f1"]) == pytest.approx(0.5714, abs=1e-3)
    assert int(full["correct"]) == 2 and int(full["fp"]) == 1
    mfe = rows["D1_mfe"]
    assert float(mfe["recall"]) == pytest.approx(0.25)


def test_per_radius_values(two_runs, tmp_path):
    runs, gt_dir = two_runs
    out, mout = tmp_path / "r.md", tmp_path / "m"
    MRM.main(["--runs", str(runs), "--gt", str(gt_dir), "--out", str(out),
              "--metrics-out", str(mout)])
    rows = {(r["run"], r["radius_px"]): r for r in _read_csv(mout / "mgmd_summary.csv")}
    # run_full：rec0 d=5 命中所有半径；rec3 d=50 仅命中 50/100；rec1 d=200 全 FP
    assert int(rows[("D1_mfe-tf-lac-lad", "10")]["correct"]) == 1
    assert int(rows[("D1_mfe-tf-lac-lad", "50")]["correct"]) == 2
    assert int(rows[("D1_mfe-tf-lac-lad", "100")]["correct"]) == 2
    assert float(rows[("D1_mfe-tf-lac-lad", "10")]["recall"]) == pytest.approx(0.25)


def _read_csv(path: Path) -> list[dict]:
    import csv
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_ablation_table_structure(two_runs, tmp_path):
    runs, gt_dir = two_runs
    out, mout = tmp_path / "r.md", tmp_path / "m"
    MRM.main(["--runs", str(runs), "--gt", str(gt_dir), "--out", str(out),
              "--metrics-out", str(mout)])
    md = out.read_text(encoding="utf-8")
    assert "模块消融表" in md and "MFE+TF+LAC+LAD（全流水线）" in md
    assert "MFE+TF" in md and "Table 3" in md
    ab = _read_csv(mout / "mgmd_ablation.csv")
    tiers = {r["modules"] for r in ab}
    assert "mfe" in tiers and "mfe,tf,lac,lad" in tiers     # 四档键（缺档以 — 占位）


def test_ap_stretch_handcomputed(two_runs, tmp_path):
    """AP 手算验证：scores 0.9(TP)/0.8(FP)/0.7(TP)，visible_den=4
    → AP=(26×1.0 + 25×2/3)/101 ≈ 0.4224。"""
    runs, gt_dir = two_runs
    loaded = MRM.load_runs(runs, gt_dir)
    full = next(r for r in loaded if r.dir == "D1_mfe-tf-lac-lad")
    ap = MRM.radius_ap(full.records, full.gt)
    assert ap == pytest.approx((26 * 1.0 + 25 * (2 / 3)) / 101, abs=1e-6)
    out, mout = tmp_path / "r.md", tmp_path / "m"
    MRM.main(["--runs", str(runs), "--gt", str(gt_dir), "--out", str(out),
              "--metrics-out", str(mout)])
    assert (mout / "mgmd_ap.csv").is_file()
    assert "AP@100" in out.read_text(encoding="utf-8")


def test_metrics_json_consistency_check(two_runs, tmp_path):
    """复算与 metrics.json 不一致 → warning（数值可复算复现的自检）。"""
    runs, gt_dir = two_runs
    gt = _gt(4)
    _write_run(runs, "D1_bad", "D1/cam9", ["mfe"], [_rec(0, 105.0, 100.0)] +
               [_rec(i, None, None) for i in range(1, 4)], gt,
               metrics_consistent=False)
    with pytest.warns(UserWarning, match="复算不一致"):
        MRM.main(["--runs", str(runs), "--gt", str(gt_dir),
                  "--out", str(tmp_path / "r.md"), "--metrics-out", str(tmp_path / "m")])


def test_missing_gt_graceful(two_runs, tmp_path):
    runs, _gt_dir = two_runs
    empty_gt = tmp_path / "gt_empty"
    empty_gt.mkdir()
    out, mout = tmp_path / "r.md", tmp_path / "m"
    rc = MRM.main(["--runs", str(runs), "--gt", str(empty_gt), "--out", str(out),
                   "--metrics-out", str(mout)])
    assert rc == 0                                     # 缺 GT → N/A，不崩
    rows = _read_csv(mout / "mgmd_summary.csv")
    assert all(r["precision"] == "" for r in rows)


def test_spa2d_comparison_table(two_runs, tmp_path):
    runs, gt_dir = two_runs
    spa2d = tmp_path / "spa2d_summary.csv"
    spa2d.write_text("run,config,sequence,backend,n_windows,window_fps,"
                     "effective_frame_fps,valid_outcome,recall@100,precision@100,"
                     "rmse_p,rmse_v,tier\n"
                     "x,D1_no_overlap,D1/cam2,jax,1,,,0.99,1.0,0.99,,,B\n",
                     encoding="utf-8")
    out, mout = tmp_path / "r.md", tmp_path / "m"
    MRM.main(["--runs", str(runs), "--gt", str(gt_dir), "--spa2d", str(spa2d),
              "--out", str(out), "--metrics-out", str(mout)])
    md = out.read_text(encoding="utf-8")
    assert "与 SPA2D 基线对比" in md and "100.00" in md     # SPA2D recall=1.0→100.00


def test_report_sections_present(two_runs, tmp_path):
    """issue-024 报告要素：达标核验 / Tier B 声明 / 失败分析 / 配置快照。"""
    runs, gt_dir = two_runs
    out, mout = tmp_path / "r.md", tmp_path / "m"
    assert MRM.main(["--runs", str(runs), "--gt", str(gt_dir), "--out", str(out),
                     "--metrics-out", str(mout)]) == 0
    md = out.read_text(encoding="utf-8")
    assert "指标达标核验" in md
    assert "域差与 Tier B 声明" in md
    assert "不外推到 onboard" in md
    assert "失败案例分析" in md
    assert "配置与环境快照" in md
    # D1 全流水线档存在 → 失败分析表含该序列行
    assert "| D1/cam9 |" in md


def test_targets_verdict_columns(two_runs, tmp_path):
    runs, gt_dir = two_runs
    out, mout = tmp_path / "r.md", tmp_path / "m"
    MRM.main(["--runs", str(runs), "--gt", str(gt_dir), "--out", str(out),
              "--metrics-out", str(mout)])
    md = out.read_text(encoding="utf-8")
    assert "✅ 达标" in md or "❌ 未达标" in md      # D1 档结论列存在


def test_no_runs_exit_3(tmp_path: Path):
    rc = MRM.main(["--runs", str(tmp_path / "empty"), "--gt", str(tmp_path),
                   "--out", str(tmp_path / "r.md")])
    assert rc == 3


def test_end_to_end_with_pipeline(tmp_path: Path):
    """复用 pipeline 合成序列：run → report 全链路（run 记录可复算复现）。"""
    from tests.test_mgmd_pipeline import _make_seq, _write_cfg
    from mgmd.config import load_config
    from mgmd import pipeline

    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    frames_dir, gt_csv = _make_seq(seq_dir)
    cfg = load_config(_write_cfg(seq_dir, frames_dir))
    runs_root = tmp_path / "runs"
    assert pipeline.run(cfg, frozenset({"mfe"}), gt_csv, runs_root / "T1_mfe",
                        None, "numpy") == 0
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "T1_cam9.csv").write_text(gt_csv.read_text(encoding="utf-8"),
                                        encoding="utf-8")
    out, mout = tmp_path / "report.md", tmp_path / "m"
    rc = MRM.main(["--runs", str(runs_root), "--gt", str(gt_dir), "--out", str(out),
                   "--metrics-out", str(mout)])
    assert rc == 0
    rows = _read_csv(mout / "mgmd_summary.csv")
    r100 = next(r for r in rows if r["radius_px"] == "100")
    assert float(r100["recall"]) > 0.5               # 合成序列复算 recall 达标
    md = out.read_text(encoding="utf-8")
    assert "T1/cam9" in md
