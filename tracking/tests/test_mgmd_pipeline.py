"""tests/test_mgmd_pipeline.py — 端到端 runner + WindowRecord 适配（SPEC §3.2/§4.1，US-006）。

合成序列冒烟（numpy 级，不依赖 GPU/真实数据）：四档模块子集、记录 schema 往返、
make_tables.py 兼容、downscale 坐标映射、数据缺失退出码。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from mgmd.config import MgmdError, load_config
from mgmd import pipeline
from spa2d import metrics as MET
from spa2d import records as R

W, H = 512, 384
N_FRAMES = 12
VEL = (6, 5)
SQ = 12
X0, Y0 = 100, 150

try:
    import torch  # noqa: F401
    _TORCH = True
except ImportError:
    _TORCH = False
_WEIGHTS_OK = (Path("data_raw/mgmd_weights/Net_best.pth").is_file()
               and Path("data_raw/mgmd_weights/yolov5s_GLAD-crop.pt").is_file()
               and (Path("third_party/yolov5-v6.0/models/experimental.py").is_file()))
_APPEARANCE = _TORCH and _WEIGHTS_OK


def _square_center(t: int) -> tuple[float, float]:
    return (X0 + VEL[0] * t + SQ / 2, Y0 + VEL[1] * t + SQ / 2)


def _make_seq(tmp_path: Path, n: int = N_FRAMES) -> tuple[Path, Path]:
    """合成序列：平滑纹理背景 + 匀速白方块；返回 (frames_dir, gt_csv)。"""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 160, size=(H, W), dtype=np.uint8)
    bg = cv2.GaussianBlur(noise, (9, 9), 0)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    gts: list[R.GTFrame] = []
    for t in range(n):
        img = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (X0 + VEL[0] * t, Y0 + VEL[1] * t),
                      (X0 + VEL[0] * t + SQ, Y0 + VEL[1] * t + SQ), (255, 255, 255), -1)
        cv2.imwrite(str(frames_dir / f"f_{t:04d}.png"), img)
        cx, cy = _square_center(t)
        gts.append(R.GTFrame(frame=t, timestamp_s=t / 50.0, visible=1, out_of_frame=0,
                             cx=cx, cy=cy))
    gt_csv = tmp_path / "T1_cam9.csv"
    R.write_gt_csv(gts, gt_csv)
    return frames_dir, gt_csv


def _write_cfg(tmp_path: Path, frames_dir: Path, downscale: float = 1.0) -> Path:
    doc = {
        "schema_version": 1, "config_name": "mgmd_T1",
        "sequence": {"name": "T1/cam9", "frames_dir": str(frames_dir),
                     "frame_glob": "*.png", "fps": 50.0, "resolution": [W, H],
                     "color_order": "BGR"},
        "mfe": {"blur_kernel": 11, "grid_spacing": [96, 72], "lk_win_size": 15,
                "lk_max_level": 3, "max_flow_px": 50, "min_inliers": 9,
                "ransac_thresh": 3.0, "frame_gap": 1, "or_aggregation": False,
                "overflow": "drop", "motion_coherence": True, "diff_thresh_base": 5, "diff_adaptive": True,
                "median_blur": 5, "open_iter": 1, "close_iter": 3, "morph_kernel": 3,
                "area_min": 30, "area_max": 3000, "ratio_min": 0.6, "ratio_max": 3.0,
                "max_boxes": 50, "enlarge_px": 2, "motion_avg_dist_min": 1.0,
                "motion_ratio_theta_max": 0.8, "motion_ratio_dist_max": 0.8},
        "tf": {"iou_thresh": 0.3, "max_age": 3, "min_hits": 1,
               "alg1_age": 3, "alg1_window": 3, "alg1_d_min": 1.0},
        "lac": {"weights": "data_raw/mgmd_weights/Net_best.pth", "input_size": 32,
                "prob_thresh": 0.5, "device": "cpu"},
        "lad": {"weights": "data_raw/mgmd_weights/yolov5s_GLAD-crop.pt",
                "crop_size": 320, "conf_thresh": 0.5, "imgsz": 320, "device": "cpu"},
        "metrics": {"radii_px": [10, 20, 50, 100], "primary_radius_px": 100,
                    "association": "round_half_up"},
        "runtime": {"downscale": downscale, "limit": None},
        "backend": "numpy", "seed": 0,
    }
    p = tmp_path / "mgmd_T1.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


def _check_run_artifacts(out_dir: Path, n_frames: int):
    """run 记录三件套 + make_tables 兼容键 + schema 往返（US-006 AC2）。"""
    wpath = out_dir / "windows.jsonl"
    assert wpath.is_file()
    recs = R.read_jsonl(wpath)                      # schema 往返
    assert len(recs) == n_frames
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["config"]["name"] == "mgmd_T1"
    assert meta["config"]["snapshot"]["sequence"]["name"] == "T1/cam9"   # make_tables GT 映射
    assert "backend" in meta and "mgmd" in meta
    timing = json.loads((out_dir / "timing.json").read_text(encoding="utf-8"))
    assert "window_fps" in timing["fps"] and "effective_frame_fps" in timing["fps"]
    assert (out_dir / "metrics.json").is_file()
    return recs, meta, timing


def test_smoke_mfe_only(tmp_path: Path):
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg = load_config(_write_cfg(tmp_path, frames_dir))
    out_dir = tmp_path / "runs" / "T1_mfe"
    rc = pipeline.run(cfg, frozenset({"mfe"}), gt_csv, out_dir, None, "numpy")
    assert rc == 0
    recs, _meta, _t = _check_run_artifacts(out_dir, N_FRAMES)
    # 逐帧伪窗口映射（SPEC §3.2）
    for i, r in enumerate(recs):
        assert (r.window_index, r.t_start, r.assoc_frame) == (i, i, i)
        assert r.t_center == float(i) and r.search_mode == "global"
        assert r.reason in ("OK", "ABSENT")
        assert r.accepted == r.target_present
    assert recs[0].reason == "ABSENT"               # 首帧无 prev（SPEC §5.8）
    # 运动目标被检出且中心误差 ≤5px（US-006 冒烟语义）
    det = [r for r in recs[1:] if r.target_present]
    assert det, "运动方块未被检出"
    errs = [np.hypot(r.est.x - _square_center(r.t_start)[0],
                     r.est.y - _square_center(r.t_start)[1]) for r in det]
    assert min(errs) <= 5.0
    # evaluate_windows 直接可算（FR-10）
    gt = R.read_gt_csv(gt_csv)
    res = MET.evaluate_windows(recs, gt, config_name="mgmd_T1")
    assert res.recall[100] > 0.5


def test_module_tiers_smoke(tmp_path: Path):
    """消融四档（SPEC §5.7）：mfe / mfe,tf / mfe,tf,lac / mfe,tf,lac,lad。"""
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg = load_config(_write_cfg(tmp_path, frames_dir))
    n_out: dict[str, int] = {}
    for i, mods in enumerate(("mfe", "mfe,tf", "mfe,tf,lac", "mfe,tf,lac,lad")):
        if ("lac" in mods or "lad" in mods) and not _APPEARANCE:
            pytest.skip("torch/权重/yolov5 未就绪：外观档 skip（FR-11）")
        out_dir = tmp_path / f"runs_tiers_{i}" / f"T1_{mods.replace(',', '-')}"
        rc = pipeline.run(cfg, pipeline.parse_modules(mods), gt_csv, out_dir, None, "numpy")
        assert rc == 0
        recs, meta, _t = _check_run_artifacts(out_dir, N_FRAMES)
        assert meta["mgmd"]["modules"] == sorted(mods.split(","))
        n_out[mods] = sum(1 for r in recs if r.target_present)
    # 过滤单调性：逐级增加模块，输出数不增（TF/LAC/LAD 只过滤不新增）
    assert n_out["mfe"] >= n_out["mfe,tf"] >= 0


def test_tf_confirmed_through_pipeline(tmp_path: Path):
    """TF 接线集成验证：alg1_d_min 调至合成 blob 适配值（合并 blob 膨胀对角线，
    理想矩形下的默认阈值确认逻辑由 test_mgmd_tf.py 覆盖），确认轨迹 → TRACKED。"""
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg_path = _write_cfg(tmp_path, frames_dir)
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    doc["tf"]["alg1_d_min"] = 0.6      # 合成合并 blob ρ/d≈0.62（实测）；默认 1.0 见单测
    cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    cfg = load_config(cfg_path)
    out_dir = tmp_path / "runs" / "T1_tf_conf"
    assert pipeline.run(cfg, frozenset({"mfe", "tf"}), gt_csv, out_dir, None,
                        "numpy") == 0
    recs = R.read_jsonl(out_dir / "windows.jsonl")
    tracked = [r for r in recs if r.target_present and r.state_out == "TRACKED"]
    assert tracked, "TF 确认轨迹未产生输出"
    assert any(abs(r.est.vx) > 1.0 or abs(r.est.vy) > 1.0 for r in tracked), \
        "TRACKED 记录应带 Kalman 速度（SPEC §3.2 est 映射）"


def test_records_state_mapping(tmp_path: Path):
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg = load_config(_write_cfg(tmp_path, frames_dir))
    out_dir = tmp_path / "runs" / "T1_state"
    assert pipeline.run(cfg, frozenset({"mfe"}), gt_csv, out_dir, None, "numpy") == 0
    recs = R.read_jsonl(out_dir / "windows.jsonl")
    for prev, cur in zip(recs, recs[1:]):
        assert cur.state_in == prev.state_out      # state_in = 上一帧 state_out
    for r in recs:
        if r.target_present:
            assert r.est is not None and r.reason == "OK"
            assert r.state_out == "INIT"           # 无 TF → INIT（SPEC §3.2）
        else:
            assert r.est is None and r.reason == "ABSENT" and r.state_out == "ABSENT"


def test_downscale_coordinate_mapping(tmp_path: Path):
    """downscale=0.5：记录坐标 ÷0.5 映射回原图尺度（SPEC §5.8/D7）。"""
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg = load_config(_write_cfg(tmp_path, frames_dir, downscale=0.5))
    assert cfg.runtime.downscale == 0.5
    out_dir = tmp_path / "runs" / "T1_ds"
    assert pipeline.run(cfg, frozenset({"mfe"}), gt_csv, out_dir, None, "numpy") == 0
    recs = R.read_jsonl(out_dir / "windows.jsonl")
    det = [r for r in recs[1:] if r.target_present]
    assert det
    errs = [np.hypot(r.est.x - _square_center(r.t_start)[0],
                     r.est.y - _square_center(r.t_start)[1]) for r in det]
    assert min(errs) <= 6.0                        # 映射回原图后仍命中


def test_frame_gap_buffering(tmp_path: Path):
    """frame_gap=k（SPEC D11）：前 k 帧 ABSENT（无差分对），之后检测对准当前帧坐标。"""
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg_path = _write_cfg(tmp_path, frames_dir)
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    doc["mfe"]["frame_gap"] = 3
    cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    cfg = load_config(cfg_path)
    out_dir = tmp_path / "runs" / "T1_gap3"
    assert pipeline.run(cfg, frozenset({"mfe"}), gt_csv, out_dir, None, "numpy") == 0
    recs = R.read_jsonl(out_dir / "windows.jsonl")
    assert all(r.reason == "ABSENT" for r in recs[:3])       # 前 k 帧无输出
    det = [r for r in recs[3:] if r.target_present]
    assert det, "k=3 下未检出"
    errs = [np.hypot(r.est.x - _square_center(r.t_start)[0],
                     r.est.y - _square_center(r.t_start)[1]) for r in det]
    # 合并 blob 覆盖 [p(t−k), p(t)] 运动轨迹，中心滞后 (k/2)·|v|≈11.7px；
    # 显著小于"滞后到 p(t−k)"的 k·|v|≈23.4px，证明差分对锚定当前帧（LAD 再精炼）
    assert min(errs) <= 17.0


def test_make_tables_compat(tmp_path: Path, repo_root: Path):
    """make_tables.py --runs <runs> 直接出表（FR-9 / US-006 AC2）。"""
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg = load_config(_write_cfg(tmp_path, frames_dir))
    runs_root = tmp_path / "runs_mgmd"
    assert pipeline.run(cfg, frozenset({"mfe"}), gt_csv, runs_root / "T1_mfe",
                        None, "numpy") == 0
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "T1_cam9.csv").write_text(gt_csv.read_text(encoding="utf-8"),
                                        encoding="utf-8")
    out = tmp_path / "metrics_out"
    r = subprocess.run([sys.executable, "scripts/make_tables.py",
                        "--runs", str(runs_root), "--gt", str(gt_dir),
                        "--out", str(out)],
                       cwd=repo_root, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    summary = (out / "summary.csv").read_text(encoding="utf-8")
    assert "mgmd_T1" in summary and "recall@100" in summary
    assert (out / "paper_vs_ours.md").is_file()     # 未知 config 不炸（SPEC A5）


def test_cli_end_to_end(tmp_path: Path, repo_root: Path):
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg_path = _write_cfg(tmp_path, frames_dir)
    out_dir = tmp_path / "runs" / "T1_cli"
    r = subprocess.run([sys.executable, "scripts/run_mgmd.py",
                        "--config", str(cfg_path), "--gt", str(gt_csv),
                        "--out", str(out_dir), "--modules", "mfe", "--limit", "6"],
                       cwd=repo_root, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert (out_dir / "windows.jsonl").is_file()
    assert "mgmd_T1" in r.stdout


def test_missing_data_exit_3(tmp_path: Path):
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg_path = _write_cfg(tmp_path, frames_dir)
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    doc["sequence"]["frames_dir"] = str(tmp_path / "no_such_dir")
    doc["sequence"]["video"] = str(tmp_path / "no_such.mp4")
    bad = tmp_path / "missing.yaml"
    bad.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    cfg = load_config(bad)
    with pytest.raises(MgmdError) as ei:
        pipeline.run(cfg, frozenset({"mfe"}), gt_csv, tmp_path / "o", None, "numpy")
    assert ei.value.exit_code == 3
    assert "E_DATA_MISSING" in ei.value.format()


def test_gt_missing_exit_3_via_cli(tmp_path: Path, repo_root: Path):
    frames_dir, _gt = _make_seq(tmp_path)
    cfg_path = _write_cfg(tmp_path, frames_dir)
    r = subprocess.run([sys.executable, "scripts/run_mgmd.py",
                        "--config", str(cfg_path), "--gt", str(tmp_path / "no.csv"),
                        "--out", str(tmp_path / "o"), "--modules", "mfe"],
                       cwd=repo_root, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "E_DATA_MISSING" in r.stderr


def test_history_alignment_homography_inverts_align_result():
    """align() 返回当前→上一帧 H；轨迹历史需要上一帧→当前帧（取逆）。"""
    H_cur_prev = np.array([[1.0, 0.0, -3.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    res = type("Res", (), {"fallback": False, "homography": H_cur_prev})()
    hist = pipeline._history_alignment_homography({}, 5, 1, None, None, res)
    assert hist is not None
    cur = hist @ np.array([100.0, 100.0, 1.0])
    assert cur[:2] == pytest.approx([103.0, 99.0])
    res.fallback = True
    assert pipeline._history_alignment_homography({}, 5, 1, None, None, res) is None
