"""tests/test_mgmd_config.py — mgmd 配置 schema + CLI 骨架 + 依赖懒加载矩阵（SPEC §9.1/§9.2）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import mgmd
from mgmd import pipeline
from mgmd.config import (EXIT_CONFIG_INVALID, EXIT_DATA, EXIT_OK, EXIT_RUNTIME,
                         MgmdError, load_config)

CONFIGS = ("mgmd_D1", "mgmd_D2", "mgmd_D3", "mgmd_D4")
SEQ_NAMES = {"mgmd_D1": "D1/cam2", "mgmd_D2": "D2/cam1",
             "mgmd_D3": "D3/cam3", "mgmd_D4": "D4/cam3"}


def _valid_doc() -> dict:
    return {
        "schema_version": 1, "config_name": "mgmd_test",
        "sequence": {"name": "D1/cam2", "frames_dir": "data_raw/mvdt/D1/cam2/frames",
                     "frame_glob": "*.jpg", "fps": 50.0, "resolution": [1920, 1080],
                     "color_order": "BGR", "video": "data_raw/mvdt/D1/cam2/cam2.mp4"},
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
                "prob_thresh": 0.5, "device": "auto"},
        "lad": {"weights": "data_raw/mgmd_weights/yolov5s_GLAD-crop.pt",
                "crop_size": 320, "conf_thresh": 0.5, "imgsz": 320, "device": "auto"},
        "metrics": {"radii_px": [10, 20, 50, 100], "primary_radius_px": 100,
                    "association": "round_half_up"},
        "runtime": {"downscale": 1.0, "limit": None},
        "backend": "numpy", "seed": 0,
    }


# ---------------- 4 份序列配置（US-002 AC2） ----------------

@pytest.mark.parametrize("name", CONFIGS)
def test_repo_configs_load(repo_root: Path, name: str):
    cfg = load_config(repo_root / "configs" / f"{name}.yaml")
    assert cfg.config_name == name
    assert cfg.sequence.name == SEQ_NAMES[name]
    assert cfg.mfe.diff_thresh_base == 5          # SPEC D2 官方 base
    assert cfg.mfe.grid_spacing == (96, 72)       # SPEC §5.1 官方间距
    assert cfg.tf.iou_thresh == 0.3 and cfg.tf.max_age == 3 and cfg.tf.min_hits == 1
    assert cfg.metrics.primary_radius_px in cfg.metrics.radii_px


def test_d4_downscale_default(repo_root: Path):
    cfg = load_config(repo_root / "configs" / "mgmd_D4.yaml")
    assert cfg.runtime.downscale == 0.5           # SPEC §1.3 D7
    assert cfg.sequence.resolution == (3840, 2160)


def test_tuned_mvdt_params_locked(repo_root: Path):
    """SPEC D11–D13 实测调参锁定（issue-024 扫描证据）。"""
    d1 = load_config(repo_root / "configs" / "mgmd_D1.yaml")
    assert (d1.mfe.frame_gap, d1.mfe.blur_kernel, d1.mfe.area_min) == (32, 3, 5)
    assert d1.mfe.or_aggregation and d1.mfe.overflow == "topk"
    d2 = load_config(repo_root / "configs" / "mgmd_D2.yaml")
    assert (d2.mfe.frame_gap, d2.mfe.blur_kernel, d2.mfe.area_min) == (3, 11, 5)
    assert not d2.mfe.or_aggregation
    d3 = load_config(repo_root / "configs" / "mgmd_D3.yaml")
    assert (d3.mfe.frame_gap, d3.mfe.area_min) == (2, 5)
    d4 = load_config(repo_root / "configs" / "mgmd_D4.yaml")
    assert (d4.mfe.frame_gap, d4.mfe.area_min) == (1, 5)
    # LAC 域失效直通（prob_thresh=0）+ Algorithm 1 慢目标阈值（issue-024 实测）
    for name, dmin in (("D1", 0.003), ("D2", 0.01), ("D3", 0.1), ("D4", 0.002)):
        c = load_config(repo_root / "configs" / f"mgmd_{name}.yaml")
        assert c.lac.prob_thresh == 0.0 and c.tf.alg1_d_min == dmin
        assert not c.mfe.motion_coherence


def test_d1_d3_downscale_identity(repo_root: Path):
    for name in ("mgmd_D1", "mgmd_D2", "mgmd_D3"):
        assert load_config(repo_root / "configs" / f"{name}.yaml").runtime.downscale == 1.0


# ---------------- schema 校验（非法 → 退出码 2） ----------------

def test_unknown_field_rejected(write_yaml):
    doc = _valid_doc()
    doc["bogus"] = 1
    with pytest.raises(MgmdError) as ei:
        load_config(write_yaml(doc, "bad.yaml"))
    assert ei.value.exit_code == EXIT_CONFIG_INVALID
    assert "未知字段" in str(ei.value)


def test_unknown_nested_field_rejected(write_yaml):
    doc = _valid_doc()
    doc["mfe"]["bogus"] = 1
    with pytest.raises(MgmdError) as ei:
        load_config(write_yaml(doc, "bad.yaml"))
    assert ei.value.exit_code == EXIT_CONFIG_INVALID


def test_missing_required_field(write_yaml):
    doc = _valid_doc()
    del doc["mfe"]["grid_spacing"]
    with pytest.raises(MgmdError) as ei:
        load_config(write_yaml(doc, "bad.yaml"))
    assert ei.value.exit_code == EXIT_CONFIG_INVALID
    assert "缺少必需字段" in str(ei.value)


@pytest.mark.parametrize("mutate", [
    lambda d: d["mfe"].update(ratio_min=3.0, ratio_max=0.6),
    lambda d: d["mfe"].update(area_min=3000, area_max=30),
    lambda d: d["lac"].update(prob_thresh=1.5),
    lambda d: d["lad"].update(conf_thresh=-0.1),
    lambda d: d["lad"].update(crop_size=100),          # 非 32 倍数
    lambda d: d["runtime"].update(downscale=1.5),
    lambda d: d["tf"].update(iou_thresh=1.0),
    lambda d: d["mfe"].update(frame_gap=0),
    lambda d: d["mfe"].update(overflow="bogus"),
    lambda d: d["tf"].update(alg1_age=1),
    lambda d: d["metrics"].update(primary_radius_px=15),
    lambda d: d["metrics"].update(association="bogus"),
    lambda d: d["lac"].update(device="tpu"),
    lambda d: d.update(schema_version=2),
    lambda d: d.update(backend="torch"),
])
def test_invalid_values(write_yaml, mutate):
    doc = _valid_doc()
    mutate(doc)
    with pytest.raises(MgmdError) as ei:
        load_config(write_yaml(doc, "bad.yaml"))
    assert ei.value.exit_code == EXIT_CONFIG_INVALID


def test_error_format(write_yaml):
    doc = _valid_doc()
    doc["schema_version"] = 2
    with pytest.raises(MgmdError) as ei:
        load_config(write_yaml(doc, "bad.yaml"))
    msg = ei.value.format()
    assert msg.startswith("[mgmd] E_CONFIG_INVALID: ") and " → " in msg


def test_missing_config_file():
    with pytest.raises(MgmdError) as ei:
        load_config("configs/__nonexistent__.yaml")
    assert ei.value.exit_code == EXIT_CONFIG_INVALID


# ---------------- CLI：python -m mgmd.config ----------------

def _run_module_config(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "mgmd.config", *args],
                          cwd=repo_root, capture_output=True, text=True, timeout=60)


def test_cli_valid_configs_exit_0(repo_root: Path):
    paths = [f"configs/{n}.yaml" for n in CONFIGS]
    r = _run_module_config(repo_root, *paths)
    assert r.returncode == EXIT_OK, r.stderr
    for n in CONFIGS:
        assert n in r.stdout


def test_cli_invalid_exit_2(repo_root: Path, write_yaml):
    bad = write_yaml({**_valid_doc(), "schema_version": 99}, "bad.yaml")
    r = _run_module_config(repo_root, str(bad))
    assert r.returncode == EXIT_CONFIG_INVALID
    assert r.stderr.startswith("[mgmd] E_CONFIG_INVALID: ")


# ---------------- 依赖懒加载矩阵（FR-11 / SPEC §9.2） ----------------

def test_package_imports_without_torch():
    import importlib
    for mod in ("mgmd", "mgmd.config", "mgmd.io_mvdt", "mgmd.align", "mgmd.diff",
                "mgmd.track", "mgmd.classify", "mgmd.detect", "mgmd.pipeline"):
        importlib.import_module(mod)  # 顶层导入不得触发 torch/scipy/filterpy


def test_lac_lazy_dep_and_weights():
    torch = pytest.importorskip("torch", reason="torch 未装：LAC 用例 skip（FR-11）")
    del torch
    from mgmd.classify import LacClassifier
    from mgmd.config import LacConfig
    with pytest.raises(MgmdError) as ei:  # 权重缺失 → 退出码 3
        LacClassifier(LacConfig(weights="/nonexistent/Net_best.pth",
                                input_size=32, prob_thresh=0.5, device="cpu"))
    assert ei.value.exit_code == EXIT_DATA


def test_lac_dep_missing_raises_exit4(monkeypatch):
    if pytest.importorskip("importlib.util", reason="").find_spec("torch") is not None:
        pytest.skip("torch 已装：走 test_lac_lazy_dep_and_weights")
    from mgmd.classify import LacClassifier
    from mgmd.config import LacConfig
    with pytest.raises(MgmdError) as ei:  # torch 缺失 → E_DEP_MISSING 退出码 4
        LacClassifier(LacConfig(weights="data_raw/mgmd_weights/Net_best.pth",
                                input_size=32, prob_thresh=0.5, device="cpu"))
    assert ei.value.exit_code == EXIT_RUNTIME
    assert "fetch_mgmd_weights" not in str(ei.value)


def test_tf_dep_missing_raises_exit4():
    import importlib.util
    if importlib.util.find_spec("filterpy") is not None:
        pytest.skip("filterpy 已装")
    from mgmd.track import SortTracker
    from mgmd.config import TfConfig
    with pytest.raises(MgmdError) as ei:
        SortTracker(TfConfig(iou_thresh=0.3, max_age=3, min_hits=1,
                             alg1_age=3, alg1_window=3, alg1_d_min=1.0))
    assert ei.value.exit_code == EXIT_RUNTIME


# ---------------- --modules 解析与 CLI 骨架（US-002 AC3） ----------------

def test_parse_modules_ok():
    assert pipeline.parse_modules("mfe") == frozenset({"mfe"})
    assert pipeline.parse_modules("mfe,tf") == frozenset({"mfe", "tf"})
    assert pipeline.parse_modules(" mfe, tf , lac ,lad ") == frozenset({"mfe", "tf", "lac", "lad"})


@pytest.mark.parametrize("spec", ["", "yolo", "tf,lac", "mfe,bogus"])
def test_parse_modules_bad(spec: str):
    with pytest.raises(MgmdError) as ei:
        pipeline.parse_modules(spec)
    assert ei.value.exit_code == EXIT_CONFIG_INVALID


def _run_cli(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "scripts/run_mgmd.py", *args],
                          cwd=repo_root, capture_output=True, text=True, timeout=120)


def test_cli_bad_config_exit_2(repo_root: Path, tmp_path: Path):
    r = _run_cli(repo_root, "--config", "configs/__nonexistent__.yaml",
                 "--gt", "x.csv", "--out", str(tmp_path / "o"))
    assert r.returncode == EXIT_CONFIG_INVALID
    assert r.stderr.startswith("[mgmd] E_CONFIG_INVALID: ") and " → " in r.stderr


def test_cli_missing_gt_exit_3(repo_root: Path, tmp_path: Path):
    r = _run_cli(repo_root, "--config", "configs/mgmd_D1.yaml",
                 "--gt", "data_derived/gt/__nonexistent__.csv", "--out", str(tmp_path / "o"))
    assert r.returncode == EXIT_DATA
    assert "E_DATA_MISSING" in r.stderr


def test_cli_bad_modules_exit_2(repo_root: Path, tmp_path: Path):
    gt = tmp_path / "gt.csv"
    gt.write_text("frame,timestamp_s,visible,out_of_frame\n", encoding="utf-8")
    r = _run_cli(repo_root, "--config", "configs/mgmd_D1.yaml",
                 "--gt", str(gt), "--out", str(tmp_path / "o"), "--modules", "yolo")
    assert r.returncode == EXIT_CONFIG_INVALID
    assert "--modules" in r.stderr


def test_cli_missing_weights_exit_3(repo_root: Path, tmp_path: Path, write_yaml):
    """lac 权重路径不存在 → E_WEIGHTS_MISSING 退出码 3 + 指向 fetch 脚本（US-005 AC3）。"""
    doc = _valid_doc()
    doc["lac"]["weights"] = str(tmp_path / "no_such_Net_best.pth")
    cfg_path = write_yaml(doc, "mgmd_missing_w.yaml")
    gt = tmp_path / "gt.csv"
    gt.write_text("frame,timestamp_s,visible,out_of_frame,cx,cy,bbox_x,bbox_y,"
                  "bbox_w,bbox_h,area_px,width_px,width_source,gt_source,"
                  "projection_rmse_px\n", encoding="utf-8")
    r = _run_cli(repo_root, "--config", str(cfg_path),
                 "--gt", str(gt), "--out", str(tmp_path / "o"),
                 "--modules", "mfe,lac", "--limit", "1")
    assert r.returncode == EXIT_DATA
    assert "E_WEIGHTS_MISSING" in r.stderr and "fetch_mgmd_weights.sh" in r.stderr


def test_cli_bad_gt_schema_exit_3(repo_root: Path, tmp_path: Path):
    """GT 列不匹配 → E_DATA_MISSING 退出码 3（SPEC §6.1 错误分类）。"""
    gt = tmp_path / "gt.csv"
    gt.write_text("frame,timestamp_s,visible,out_of_frame\n", encoding="utf-8")
    r = _run_cli(repo_root, "--config", "configs/mgmd_D1.yaml",
                 "--gt", str(gt), "--out", str(tmp_path / "o"),
                 "--modules", "mfe", "--limit", "2")
    assert r.returncode == EXIT_DATA
    assert "E_DATA_MISSING" in r.stderr and "make_derived_gt" in r.stderr


def test_cli_help_exit_0(repo_root: Path):
    r = subprocess.run([sys.executable, "scripts/run_mgmd.py", "--help"],
                       cwd=repo_root, capture_output=True, text=True, timeout=60)
    assert r.returncode == EXIT_OK and "--modules" in r.stdout


def test_version_exposed():
    assert isinstance(mgmd.__version__, str)
