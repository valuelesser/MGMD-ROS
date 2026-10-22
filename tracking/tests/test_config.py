"""test_config — schema 校验与六配置对照 PRD §5.2/§5.4（SPEC §9.1）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from spa2d.config import (
    ConfigError,
    Config,
    effective_thresholds,
    load_config,
    load_frozen,
    validate,
)

EXPECTED = {
    "D1_no_overlap": dict(name="D1/cam2", N=5, stride=5, vmax=1.0, delta_c=0.5,
                          rf=0.2, delta_f=0.1, rl=0.3, delta_l=0.1,
                          fps=50.00, resolution=(1920, 1080)),
    "D1_overlap": dict(name="D1/cam2", N=5, stride=1, vmax=1.0, delta_c=0.5,
                       rf=0.2, delta_f=0.1, rl=0.3, delta_l=0.1,
                       fps=50.00, resolution=(1920, 1080)),
    "D2_no_overlap": dict(name="D2/cam1", N=5, stride=5, vmax=4.0, delta_c=0.5,
                          rf=0.2, delta_f=0.1, rl=0.3, delta_l=0.1,
                          fps=29.84, resolution=(1920, 1080)),
    "D2_overlap": dict(name="D2/cam1", N=5, stride=1, vmax=5.0, delta_c=0.5,
                       rf=0.2, delta_f=0.1, rl=0.3, delta_l=0.1,
                       fps=29.84, resolution=(1920, 1080)),
    "D3_overlap": dict(name="D3/cam3", N=8, stride=4, vmax=15.0, delta_c=2.0,
                       rf=0.4, delta_f=0.2, rl=0.6, delta_l=0.25,
                       fps=25.00, resolution=(1440, 1080)),
    "D4_overlap": dict(name="D4/cam3", N=8, stride=4, vmax=15.0, delta_c=2.0,
                       rf=1.0, delta_f=0.2, rl=1.0, delta_l=0.25,
                       fps=21.24, resolution=(3840, 2160)),
}


def _load_base(configs_dir: Path) -> dict:
    return yaml.safe_load((configs_dir / "D1_no_overlap.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("config_name", sorted(EXPECTED))
def test_all_configs_load_and_validate(configs_dir: Path, config_name: str):
    cfg = load_config(configs_dir / f"{config_name}.yaml")
    assert isinstance(cfg, Config)
    assert cfg.config_name == config_name
    assert cfg.backend == "jax"
    assert cfg.seed == 0
    assert cfg.metrics.radii_px == (10, 20, 50, 100)
    assert cfg.metrics.primary_radius_px == 100
    assert cfg.search.top_k == 5
    assert cfg.search.batch_logical == 512
    assert cfg.search.gpu_chunk_bytes == 1610612736
    assert cfg.cache.host_max_bytes == 2147483648
    assert cfg.timing.warmup_windows == 20 and cfg.timing.repeats == 3
    assert cfg.preprocess.kernel_size == 5
    assert cfg.thresholds_file == "configs/thresholds_frozen.yaml"


@pytest.mark.parametrize("config_name", sorted(EXPECTED))
def test_config_values_match_prd(configs_dir: Path, config_name: str):
    exp = EXPECTED[config_name]
    cfg = load_config(configs_dir / f"{config_name}.yaml")
    assert cfg.sequence.name == exp["name"]
    assert cfg.sequence.fps == exp["fps"]
    assert cfg.sequence.resolution == exp["resolution"]
    assert cfg.window.N == exp["N"]
    assert cfg.window.stride == exp["stride"]
    assert cfg.search.coarse.vmax == exp["vmax"]
    assert cfg.search.coarse.delta_c == exp["delta_c"]
    assert cfg.search.fine.rf == exp["rf"]
    assert cfg.search.fine.delta_f == exp["delta_f"]
    assert cfg.search.local.rl == exp["rl"]
    assert cfg.search.local.delta_l == exp["delta_l"]


def _mut_coarse_not_divisible(doc):
    doc["search"]["global"] = {"vmax": 1.0, "delta_c": 0.3}


def _mut_fine_not_divisible(doc):
    doc["search"]["fine"] = {"rf": 0.25, "delta_f": 0.1}


def _mut_local_zero_steps(doc):
    doc["search"]["local"] = {"rl": 0.3, "delta_l": 0.5}


def _mut_stride_gt_N(doc):
    doc["window"] = {"N": 5, "stride": 6}


def _mut_N_lt_2(doc):
    doc["window"] = {"N": 1, "stride": 1}


def _mut_kernel_even(doc):
    doc["preprocess"]["kernel_size"] = 4


def _mut_kernel_out_of_set(doc):
    doc["preprocess"]["kernel_size"] = 6


def _mut_primary_not_in_radii(doc):
    doc["metrics"]["primary_radius_px"] = 33


def _mut_bad_backend(doc):
    doc["backend"] = "torch"


def _mut_bad_association(doc):
    doc["metrics"]["association"] = "nearest"


def _mut_bad_kernel_shape(doc):
    doc["preprocess"]["kernel_shape"] = "disk"


def _mut_unknown_field(doc):
    doc["surprise"] = 1


def _mut_unknown_nested(doc):
    doc["search"]["global"]["extra"] = 2


def _mut_missing_section(doc):
    doc.pop("search")


def _mut_missing_nested(doc):
    doc["window"].pop("stride")


def _mut_bad_resolution(doc):
    doc["sequence"]["resolution"] = [1920]


def _mut_bad_fps(doc):
    doc["sequence"]["fps"] = 0


MUTATORS = [
    ("coarse_not_divisible", _mut_coarse_not_divisible),
    ("fine_not_divisible", _mut_fine_not_divisible),
    ("local_zero_steps", _mut_local_zero_steps),
    ("stride_gt_N", _mut_stride_gt_N),
    ("N_lt_2", _mut_N_lt_2),
    ("kernel_even", _mut_kernel_even),
    ("kernel_out_of_set", _mut_kernel_out_of_set),
    ("primary_not_in_radii", _mut_primary_not_in_radii),
    ("bad_backend", _mut_bad_backend),
    ("bad_association", _mut_bad_association),
    ("bad_kernel_shape", _mut_bad_kernel_shape),
    ("unknown_field", _mut_unknown_field),
    ("unknown_nested", _mut_unknown_nested),
    ("missing_section", _mut_missing_section),
    ("missing_nested", _mut_missing_nested),
    ("bad_resolution", _mut_bad_resolution),
    ("bad_fps", _mut_bad_fps),
]


@pytest.mark.parametrize("mid,mutator", MUTATORS, ids=[m[0] for m in MUTATORS])
def test_invalid_configs_rejected(configs_dir, write_yaml, mid, mutator):
    doc = _load_base(configs_dir)
    mutator(doc)
    path = write_yaml(doc, f"bad_{mid}.yaml")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert excinfo.value.code == "E_CONFIG_INVALID"
    assert excinfo.value.exit_code == 2
    assert excinfo.value.format().startswith("[spa2d] E_CONFIG_INVALID:")


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError) as excinfo:
        load_config(tmp_path / "nope.yaml")
    assert excinfo.value.code == "E_CONFIG_INVALID"


def test_validate_returns_config(configs_dir):
    cfg = load_config(configs_dir / "D1_no_overlap.yaml")
    assert validate(cfg) is cfg


def test_cli_exit_code_2_on_invalid(configs_dir, write_yaml, repo_root):
    doc = _load_base(configs_dir)
    _mut_kernel_even(doc)
    path = write_yaml(doc, "bad_cli.yaml")
    proc = subprocess.run(
        [sys.executable, "-m", "spa2d.config", str(path)],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "E_CONFIG_INVALID" in proc.stderr


def test_cli_exit_code_0_on_valid(configs_dir, repo_root):
    proc = subprocess.run(
        [sys.executable, "-m", "spa2d.config", str(configs_dir / "D1_no_overlap.yaml")],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "OK" in proc.stdout


def _frozen_doc():
    return {
        "schema_version": 1,
        "frozen_at": "2026-10-10T00:00:00Z",
        "calibration": {"validation": ["D1/cam2:first500", "D3/cam3:first500"]},
        "thresholds": {
            "theta_high_pnr": 8.0,
            "theta_low_pnr": 5.0,
            "theta_margin": 0.02,
            "theta_edge_pnr": 10.0,
            "v_stat": 0.05,
            "m_lost": 3,
            "reacq_max": 5,
            "gap_prior_windows": 2,
            "rho_formula": "max(8, ceil(2*width_ref))",
            "r_cont_high_formula": "max(25, 3*width_ref)",
            "r_cont_low_formula": "max(10, 1.5*width_ref)",
        },
        "width_ref": {"D1/cam2": 12.0, "D2/cam1": None, "D3/cam3": 10.0, "D4/cam3": None},
        "preprocess": {"kernel_size": 5},
    }


def test_load_frozen_and_effective_thresholds(configs_dir, write_yaml):
    path = write_yaml(_frozen_doc(), "thresholds_frozen.yaml")
    frozen = load_frozen(path)
    assert len(frozen.sha256) == 64
    assert frozen.kernel_size == 5
    assert frozen.theta_edge_pnr == 10.0
    cfg = load_config(configs_dir / "D1_no_overlap.yaml")
    th = effective_thresholds(cfg, frozen, width_ref=12.0)
    assert th.rho == 24
    assert th.r_cont_high == 36.0
    assert th.r_cont_low == 18.0
    assert th.kernel_size == 5
    assert th.theta_high_pnr == 8.0


def test_effective_thresholds_width_ref_default(configs_dir, write_yaml):
    path = write_yaml(_frozen_doc(), "thresholds_frozen.yaml")
    frozen = load_frozen(path)
    cfg = load_config(configs_dir / "D1_no_overlap.yaml")
    th = effective_thresholds(cfg, frozen, width_ref=None)
    assert th.width_ref == 8.0
    assert th.rho == 16
    assert th.r_cont_high == 25.0
    assert th.r_cont_low == 12.0


def test_frozen_edge_rule_violation(write_yaml):
    doc = _frozen_doc()
    doc["thresholds"]["theta_edge_pnr"] = 9.0
    path = write_yaml(doc, "thresholds_frozen.yaml")
    with pytest.raises(ConfigError) as excinfo:
        load_frozen(path)
    assert excinfo.value.code == "E_FROZEN_TAMPER"
    assert excinfo.value.exit_code == 5


def test_frozen_low_gt_high(write_yaml):
    doc = _frozen_doc()
    doc["thresholds"]["theta_low_pnr"] = 9.0
    doc["thresholds"]["theta_edge_pnr"] = 10.0
    path = write_yaml(doc, "thresholds_frozen.yaml")
    with pytest.raises(ConfigError) as excinfo:
        load_frozen(path)
    assert excinfo.value.exit_code == 5


def test_frozen_missing_file(tmp_path):
    with pytest.raises(ConfigError) as excinfo:
        load_frozen(tmp_path / "absent.yaml")
    assert excinfo.value.code == "E_FROZEN_MISSING"
    assert excinfo.value.exit_code == 5


def test_effective_thresholds_kernel_conflict(configs_dir, write_yaml):
    doc = _frozen_doc()
    doc["preprocess"]["kernel_size"] = 7
    path = write_yaml(doc, "thresholds_frozen.yaml")
    frozen = load_frozen(path)
    cfg = load_config(configs_dir / "D1_no_overlap.yaml")
    with pytest.raises(ConfigError) as excinfo:
        effective_thresholds(cfg, frozen, width_ref=12.0)
    assert excinfo.value.exit_code == 5
