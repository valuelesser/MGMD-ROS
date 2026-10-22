"""test_run_all_report — run_all 一键复跑 + report（SPEC §4.1/PRD §11，I-16 验收）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _run(mode: str, backend: str = "numpy"):
    env = {"SPA2D_RUN_ALL_MODE": mode, "SPA2D_BACKEND": backend,
           "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    return subprocess.run(["bash", str(REPO / "scripts" / "run_all.sh")],
                          cwd=str(REPO), capture_output=True, text=True, env=env, timeout=900)


def test_run_all_smoke_end_to_end():
    proc = _run("smoke", "numpy")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Tier B" in proc.stdout
    report = REPO / "outputs" / "report.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "Tier B" in text and "static ground-camera proxy" in text
    # 报告九节标题
    for sec in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8.", "## 9."):
        assert sec in text
    assert (REPO / "outputs" / "metrics" / "summary.csv").is_file()
    assert (REPO / "outputs" / "metrics" / "paper_vs_ours.md").is_file()
    assert (REPO / "outputs" / "figures" / "fig7_recall_fps.png").is_file()
    assert (REPO / "outputs" / "figures" / "fig8_trajectories.png").is_file()
    assert (REPO / "outputs" / "runs" / "smoke_D1" / "windows.jsonl").is_file()
    assert (REPO / "outputs" / "runs" / "smoke_D1" / "repeat_ranges.json").is_file()
    assert (REPO / "outputs" / "runs_baselines" / "bgsub_smoke_D1" / "metrics.json").is_file()


def test_run_all_real_requires_frozen():
    # real 模式：Tier B 声明；缺冻结阈值 → 退出码 5（前置失败即停）
    if (REPO / "configs" / "thresholds_frozen.yaml").exists():
        pytest.skip("仓库已存在冻结阈值")
    proc = _run("real", "numpy")
    assert "Tier B" in proc.stdout
    assert proc.returncode == 5
