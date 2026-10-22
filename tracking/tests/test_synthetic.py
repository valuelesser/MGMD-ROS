"""test_synthetic — 合成数据生成器（SPEC §5.11/§3.4，I-02 验收）。"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from spa2d import records as R
from spa2d import synthetic as S

TINY_CANVAS = (96, 128)


def _read_manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def test_sanity_spec_coverage():
    specs = S.build_sanity_specs(n_seqs=30, seed=0, n_frames=120)
    assert len(specs) == 30
    assert {s.width_px for s in specs} == set(S.WIDTHS)
    assert {s.snr for s in specs} == set(S.SNR_LEVELS)
    assert all(s.n_frames >= 120 for s in specs)

    d12 = [s for s in specs if s.seq_id < f"seq_{15:03d}"]
    d34 = [s for s in specs if s.seq_id >= f"seq_{15:03d}"]
    assert d12 and d34
    assert max(max(abs(s.vx), abs(s.vy)) for s in d12) <= 5.0
    assert max(max(abs(s.vx), abs(s.vy)) for s in d34) >= 15.0
    has_half = any(abs(s.vx) % 1 == 0.5 or abs(s.vy) % 1 == 0.5 for s in specs)
    assert has_half

    negs = {n for s in specs for n in s.negatives}
    assert {"bird", "insect", "cloud"} <= negs
    for s in specs:
        if s.negatives:
            assert s.visible_range != (0, s.n_frames)
        assert abs(s.ax) < 0.02 and abs(s.ay) < 0.02


def test_reproducible_same_seed(tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    S.generate_sanity_set(out1, n_seqs=2, seed=7, n_frames=8, canvas=TINY_CANVAS)
    S.generate_sanity_set(out2, n_seqs=2, seed=7, n_frames=8, canvas=TINY_CANVAS)
    assert (out1 / "manifest.json").read_bytes() == (out2 / "manifest.json").read_bytes()
    f1 = sorted((out1 / "seq_000" / "frames").glob("*.png"))
    f2 = sorted((out2 / "seq_000" / "frames").glob("*.png"))
    assert [p.name for p in f1] == [p.name for p in f2]
    for a, b in zip(f1, f2):
        assert a.read_bytes() == b.read_bytes()
    assert (out1 / "seq_000" / "gt.csv").read_bytes() == (out2 / "seq_000" / "gt.csv").read_bytes()


def test_seed_changes_output(tmp_path):
    S.generate_sanity_set(tmp_path / "a", n_seqs=1, seed=1, n_frames=6, canvas=TINY_CANVAS)
    S.generate_sanity_set(tmp_path / "b", n_seqs=1, seed=2, n_frames=6, canvas=TINY_CANVAS)
    ha = _read_manifest(tmp_path / "a")["sequences"][0]["frames_sha256"]
    hb = _read_manifest(tmp_path / "b")["sequences"][0]["frames_sha256"]
    assert ha != hb


def test_outputs_structure_and_gt_schema(tmp_path):
    out = tmp_path / "synthetic"
    manifest = S.generate_sanity_set(out, n_seqs=2, seed=3, n_frames=8, canvas=TINY_CANVAS)
    assert (out / "manifest.json").exists()
    for seq in manifest["sequences"]:
        seq_dir = out / seq["dir"]
        assert (seq_dir / "frames").is_dir()
        assert (seq_dir / "gt.csv").exists()
        assert (seq_dir / "meta.json").exists()
        assert len(seq["frames_sha256"]) == 64
        assert len(seq["gt_sha256"]) == 64
        frames = sorted((seq_dir / "frames").glob("*.png"))
        assert len(frames) == manifest["suite_meta"]["sanity"]["default_n_frames"]
        rows = R.read_gt_csv(seq_dir / "gt.csv")
        assert len(rows) == seq["n_frames"]
        for gt in rows:
            if gt.visible == 1:
                assert gt.cx is not None and gt.cy is not None
                assert gt.out_of_frame == 0
                assert gt.width_source == "bbox"


def test_gt_pixel_consistency():
    spec = S.SequenceSpec(
        seq_id="consistency", width_px=8, snr="high", vx=1.0, vy=-0.5, ax=0.0, ay=0.0,
        n_frames=10, height=TINY_CANVAS[0], width=TINY_CANVAS[1], seed=123,
        p0=(60.0, 48.0), visible_range=(0, 10), negatives=(), static_bg=True)
    renderer = S.SequenceRenderer(spec)
    for f in (0, 3, 7):
        bg = renderer.render(f, with_target=False).astype(np.int32)
        fg = renderer.render(f, with_target=True).astype(np.int32)
        diff = bg - fg
        assert diff.max() > 0
        iy, ix = np.unravel_index(int(np.argmax(diff)), diff.shape)
        cx, cy = renderer.target_center(f)
        assert abs(ix - round(cx)) <= 1 and abs(iy - round(cy)) <= 1
        gt = renderer.gt_row(f)
        assert gt.visible == 1
        assert abs(gt.cx - cx) < 1e-6 and abs(gt.cy - cy) < 1e-6


def test_hard_negative_frames_invisible(tmp_path):
    out = tmp_path / "syn"
    manifest = S.generate_sanity_set(out, n_seqs=6, seed=5, n_frames=24, canvas=TINY_CANVAS)
    types = {n for seq in manifest["sequences"] for n in seq["negatives"]}
    assert {"bird", "insect", "cloud"} <= types
    checked = 0
    for seq in manifest["sequences"]:
        if not seq["negatives"]:
            continue
        rows = R.read_gt_csv(out / seq["dir"] / "gt.csv")
        absent = sum(1 for g in rows if g.visible == 0 and g.out_of_frame == 0)
        assert absent > 0
        meta = json.loads((out / seq["dir"] / "meta.json").read_text())
        assert len(meta["negatives"]) == len(seq["negatives"])
        checked += 1
    assert checked == 5


def test_out_of_frame_flag(tmp_path):
    spec = S.SequenceSpec(
        seq_id="oof", width_px=4, snr="high", vx=20.0, vy=0.0, ax=0.0, ay=0.0,
        n_frames=20, height=TINY_CANVAS[0], width=TINY_CANVAS[1], seed=9,
        p0=(10.0, 48.0), visible_range=(0, 20), negatives=(), static_bg=True)
    entry = S.render_sequence(spec, tmp_path / "oof")
    rows = R.read_gt_csv(tmp_path / "oof" / "gt.csv")
    assert any(g.out_of_frame == 1 and g.visible == 0 for g in rows)
    assert any(g.visible == 1 for g in rows)
    assert entry["seq_id"] == "oof"


def test_benchmark_grid_truth(tmp_path):
    out = tmp_path / "bench"
    manifest = S.generate_benchmark(out, seed=0, n_frames=6, canvas=TINY_CANVAS)
    assert manifest["suite"] == "benchmark"
    assert manifest["suite_meta"]["benchmark"]["true_velocity"] == {"vx": 0.7, "vy": 0.0}
    grid = out / "benchmark" / "grid_truth.csv"
    with grid.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 441
    trues = [r for r in rows if r["is_true"] == "1"]
    assert len(trues) == 1
    assert float(trues[0]["vx"]) == pytest.approx(0.7)
    assert float(trues[0]["vy"]) == pytest.approx(0.0)
    meta = json.loads((out / "benchmark" / "meta.json").read_text())
    assert meta["velocity"] == {"vx": 0.7, "vy": 0.0, "ax": 0.0, "ay": 0.0}
    assert meta["static_bg"] is True
    gt = R.read_gt_csv(out / "benchmark" / "gt.csv")
    assert len(gt) == 6
    assert all(g.visible == 1 for g in gt)


def test_benchmark_motion_is_point_seven():
    spec = S.build_benchmark_spec(seed=0, n_frames=50, canvas=TINY_CANVAS)
    renderer = S.SequenceRenderer(spec)
    x0, y0 = renderer.target_center(0)
    x1, y1 = renderer.target_center(1)
    assert x1 - x0 == pytest.approx(0.7)
    assert y1 - y0 == pytest.approx(0.0)
    assert np.allclose(renderer.bg.drift, 0.0)


def test_cli_sanity(tmp_path, repo_root):
    out = tmp_path / "cli_sanity"
    proc = subprocess.run(
        [sys.executable, "scripts/make_synthetic.py", "--suite", "sanity",
         "--out", str(out), "--seqs", "2", "--frames", "6",
         "--width", "128", "--height", "96", "--seed", "3"],
        cwd=str(repo_root), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (out / "manifest.json").exists()
    assert (out / "seq_000" / "gt.csv").exists()
    assert (out / "seq_001").exists()
    assert "suite=sanity" in proc.stdout


def test_cli_benchmark_and_overwrite_guard(tmp_path, repo_root):
    out = tmp_path / "cli_bench"
    cmd = [sys.executable, "scripts/make_synthetic.py", "--suite", "benchmark",
           "--out", str(out), "--frames", "6", "--width", "128", "--height", "96",
           "--seed", "0"]
    first = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert (out / "benchmark" / "grid_truth.csv").exists()
    second = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert second.returncode == 2
    third = subprocess.run(cmd + ["--overwrite"], cwd=str(repo_root),
                           capture_output=True, text=True)
    assert third.returncode == 0


def test_cli_bad_canvas(tmp_path, repo_root):
    proc = subprocess.run(
        [sys.executable, "scripts/make_synthetic.py", "--suite", "sanity",
         "--out", str(tmp_path / "x"), "--width", "128"],
        cwd=str(repo_root), capture_output=True, text=True)
    assert proc.returncode == 2


def test_manifest_aggregates_suites(tmp_path):
    out = tmp_path / "syn"
    S.generate_sanity_set(out, n_seqs=1, seed=0, n_frames=5, canvas=TINY_CANVAS)
    manifest = S.generate_benchmark(out, seed=0, n_frames=5, canvas=TINY_CANVAS)
    ids = {s["seq_id"] for s in manifest["sequences"]}
    assert ids == {"seq_000", "benchmark"}
    assert manifest["suites"] == ["benchmark", "sanity"]
    with pytest.raises(S.SyntheticError):
        S.generate_benchmark(out, seed=0, n_frames=5, canvas=TINY_CANVAS)
    ok = S.generate_benchmark(out, seed=0, n_frames=5, canvas=TINY_CANVAS, overwrite=True)
    assert ok["n_sequences"] == 2

