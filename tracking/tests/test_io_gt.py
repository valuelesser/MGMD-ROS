"""test_io_gt — MVDT I/O、派生 GT、fetch 与 assoc_frame（SPEC §4.1/§5.10/§3.2，I-08）。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from spa2d import io_mvdt as io
from spa2d import records as R
from spa2d.metrics import associate_frame, gt_velocity

CANVAS = (96, 128)  # (H, W)


# ---------------- metrics / GT association ----------------

def test_assoc_frame_modes():
    assert associate_frame(0, 5, "round_half_up") == 2
    assert associate_frame(0, 5, "floor") == 2
    assert associate_frame(0, 8, "round_half_up") == 4
    assert associate_frame(0, 8, "floor") == 3
    assert associate_frame(4, 8, "round_half_up") == 8
    assert associate_frame(4, 8, "floor") == 7
    with pytest.raises(ValueError):
        associate_frame(0, 5, "nearest")


def test_gt_velocity_central_and_endpoints():
    gt = [R.GTFrame(frame=i, timestamp_s=i / 10, visible=1, out_of_frame=0,
                    cx=10.0 + 0.7 * i, cy=20.0 - 0.5 * i) for i in range(6)]
    vel = gt_velocity(gt)
    assert np.allclose(vel, [0.7, -0.5], atol=1e-9)
    assert np.allclose(vel[0], [0.7, -0.5])
    assert np.allclose(vel[-1], [0.7, -0.5])


def test_gt_velocity_nan_on_invisible():
    gt = [
        R.GTFrame(frame=0, timestamp_s=0, visible=1, out_of_frame=0, cx=0.0, cy=0.0),
        R.GTFrame(frame=1, timestamp_s=0.1, visible=0, out_of_frame=1, cx=1.0, cy=0.0),
        R.GTFrame(frame=2, timestamp_s=0.2, visible=1, out_of_frame=0, cx=2.0, cy=0.0),
    ]
    vel = gt_velocity(gt)
    assert np.isnan(vel).all()


def test_read_gt_csv_nan_and_out_of_frame(tmp_path):
    rows = [
        R.GTFrame(frame=0, timestamp_s=0.0, visible=1, out_of_frame=0, cx=10.0, cy=20.0,
                  width_px=4.0, width_source="bbox", gt_source="published2d"),
        R.GTFrame(frame=1, timestamp_s=0.02, visible=0, out_of_frame=1, cx=-5.0, cy=20.0,
                  gt_source="published2d"),
        R.GTFrame(frame=2, timestamp_s=0.04, visible=0, out_of_frame=0, gt_source="published2d"),
    ]
    path = tmp_path / "gt.csv"
    R.write_gt_csv(rows, path)
    back = R.read_gt_csv(path)
    assert back == rows
    text = path.read_text()
    assert ",," in text  # NaN 写空串
    assert back[1].out_of_frame == 1 and back[1].cx == -5.0
    assert back[2].cx is None and back[2].cy is None


# ---------------- io_mvdt ----------------

def _write_frames(dirpath: Path, names, h=CANVAS[0], w=CANVAS[1]):
    dirpath.mkdir(parents=True, exist_ok=True)
    for n in names:
        cv2.imwrite(str(dirpath / n), np.zeros((h, w, 3), np.uint8))


def test_open_sequence_numeric_order(tmp_path):
    d = tmp_path / "frames"
    _write_frames(d, ["2.png", "10.png", "1.png"])
    src = io.open_sequence(frames_dir=str(d), glob="*.png")
    assert len(src) == 3
    assert [p.name for p in src._paths] == ["1.png", "2.png", "10.png"]
    info = src.probe()
    assert info.source_kind == "frames" and info.frame_count == 3
    assert (info.width, info.height) == (CANVAS[1], CANVAS[0])
    assert info.color_order == "BGR"


def test_open_sequence_video_decode(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
                             (CANVAS[1], CANVAS[0]))
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((CANVAS[0], CANVAS[1], 3), i * 10, np.uint8))
    writer.release()
    src = io.open_sequence(video=path)
    frames = list(src)
    assert len(frames) >= 4
    assert src.probe().source_kind == "video"


def test_open_sequence_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.open_sequence(frames_dir=str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        io.open_sequence(frames_dir=str(empty))


def test_frame_timestamps():
    ts = io.frame_timestamps(50.0, 3)
    assert np.allclose(ts, [0.0, 0.02, 0.04])
    given = np.array([1.0, 2.0, 3.0])
    assert np.allclose(io.frame_timestamps(50.0, 3, given), given)
    with pytest.raises(ValueError):
        io.frame_timestamps(50.0, 3, np.array([1.0, 2.0]))


def test_load_calibration_json(tmp_path):
    doc = {
        "image_size": [128, 96],
        "K": [[100, 0, 64], [0, 100, 48], [0, 0, 1]],
        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "t": [0, 0, 0],
        "distortion": {"model": "none", "coeffs": []},
    }
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps(doc))
    cam = io.load_calibration(p)
    assert cam is not None
    assert np.allclose(cam.K, doc["K"])
    assert cam.distortion_model == "none"
    assert cam.image_size == (128, 96)
    assert io.load_calibration(tmp_path / "absent.json") is None


def test_load_trajectory_published2d_with_bbox(tmp_path):
    p = tmp_path / "trajectory.csv"
    p.write_text("frame,cx,cy,bbox_x,bbox_y,bbox_w,bbox_h\n"
                 "0,10.0,20.0,8,18,4,4\n1,10.7,20.0,8.7,18,4,4\n")
    traj = io.load_trajectory(p)
    assert traj.uv is not None and traj.xyz is None
    assert np.allclose(traj.uv[1], [10.7, 20.0])
    assert traj.bbox is not None and traj.bbox[0, 2] == 4.0


def test_load_trajectory_3d_and_whitespace(tmp_path):
    p = tmp_path / "traj.txt"
    p.write_text("0 1.0 2.0 3.0\n1 1.5 2.5 3.5\n")
    traj = io.load_trajectory(p)
    assert traj.xyz is not None
    assert np.allclose(traj.xyz[1], [1.5, 2.5, 3.5])


def test_project_3d_to_2d_pinhole_and_rmse(tmp_path):
    calib = tmp_path / "calibration.json"
    calib.write_text(json.dumps({
        "K": [[100, 0, 64], [0, 100, 48], [0, 0, 1]], "R": np.eye(3).tolist(),
        "t": [0, 0, 0], "distortion_model": "none"}))
    cam = io.load_calibration(calib)
    uv_true = np.array([[64.0, 48.0], [74.0, 58.0]])
    xyz = np.stack([(uv_true[:, 0] - 64) / 100.0, (uv_true[:, 1] - 48) / 100.0,
                    np.ones(2)], axis=1)
    traj = io.Trajectory(frames=np.arange(2), xyz=xyz, uv=uv_true)
    uv, rmse = io.project_3d_to_2d(traj, cam)
    assert np.allclose(uv, uv_true, atol=1e-9)
    assert rmse == pytest.approx(0.0, abs=1e-6)
    traj2 = io.Trajectory(frames=np.arange(2), xyz=xyz, uv=uv_true + 3.0)
    _, rmse2 = io.project_3d_to_2d(traj2, cam)
    assert rmse2 == pytest.approx(np.sqrt(9.0 + 9.0), abs=1e-6)


def test_project_radtan_changes_result():
    cam = io.CameraModel(K=np.array([[100, 0, 64], [0, 100, 48], [0, 0, 1.0]]),
                         R=np.eye(3), t=np.zeros(3),
                         dist=np.array([0.1, 0.01, 0.0, 0.0, 0.0]),
                         distortion_model="radtan", image_size=(128, 96))
    xyz = np.array([[0.5, 0.4, 1.0]])
    uv, _ = io.project_3d_to_2d(io.Trajectory(frames=np.array([0]), xyz=xyz), cam)
    assert not np.allclose(uv, [[114.0, 88.0]])  # 与无畸变不同


# ---------------- fetch_data.sh ----------------

def _license(tmp_path: Path) -> Path:
    lic = tmp_path / "LICENSE.txt"
    lic.write_text("MVDT synthetic test license\n")
    return lic


def _run_fetch(repo_root, *args):
    return subprocess.run(["bash", str(repo_root / "scripts" / "fetch_data.sh"), *args],
                          capture_output=True, text=True)


def test_fetch_local_dir_and_check(tmp_path, repo_root):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.bin").write_bytes(b"hello")
    (src / "sub" / "b.bin").write_bytes(b"world")
    dest = tmp_path / "dest"
    lic = _license(tmp_path)
    r = _run_fetch(repo_root, "--source", str(src), "--dest", str(dest),
                   "--accept-license", str(lic))
    assert r.returncode == 0, r.stderr
    assert (dest / "a.bin").read_bytes() == b"hello"
    assert (dest / "sub" / "b.bin").read_bytes() == b"world"
    assert (dest / "checksums.sha256").exists()
    prov = json.loads((dest / "provenance.json").read_text())
    assert prov["mode"] == "local" and prov["n_files"] >= 2
    assert prov["accepted_license_sha256"] is not None
    assert "a.bin" in prov["files"]

    assert _run_fetch(repo_root, "--dest", str(dest), "--check").returncode == 0
    (dest / "a.bin").write_bytes(b"tampered")
    bad = _run_fetch(repo_root, "--dest", str(dest), "--check")
    assert bad.returncode == 3
    assert "a.bin" in (bad.stdout + bad.stderr)


def test_fetch_url_list_file_scheme(tmp_path, repo_root):
    up = tmp_path / "up.bin"
    up.write_bytes(b"payload-1")
    urls = tmp_path / "urls.txt"
    urls.write_text(f"# comment\nfile://{up} nested/x.bin\n")
    dest = tmp_path / "dest"
    lic = _license(tmp_path)
    r = _run_fetch(repo_root, "--source", str(urls), "--dest", str(dest),
                   "--accept-license", str(lic))
    assert r.returncode == 0, r.stderr
    assert (dest / "nested" / "x.bin").read_bytes() == b"payload-1"
    assert _run_fetch(repo_root, "--dest", str(dest), "--check").returncode == 0


def test_fetch_requires_license(tmp_path, repo_root):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a").write_bytes(b"x")
    r = _run_fetch(repo_root, "--source", str(src), "--dest", str(tmp_path / "d"))
    assert r.returncode == 2
    assert "许可" in (r.stdout + r.stderr)


def test_fetch_check_missing_dest(tmp_path, repo_root):
    r = _run_fetch(repo_root, "--dest", str(tmp_path / "nope"), "--check")
    assert r.returncode == 3


# ---------------- make_derived_gt.py ----------------

def _make_mvdt_seq(raw: Path, seq: str, traj_text: str, calib_doc=None, n_frames=4):
    seq_dir = raw / Path(seq)
    frames = seq_dir / "frames"
    frames.mkdir(parents=True)
    for i in range(n_frames):
        cv2.imwrite(str(frames / f"{i:06d}.png"), np.zeros((CANVAS[0], CANVAS[1], 3), np.uint8))
    (seq_dir / "trajectory.csv").write_text(traj_text)
    if calib_doc is not None:
        (seq_dir / "calibration.json").write_text(json.dumps(calib_doc))
    return seq_dir


def _run_gt(repo_root, *args):
    return subprocess.run([sys.executable, str(repo_root / "scripts" / "make_derived_gt.py"), *args],
                          cwd=str(repo_root), capture_output=True, text=True)


def test_make_derived_gt_published2d(tmp_path, repo_root):
    raw = tmp_path / "mvdt"
    traj = "frame,cx,cy\n0,10.0,20.0\n1,10.7,20.0\n2,11.4,20.0\n3,12.1,20.0\n"
    _make_mvdt_seq(raw, "D1/cam2", traj)
    out = tmp_path / "gt" / "D1_cam2.csv"
    r = _run_gt(repo_root, "--seq", "D1/cam2", "--raw", str(raw), "--out", str(out))
    assert r.returncode == 0, r.stderr
    rows = R.read_gt_csv(out)
    assert len(rows) == 4
    assert all(g.gt_source == "published2d" for g in rows)
    assert rows[0].visible == 1 and rows[0].cx == 10.0
    assert rows[0].width_px is None
    readme = out.parent / "README.md"
    assert readme.exists() and "published2d" in readme.read_text()


def test_make_derived_gt_auto_crosscheck_rmse(tmp_path, repo_root):
    raw = tmp_path / "mvdt"
    calib = {"K": [[100, 0, 64], [0, 100, 48], [0, 0, 1]],
             "R": np.eye(3).tolist(), "t": [0, 0, 0], "distortion_model": "none"}
    uv = [(10.0, 20.0), (10.7, 20.0), (11.4, 20.0), (12.1, 20.0)]
    lines = ["frame,cx,cy,X,Y,Z"]
    for i, (u, v) in enumerate(uv):
        lines.append(f"{i},{u},{v},{(u-64)/100},{(v-48)/100},1.0")
    _make_mvdt_seq(raw, "D1/cam2", "\n".join(lines) + "\n", calib)
    out = tmp_path / "gt" / "D1_cam2.csv"
    r = _run_gt(repo_root, "--seq", "D1/cam2", "--raw", str(raw), "--out", str(out),
                "--emit-sensitivity")
    assert r.returncode == 0, r.stderr
    rows = R.read_gt_csv(out)
    assert all(g.gt_source == "published2d" for g in rows)
    assert all(g.projection_rmse_px is not None and g.projection_rmse_px < 1e-6 for g in rows)
    sens = out.parent / "sensitivity" / "D1_cam2_shift_pm0.5.csv"
    assert sens.exists()
    text = sens.read_text().splitlines()
    assert text[0] == "frame,cx_plus,cy_plus,cx_minus,cy_minus"
    assert len(text) == 1 + 4


def test_make_derived_gt_project3d(tmp_path, repo_root):
    raw = tmp_path / "mvdt"
    calib = {"K": [[100, 0, 64], [0, 100, 48], [0, 0, 1]],
             "R": np.eye(3).tolist(), "t": [0, 0, 0], "distortion_model": "none"}
    lines = ["frame,X,Y,Z"]
    for i in range(4):
        u = 10.0 + 0.7 * i
        lines.append(f"{i},{(u-64)/100},{(20.0-48)/100},1.0")
    _make_mvdt_seq(raw, "D2/cam1", "\n".join(lines) + "\n", calib)
    out = tmp_path / "gt" / "D2_cam1.csv"
    r = _run_gt(repo_root, "--seq", "D2/cam1", "--raw", str(raw), "--out", str(out),
                "--method", "auto")
    assert r.returncode == 0, r.stderr
    rows = R.read_gt_csv(out)
    assert all(g.gt_source == "project3d" for g in rows)
    assert rows[0].visible == 1
    assert abs(rows[0].cx - 10.0) < 1e-6


def test_make_derived_gt_missing_data(tmp_path, repo_root):
    r = _run_gt(repo_root, "--seq", "D1/cam2", "--raw", str(tmp_path / "none"),
                "--out", str(tmp_path / "o.csv"))
    assert r.returncode == 3
    assert "make_synthetic" in (r.stdout + r.stderr)


def test_make_derived_gt_fps_assert(tmp_path, repo_root):
    raw = tmp_path / "mvdt"
    _make_mvdt_seq(raw, "D1/cam2", "frame,cx,cy\n0,10,20\n")
    r = _run_gt(repo_root, "--seq", "D1/cam2", "--raw", str(raw),
                "--out", str(tmp_path / "o.csv"), "--fps", "40.0")
    assert r.returncode == 2


def test_make_derived_gt_config_assert(tmp_path, repo_root):
    raw = tmp_path / "mvdt"
    _make_mvdt_seq(raw, "D1/cam2", "frame,cx,cy\n0,10,20\n1,11,20\n")
    out = tmp_path / "D1_cam2.csv"
    r = _run_gt(repo_root, "--seq", "D1/cam2", "--raw", str(raw), "--out", str(out),
                "--config", str(repo_root / "configs" / "D1_no_overlap.yaml"))
    assert r.returncode == 2  # 分辨率 128x96 != 1920x1080


# ---------------- real MVDT formats ----------------

def test_load_calibration_mvdt_style(tmp_path):
    doc = {
        "K-matrix": [[1506.6612, 0, 966.6623], [0, 1505.2822, 530.2608], [0, 0, 1]],
        "distCoeff": [-0.006673507597820779, 0.007775663251591633, 0, 0],
        "fps": 50, "resolution": [1920, 1080],
    }
    p = tmp_path / "sonyG_1.json"
    p.write_text(json.dumps(doc))
    cam = io.load_calibration(p)
    assert cam is not None
    assert cam.K[0, 0] == pytest.approx(1506.6612)
    assert cam.dist.size == 4
    assert cam.distortion_model == "radtan"
    assert cam.image_size == (1920, 1080)
    assert cam.fps == 50.0
    assert cam.has_extrinsics is False


def test_load_detections_and_rtk_formats(tmp_path):
    det = tmp_path / "cam2.txt"
    det.write_text("1.000000 1766.58 499.30\n2.000000 1766.86 499.92\n")
    traj = io.load_trajectory(det)
    assert traj.uv is not None and traj.xyz is None
    assert list(traj.frames) == [1, 2]
    assert traj.uv[0, 0] == pytest.approx(1766.58)

    rtk = tmp_path / "rtk.txt"
    rtk.write_text("  7.4667080e+00   1.0220384e+01  -7.4575476e+00\n"
                   "  7.4876051e+00   1.0200182e+01  -7.4532373e+00\n")
    t3 = io.load_trajectory(rtk)
    assert t3.xyz is not None and t3.uv is None
    assert list(t3.frames) == [0, 1]
    assert t3.xyz[0, 0] == pytest.approx(7.4667080)


def test_make_derived_gt_mvdt_layout(tmp_path, repo_root):
    raw = tmp_path / "mvdt"
    (raw / "dataset1" / "detections").mkdir(parents=True)
    (raw / "dataset1" / "trajectory").mkdir(parents=True)
    (raw / "calibration" / "sonyG").mkdir(parents=True)
    (raw / "dataset1" / "cameras.txt").write_text("cam0 - iphone6\ncam1 - p20pro\ncam2 - sonyG\n")
    (raw / "dataset1" / "detections" / "cam2.txt").write_text(
        "1.000000 1766.58 499.30\n2.000000 1766.86 499.92\n3.000000 1767.60 499.50\n")
    (raw / "dataset1" / "trajectory" / "rtk.txt").write_text(
        "7.1 1.0 -7.0\n7.2 1.1 -7.1\n7.3 1.2 -7.2\n")
    (raw / "calibration" / "sonyG" / "sonyG_1.json").write_text(json.dumps({
        "K-matrix": [[1506.6612, 0, 966.6623], [0, 1505.2822, 530.2608], [0, 0, 1]],
        "distCoeff": [-0.0066, 0.0077, 0, 0], "fps": 50, "resolution": [1920, 1080]}))

    out = tmp_path / "gt" / "D1_cam2.csv"
    r = _run_gt(repo_root, "--seq", "dataset1/cam2",
                "--raw", str(raw), "--out", str(out))
    assert r.returncode == 0, r.stderr
    rows = R.read_gt_csv(out)
    assert len(rows) == 3
    assert all(g.gt_source == "published2d" for g in rows)
    assert rows[0].cx == pytest.approx(1766.58)
    assert rows[0].visible == 1  # 在 1920x1080 内
    assert rows[0].width_px is None
    assert "published2d" in (out.parent / "README.md").read_text()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data_raw" / "mvdt" / "dataset1"
         / "detections" / "cam2.txt").exists(),
    reason="真实 MVDT 未下载（运行 scripts/fetch_data.sh 后可测）")
def test_make_derived_gt_real_mvdt(tmp_path, repo_root):
    out = tmp_path / "dataset1_cam2.csv"
    r = _run_gt(repo_root, "--seq", "dataset1/cam2",
                "--raw", str(repo_root / "data_raw" / "mvdt"), "--out", str(out))
    assert r.returncode == 0, r.stderr
    rows = R.read_gt_csv(out)
    assert len(rows) == 8016
    assert all(g.gt_source == "published2d" for g in rows)
    assert sum(g.visible for g in rows) > 3000  # (0,0) 占位视为 absent

