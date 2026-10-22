"""tests/test_mgmd_viz.py — 可视化核心单测（SPEC V1）。

覆盖：classify 四分类、RingBuffer 滚动、Navigator 键位/扫描、draw_overlay
（含 downscale 坐标与颜色）、FakeWindow、Visualizer 暂停/退出。
全部用例不创建真实 GUI。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mgmd import viz
from spa2d import records as R


# ---------------------------------------------------------------- helpers

def _cfg(downscale: float = 1.0, gap: int = 1, or_mode: bool = False,
         radius: float = 100.0, resolution=(400, 200)) -> SimpleNamespace:
    return SimpleNamespace(
        config_name="mgmd_T1",
        runtime=SimpleNamespace(downscale=downscale),
        mfe=SimpleNamespace(frame_gap=gap, or_aggregation=or_mode),
        metrics=SimpleNamespace(primary_radius_px=radius),
        sequence=SimpleNamespace(resolution=list(resolution)),
    )


def _record(index: int = 10, x: float = 100.0, y: float = 50.0,
            present: bool = True, vx: float = 0.0, vy: float = 0.0,
            score: float = 1.0) -> R.WindowRecord:
    est = R.Estimate(x=x, y=y, vx=vx, vy=vy) if present else None
    conf = R.Confidence(pnr=score, mu_bg=0.0, sigma_bg=0.0, margin=score,
                        single_candidate=False)
    return R.WindowRecord(
        window_index=index, t_start=index, t_center=float(index),
        assoc_frame=index, search_mode="global",
        state_in="ABSENT" if not present else "TRACKED",
        state_out="ABSENT" if not present else "TRACKED",
        accepted=present, target_present=present,
        reason="OK" if present else "ABSENT",
        est=est, peak=None, confidence=conf, continuity=None,
        flags=R.Flags(), timing_ms=None)


def _gt(index: int = 10, x: float = 100.0, y: float = 50.0,
        visible: int = 1) -> R.GTFrame:
    return R.GTFrame(frame=index, timestamp_s=index / 25.0, visible=visible,
                     out_of_frame=0, cx=x, cy=y,
                     bbox_x=x - 5.0, bbox_y=y - 5.0, bbox_w=10.0, bbox_h=10.0)


def _view(index: int, outcome: str, rec: R.WindowRecord | None = None,
          gt: R.GTFrame | None = None) -> viz.FrameView:
    return viz.FrameView(index=index, record=rec or _record(index), rect=None,
                         gt=gt, outcome=outcome)


class PauseKeyWindow(viz.FakeWindow):
    """只在暂停循环的 waitKey(30) 返回队列键；逐帧 waitKey(1) 不消耗键。"""

    def wait_key(self, delay_ms: int = 1) -> int:
        if delay_ms >= 20 and self._keys:
            return int(self._keys.popleft())
        if delay_ms >= 20:
            self.closed = True
        return -1


# ---------------------------------------------------------------- classify

def test_classify_correct():
    rec = _record(x=100.0, y=50.0)
    assert viz.classify(rec, _gt(x=101.0, y=51.0), 100.0) == "correct"


def test_classify_fp_without_gt():
    rec = _record(x=100.0, y=50.0)
    assert viz.classify(rec, _gt(visible=0), 100.0) == "fp"


def test_classify_fp_out_of_radius():
    rec = _record(x=0.0, y=0.0)
    assert viz.classify(rec, _gt(x=100.0, y=100.0), 50.0) == "fp"


def test_classify_miss_and_absent():
    assert viz.classify(_record(present=False), _gt(), 100.0) == "miss"
    assert viz.classify(_record(present=False), None, 100.0) == "absent"


def test_classify_present_without_est_is_fp():
    rec = _record(present=True)
    rec.est = None
    assert viz.classify(rec, _gt(), 100.0) == "fp"


# ---------------------------------------------------------------- RingBuffer

def test_ring_buffer_rollover_and_lookup():
    buf = viz.RingBuffer(2)
    for i in range(3):
        frame = np.full((2, 2, 3), i, dtype=np.uint8)
        buf.append(_view(i, "absent"), frame)
    assert len(buf) == 2
    assert (buf.first, buf.last) == (1, 2)
    item = buf.get(2)
    assert item is not None and item[0].index == 2 and int(item[1][0, 0, 0]) == 2
    assert buf.get(0) is None
    with pytest.raises(ValueError):
        viz.RingBuffer(0)


# ---------------------------------------------------------------- Navigator

def test_navigator_toggle_and_steps():
    nav = viz.Navigator()
    assert nav.mode == "play"
    assert nav.on_key(viz.KEY_SPACE) == "toggle" and nav.mode == "pause"
    assert nav.on_key(viz.KEY_SPACE) == "toggle" and nav.mode == "play"
    assert nav.on_key(viz.KEY_LEFT) == "back1"
    assert nav.on_key(viz.KEY_RIGHT) == "forward1"
    assert nav.on_key(viz.KEY_UP) == "back10"
    assert nav.on_key(viz.KEY_DOWN) == "forward10"
    assert nav.on_key(viz.KEY_A) == "back10"
    assert nav.on_key(viz.KEY_D) == "forward10"
    assert nav.on_key(viz.KEY_HOME) == "start"
    assert nav.on_key(viz.KEY_END) == "end"
    assert nav.on_key(viz.KEY_H) == "start"
    assert nav.on_key(viz.KEY_E) == "end"
    assert nav.on_key(viz.KEY_P) == "prev_fail"


def test_navigator_quit_stops_pipeline():
    nav = viz.Navigator()
    assert nav.on_key(viz.KEY_Q) == "quit"
    assert nav.mode == "stop"
    assert nav.decide(_view(0, "absent"), 1) is False


def test_navigator_scan_hits_fp_or_miss():
    nav = viz.Navigator(scan_max=100, clock=lambda: 0.0)
    assert nav.on_key(viz.KEY_N) == "scan" and nav.mode == "scan"
    assert nav.decide(_view(1, "absent"), 1) is True and nav.mode == "scan"
    assert nav.should_render(_view(1, "absent"), 1) is False
    assert nav.decide(_view(2, "fp"), 1) is True
    assert nav.mode == "pause" and nav.cursor == 2 and nav.jumps == 1
    assert nav.should_render(_view(2, "fp"), 1) is True


def test_navigator_scan_exhausted():
    nav = viz.Navigator(scan_max=2, clock=lambda: 0.0)
    nav.begin_scan()
    assert nav.decide(_view(1, "absent"), 1) is True
    assert nav.decide(_view(2, "absent"), 1) is True
    assert nav.mode == "pause" and nav.scan_exhausted is True


def test_navigator_scan_timeout():
    now = [0.0]
    nav = viz.Navigator(scan_max=1000, scan_timeout_s=10.0, clock=lambda: now[0])
    nav.begin_scan()
    now[0] = 11.0
    assert nav.decide(_view(1, "absent"), 1) is True
    assert nav.mode == "pause" and nav.scan_exhausted is True


def test_navigator_every_n_frames():
    nav = viz.Navigator()
    assert nav.should_render(_view(4, "correct"), 2) is True
    assert nav.should_render(_view(5, "correct"), 2) is False


# ---------------------------------------------------------------- overlay

def _pixels_of(frame: np.ndarray, color) -> np.ndarray:
    return np.argwhere(np.all(frame == np.asarray(color, dtype=np.uint8), axis=2))


def test_draw_overlay_maps_downscale_and_colors():
    cfg = _cfg(downscale=0.5, radius=100.0, resolution=(400, 200))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    rec = _record(index=10, x=100.0, y=50.0, vx=2.0, vy=1.0)
    gt = _gt(index=10, x=200.0, y=100.0)
    view = viz.FrameView(index=10, record=rec, rect=(45.0, 20.0, 10.0, 10.0),
                         gt=gt, outcome="correct")
    state = viz.OverlayState(mode="pause", fp=0, miss=0, total=5, start=0,
                             buffer_first=10, buffer_last=10)
    out = viz.draw_overlay(frame, view, cfg, {"mfe", "tf"}, state)
    assert out.shape == frame.shape and out is not frame

    green = _pixels_of(out, viz.COLOR_CORRECT)
    assert green.size > 0
    # 检测框/中心在显示坐标 (50,25)（原图 100,50 × downscale 0.5）
    assert abs(green[:, 1].mean() - 50.0) < 8.0
    assert abs(green[:, 0].mean() - 25.0) < 8.0

    yellow = _pixels_of(out, viz.COLOR_GT)
    assert yellow.size > 0
    assert abs(yellow[:, 1].mean() - 100.0) < 8.0
    assert abs(yellow[:, 0].mean() - 50.0) < 8.0

    cyan = _pixels_of(out, viz.COLOR_ARROW)
    assert cyan.size > 0


def test_draw_overlay_fp_color():
    cfg = _cfg(downscale=1.0, radius=10.0, resolution=(200, 100))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    rec = _record(index=2, x=50.0, y=50.0)
    gt = _gt(index=2, x=150.0, y=50.0)
    view = viz.FrameView(index=2, record=rec, rect=(45.0, 45.0, 10.0, 10.0),
                         gt=gt, outcome="fp")
    out = viz.draw_overlay(frame, view, cfg, {"mfe"},
                           viz.OverlayState("play", 1, 0, 5, 0, 2, 2))
    assert _pixels_of(out, viz.COLOR_FP).size > 0


def test_draw_overlay_miss_and_absent_do_not_draw_detection():
    cfg = _cfg(downscale=1.0, resolution=(200, 100))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    gt = _gt(index=3, x=120.0, y=40.0)
    miss = viz.FrameView(index=3, record=_record(index=3, present=False),
                         rect=None, gt=gt, outcome="miss")
    out = viz.draw_overlay(frame, miss, cfg, {"mfe"},
                           viz.OverlayState("play", 0, 1, 5, 0, 3, 3))
    assert _pixels_of(out, viz.COLOR_MISS).size == 0  # miss 只画 GT，无检测色
    assert _pixels_of(out, viz.COLOR_GT).size > 0

    # absent：即便 record 残留 est，也不得画检测（取文本面板外的位置验证）
    cfg2 = _cfg(downscale=1.0, resolution=(300, 200))
    frame2 = np.zeros((200, 300, 3), dtype=np.uint8)
    rec_abs = _record(index=4, present=False)
    rec_abs.est = R.Estimate(x=250.0, y=150.0, vx=0.0, vy=0.0)
    absent = viz.FrameView(index=4, record=rec_abs, rect=None, gt=None,
                            outcome="absent")
    out2 = viz.draw_overlay(frame2, absent, cfg2, {"mfe"},
                            viz.OverlayState("play", 0, 1, 5, 0, 4, 4))
    assert _pixels_of(out2, viz.COLOR_GT).size == 0
    assert np.all(out2[140:160, 240:260] == 0)


# ---------------------------------------------------------------- windows

def test_fake_window_and_trackbar():
    win = viz.FakeWindow(keys=[viz.KEY_SPACE, viz.KEY_Q])
    win.show(np.zeros((2, 2, 3), dtype=np.uint8))
    assert len(win.frames) == 1
    assert win.wait_key(1) == viz.KEY_SPACE
    assert win.wait_key(1) == viz.KEY_Q
    assert win.wait_key(1) == -1
    win.create_trackbar("frame", 9, None)
    win.set_trackbar("frame", 4)
    assert win.get_trackbar("frame") == 4
    win.close()
    assert win.closed is True


def test_visualizer_renders_and_quits():
    cfg = _cfg(downscale=0.5, radius=100.0, resolution=(400, 200))
    gt = [_gt(index=0, x=200.0, y=100.0)]
    win = viz.FakeWindow(keys=[viz.KEY_SPACE, viz.KEY_Q])
    vis = viz.Visualizer(cfg, {"mfe"}, gt, total=3, start=0, every=1,
                         buffer=10, width=200, window=win)
    rec = _record(index=0, x=100.0, y=50.0)
    ok = vis.update(0, np.zeros((100, 200, 3), dtype=np.uint8), rec,
                    (45.0, 20.0, 10.0, 10.0))
    assert ok is False
    summary = vis.summary()
    assert summary["aborted"] is True and summary["viewed"] == 1
    assert summary["fp"] == 1  # 原图 (100,50) vs GT (200,100) 距离约 111.8 > 100
    assert len(win.frames) >= 1
    vis.close()
    assert win.closed is True


def test_visualizer_step_then_resume():
    cfg = _cfg(downscale=1.0, resolution=(200, 100))
    win = viz.FakeWindow(keys=[viz.KEY_SPACE, viz.KEY_RIGHT, viz.KEY_SPACE])
    vis = viz.Visualizer(cfg, {"mfe"}, [], total=3, width=200, buffer=10,
                         window=win)
    rec = _record(index=0, present=False)
    assert vis.update(0, np.zeros((100, 200, 3), dtype=np.uint8), rec, None) is True
    assert vis.nav.mode == "play"
    assert vis.viewed == 1
    vis.close()


# ------------------------------------------------- pipeline integration

from mgmd import pipeline  # noqa: E402
from mgmd.config import MgmdError, load_config  # noqa: E402
from mgmd.io_mvdt import MgmdFrameSource  # noqa: E402
from tests.test_mgmd_pipeline import H, N_FRAMES, W, _make_seq, _write_cfg  # noqa: E402


def test_pipeline_on_frame_callback_and_metrics_invariance(tmp_path):
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg = load_config(_write_cfg(tmp_path, frames_dir))
    calls = []

    def cb(index, frame, record, rect):
        calls.append((index, frame.shape, record.window_index, rect))
        return True

    out_a, out_b = tmp_path / "run_a", tmp_path / "run_b"
    assert pipeline.run(cfg, frozenset({"mfe"}), gt_csv, out_a, None, "numpy") == 0
    assert pipeline.run(cfg, frozenset({"mfe"}), gt_csv, out_b, None, "numpy",
                        on_frame=cb) == 0

    recs = R.read_jsonl(out_a / "windows.jsonl")
    assert len(calls) == len(recs) == N_FRAMES
    assert [c[0] for c in calls] == [r.window_index for r in recs]
    assert all(c[1] == (H, W, 3) for c in calls)
    for rec, call in zip(recs, calls):
        if rec.target_present:
            assert call[3] is not None
    assert (out_a / "metrics.json").read_bytes() == (out_b / "metrics.json").read_bytes()
    timing = json.loads((out_b / "timing.json").read_text())
    assert set(timing["layers"]) == {"t_pre", "t_search", "t_state", "t_decode",
                                     "t_window"}
    assert "viz" not in timing
    meta = json.loads((out_b / "run_meta.json").read_text())
    assert meta["viz"]["enabled"] is True
    assert meta["viz"]["viewed"] == N_FRAMES
    assert meta["viz"]["aborted"] is False


def test_pipeline_on_frame_early_stop(tmp_path):
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg = load_config(_write_cfg(tmp_path, frames_dir))
    seen = []

    def cb(index, frame, record, rect):
        seen.append(index)
        return len(seen) < 5

    out = tmp_path / "stop"
    assert pipeline.run(cfg, frozenset({"mfe"}), gt_csv, out, None, "numpy",
                        on_frame=cb) == 0
    recs = R.read_jsonl(out / "windows.jsonl")
    assert len(recs) == 5
    meta = json.loads((out / "run_meta.json").read_text())
    viz = meta["viz"]
    assert viz["aborted"] is True
    assert viz["abort_frame"] == 4
    assert viz["viewed"] == 5
    assert viz["jumps"] == 0 and viz["fp"] == 0 and viz["miss"] == 0
    assert (out / "metrics.json").is_file()


def test_pipeline_on_frame_exception_writes_partial_records(tmp_path):
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg = load_config(_write_cfg(tmp_path, frames_dir))

    def cb(index, frame, record, rect):
        if record.window_index == 2:
            raise RuntimeError("boom")
        return True

    out = tmp_path / "err"
    with pytest.raises(MgmdError) as ei:
        pipeline.run(cfg, frozenset({"mfe"}), gt_csv, out, None, "numpy",
                     on_frame=cb)
    assert ei.value.exit_code == 4
    recs = R.read_jsonl(out / "windows.jsonl")
    assert len(recs) == 3
    meta = json.loads((out / "run_meta.json").read_text())
    assert "boom" in meta["viz"]["error"]
    assert (out / "metrics.json").is_file()


def test_mgmd_frame_source_close(tmp_path):
    import cv2

    video = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"),
                             25.0, (W, H))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    writer.release()

    seq = SimpleNamespace(video=str(video), frames_dir="", frame_glob="*.png",
                          resolution=[W, H], color_order="BGR", name="T1/cam9",
                          fps=25.0)
    src = MgmdFrameSource(seq, SimpleNamespace(downscale=1.0), use_video=True)
    frame = src.read(0)
    assert frame.shape == (H, W, 3)
    assert src._src._cap is not None
    src.close()
    assert src._src._cap is None


# ------------------------------------------------------------ V3 CLI / window

import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_mgmd  # noqa: E402


def test_run_mgmd_help_lists_viz_flags():
    r = subprocess.run([sys.executable, "scripts/run_mgmd.py", "--help"],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    for flag in ("--viz", "--viz-every", "--viz-buffer", "--viz-width",
                 "--viz-scan-max"):
        assert flag in r.stdout


@pytest.mark.parametrize("extra", [
    ["--viz-every", "0"],
    ["--viz-buffer", "5"],
    ["--viz-width", "100"],
    ["--viz-scan-max", "0"],
])
def test_run_mgmd_invalid_viz_args_exit_2(tmp_path, extra):
    rc = run_mgmd.main([
        "--config", str(REPO / "configs" / "mgmd_D1.yaml"),
        "--gt", str(tmp_path / "g.csv"),
        "--out", str(tmp_path / "o"),
        *extra,
    ])
    assert rc == 2


def test_run_mgmd_viz_unavailable(tmp_path, monkeypatch, capsys):
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg_path = _write_cfg(tmp_path, frames_dir)

    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no display")

    monkeypatch.setattr(viz, "CvWindow", Boom)
    rc = run_mgmd.main([
        "--config", str(cfg_path), "--gt", str(gt_csv),
        "--out", str(tmp_path / "viz_out"), "--modules", "mfe",
        "--viz", "--limit", "4",
    ])
    assert rc == 4
    assert "E_VIZ_UNAVAILABLE" in capsys.readouterr().err


def test_visualizer_scan_jumps_to_fp():
    cfg = _cfg(downscale=1.0, resolution=(200, 100))
    win = PauseKeyWindow(keys=[viz.KEY_Q])
    vis = viz.Visualizer(cfg, {"mfe"}, [], total=10, width=200, buffer=10,
                         window=win)
    rec0 = _record(index=0, present=False)
    assert vis.update(0, np.zeros((100, 200, 3), dtype=np.uint8), rec0, None) is True
    vis.nav.begin_scan()
    rec1 = _record(index=1, x=50.0, y=50.0)
    assert vis.update(1, np.zeros((100, 200, 3), dtype=np.uint8), rec1,
                      (45.0, 45.0, 10.0, 10.0)) is False
    summary = vis.summary()
    assert summary["jumps"] == 1 and summary["aborted"] is True


def test_visualizer_prev_fail_and_trackbar():
    cfg = _cfg(downscale=1.0, resolution=(200, 100))
    win = viz.FakeWindow()
    vis = viz.Visualizer(cfg, {"mfe"}, [], total=5, width=200, buffer=10,
                         window=win)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    views = [
        viz.FrameView(0, _record(0), None, _gt(0), "correct"),
        viz.FrameView(1, _record(1), None, None, "fp"),
        viz.FrameView(2, _record(2, present=False), None, _gt(2), "miss"),
    ]
    for view in views:
        vis.buffer.append(view, frame.copy())
    vis.nav.cursor = 2
    vis._prev_fail()
    assert vis.nav.cursor == 1
    win.trackbars["frame"] = 0
    vis._last_tb = -1
    vis._poll_trackbar()
    assert vis.nav.cursor == 0
    vis.close()
    assert win.closed is True


def test_run_mgmd_ctrl_c_closes_visualizer(tmp_path, monkeypatch):
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg_path = _write_cfg(tmp_path, frames_dir)
    win = viz.FakeWindow()
    monkeypatch.setattr(viz, "CvWindow", lambda *a, **k: win)

    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_mgmd.pipeline, "run", boom)
    rc = run_mgmd.main([
        "--config", str(cfg_path), "--gt", str(gt_csv),
        "--out", str(tmp_path / "viz_cc"), "--modules", "mfe",
        "--viz", "--limit", "4",
    ])
    assert rc == 130
    assert win.closed is True


def test_run_mgmd_viz_q_early_exit(tmp_path, monkeypatch):
    frames_dir, gt_csv = _make_seq(tmp_path)
    cfg_path = _write_cfg(tmp_path, frames_dir)
    win = viz.FakeWindow(keys=[viz.KEY_Q])
    monkeypatch.setattr(viz, "CvWindow", lambda *a, **k: win)
    out = tmp_path / "viz_q"
    rc = run_mgmd.main([
        "--config", str(cfg_path), "--gt", str(gt_csv), "--out", str(out),
        "--modules", "mfe", "--viz", "--limit", "10",
    ])
    assert rc == 0
    assert win.closed is True
    recs = R.read_jsonl(out / "windows.jsonl")
    assert len(recs) == 1
    meta = json.loads((out / "run_meta.json").read_text())
    meta_viz = meta["viz"]
    assert meta_viz["aborted"] is True
    assert meta_viz["viewed"] == 1
    assert meta_viz["jumps"] == 0
    assert meta_viz["fp"] == 0 and meta_viz["miss"] == 1
    assert meta_viz["every"] == 1 and meta_viz["buffer"] == 200
    assert meta_viz["width"] == 1280


def test_draw_overlay_or_mode_gap_and_score_text():
    cfg = _cfg(downscale=1.0, gap=2, or_mode=True, resolution=(300, 200))
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    rec = _record(index=5, x=150.0, y=100.0, score=0.75)
    view = viz.FrameView(index=5, record=rec, rect=(145.0, 95.0, 10.0, 10.0),
                         gt=_gt(index=5, x=150.0, y=100.0), outcome="correct")
    out = viz.draw_overlay(frame, view, cfg, {"mfe", "tf"},
                           viz.OverlayState("play", 0, 0, 10, 0, 5, 5))
    assert out.shape == frame.shape
    # 文本面板 + 叠加已绘制（左上区域非全黑）
    assert np.any(out[8:110, 8:300] != 0)


def test_visualizer_trackbar_drag_pauses_and_seeks():
    cfg = _cfg(downscale=1.0, resolution=(200, 100))
    win = PauseKeyWindow(keys=[viz.KEY_Q])
    vis = viz.Visualizer(cfg, {"mfe"}, [], total=10, width=200, buffer=10,
                         window=win)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    for i in range(3):
        rec = _record(index=i, present=False)
        assert vis.update(i, frame, rec, None) is True
    assert vis.nav.mode == "play"
    vis._on_trackbar(1)               # 模拟播放态拖动进度条
    rec = _record(index=3, present=False)
    assert vis.update(3, frame, rec, None) is False
    assert vis.nav.cursor == 1
    assert vis.summary()["aborted"] is True


def test_visualizer_window_close_in_play_aborts():
    cfg = _cfg(downscale=1.0, resolution=(200, 100))
    win = viz.FakeWindow()
    vis = viz.Visualizer(cfg, {"mfe"}, [], total=5, width=200, buffer=10,
                         window=win)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert vis.update(0, frame, _record(index=0, present=False), None) is True
    win.closed = True
    assert vis.update(1, frame, _record(index=1, present=False), None) is False
    assert vis.summary()["aborted"] is True


def test_visualizer_prev_fail_during_play():
    cfg = _cfg(downscale=1.0, resolution=(200, 100))
    win = viz.FakeWindow()
    vis = viz.Visualizer(cfg, {"mfe"}, [], total=6, width=200, buffer=10,
                         window=win)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    views = [
        viz.FrameView(0, _record(0), None, None, "fp"),
        viz.FrameView(1, _record(1), None, None, "correct"),
        viz.FrameView(2, _record(2), None, None, "miss"),
        viz.FrameView(3, _record(3), None, None, "correct"),
    ]
    for view in views:
        vis.buffer.append(view, frame.copy())
    win._keys.extend([viz.KEY_P, viz.KEY_Q])
    assert vis.update(4, frame, _record(index=4, present=False), None) is False
    assert vis.nav.cursor == 2          # 跳过 frame3(correct)，回退到 frame2(miss)
    assert vis.summary()["aborted"] is True


def test_cvwindow_unavailable_without_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(viz.VizUnavailable):
        viz.CvWindow("__viz_no_display__")


def test_draw_overlay_status_right_and_pnr_label(monkeypatch):
    calls = []
    orig = viz.cv2.putText

    def fake(img, text, org, *args, **kwargs):
        calls.append((text, org))
        return orig(img, text, org, *args, **kwargs)

    monkeypatch.setattr(viz.cv2, "putText", fake)
    cfg = _cfg(downscale=1.0, resolution=(800, 200))
    frame = np.zeros((200, 800, 3), dtype=np.uint8)
    rec = _record(index=5, x=400.0, y=100.0)
    view = viz.FrameView(index=5, record=rec, rect=(395.0, 95.0, 10.0, 10.0),
                         gt=_gt(index=5, x=400.0, y=100.0), outcome="correct")
    viz.draw_overlay(frame, view, cfg, {"mfe"},
                     viz.OverlayState("play", 0, 0, 10, 0, 5, 5))
    assert any("pnr=" in text for text, _ in calls)
    status = [(text, org) for text, org in calls if "PLAY" in text]
    assert status and status[0][1][0] > 400


def test_visualizer_prev_fail_during_play_single_key():
    """审计探针：只给 p 键时也应进入 PAUSE 并定位上一个 fail（不挂起）。"""
    cfg = _cfg(downscale=1.0, resolution=(200, 100))
    win = viz.FakeWindow()
    vis = viz.Visualizer(cfg, {"mfe"}, [], total=6, width=200, buffer=10,
                         window=win)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    for view in [
        viz.FrameView(0, _record(0), None, None, "correct"),
        viz.FrameView(1, _record(1), None, None, "fp"),
        viz.FrameView(2, _record(2), None, None, "correct"),
    ]:
        vis.buffer.append(view, frame.copy())
    win._keys.append(viz.KEY_P)
    assert vis.update(3, frame, _record(index=3, present=False), None) is False
    assert vis.nav.mode == "pause"
    assert vis.nav.cursor == 1


def test_visualizer_prev_fail_during_play_after_updates():
    """审计探针：真实 play 更新 4 帧后按 p，应暂停并回退到上一个 fail。"""
    cfg = _cfg(downscale=1.0, resolution=(200, 100))
    gts = [_gt(index=i, x=100.0, y=50.0) for i in range(4)]
    gts += [_gt(index=4, visible=0), _gt(index=5, visible=0)]
    win = viz.FakeWindow()
    vis = viz.Visualizer(cfg, {"mfe"}, gts, total=6, width=200, buffer=10,
                         window=win)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    # 0:correct, 1:fp, 2:correct, 3:miss
    assert vis.update(0, frame, _record(index=0, x=100.0, y=50.0),
                      (95.0, 45.0, 10.0, 10.0)) is True
    assert vis.update(1, frame, _record(index=1, x=0.0, y=0.0),
                      (45.0, 45.0, 10.0, 10.0)) is True
    assert vis.update(2, frame, _record(index=2, x=100.0, y=50.0),
                      (95.0, 45.0, 10.0, 10.0)) is True
    assert vis.update(3, frame, _record(index=3, x=100.0, y=50.0),
                      (95.0, 45.0, 10.0, 10.0)) is True
    assert vis.nav.cursor == 3
    win._keys.append(viz.KEY_P)
    assert vis.update(4, frame, _record(index=4, present=False), None) is False
    assert vis.nav.mode == "pause"
    assert vis.nav.cursor == 1          # 跳过 frame3/frame2(correct)，回退到 frame1(fp)
    assert vis.summary()["aborted"] is True


def test_visualizer_prev_fail_includes_immediately_previous_frame():
    cfg = _cfg(downscale=1.0, resolution=(200, 100))
    win = viz.FakeWindow()
    vis = viz.Visualizer(cfg, {"mfe"}, [], total=6, width=200, buffer=10,
                         window=win)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    for view in [
        viz.FrameView(0, _record(0), None, None, "correct"),
        viz.FrameView(1, _record(1), None, None, "correct"),
        viz.FrameView(2, _record(2), None, None, "correct"),
        viz.FrameView(3, _record(3), None, None, "fp"),
    ]:
        vis.buffer.append(view, frame.copy())
    vis.nav.cursor = 3  # 上一帧（当前实时帧将是 4）
    win._keys.append(viz.KEY_P)
    assert vis.update(4, frame, _record(index=4, present=False), None) is False
    assert vis.nav.mode == "pause"
    assert vis.nav.cursor == 3
