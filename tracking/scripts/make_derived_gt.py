"""scripts/make_derived_gt.py — 生成派生 GT（SPEC §4.1 / §5.10 / §3.2）。

用法：
    python scripts/make_derived_gt.py --seq D1/cam2 --raw data_raw/mvdt \
        --out data_derived/gt/D1_cam2.csv [--method auto|published2d|project3d] \
        [--fps 50.0] [--config configs/D1_no_overlap.yaml] [--emit-sensitivity]

方法 auto：有发布 2D 位置 → primary=published2d 并用 3D+标定投影 cross-check 写
projection_rmse_px；否则仅 3D 轨迹 → project3d。数据不可达 → 退出码 3。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spa2d import io_mvdt as io  # noqa: E402
from spa2d import records as R  # noqa: E402

KNOWN_FPS = {"D1/cam2": 50.00, "D2/cam1": 29.84, "D3/cam3": 25.00, "D4/cam3": 21.24}
METHODS = ("auto", "published2d", "project3d")
RMSE_WARN_PX = 20.0
_CALIB_STEMS = ("calibration", "calib", "camera")
_TRAJ_STEMS = ("trajectory", "traj", "gt", "positions", "centers")
_STRUCT_EXTS = (".json", ".yaml", ".yml")
_TABULAR_EXTS = (".csv", ".txt", ".json")


class DataError(Exception):
    def __init__(self, code: str, message: str, remediation: str, exit_code: int = 3):
        self.code, self.message, self.remediation, self.exit_code = code, message, remediation, exit_code

    def format(self) -> str:
        return f"[spa2d] {self.code}: {self.message} → {self.remediation}"


def _find(seq_dir: Path, stems, exts) -> Path | None:
    for stem in stems:
        for ext in exts:
            p = seq_dir / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def _find_video(seq_dir: Path) -> Path | None:
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        hits = sorted(seq_dir.glob(f"*{ext}"))
        if hits:
            return hits[0]
    return None


def _resolve_fps(seq: str, given: float | None, cam_fps: float | None = None) -> float:
    known = KNOWN_FPS.get(seq)
    if given is not None:
        if known is not None and abs(float(given) - known) > 0.01:
            raise DataError("E_CONFIG_INVALID",
                            f"--fps {given} 与序列表 {known} 不一致",
                            "修正 --fps 或移除该参数", 2)
        return float(given)
    if known is not None:
        return known
    if cam_fps is not None:
        return float(cam_fps)
    raise DataError("E_CONFIG_INVALID", f"未知序列 {seq} 的 fps",
                    "用 --fps 显式指定", 2)


def _map_by_frame(values: np.ndarray | None, frames: np.ndarray, n: int,
                  width: int, offset: int | None = None) -> np.ndarray | None:
    if values is None:
        return None
    frames = np.asarray(frames).ravel()
    if offset is None:
        offset = 1 if (frames.size and int(frames.min()) == 1) else 0
    out = np.full((n, width), np.nan, dtype=np.float64)
    index = {int(f) - offset: k for k, f in enumerate(frames)}
    for i in range(n):
        k = index.get(i)
        if k is not None and k < len(values):
            out[i] = np.asarray(values[k], dtype=np.float64).ravel()[:width]
    return out


def _width_arrays(traj, n: int) -> tuple[np.ndarray, list[str]]:
    width = np.full(n, np.nan, dtype=np.float64)
    source: list[str] = [""] * n
    if traj is None:
        return width, source
    if traj.bbox is not None:
        mapped = _map_by_frame(traj.bbox[:, 2:3], traj.frames, n, 1)
        width = mapped[:, 0].copy()
        source = ["bbox_w"] * n
    elif traj.width is not None:
        mapped = _map_by_frame(traj.width.reshape(-1, 1), traj.frames, n, 1)
        width = mapped[:, 0].copy()
        source = ["bbox_w"] * n
    elif traj.area is not None:
        mapped = _map_by_frame(traj.area.reshape(-1, 1), traj.frames, n, 1)
        width = np.sqrt(np.clip(mapped[:, 0], 0, None))
        source = ["sqrt_area"] * n
    return width, source


def _resolve_method(method: str, traj, cam) -> tuple[str, np.ndarray, np.ndarray, float | None]:
    if traj is None:
        raise DataError("E_DATA_MISSING", "未找到轨迹文件",
                        "提供 trajectory.csv/txt/json，或改用合成数据 make_synthetic.py")
    has_uv = traj.uv is not None
    has_xyz = traj.xyz is not None
    if method in ("auto", "published2d"):
        if has_uv:
            rmse = None
            if has_xyz and cam is not None and cam.has_extrinsics:
                _, rmse = io.project_3d_to_2d(traj, cam)
            return "published2d", np.asarray(traj.uv, dtype=np.float64), traj.frames, rmse
        if method == "published2d":
            raise DataError("E_DATA_MISSING", "轨迹无发布 2D 位置",
                            "改用 --method project3d 或提供 2D 列")
    if method in ("auto", "project3d"):
        if not has_xyz or cam is None or not cam.has_extrinsics:
            raise DataError("E_DATA_MISSING", "缺少 3D 轨迹或相机外参，无法投影",
                            "提供 3D 轨迹 + 含 R/t 的标定，或用 make_synthetic.py")
        uv, rmse = io.project_3d_to_2d(traj, cam)
        return "project3d", uv, traj.frames, rmse
    raise DataError("E_CONFIG_INVALID", f"未知方法 {method}", "使用 auto|published2d|project3d", 2)


def _build_rows(seq: str, method: str, centers: np.ndarray, traj_frames, traj, n: int,
                fps: float, info, rmse: float | None) -> list[R.GTFrame]:
    centers = _map_by_frame(centers, traj_frames, n, 2)
    widths, wsource = _width_arrays(traj, n)
    ts = _map_by_frame(traj.timestamps.reshape(-1, 1), traj.frames, n, 1) if (
        traj is not None and traj.timestamps is not None) else None

    rows: list[R.GTFrame] = []
    for i in range(n):
        timestamp = float(ts[i, 0]) if ts is not None and not np.isnan(ts[i, 0]) else i / fps
        cx, cy = centers[i]
        w = None if np.isnan(widths[i]) else float(widths[i])
        src = wsource[i] or None
        if np.isnan(cx) or np.isnan(cy):
            rows.append(R.GTFrame(frame=i, timestamp_s=timestamp, visible=0,
                                  out_of_frame=0, width_px=w, width_source=src,
                                  gt_source=method, projection_rmse_px=rmse))
            continue
        if float(cx) == 0.0 and float(cy) == 0.0:
            # MVDT 标注用 (0,0) 表示目标未出现/未标注 → 视为 absent
            rows.append(R.GTFrame(frame=i, timestamp_s=timestamp, visible=0,
                                  out_of_frame=0, width_px=w, width_source=src,
                                  gt_source=method, projection_rmse_px=rmse))
            continue
        in_frame = (0.0 <= cx < info.width) and (0.0 <= cy < info.height)
        visible = 1 if in_frame else 0
        oof = 0 if in_frame else 1
        rows.append(R.GTFrame(
            frame=i, timestamp_s=timestamp, visible=visible, out_of_frame=oof,
            cx=round(float(cx), 4), cy=round(float(cy), 4),
            bbox_x=round(float(cx - w / 2), 4) if w is not None else None,
            bbox_y=round(float(cy - w / 2), 4) if w is not None else None,
            bbox_w=w, bbox_h=w, area_px=(w * w) if w is not None else None,
            width_px=w, width_source=src, gt_source=method, projection_rmse_px=rmse))
    return rows


def _shift_centers(centers: np.ndarray, delta: float) -> np.ndarray:
    n = centers.shape[0]
    out = np.full_like(centers, np.nan)
    for i in range(n):
        t = i + delta
        i0 = int(np.floor(t))
        frac = t - i0
        i1 = i0 + 1
        if 0 <= i0 < n and 0 <= i1 < n:
            a, b = centers[i0], centers[i1]
            if not np.isnan(a).any() and not np.isnan(b).any():
                out[i] = a * (1 - frac) + b * frac
        elif 0 <= i0 < n and not np.isnan(centers[i0]).any():
            out[i] = centers[i0]
        elif 0 <= i1 < n and not np.isnan(centers[i1]).any():
            out[i] = centers[i1]
    return out


def _write_sensitivity(sens_dir: Path, seq: str, centers: np.ndarray) -> Path:
    sens_dir.mkdir(parents=True, exist_ok=True)
    path = sens_dir / f"{seq.replace('/', '_')}_shift_pm0.5.csv"
    plus = _shift_centers(centers, 0.5)
    minus = _shift_centers(centers, -0.5)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("frame,cx_plus,cy_plus,cx_minus,cy_minus\n")
        for i in range(centers.shape[0]):
            def fmt(v):
                return "" if np.isnan(v) else f"{v:.6f}"
            fh.write(f"{i},{fmt(plus[i,0])},{fmt(plus[i,1])},"
                     f"{fmt(minus[i,0])},{fmt(minus[i,1])}\n")
    return path


def _update_readme(readme: Path, seq: str, method: str, calib_path, rmse, gt_sha: str, cmd: str) -> None:
    start = f"<!-- spa2d:seq {seq} -->"
    end = "<!-- /spa2d:seq -->"
    warn = ""
    if rmse is not None and rmse > RMSE_WARN_PX:
        warn = f"\n- ⚠️ cross-check RMSE={rmse:.2f}px > {RMSE_WARN_PX}px（published2d 与投影不一致）"
    block = (
        f"{start}\n### {seq}\n"
        f"- method: {method}\n"
        f"- calibration: {calib_path if calib_path else 'none'}\n"
        f"- distortion_model: (见标定)\n"
        f"- projection_rmse_px: {'N/A' if rmse is None else f'{rmse:.4f}'}{warn}\n"
        f"- gt_sha256: {gt_sha}\n"
        f"- command: `{cmd}`\n"
        f"{end}\n")
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
    else:
        text = "# Derived GT\n\n由 `scripts/make_derived_gt.py` 生成（SPEC §3.2/§5.10）。\n\n"
    if start in text and end in text:
        pre = text[:text.index(start)]
        post = text[text.index(end) + len(end):]
        text = pre + block + post
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + block
    readme.write_text(text, encoding="utf-8")


def _mvdt_calibration(raw, dataset_dir: Path, cam: str) -> Path | None:
    model = None
    cameras_txt = dataset_dir / "cameras.txt"
    if cameras_txt.exists():
        for line in cameras_txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.split()[0] == cam and "-" in line:
                model = line.split("-", 1)[1].strip()
    if not model:
        return None
    for suffix in ("_1", "_2", ""):
        p = Path(raw) / "calibration" / model / f"{model}{suffix}.json"
        if p.exists():
            return p
    return None


def _is_mvdt_layout(raw, seq: str) -> bool:
    parts = seq.split("/")
    if len(parts) != 2:
        return False
    ds_dir = Path(raw) / parts[0]
    return (ds_dir / "detections" / f"{parts[1]}.txt").exists() or \
        (ds_dir / "trajectory" / "rtk.txt").exists()


def _run_mvdt(args) -> int:
    dataset, cam = args.seq.split("/")[0], args.seq.split("/")[1]
    ds_dir = Path(args.raw) / dataset
    det_path = ds_dir / "detections" / f"{cam}.txt"
    rtk_path = ds_dir / "trajectory" / "rtk.txt"
    calib_path = _mvdt_calibration(args.raw, ds_dir, cam)

    try:
        det = io.load_trajectory(det_path) if det_path.exists() else None
        rtk = io.load_trajectory(rtk_path) if rtk_path.exists() else None
        cam_model = io.load_calibration(calib_path) if calib_path else None
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[spa2d] E_DATA_MISSING: MVDT 解析失败: {exc}", file=sys.stderr)
        return 3
    if det is None and rtk is None:
        print(f"[spa2d] E_DATA_MISSING: {ds_dir} 无 detections/ 或 trajectory/ → "
              f"用 make_synthetic.py 生成合成数据", file=sys.stderr)
        return 3

    if cam_model is not None and cam_model.image_size is not None:
        width, height = cam_model.image_size
    elif args.config is not None:
        from spa2d.config import load_config
        cfg = load_config(args.config)
        width, height = cfg.sequence.resolution
    else:
        print("[spa2d] E_DATA_MISSING: 缺少标定分辨率，且未提供 --config",
              file=sys.stderr)
        return 3

    try:
        fps = _resolve_fps(args.seq, args.fps, cam_model.fps if cam_model else None)
    except DataError as exc:
        print(exc.format(), file=sys.stderr)
        return exc.exit_code

    if det is not None:
        uv, frames = det.uv, det.frames
        xyz = rtk.xyz if (rtk is not None and rtk.xyz is not None
                          and len(rtk.xyz) == len(uv)) else None
    else:
        uv, frames, xyz = None, rtk.frames, rtk.xyz
    traj = io.Trajectory(frames=frames, uv=uv, xyz=xyz)
    n = len(frames)
    info = io.SequenceInfo(n, int(width), int(height), "BGR", "mvdt")

    if args.config is not None:
        from spa2d.config import load_config
        cfg = load_config(args.config)
        if (int(width), int(height)) != tuple(cfg.sequence.resolution):
            print(f"[spa2d] E_CONFIG_INVALID: 标定分辨率 {(width, height)} "
                  f"!= config {tuple(cfg.sequence.resolution)}", file=sys.stderr)
            return 2

    try:
        method, centers, traj_frames, rmse = _resolve_method(args.method, traj, cam_model)
    except DataError as exc:
        print(exc.format(), file=sys.stderr)
        return exc.exit_code

    rows = _build_rows(args.seq, method, centers, traj_frames, traj, n, fps, info, rmse)
    R.write_gt_csv(rows, args.out)
    gt_sha = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    cmd = "python scripts/make_derived_gt.py " + " ".join(sys.argv[1:])
    _update_readme(Path(args.out).parent / "README.md", args.seq, method, calib_path,
                   rmse, gt_sha, cmd)
    if args.emit_sensitivity:
        _write_sensitivity(Path(args.out).parent / "sensitivity", args.seq,
                           _map_by_frame(centers, traj_frames, n, 2))
    print(f"{args.seq}: method={method} frames={n} {width}x{height} fps={fps} "
          f"rmse={'NA' if rmse is None else f'{rmse:.3f}'} -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="make_derived_gt.py",
                                     description="生成 MVDT 派生 GT（SPEC §3.2）")
    parser.add_argument("--seq", required=True, help="序列名，如 D1/cam2")
    parser.add_argument("--raw", required=True, help="原始数据根，如 data_raw/mvdt")
    parser.add_argument("--out", required=True, help="输出 CSV，如 data_derived/gt/D1_cam2.csv")
    parser.add_argument("--method", choices=METHODS, default="auto")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--config", default=None, help="可选 config YAML，用于尺寸/色彩顺序断言")
    parser.add_argument("--emit-sensitivity", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _is_mvdt_layout(args.raw, args.seq):
        return _run_mvdt(args)
    seq_dir = Path(args.raw) / Path(args.seq)
    if not seq_dir.is_dir():
        print(f"[spa2d] E_DATA_MISSING: 序列目录不存在: {seq_dir} → "
              f"先获取数据（fetch_data.sh）或用 make_synthetic.py 生成合成数据", file=sys.stderr)
        return 3

    frames_dir = seq_dir / "frames"
    source_kwargs = {}
    if frames_dir.is_dir():
        source_kwargs["frames_dir"] = str(frames_dir)
    else:
        video = _find_video(seq_dir)
        if video is None:
            print(f"[spa2d] E_DATA_MISSING: {seq_dir} 下无 frames/ 或视频 → "
                  f"用 make_synthetic.py 生成合成数据", file=sys.stderr)
            return 3
        source_kwargs["video"] = str(video)

    try:
        source = io.open_sequence(**source_kwargs)
        info = source.probe()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[spa2d] E_DECODE: {exc} → 检查数据完整性", file=sys.stderr)
        return 4

    if args.config is not None:
        from spa2d.config import load_config
        cfg = load_config(args.config)
        if (info.width, info.height) != tuple(cfg.sequence.resolution):
            print(f"[spa2d] E_CONFIG_INVALID: 解码分辨率 {(info.width, info.height)} "
                  f"!= config {tuple(cfg.sequence.resolution)}", file=sys.stderr)
            return 2
        if info.color_order != cfg.sequence.color_order:
            print(f"[spa2d] E_CONFIG_INVALID: 色彩顺序 {info.color_order} "
                  f"!= config {cfg.sequence.color_order}", file=sys.stderr)
            return 2

    calib_path = _find(seq_dir, _CALIB_STEMS, _STRUCT_EXTS)
    traj_path = _find(seq_dir, _TRAJ_STEMS, _TABULAR_EXTS)
    try:
        cam = io.load_calibration(calib_path)
        traj = io.load_trajectory(traj_path)
    except (ValueError, json.JSONDecodeError) as exc:  # type: ignore[name-defined]
        print(f"[spa2d] E_DATA_MISSING: 标定/轨迹解析失败: {exc}", file=sys.stderr)
        return 3

    try:
        fps = _resolve_fps(args.seq, args.fps, cam.fps if cam is not None else None)
    except DataError as exc:
        print(exc.format(), file=sys.stderr)
        return exc.exit_code

    try:
        method, centers, traj_frames, rmse = _resolve_method(args.method, traj, cam)
    except DataError as exc:
        print(exc.format(), file=sys.stderr)
        return exc.exit_code

    rows = _build_rows(args.seq, method, centers, traj_frames, traj, info.frame_count,
                       fps, info, rmse)
    R.write_gt_csv(rows, args.out)
    gt_sha = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()

    cmd = "python scripts/make_derived_gt.py " + " ".join(sys.argv[1:])
    readme = Path(args.out).parent / "README.md"
    _update_readme(readme, args.seq, method, calib_path, rmse, gt_sha, cmd)

    if args.emit_sensitivity:
        _write_sensitivity(Path(args.out).parent / "sensitivity", args.seq,
                           _map_by_frame(centers, traj_frames, info.frame_count, 2))

    print(f"{args.seq}: method={method} frames={info.frame_count} "
          f"{info.width}x{info.height} fps={fps} "
          f"rmse={'NA' if rmse is None else f'{rmse:.3f}'} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
