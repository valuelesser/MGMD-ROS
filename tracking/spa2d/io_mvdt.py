"""spa2d.io_mvdt — MVDT 数据读取（SPEC §4.1 / §5.10）。

提供：图像序列 / 视频解码（``open_sequence``）、帧时间戳（``frame_timestamps``）、
标定读取（``load_calibration``）、轨迹读取（``load_trajectory``）、
3D→2D 投影含畸变与 RMSE（``project_3d_to_2d``），以及序列探针（分辨率/帧数/色彩顺序）。
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yaml

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class SequenceInfo:
    frame_count: int
    width: int
    height: int
    color_order: str
    source_kind: str  # "frames" | "video"


@dataclass
class CameraModel:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(0))
    distortion_model: str = "none"
    image_size: tuple[int, int] | None = None
    fps: float | None = None
    has_extrinsics: bool = False


@dataclass
class Trajectory:
    frames: np.ndarray
    xyz: np.ndarray | None = None
    uv: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    bbox: np.ndarray | None = None
    area: np.ndarray | None = None
    width: np.ndarray | None = None
    fps: float | None = None


def _natural_key(name: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


class FrameSource:
    def __init__(self, frames_dir: str | None = None, video: str | None = None,
                 glob: str = "*.jpg") -> None:
        if (frames_dir is None) == (video is None):
            raise ValueError("open_sequence 需要且仅需要 frames_dir 或 video 之一")
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.video = Path(video) if video else None
        self.glob = glob
        self._cap = None
        self._pos = -1
        self._paths: list[Path] = []
        if self.frames_dir is not None:
            if not self.frames_dir.is_dir():
                raise FileNotFoundError(f"帧目录不存在: {self.frames_dir}")
            paths = [p for p in self.frames_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            self._paths = sorted(paths, key=lambda p: _natural_key(p.name))
            if not self._paths:
                raise FileNotFoundError(f"帧目录无图像: {self.frames_dir}")

    def __len__(self) -> int:
        if self.frames_dir is not None:
            return len(self._paths)
        cap = cv2.VideoCapture(str(self.video))
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        if self.frames_dir is not None:
            for i, p in enumerate(self._paths):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"解码失败: {p}")
                yield i, img
            return
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {self.video}")
        try:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, frame
                i += 1
        finally:
            cap.release()

    def read(self, idx: int) -> np.ndarray:
        """随机/顺序读取第 idx 帧（BGR）。视频用常驻 capture 顺序前进，避免重复 open/seek。"""
        if self.frames_dir is not None:
            if not (0 <= idx < len(self._paths)):
                raise IndexError(f"帧索引越界: {idx}")
            img = cv2.imread(str(self._paths[idx]), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"解码失败: {self._paths[idx]}")
            return img
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.video))
            if not self._cap.isOpened():
                raise ValueError(f"无法打开视频: {self.video}")
            self._pos = 0
        if idx < self._pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._pos = idx
        frame = None
        while self._pos <= idx:
            ok, frame = self._cap.read()
            if not ok:
                raise ValueError(f"视频解码失败: {self.video} @ {idx}")
            self._pos += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._pos = -1

    def probe(self) -> SequenceInfo:
        if self.frames_dir is not None:
            img = cv2.imread(str(self._paths[0]), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"解码失败: {self._paths[0]}")
            h, w = img.shape[:2]
            return SequenceInfo(len(self._paths), w, h, "BGR", "frames")
        cap = cv2.VideoCapture(str(self.video))
        try:
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {self.video}")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return SequenceInfo(n, w, h, "BGR", "video")
        finally:
            cap.release()


def open_sequence(frames_dir: str | None = None, video: str | None = None,
                  glob: str = "*.jpg") -> FrameSource:
    return FrameSource(frames_dir=frames_dir, video=video, glob=glob)


def frame_timestamps(fps: float, n: int, dataset_ts: np.ndarray | None = None) -> np.ndarray:
    if dataset_ts is not None:
        arr = np.asarray(dataset_ts, dtype=np.float64)
        if arr.shape[0] != n:
            raise ValueError(f"数据集时间戳长度 {arr.shape[0]} != 帧数 {n}")
        return arr
    return np.arange(n, dtype=np.float64) / float(fps)


def _load_struct(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"标定/轨迹根节点必须是映射: {path}")
    return data


def _get_first(data: dict, keys) -> object | None:
    for k in keys:
        if k in data:
            return data[k]
    return None


def load_calibration(path: str | Path | None) -> CameraModel | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = _load_struct(p)

    K = _get_first(data, ("K", "intrinsics", "camera_matrix", "K-matrix", "K_matrix"))
    if K is None:
        raise ValueError(f"标定缺少内参 K: {p}")
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)

    R = _get_first(data, ("R", "rotation", "R-matrix", "R_matrix"))
    t = _get_first(data, ("t", "translation", "t-vec", "t_vec", "T-vec"))
    has_extrinsics = R is not None or t is not None
    if R is None and "T" in data:
        T = np.asarray(data["T"], dtype=np.float64).reshape(4, 4)
        R, t = T[:3, :3], T[:3, 3]
        has_extrinsics = True
    R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64).reshape(3)

    dist = _get_first(data, ("dist", "dist_coeffs", "D", "distortion_coeffs", "distCoeff"))
    dist = np.zeros(0) if dist is None else np.asarray(dist, dtype=np.float64).ravel()

    dist_block = data.get("distortion")
    model = data.get("distortion_model")
    if model is None and isinstance(dist_block, dict):
        model = dist_block.get("model")
        if dist.size == 0 and "coeffs" in dist_block:
            dist = np.asarray(dist_block["coeffs"], dtype=np.float64).ravel()
    if model is None:
        model = "none" if dist.size == 0 else "radtan"

    size = _get_first(data, ("image_size", "resolution"))
    image_size = None if size is None else (int(size[0]), int(size[1]))
    fps = data.get("fps")
    return CameraModel(K=K, R=R, t=t, dist=dist, distortion_model=str(model),
                       image_size=image_size,
                       fps=None if fps is None else float(fps),
                       has_extrinsics=has_extrinsics)


_FRAME_KEYS = ("frame", "f", "index", "idx")
_X_KEYS = ("x", "x_m")
_Y_KEYS = ("y", "y_m")
_Z_KEYS = ("z", "z_m")
_U_KEYS = ("cx", "u", "x2d", "col")
_V_KEYS = ("cy", "v", "y2d", "row")
_TS_KEYS = ("timestamp", "timestamp_s", "time", "t")


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def load_trajectory(path: str | Path | None) -> Trajectory | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        frames = np.asarray(data.get("frame", data.get("frames")), dtype=np.int64)
        xyz = np.asarray(data["xyz"], dtype=np.float64) if "xyz" in data else None
        uv = np.asarray(data["uv"], dtype=np.float64) if "uv" in data else None
        ts = np.asarray(data["timestamp"], dtype=np.float64) if "timestamp" in data else None
        return Trajectory(frames=frames, xyz=xyz, uv=uv, timestamps=ts,
                          fps=data.get("fps"))

    text_lines = p.read_text(encoding="utf-8").splitlines()
    meaningful = [ln for ln in text_lines
                  if ln.strip() and not ln.strip().startswith("#")]
    if not meaningful:
        return Trajectory(frames=np.zeros(0, dtype=np.int64))

    if "," in meaningful[0]:
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            raw_rows = [r for r in reader if any(v not in (None, "") for v in r.values())]
        if raw_rows:
            header = [_norm(k) for k in raw_rows[0].keys()]
            norm_rows = [{_norm(k): v for k, v in r.items()} for r in raw_rows]
            return _trajectory_from_rows(norm_rows, header)
        return Trajectory(frames=np.zeros(0, dtype=np.int64))

    rows = [ln.split() for ln in meaningful if all(_is_number(t) for t in ln.split())]
    if not rows:
        raise ValueError(f"无法解析轨迹（无数值行）: {p}")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"轨迹列数不一致: {sorted(widths)}: {p}")
    arr = np.asarray(rows, dtype=np.float64)
    n = arr.shape[0]
    if arr.shape[1] >= 4:
        return Trajectory(frames=arr[:, 0].astype(np.int64), xyz=arr[:, 1:4])
    if arr.shape[1] == 3:
        col0 = arr[:, 0]
        looks_frame = np.allclose(col0, np.round(col0), atol=1e-6) and (
            n <= 1 or np.all(np.diff(col0) >= 1.0 - 1e-6))
        if looks_frame:
            return Trajectory(frames=col0.astype(np.int64), uv=arr[:, 1:3])
        return Trajectory(frames=np.arange(n, dtype=np.int64), xyz=arr[:, :3])
    raise ValueError(f"无法解析轨迹（列数 {arr.shape[1]}）: {p}")


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _col(header: list[str], keys) -> int | None:
    for k in keys:
        if k in header:
            return header.index(k)
    return None


def _trajectory_from_rows(rows: list[dict], header: list[str]) -> Trajectory:
    def col(keys):
        return _col(header, keys)

    fcol = col(_FRAME_KEYS)
    frames = np.array([int(float(r[header[fcol]])) for r in rows], dtype=np.int64) \
        if fcol is not None else np.arange(len(rows), dtype=np.int64)

    def gather(keys):
        c = col(keys)
        if c is None:
            return None
        name = header[c]
        return np.array([float(r[name]) if r[name] not in ("", None) else np.nan for r in rows],
                        dtype=np.float64)

    x, y, z = gather(_X_KEYS), gather(_Y_KEYS), gather(_Z_KEYS)
    u, v = gather(_U_KEYS), gather(_V_KEYS)
    ts = gather(_TS_KEYS)

    xyz = None
    if x is not None and y is not None and z is not None:
        xyz = np.stack([x, y, z], axis=1)
    uv = None
    if u is not None and v is not None:
        uv = np.stack([u, v], axis=1)
    elif xyz is None and x is not None and y is not None:
        uv = np.stack([x, y], axis=1)

    bx, by, bw, bh = (gather(("bbox_x",)), gather(("bbox_y",)),
                      gather(("bbox_w",)), gather(("bbox_h",)))
    bbox = np.stack([bx, by, bw, bh], axis=1) if all(
        v is not None for v in (bx, by, bw, bh)) else None
    area = gather(("area_px", "area"))
    width = gather(("width_px", "width"))
    return Trajectory(frames=frames, xyz=xyz, uv=uv, timestamps=ts,
                      bbox=bbox, area=area, width=width)


def _radtan(xn: np.ndarray, dist: np.ndarray) -> np.ndarray:
    k = list(dist) + [0.0] * max(0, 5 - dist.size)
    k1, k2, p1, p2, k3 = k[:5]
    x, y = xn[:, 0], xn[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([xd, yd], axis=1)


def project_3d_to_2d(traj: Trajectory, cam: CameraModel, fps: float | None = None
                     ) -> tuple[np.ndarray, float | None]:
    """x = K [R|t] X（含已声明畸变）；返回 (uv(N,2), rmse 或 None)。

    rmse 仅当轨迹同时含发布 2D 位置（``traj.uv``）时计算，用于 cross-check。
    """
    if traj.xyz is None or traj.xyz.size == 0:
        raise ValueError("轨迹缺少 3D 坐标，无法投影")
    X = np.asarray(traj.xyz, dtype=np.float64).reshape(-1, 3)
    Xc = X @ cam.R.T + cam.t.reshape(1, 3)
    z = Xc[:, 2:3]
    xn = Xc[:, :2] / np.where(np.abs(z) < 1e-12, 1e-12, z)
    model = cam.distortion_model.lower()
    if model in ("none", "pinhole", "") or cam.dist.size == 0:
        undist = xn
    elif model in ("radtan", "plumb_bob", "rational"):
        undist = _radtan(xn, cam.dist)
    else:
        raise ValueError(f"未支持的畸变模型: {cam.distortion_model}")
    homog = np.concatenate([undist, np.ones((undist.shape[0], 1))], axis=1)
    uv = (homog @ cam.K.T)[:, :2]

    rmse = None
    if traj.uv is not None and traj.uv.shape == uv.shape:
        d = uv - np.asarray(traj.uv, dtype=np.float64)
        valid = ~np.isnan(d).any(axis=1)
        if valid.any():
            rmse = float(np.sqrt(np.mean(np.sum(d[valid] ** 2, axis=1))))
    return uv, rmse


def sequence_frames_hash(frames_dir: str | Path, glob: str = "*") -> str:
    import hashlib

    h = hashlib.sha256()
    root = Path(frames_dir)
    for p in sorted((q for q in root.rglob(glob) if q.is_file()),
                    key=lambda q: _natural_key(str(q))):
        h.update(str(p.relative_to(root)).encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()
