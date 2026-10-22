"""spa2d.synthetic — 确定性合成序列生成器（SPEC §5.11 / §3.4）。

生成两类数据：
- ``sanity``：≥30 条短序列，暗高斯斑目标，宽度 ∈ {2,4,8,16}px，SNR ∈ {low,mid,high}，
  速度覆盖 D1/D2 的 [−5,5] 与 D3/D4 的 [−15,15]，含鸟/虫/云 hard negative。
- ``benchmark``：单序列，目标 (0.7, 0) px/frame、200 帧、静止背景纹理 + 噪声，附全网格真值表。

所有随机性来自 ``numpy.random.Generator``（seed 派生），同 seed 逐位可复现。
背景 = 天空渐变 + Perlin-like 平滑云层（低频正弦粗网格上采样）+ 高斯噪声（σ 由 SNR 档位决定）。
目标 = 暗高斯斑（配合预处理 D = close(G,B) − G 的暗目标语义）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from spa2d import records as R

GENERATOR_VERSION = "0.1.0"
SNR_LEVELS = ("low", "mid", "high")
WIDTHS = (2, 4, 8, 16)

# SNR 档位：noise_sigma 与目标对比度（0..255 强度单位）
SNR_TABLE = {
    "low": {"noise_sigma": 8.0, "contrast": 10.0},
    "mid": {"noise_sigma": 4.0, "contrast": 20.0},
    "high": {"noise_sigma": 2.0, "contrast": 34.0},
}

_SKY_TOP = 210.0
_SKY_BOTTOM = 150.0

_D12_SPEEDS = [
    (0.0, 0.0), (0.5, -0.5), (-1.0, 1.0), (2.0, -1.5), (3.5, 2.5),
    (-4.0, 4.5), (5.0, -5.0), (1.5, 0.5), (-2.5, -3.5), (4.5, 1.5),
    (-0.5, 2.0), (2.5, 3.0), (-3.0, -1.0), (3.0, 4.0), (-5.0, 2.5),
]
_D34_SPEEDS = [
    (0.0, 0.0), (0.5, -0.5), (-1.5, 1.5), (6.0, -4.0), (9.5, 7.5),
    (-12.0, 13.0), (15.0, -15.0), (3.5, -2.5), (-7.5, -8.5), (11.0, 4.5),
    (-0.5, 10.0), (7.5, 9.0), (-9.0, -3.0), (13.0, 12.0), (-15.0, 5.5),
]
_ACCELS = (0.0, 0.01, -0.015, 0.005)
_NEG_PATTERNS = [(), ("cloud",), ("bird",), ("insect",), ("bird", "cloud"), ("insect", "cloud")]


class SyntheticError(Exception):
    def __init__(self, message: str, exit_code: int = 2) -> None:
        self.message = message
        self.exit_code = exit_code
        super().__init__(message)


@dataclass(frozen=True)
class SequenceSpec:
    seq_id: str
    width_px: int
    snr: str
    vx: float
    vy: float
    ax: float
    ay: float
    n_frames: int
    height: int
    width: int
    seed: int
    p0: tuple[float, float]
    visible_range: tuple[int, int]
    negatives: tuple[str, ...] = ()
    fps: float = 25.0
    kind: str = "sanity"
    static_bg: bool = False
    clouds: bool = True


def _sky_gradient(h: int, w: int) -> np.ndarray:
    col = np.linspace(_SKY_TOP, _SKY_BOTTOM, h, dtype=np.float32)[:, None]
    return np.repeat(col, w, axis=1)


def _cloud_field(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    ch = max(2, h // 16)
    cw = max(2, w // 16)
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    field = np.zeros((ch, cw), np.float32)
    for k in range(4):
        fx = rng.uniform(0.5, 2.5) * np.pi / cw
        fy = rng.uniform(0.5, 2.5) * np.pi / ch
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.3, 1.0) / (k + 1)
        field += amp * np.sin(fx * xx + fy * yy + phase)
    field -= field.min()
    rng_ptp = max(float(field.max()), 1e-6)
    field /= rng_ptp
    cloud = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC) * 45.0
    return cloud.astype(np.float32)


class BackgroundModel:
    def __init__(self, seed: int, h: int, w: int, snr: str, static: bool = False,
                 clouds: bool = True) -> None:
        if snr not in SNR_TABLE:
            raise SyntheticError(f"未知 SNR 档位: {snr}")
        self.h = h
        self.w = w
        self.rng = np.random.default_rng([seed, 0])
        self.base_cloud = (_cloud_field(h, w, self.rng) if clouds
                           else np.zeros((h, w), np.float32))
        self.drift = np.zeros(2, np.float32) if static else \
            self.rng.uniform(-0.6, 0.6, size=2).astype(np.float32)
        self.sky = _sky_gradient(h, w)
        self.sigma = float(SNR_TABLE[snr]["noise_sigma"])

    def frame(self, f: int, rng: np.random.Generator) -> np.ndarray:
        if abs(self.drift[0]) < 1e-9 and abs(self.drift[1]) < 1e-9:
            cloud = self.base_cloud
        else:
            m = np.array([[1.0, 0.0, float(self.drift[0] * f)],
                          [0.0, 1.0, float(self.drift[1] * f)]], np.float32)
            cloud = cv2.warpAffine(self.base_cloud, m, (self.w, self.h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        noise = rng.standard_normal((self.h, self.w), dtype=np.float32) * self.sigma
        return self.sky + cloud + noise


def _draw_dark_blob(img: np.ndarray, cx: float, cy: float, sigma: float, amplitude: float) -> None:
    h, w = img.shape
    radius = int(np.ceil(3.0 * sigma))
    x0, x1 = max(0, int(np.floor(cx)) - radius), min(w, int(np.ceil(cx)) + radius + 1)
    y0, y1 = max(0, int(np.floor(cy)) - radius), min(h, int(np.ceil(cy)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    g = np.exp(-(((gx - cx) ** 2 + (gy - cy) ** 2) / (2.0 * sigma * sigma)))
    img[y0:y1, x0:x1] -= amplitude * g


class SequenceRenderer:
    """给定 SequenceSpec 的纯函数式渲染器：render(f, with_target) 确定性。"""

    def __init__(self, spec: SequenceSpec) -> None:
        self.spec = spec
        self.bg = BackgroundModel(spec.seed, spec.height, spec.width, spec.snr,
                                  spec.static_bg, spec.clouds)
        self.contrast = float(SNR_TABLE[spec.snr]["contrast"])
        self.neg_params = self._build_negatives()

    def _build_negatives(self) -> list[dict]:
        rng = np.random.default_rng([self.spec.seed, 7])
        n = self.spec.n_frames
        out: list[dict] = []
        if "cloud" in self.spec.negatives:
            out.append({
                "type": "cloud",
                "sigma": round(0.18 * min(self.spec.height, self.spec.width), 3),
                "amplitude": 14.0,
                "vx": round(float(rng.uniform(-0.3, 0.3)), 4),
                "vy": round(float(rng.uniform(-0.3, 0.3)), 4),
                "center": [round(self.spec.width * float(rng.uniform(0.3, 0.7)), 2),
                           round(self.spec.height * float(rng.uniform(0.3, 0.7)), 2)],
                "frame_range": [0, n],
            })
        if "bird" in self.spec.negatives:
            out.append({
                "type": "bird",
                "sigma": round(float(rng.uniform(6.0, 10.0)), 3),
                "amplitude": round(self.contrast * 1.2, 3),
                "vx": round(float(rng.uniform(-3.0, 3.0)), 4),
                "vy": round(float(rng.uniform(-3.0, 3.0)), 4),
                "center": [round(self.spec.width * float(rng.uniform(0.2, 0.8)), 2),
                           round(self.spec.height * float(rng.uniform(0.2, 0.8)), 2)],
                "frame_range": [n // 6, 5 * n // 6],
            })
        if "insect" in self.spec.negatives:
            out.append({
                "type": "insect",
                "sigma": 1.25,
                "amplitude": round(self.contrast * 1.1, 3),
                "vx": round(float(rng.uniform(-15.0, 15.0)), 4),
                "vy": round(float(rng.uniform(-15.0, 15.0)), 4),
                "center": [round(self.spec.width * float(rng.uniform(0.2, 0.8)), 2),
                           round(self.spec.height * float(rng.uniform(0.2, 0.8)), 2)],
                "frame_range": [n // 3, 2 * n // 3],
            })
        return out

    def target_center(self, f: float) -> tuple[float, float]:
        s = self.spec
        x = s.p0[0] + s.vx * f + 0.5 * s.ax * f * f
        y = s.p0[1] + s.vy * f + 0.5 * s.ay * f * f
        return float(x), float(y)

    def _target_in_range(self, f: int) -> bool:
        s, e = self.spec.visible_range
        return s <= f < e

    def render(self, f: int, with_target: bool = True) -> np.ndarray:
        rng = np.random.default_rng([self.spec.seed, f])
        img = self.bg.frame(f, rng)
        for neg in self.neg_params:
            lo, hi = neg["frame_range"]
            if lo <= f < hi:
                cx = neg["center"][0] + neg["vx"] * f
                cy = neg["center"][1] + neg["vy"] * f
                _draw_dark_blob(img, cx, cy, neg["sigma"], neg["amplitude"])
        if with_target and self._target_in_range(f):
            cx, cy = self.target_center(f)
            _draw_dark_blob(img, cx, cy, self.spec.width_px / 2.0, self.contrast)
        return np.clip(np.rint(img), 0.0, 255.0).astype(np.uint8)

    def gt_row(self, f: int) -> R.GTFrame:
        s = self.spec
        cx, cy = self.target_center(f)
        in_range = self._target_in_range(f)
        in_frame = (0.0 <= cx < s.width) and (0.0 <= cy < s.height)
        w = float(s.width_px)
        if in_range and in_frame:
            return R.GTFrame(
                frame=f, timestamp_s=f / s.fps, visible=1, out_of_frame=0,
                cx=round(cx, 4), cy=round(cy, 4),
                bbox_x=round(cx - w / 2, 4), bbox_y=round(cy - w / 2, 4),
                bbox_w=w, bbox_h=w, area_px=w * w, width_px=w,
                width_source="bbox", gt_source="synthetic")
        if in_range:
            return R.GTFrame(
                frame=f, timestamp_s=f / s.fps, visible=0, out_of_frame=1,
                cx=round(cx, 4), cy=round(cy, 4), width_px=w,
                width_source="bbox", gt_source="synthetic")
        return R.GTFrame(
            frame=f, timestamp_s=f / s.fps, visible=0, out_of_frame=0,
            gt_source="synthetic")


def build_sanity_specs(n_seqs: int = 30, seed: int = 0, n_frames: int = 120,
                       canvas: tuple[int, int] | None = None) -> list[SequenceSpec]:
    half = (n_seqs + 1) // 2
    specs: list[SequenceSpec] = []
    for i in range(n_seqs):
        w = WIDTHS[i % len(WIDTHS)]
        snr = SNR_LEVELS[i % len(SNR_LEVELS)]
        if i < half:
            vx, vy = _D12_SPEEDS[i % len(_D12_SPEEDS)]
            res = (1080, 1920)
        else:
            vx, vy = _D34_SPEEDS[(i - half) % len(_D34_SPEEDS)]
            res = (1080, 1440)
        if i == n_seqs - 1:
            res = (2160, 3840)
        if canvas is not None:
            res = canvas
        h, wd = res
        ax = _ACCELS[i % len(_ACCELS)]
        ay = -_ACCELS[(i + 1) % len(_ACCELS)] * 0.5
        p0 = (wd / 2.0 - vx * n_frames / 2.0, h / 2.0 - vy * n_frames / 2.0)
        negs = _NEG_PATTERNS[i % len(_NEG_PATTERNS)]
        if negs:
            vis = (n_frames // 4, 3 * n_frames // 4)
        else:
            vis = (0, n_frames)
        specs.append(SequenceSpec(
            seq_id=f"seq_{i:03d}", width_px=w, snr=snr, vx=float(vx), vy=float(vy),
            ax=float(ax), ay=float(ay), n_frames=n_frames, height=h, width=wd,
            seed=seed + 1000 * i + 1, p0=p0, visible_range=vis, negatives=negs,
            fps=25.0, kind="sanity"))
    return specs


def build_benchmark_spec(seed: int = 0, n_frames: int = 200,
                         canvas: tuple[int, int] = (480, 640)) -> SequenceSpec:
    h, w = canvas
    return SequenceSpec(
        seq_id="benchmark", width_px=4, snr="high", vx=0.7, vy=0.0, ax=0.0, ay=0.0,
        n_frames=n_frames, height=h, width=w, seed=seed + 99991,
        p0=(w / 2.0, h / 2.0), visible_range=(0, n_frames), negatives=(),
        fps=50.0, kind="benchmark", static_bg=True, clouds=False)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def render_sequence(spec: SequenceSpec, seq_dir: Path) -> dict:
    seq_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = seq_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    renderer = SequenceRenderer(spec)

    gt_rows: list[R.GTFrame] = []
    for f in range(spec.n_frames):
        frame = renderer.render(f, with_target=True)
        cv2.imwrite(str(frames_dir / f"{f:06d}.png"), frame)
        gt_rows.append(renderer.gt_row(f))
    R.write_gt_csv(gt_rows, seq_dir / "gt.csv")

    frame_paths = sorted(frames_dir.glob("*.png"))
    frames_hash = hashlib.sha256()
    for p in frame_paths:
        frames_hash.update(p.name.encode("utf-8"))
        frames_hash.update(p.read_bytes())
    meta = {
        "seq_id": spec.seq_id,
        "kind": spec.kind,
        "width_px": spec.width_px,
        "snr": spec.snr,
        "contrast": renderer.contrast,
        "noise_sigma": renderer.bg.sigma,
        "velocity": {"vx": spec.vx, "vy": spec.vy, "ax": spec.ax, "ay": spec.ay},
        "p0": [spec.p0[0], spec.p0[1]],
        "n_frames": spec.n_frames,
        "resolution": [spec.width, spec.height],
        "fps": spec.fps,
        "seed": spec.seed,
        "static_bg": spec.static_bg,
        "clouds": spec.clouds,
        "target_visible_range": [spec.visible_range[0], spec.visible_range[1]],
        "negatives": renderer.neg_params,
        "frames_sha256": frames_hash.hexdigest(),
        "gt_sha256": _hash_file(seq_dir / "gt.csv"),
    }
    (seq_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    return {
        "seq_id": spec.seq_id,
        "kind": spec.kind,
        "dir": seq_dir.name,
        "width_px": spec.width_px,
        "snr": spec.snr,
        "velocity": meta["velocity"],
        "p0": meta["p0"],
        "n_frames": spec.n_frames,
        "resolution": meta["resolution"],
        "fps": spec.fps,
        "seed": spec.seed,
        "target_visible_range": meta["target_visible_range"],
        "negatives": [n["type"] for n in renderer.neg_params],
        "static_bg": spec.static_bg,
        "frames_sha256": meta["frames_sha256"],
        "gt_sha256": meta["gt_sha256"],
    }


def _load_manifest(out_dir: Path) -> dict | None:
    p = out_dir / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(out_dir: Path, suite: str, seed: int, sequences: list[dict],
                   extra: dict | None = None) -> dict:
    existing = _load_manifest(out_dir) or {}
    merged = {s["seq_id"]: s for s in existing.get("sequences", [])}
    for s in sequences:
        merged[s["seq_id"]] = s
    prior_suites = existing.get("suites")
    if prior_suites is None:
        prior_suites = [existing["suite"]] if existing.get("suite") else []
    suites = sorted(set(prior_suites) | {suite})
    merged_extra = dict(existing.get("suite_meta", {}))
    if extra:
        merged_extra[suite] = extra
    manifest = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "suites": suites,
        "suite": suite,
        "seed": seed,
        "n_sequences": len(merged),
        "sequences": [merged[k] for k in sorted(merged)],
        "suite_meta": merged_extra,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return manifest


def _guard_entries(out_dir: Path, new_ids: Iterable[str], overwrite: bool) -> None:
    if overwrite:
        return
    existing = _load_manifest(out_dir)
    if not existing:
        return
    have = {s["seq_id"] for s in existing.get("sequences", [])}
    clash = sorted(set(new_ids) & have)
    if clash:
        raise SyntheticError(
            f"序列已存在: {clash}（使用 --overwrite 覆盖）")


def generate_sanity_set(out_dir: str | Path, n_seqs: int = 30, seed: int = 0,
                        n_frames: int = 120, canvas: tuple[int, int] | None = None,
                        overwrite: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = build_sanity_specs(n_seqs=n_seqs, seed=seed, n_frames=n_frames, canvas=canvas)
    _guard_entries(out, (s.seq_id for s in specs), overwrite)
    entries = [render_sequence(spec, out / spec.seq_id) for spec in specs]
    return write_manifest(out, "sanity", seed, entries,
                          extra={"default_n_frames": n_frames})


def generate_benchmark(out_dir: str | Path, seed: int = 0, n_frames: int = 200,
                       canvas: tuple[int, int] = (480, 640),
                       overwrite: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = build_benchmark_spec(seed=seed, n_frames=n_frames, canvas=canvas)
    _guard_entries(out, [spec.seq_id], overwrite)
    entry = render_sequence(spec, out / "benchmark")

    grid_path = out / "benchmark" / "grid_truth.csv"
    vxs = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.1), 4)
    vys = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.1), 4)
    with grid_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("vx,vy,is_true\n")
        for vy in vys:
            for vx in vxs:
                is_true = int(abs(vx - spec.vx) <= 1e-9 and abs(vy - spec.vy) <= 1e-9)
                fh.write(f"{vx:.4f},{vy:.4f},{is_true}\n")

    entry["grid_truth"] = grid_path.name
    entry["grid_size"] = int(len(vxs) * len(vys))
    return write_manifest(out, "benchmark", seed, [entry],
                          extra={"default_n_frames": n_frames,
                                 "true_velocity": {"vx": spec.vx, "vy": spec.vy}})


def generate(suite: str, out_dir: str | Path, seqs: int = 30, seed: int = 0,
             n_frames: int | None = None, canvas: tuple[int, int] | None = None,
             overwrite: bool = False) -> dict:
    if suite == "sanity":
        return generate_sanity_set(out_dir, n_seqs=seqs, seed=seed,
                                   n_frames=120 if n_frames is None else n_frames,
                                   canvas=canvas, overwrite=overwrite)
    if suite == "benchmark":
        return generate_benchmark(out_dir, seed=seed,
                                  n_frames=200 if n_frames is None else n_frames,
                                  canvas=canvas or (480, 640), overwrite=overwrite)
    raise SyntheticError(f"未知 suite: {suite!r}（可选 sanity|benchmark）")
