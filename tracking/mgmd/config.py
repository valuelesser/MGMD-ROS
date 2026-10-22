"""mgmd.config — MGMD 配置 YAML 加载与 schema 校验（SPEC §3.1 / §4.3）。

公开接口：load_config / validate。错误信息统一格式：
``[mgmd] <code>: <message> → <remediation>``。
CLI：``.venv/bin/python -m mgmd.config configs/mgmd_*.yaml``（非法 → 退出码 2）。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUITE = "mgmd"

EXIT_OK = 0
EXIT_CONFIG_INVALID = 2
EXIT_DATA = 3
EXIT_RUNTIME = 4
EXIT_FROZEN = 5  # 保留：MGMD 无冻结阈值机制，不使用（SPEC §4.3）
EXIT_INTERRUPT = 130

SCHEMA_VERSION = 1
ALLOWED_COLOR_ORDER = ("BGR", "RGB", "GRAY")
ALLOWED_ASSOCIATION = ("round_half_up", "floor")
ALLOWED_DEVICE = ("auto", "cpu", "cuda")
ALLOWED_BACKEND = ("numpy", "jax")
ALLOWED_MODULES = ("mfe", "tf", "lac", "lad")

_TOP_KEYS = ("schema_version", "config_name", "sequence", "mfe", "tf", "lac", "lad",
             "metrics", "runtime", "backend", "seed")
_SEQ_REQUIRED = ("name", "frames_dir", "frame_glob", "fps", "resolution", "color_order")
_SEQ_OPTIONAL = ("video", "calibration", "trajectory")
_MFE_KEYS = ("blur_kernel", "grid_spacing", "lk_win_size", "lk_max_level",
             "max_flow_px", "min_inliers", "ransac_thresh", "diff_thresh_base",
             "diff_adaptive", "median_blur", "open_iter", "close_iter", "morph_kernel",
             "area_min", "area_max", "ratio_min", "ratio_max", "max_boxes",
             "enlarge_px", "motion_avg_dist_min", "motion_ratio_theta_max",
             "motion_ratio_dist_max", "frame_gap", "or_aggregation", "overflow",
             "motion_coherence")
_TF_KEYS = ("iou_thresh", "max_age", "min_hits", "alg1_age", "alg1_window", "alg1_d_min")
_LAC_KEYS = ("weights", "input_size", "prob_thresh", "device")
_LAD_KEYS = ("weights", "crop_size", "conf_thresh", "imgsz", "device")
_METRICS_KEYS = ("radii_px", "primary_radius_px", "association")
_RUNTIME_KEYS = ("downscale", "limit")


class MgmdError(Exception):
    """MGMD 配置/数据/运行期错误。``exit_code`` 见 SPEC §4.3。"""

    def __init__(self, code: str, message: str, remediation: str,
                 exit_code: int = EXIT_CONFIG_INVALID) -> None:
        self.code = code
        self.message = message
        self.remediation = remediation
        self.exit_code = exit_code
        super().__init__(self.format())

    def format(self) -> str:
        return f"[{SUITE}] {self.code}: {self.message} → {self.remediation}"

    def __str__(self) -> str:
        return self.format()


ConfigError = MgmdError  # 与 spa2d.config 命名对齐


@dataclass(frozen=True)
class SequenceConfig:
    name: str
    frames_dir: str
    frame_glob: str
    fps: float
    resolution: tuple[int, int]
    color_order: str
    video: str | None = None
    calibration: str | None = None
    trajectory: str | None = None


@dataclass(frozen=True)
class MfeConfig:
    blur_kernel: int
    grid_spacing: tuple[int, int]
    lk_win_size: int
    lk_max_level: int
    max_flow_px: float
    min_inliers: int
    ransac_thresh: float
    frame_gap: int                 # 差分帧间隔 k（1=官方两帧口径；SPEC D11）
    or_aggregation: bool           # 论文口径：|It−Ît−k| OR |It−Ît+k|（SPEC D12）
    overflow: str                  # 候选超 max_boxes 时：drop（官方整帧弃检）/topk（SPEC D13）
    motion_coherence: bool         # 逐框运动一致性过滤（官方；MVDT 静态相机关闭，SPEC D14）
    diff_thresh_base: float
    diff_adaptive: bool
    median_blur: int
    open_iter: int
    close_iter: int
    morph_kernel: int
    area_min: float
    area_max: float
    ratio_min: float
    ratio_max: float
    max_boxes: int
    enlarge_px: int
    motion_avg_dist_min: float
    motion_ratio_theta_max: float
    motion_ratio_dist_max: float


@dataclass(frozen=True)
class TfConfig:
    iou_thresh: float
    max_age: int
    min_hits: int
    alg1_age: int
    alg1_window: int
    alg1_d_min: float


@dataclass(frozen=True)
class LacConfig:
    weights: str
    input_size: int
    prob_thresh: float
    device: str


@dataclass(frozen=True)
class LadConfig:
    weights: str
    crop_size: int
    conf_thresh: float
    imgsz: int
    device: str


@dataclass(frozen=True)
class MetricsConfig:
    radii_px: tuple[int, ...]
    primary_radius_px: int
    association: str


@dataclass(frozen=True)
class RuntimeConfig:
    downscale: float
    limit: int | None


@dataclass(frozen=True)
class Config:
    schema_version: int
    config_name: str
    sequence: SequenceConfig
    mfe: MfeConfig
    tf: TfConfig
    lac: LacConfig
    lad: LadConfig
    metrics: MetricsConfig
    runtime: RuntimeConfig
    backend: str
    seed: int
    source_path: str | None = None


def _reject_unknown(data: dict, allowed, where: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise MgmdError("E_CONFIG_INVALID", f"{where} 含未知字段: {unknown}",
                        "删除未知字段或更新 schema 定义（SPEC §3.1）")


def _require(data: dict, keys, where: str) -> None:
    if not isinstance(data, dict):
        raise MgmdError("E_CONFIG_INVALID", f"{where} 必须是映射（mapping）",
                        f"修正 {where} 结构")
    missing = sorted(set(keys) - set(data))
    if missing:
        raise MgmdError("E_CONFIG_INVALID", f"{where} 缺少必需字段: {missing}",
                        "补齐必需字段（SPEC §3.1）")


def _as_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MgmdError("E_CONFIG_INVALID",
                        f"{where} 必须是数值，实际为 {type(value).__name__}",
                        f"将 {where} 改为数值")
    return float(value)


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MgmdError("E_CONFIG_INVALID",
                        f"{where} 必须是整数，实际为 {type(value).__name__}",
                        f"将 {where} 改为整数")
    return value


def _as_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise MgmdError("E_CONFIG_INVALID",
                        f"{where} 必须是布尔，实际为 {type(value).__name__}",
                        f"将 {where} 改为 true/false")
    return value


def _build_config(doc: dict, source_path: str | None) -> Config:
    if not isinstance(doc, dict):
        raise MgmdError("E_CONFIG_INVALID", "顶层必须是映射（mapping）",
                        "修正 YAML 顶层结构")
    _reject_unknown(doc, _TOP_KEYS, "<root>")
    _require(doc, _TOP_KEYS, "<root>")

    seq = doc["sequence"]
    _reject_unknown(seq, _SEQ_REQUIRED + _SEQ_OPTIONAL, "sequence")
    _require(seq, _SEQ_REQUIRED, "sequence")
    resolution = seq["resolution"]
    if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in resolution)):
        raise MgmdError("E_CONFIG_INVALID", "sequence.resolution 必须是 [W, H] 两个整数",
                        "修正 resolution，例如 [1920, 1080]")
    sequence = SequenceConfig(
        name=str(seq["name"]),
        frames_dir=str(seq["frames_dir"]),
        frame_glob=str(seq["frame_glob"]),
        fps=_as_number(seq["fps"], "sequence.fps"),
        resolution=(int(resolution[0]), int(resolution[1])),
        color_order=str(seq["color_order"]),
        video=None if seq.get("video") is None else str(seq["video"]),
        calibration=None if seq.get("calibration") is None else str(seq["calibration"]),
        trajectory=None if seq.get("trajectory") is None else str(seq["trajectory"]),
    )

    m = doc["mfe"]
    _reject_unknown(m, _MFE_KEYS, "mfe")
    _require(m, _MFE_KEYS, "mfe")
    gs = m["grid_spacing"]
    if (not isinstance(gs, (list, tuple)) or len(gs) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in gs)):
        raise MgmdError("E_CONFIG_INVALID", "mfe.grid_spacing 必须是 [W, H] 两个整数",
                        "例如 [96, 72]（SPEC §5.1 官方间距）")
    mfe = MfeConfig(
        blur_kernel=_as_int(m["blur_kernel"], "mfe.blur_kernel"),
        grid_spacing=(int(gs[0]), int(gs[1])),
        lk_win_size=_as_int(m["lk_win_size"], "mfe.lk_win_size"),
        lk_max_level=_as_int(m["lk_max_level"], "mfe.lk_max_level"),
        max_flow_px=_as_number(m["max_flow_px"], "mfe.max_flow_px"),
        min_inliers=_as_int(m["min_inliers"], "mfe.min_inliers"),
        ransac_thresh=_as_number(m["ransac_thresh"], "mfe.ransac_thresh"),
        frame_gap=_as_int(m["frame_gap"], "mfe.frame_gap"),
        or_aggregation=_as_bool(m["or_aggregation"], "mfe.or_aggregation"),
        overflow=str(m["overflow"]),
        motion_coherence=_as_bool(m["motion_coherence"], "mfe.motion_coherence"),
        diff_thresh_base=_as_number(m["diff_thresh_base"], "mfe.diff_thresh_base"),
        diff_adaptive=_as_bool(m["diff_adaptive"], "mfe.diff_adaptive"),
        median_blur=_as_int(m["median_blur"], "mfe.median_blur"),
        open_iter=_as_int(m["open_iter"], "mfe.open_iter"),
        close_iter=_as_int(m["close_iter"], "mfe.close_iter"),
        morph_kernel=_as_int(m["morph_kernel"], "mfe.morph_kernel"),
        area_min=_as_number(m["area_min"], "mfe.area_min"),
        area_max=_as_number(m["area_max"], "mfe.area_max"),
        ratio_min=_as_number(m["ratio_min"], "mfe.ratio_min"),
        ratio_max=_as_number(m["ratio_max"], "mfe.ratio_max"),
        max_boxes=_as_int(m["max_boxes"], "mfe.max_boxes"),
        enlarge_px=_as_int(m["enlarge_px"], "mfe.enlarge_px"),
        motion_avg_dist_min=_as_number(m["motion_avg_dist_min"], "mfe.motion_avg_dist_min"),
        motion_ratio_theta_max=_as_number(m["motion_ratio_theta_max"], "mfe.motion_ratio_theta_max"),
        motion_ratio_dist_max=_as_number(m["motion_ratio_dist_max"], "mfe.motion_ratio_dist_max"),
    )

    t = doc["tf"]
    _reject_unknown(t, _TF_KEYS, "tf")
    _require(t, _TF_KEYS, "tf")
    tf = TfConfig(
        iou_thresh=_as_number(t["iou_thresh"], "tf.iou_thresh"),
        max_age=_as_int(t["max_age"], "tf.max_age"),
        min_hits=_as_int(t["min_hits"], "tf.min_hits"),
        alg1_age=_as_int(t["alg1_age"], "tf.alg1_age"),
        alg1_window=_as_int(t["alg1_window"], "tf.alg1_window"),
        alg1_d_min=_as_number(t["alg1_d_min"], "tf.alg1_d_min"),
    )

    lc = doc["lac"]
    _reject_unknown(lc, _LAC_KEYS, "lac")
    _require(lc, _LAC_KEYS, "lac")
    lac = LacConfig(
        weights=str(lc["weights"]),
        input_size=_as_int(lc["input_size"], "lac.input_size"),
        prob_thresh=_as_number(lc["prob_thresh"], "lac.prob_thresh"),
        device=str(lc["device"]),
    )

    ld = doc["lad"]
    _reject_unknown(ld, _LAD_KEYS, "lad")
    _require(ld, _LAD_KEYS, "lad")
    lad = LadConfig(
        weights=str(ld["weights"]),
        crop_size=_as_int(ld["crop_size"], "lad.crop_size"),
        conf_thresh=_as_number(ld["conf_thresh"], "lad.conf_thresh"),
        imgsz=_as_int(ld["imgsz"], "lad.imgsz"),
        device=str(ld["device"]),
    )

    met = doc["metrics"]
    _reject_unknown(met, _METRICS_KEYS, "metrics")
    _require(met, _METRICS_KEYS, "metrics")
    radii = met["radii_px"]
    if not isinstance(radii, (list, tuple)) or not radii:
        raise MgmdError("E_CONFIG_INVALID", "metrics.radii_px 必须是非空列表",
                        "例如 [10, 20, 50, 100]")
    metrics = MetricsConfig(
        radii_px=tuple(int(r) for r in radii),
        primary_radius_px=_as_int(met["primary_radius_px"], "metrics.primary_radius_px"),
        association=str(met["association"]),
    )

    rt = doc["runtime"]
    _reject_unknown(rt, _RUNTIME_KEYS, "runtime")
    _require(rt, ("downscale",), "runtime")
    limit = rt.get("limit")
    if limit is not None:
        limit = _as_int(limit, "runtime.limit")
    runtime = RuntimeConfig(
        downscale=_as_number(rt["downscale"], "runtime.downscale"),
        limit=limit,
    )

    return Config(
        schema_version=_as_int(doc["schema_version"], "schema_version"),
        config_name=str(doc["config_name"]),
        sequence=sequence,
        mfe=mfe,
        tf=tf,
        lac=lac,
        lad=lad,
        metrics=metrics,
        runtime=runtime,
        backend=str(doc["backend"]),
        seed=_as_int(doc["seed"], "seed"),
        source_path=source_path,
    )


def validate(cfg: Config) -> Config:
    """校验 SPEC §3.1 全部规则；收集所有违规项后抛出单个 MgmdError（退出码 2）。"""
    errs: list[str] = []

    if cfg.schema_version != SCHEMA_VERSION:
        errs.append(f"schema_version 必须为 {SCHEMA_VERSION}，实际 {cfg.schema_version}")
    if not cfg.config_name:
        errs.append("config_name 不能为空")

    s = cfg.sequence
    if s.fps <= 0:
        errs.append(f"sequence.fps 必须为正，实际 {s.fps}")
    if s.color_order not in ALLOWED_COLOR_ORDER:
        errs.append(f"sequence.color_order 必须是 {ALLOWED_COLOR_ORDER} 之一，实际 {s.color_order!r}")
    if not (0 < s.resolution[0] and 0 < s.resolution[1]):
        errs.append(f"sequence.resolution 必须为正，实际 {s.resolution}")

    m = cfg.mfe
    if m.blur_kernel < 1 or m.blur_kernel % 2 == 0:
        errs.append(f"mfe.blur_kernel 必须是正奇数，实际 {m.blur_kernel}")
    if m.grid_spacing[0] < 1 or m.grid_spacing[1] < 1:
        errs.append(f"mfe.grid_spacing 必须为正，实际 {m.grid_spacing}")
    if m.lk_win_size < 1:
        errs.append(f"mfe.lk_win_size 必须为正，实际 {m.lk_win_size}")
    if m.lk_max_level < 0:
        errs.append(f"mfe.lk_max_level 必须 ≥ 0，实际 {m.lk_max_level}")
    if m.max_flow_px <= 0:
        errs.append(f"mfe.max_flow_px 必须为正，实际 {m.max_flow_px}")
    if m.min_inliers < 4:
        errs.append(f"mfe.min_inliers 必须 ≥ 4（单应最少点数），实际 {m.min_inliers}")
    if m.ransac_thresh <= 0:
        errs.append(f"mfe.ransac_thresh 必须为正，实际 {m.ransac_thresh}")
    if m.frame_gap < 1:
        errs.append(f"mfe.frame_gap 必须 ≥ 1，实际 {m.frame_gap}")
    if m.overflow not in ("drop", "topk"):
        errs.append(f"mfe.overflow 必须是 drop/topk 之一，实际 {m.overflow!r}")
    if m.diff_thresh_base <= 0:
        errs.append(f"mfe.diff_thresh_base 必须为正，实际 {m.diff_thresh_base}")
    if m.median_blur < 1 or m.median_blur % 2 == 0:
        errs.append(f"mfe.median_blur 必须是正奇数，实际 {m.median_blur}")
    if m.morph_kernel < 1 or m.morph_kernel % 2 == 0:
        errs.append(f"mfe.morph_kernel 必须是正奇数，实际 {m.morph_kernel}")
    if m.open_iter < 0 or m.close_iter < 0:
        errs.append(f"mfe.open_iter/close_iter 必须 ≥ 0，实际 {m.open_iter}/{m.close_iter}")
    if not (0 <= m.area_min < m.area_max):
        errs.append(f"mfe.area_min<area_max 不成立：{m.area_min}/{m.area_max}")
    if not (0 < m.ratio_min < m.ratio_max):
        errs.append(f"mfe 要求 0<ratio_min<ratio_max，实际 {m.ratio_min}/{m.ratio_max}")
    if m.max_boxes < 1:
        errs.append(f"mfe.max_boxes 必须 ≥ 1，实际 {m.max_boxes}")
    if m.enlarge_px < 0:
        errs.append(f"mfe.enlarge_px 必须 ≥ 0，实际 {m.enlarge_px}")
    if m.motion_avg_dist_min <= 0:
        errs.append(f"mfe.motion_avg_dist_min 必须为正，实际 {m.motion_avg_dist_min}")
    if m.motion_ratio_theta_max <= 0 or m.motion_ratio_dist_max <= 0:
        errs.append("mfe.motion_ratio_{theta,dist}_max 必须为正，"
                    f"实际 {m.motion_ratio_theta_max}/{m.motion_ratio_dist_max}")

    t = cfg.tf
    if not (0 < t.iou_thresh < 1):
        errs.append(f"tf.iou_thresh 必须在 (0,1)，实际 {t.iou_thresh}")
    if t.max_age < 1:
        errs.append(f"tf.max_age 必须 ≥ 1，实际 {t.max_age}")
    if t.min_hits < 1:
        errs.append(f"tf.min_hits 必须 ≥ 1，实际 {t.min_hits}")
    if t.alg1_age < 2:
        errs.append(f"tf.alg1_age 必须 ≥ 2，实际 {t.alg1_age}")
    if t.alg1_window < 1:
        errs.append(f"tf.alg1_window 必须 ≥ 1，实际 {t.alg1_window}")
    if t.alg1_d_min <= 0:
        errs.append(f"tf.alg1_d_min 必须为正，实际 {t.alg1_d_min}")

    if cfg.lac.input_size < 8:
        errs.append(f"lac.input_size 必须 ≥ 8，实际 {cfg.lac.input_size}")
    if not (0 <= cfg.lac.prob_thresh <= 1):
        errs.append(f"lac.prob_thresh 必须在 [0,1]，实际 {cfg.lac.prob_thresh}")
    if cfg.lac.device not in ALLOWED_DEVICE:
        errs.append(f"lac.device 必须是 {ALLOWED_DEVICE} 之一，实际 {cfg.lac.device!r}")

    if cfg.lad.crop_size < 64 or cfg.lad.crop_size % 32 != 0:
        errs.append(f"lad.crop_size 必须 ≥64 且为 32 倍数，实际 {cfg.lad.crop_size}")
    if cfg.lad.imgsz < 32 or cfg.lad.imgsz % 32 != 0:
        errs.append(f"lad.imgsz 必须 ≥32 且为 32 倍数，实际 {cfg.lad.imgsz}")
    if not (0 <= cfg.lad.conf_thresh <= 1):
        errs.append(f"lad.conf_thresh 必须在 [0,1]，实际 {cfg.lad.conf_thresh}")
    if cfg.lad.device not in ALLOWED_DEVICE:
        errs.append(f"lad.device 必须是 {ALLOWED_DEVICE} 之一，实际 {cfg.lad.device!r}")

    if cfg.metrics.primary_radius_px not in cfg.metrics.radii_px:
        errs.append(f"metrics.primary_radius_px={cfg.metrics.primary_radius_px} "
                    f"不在 radii_px={cfg.metrics.radii_px} 中")
    if cfg.metrics.association not in ALLOWED_ASSOCIATION:
        errs.append(f"metrics.association 必须是 {ALLOWED_ASSOCIATION} 之一，"
                    f"实际 {cfg.metrics.association!r}")

    if not (0 < cfg.runtime.downscale <= 1):
        errs.append(f"runtime.downscale 必须在 (0,1]，实际 {cfg.runtime.downscale}")
    if cfg.runtime.limit is not None and cfg.runtime.limit < 1:
        errs.append(f"runtime.limit 必须 ≥ 1 或 null，实际 {cfg.runtime.limit}")

    if cfg.backend not in ALLOWED_BACKEND:
        errs.append(f"backend 必须是 {ALLOWED_BACKEND} 之一，实际 {cfg.backend!r}")

    if errs:
        raise MgmdError("E_CONFIG_INVALID",
                        f"{cfg.config_name} 校验失败（{len(errs)} 项）: " + "; ".join(errs),
                        "修正 configs/mgmd_*.yaml 后重试")
    return cfg


def load_config(path: str | Path) -> Config:
    """加载并校验单个 config YAML，返回冻结的 Config。"""
    p = Path(path)
    if not p.exists():
        raise MgmdError("E_CONFIG_INVALID", f"配置文件不存在: {p}",
                        "检查路径，或在 configs/ 下创建对应 YAML")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MgmdError("E_CONFIG_INVALID", f"YAML 解析失败: {p}: {exc}",
                        "修正 YAML 语法") from exc
    cfg = _build_config(doc, source_path=str(p))
    return validate(cfg)


def _main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="mgmd.config", description="校验 mgmd config YAML（SPEC §3.1），非法时退出码 2")
    parser.add_argument("paths", nargs="+", help="一个或多个 config YAML 路径")
    args = parser.parse_args(argv)
    failed = False
    for path in args.paths:
        try:
            cfg = load_config(path)
            print(f"OK {path}: {cfg.config_name} (downscale={cfg.runtime.downscale}, "
                  f"crop={cfg.lad.crop_size})")
        except MgmdError as exc:
            failed = True
            print(exc.format(), file=sys.stderr)
    return EXIT_CONFIG_INVALID if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
