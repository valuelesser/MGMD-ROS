"""spa2d.config — config YAML 加载、校验与冻结阈值（SPEC §3.1 / §3.5 / §4.2/§4.3）。

公开接口：load_config / validate / load_frozen / effective_thresholds。
错误信息统一格式：``[spa2d] <code>: <message> → <remediation>``。

校验规则说明（SPEC §3.1 的一个修正）：
- 粗网格必须关于 0 对称且命中 ±vmax，因此要求 ``2*vmax`` 能被 ``delta_c`` 整除
  （SPEC 原文写作 ``vmax % delta_c == 0``，但 D3/D4 的 (vmax=15, delta_c=2.0)
  并不满足该式，而 2*15/2=15 为整数，与实际粗网格 16 点/轴一致）。
- fine 邻域要求 ``rf % delta_f == 0``（六配置均满足）。
- local 邻域使用 ``floor(rl / delta_l)`` 计算步数（SPEC §5.3），要求步数 ≥ 1；
  D3 的 (rl=0.6, delta_l=0.25) 步数为 2，不要求整除。
"""
from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from spa2d.confidence import cont_radii, rho_for

SUITE = "spa2d"

EXIT_OK = 0
EXIT_CONFIG_INVALID = 2
EXIT_DATA = 3
EXIT_RUNTIME = 4
EXIT_FROZEN = 5
EXIT_INTERRUPT = 130

SCHEMA_VERSION = 1
ALLOWED_KERNEL_SIZES = (3, 5, 7, 9, 11)
ALLOWED_BACKENDS = ("jax", "numpy")
ALLOWED_KERNEL_SHAPES = ("ellipse", "rect", "cross")
ALLOWED_NORMALIZE = ("divide_255",)
ALLOWED_ASSOCIATION = ("round_half_up", "floor")
ALLOWED_SEARCH_MODES = ("coarse_to_fine",)

_TOP_KEYS = (
    "schema_version", "config_name", "sequence", "window", "preprocess",
    "search", "thresholds_file", "cache", "metrics", "timing", "backend", "seed",
)
_SEQ_REQUIRED = ("name", "frames_dir", "frame_glob", "fps", "resolution", "color_order")
_SEQ_OPTIONAL = ("calibration", "trajectory", "video")
_WINDOW_KEYS = ("N", "stride")
_PRE_KEYS = ("kernel_shape", "kernel_size", "normalize")
_SEARCH_KEYS = ("mode", "global", "fine", "local", "top_k", "batch_logical", "gpu_chunk_bytes")
_GLOBAL_KEYS = ("vmax", "delta_c")
_FINE_KEYS = ("rf", "delta_f")
_LOCAL_KEYS = ("rl", "delta_l")
_CACHE_KEYS = ("host_max_bytes",)
_METRICS_KEYS = ("radii_px", "primary_radius_px", "association")
_TIMING_KEYS = ("warmup_windows", "repeats")


class ConfigError(Exception):
    """配置或冻结阈值校验失败。``exit_code`` 见 SPEC §4.3。"""

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


@dataclass(frozen=True)
class SequenceConfig:
    name: str
    frames_dir: str
    frame_glob: str
    fps: float
    resolution: tuple[int, int]
    color_order: str
    calibration: str | None = None
    trajectory: str | None = None
    video: str | None = None


@dataclass(frozen=True)
class WindowConfig:
    N: int
    stride: int


@dataclass(frozen=True)
class PreprocessConfig:
    kernel_shape: str
    kernel_size: int
    normalize: str


@dataclass(frozen=True)
class GlobalSearch:
    vmax: float
    delta_c: float


@dataclass(frozen=True)
class FineSearch:
    rf: float
    delta_f: float


@dataclass(frozen=True)
class LocalSearch:
    rl: float
    delta_l: float


@dataclass(frozen=True)
class SearchConfig:
    mode: str
    coarse: GlobalSearch
    fine: FineSearch
    local: LocalSearch
    top_k: int
    batch_logical: int
    gpu_chunk_bytes: int


@dataclass(frozen=True)
class CacheConfig:
    host_max_bytes: int


@dataclass(frozen=True)
class MetricsConfig:
    radii_px: tuple[int, ...]
    primary_radius_px: int
    association: str


@dataclass(frozen=True)
class TimingConfig:
    warmup_windows: int
    repeats: int


@dataclass(frozen=True)
class Config:
    schema_version: int
    config_name: str
    sequence: SequenceConfig
    window: WindowConfig
    preprocess: PreprocessConfig
    search: SearchConfig
    thresholds_file: str
    cache: CacheConfig
    metrics: MetricsConfig
    timing: TimingConfig
    backend: str
    seed: int
    source_path: str | None = None


@dataclass(frozen=True)
class FrozenThresholds:
    schema_version: int
    sha256: str
    frozen_at: str
    kernel_size: int | None
    theta_high_pnr: float
    theta_low_pnr: float
    theta_margin: float
    theta_edge_pnr: float
    v_stat: float
    m_lost: int
    reacq_max: int
    gap_prior_windows: int
    width_ref: dict[str, float | None]
    matched_sigma: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class Thresholds:
    theta_high_pnr: float
    theta_low_pnr: float
    theta_margin: float
    theta_edge_pnr: float
    v_stat: float
    m_lost: int
    reacq_max: int
    gap_prior_windows: int
    rho: int
    r_cont_high: float
    r_cont_low: float
    width_ref: float
    kernel_size: int


def _reject_unknown(data: dict, allowed, where: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError("E_CONFIG_INVALID", f"{where} 含未知字段: {unknown}",
                          "删除未知字段或更新 schema 定义")


def _require(data: dict, keys, where: str) -> None:
    missing = sorted(set(keys) - set(data))
    if missing:
        raise ConfigError("E_CONFIG_INVALID", f"{where} 缺少必需字段: {missing}",
                          "补齐必需字段")


def _as_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("E_CONFIG_INVALID", f"{where} 必须是数值，实际为 {type(value).__name__}",
                          f"将 {where} 改为数值")
    return float(value)


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("E_CONFIG_INVALID", f"{where} 必须是整数，实际为 {type(value).__name__}",
                          f"将 {where} 改为整数")
    return value


def _is_integer_multiple(value: float, step: float, tol: float = 1e-9) -> bool:
    return abs(value / step - round(value / step)) <= tol


def _radius_steps(radius: float, step: float) -> int:
    return math.floor(radius / step + 1e-9)


def _build_config(doc: dict, source_path: str | None) -> Config:
    if not isinstance(doc, dict):
        raise ConfigError("E_CONFIG_INVALID", "顶层必须是映射（mapping）",
                          "修正 YAML 顶层结构")
    _reject_unknown(doc, _TOP_KEYS, "<root>")
    _require(doc, _TOP_KEYS, "<root>")

    seq = doc["sequence"]
    if not isinstance(seq, dict):
        raise ConfigError("E_CONFIG_INVALID", "sequence 必须是映射", "修正 sequence")
    _reject_unknown(seq, _SEQ_REQUIRED + _SEQ_OPTIONAL, "sequence")
    _require(seq, _SEQ_REQUIRED, "sequence")
    resolution = seq["resolution"]
    if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in resolution)):
        raise ConfigError("E_CONFIG_INVALID", "sequence.resolution 必须是 [W, H] 两个整数",
                          "修正 resolution，例如 [1920, 1080]")
    sequence = SequenceConfig(
        name=str(seq["name"]),
        frames_dir=str(seq["frames_dir"]),
        frame_glob=str(seq["frame_glob"]),
        fps=_as_number(seq["fps"], "sequence.fps"),
        resolution=(int(resolution[0]), int(resolution[1])),
        color_order=str(seq["color_order"]),
        calibration=None if seq.get("calibration") is None else str(seq["calibration"]),
        trajectory=None if seq.get("trajectory") is None else str(seq["trajectory"]),
        video=None if seq.get("video") is None else str(seq["video"]),
    )

    win = doc["window"]
    _reject_unknown(win, _WINDOW_KEYS, "window")
    _require(win, _WINDOW_KEYS, "window")
    window = WindowConfig(N=_as_int(win["N"], "window.N"),
                          stride=_as_int(win["stride"], "window.stride"))

    pre = doc["preprocess"]
    _reject_unknown(pre, _PRE_KEYS, "preprocess")
    _require(pre, _PRE_KEYS, "preprocess")
    preprocess = PreprocessConfig(
        kernel_shape=str(pre["kernel_shape"]),
        kernel_size=_as_int(pre["kernel_size"], "preprocess.kernel_size"),
        normalize=str(pre["normalize"]),
    )

    srch = doc["search"]
    _reject_unknown(srch, _SEARCH_KEYS, "search")
    _require(srch, _SEARCH_KEYS, "search")
    g = srch["global"]
    _reject_unknown(g, _GLOBAL_KEYS, "search.global")
    _require(g, _GLOBAL_KEYS, "search.global")
    f = srch["fine"]
    _reject_unknown(f, _FINE_KEYS, "search.fine")
    _require(f, _FINE_KEYS, "search.fine")
    lc = srch["local"]
    _reject_unknown(lc, _LOCAL_KEYS, "search.local")
    _require(lc, _LOCAL_KEYS, "search.local")
    search = SearchConfig(
        mode=str(srch["mode"]),
        coarse=GlobalSearch(vmax=_as_number(g["vmax"], "search.global.vmax"),
                            delta_c=_as_number(g["delta_c"], "search.global.delta_c")),
        fine=FineSearch(rf=_as_number(f["rf"], "search.fine.rf"),
                        delta_f=_as_number(f["delta_f"], "search.fine.delta_f")),
        local=LocalSearch(rl=_as_number(lc["rl"], "search.local.rl"),
                          delta_l=_as_number(lc["delta_l"], "search.local.delta_l")),
        top_k=_as_int(srch["top_k"], "search.top_k"),
        batch_logical=_as_int(srch["batch_logical"], "search.batch_logical"),
        gpu_chunk_bytes=_as_int(srch["gpu_chunk_bytes"], "search.gpu_chunk_bytes"),
    )

    cache_doc = doc["cache"]
    _reject_unknown(cache_doc, _CACHE_KEYS, "cache")
    _require(cache_doc, _CACHE_KEYS, "cache")
    cache = CacheConfig(host_max_bytes=_as_int(cache_doc["host_max_bytes"], "cache.host_max_bytes"))

    metrics_doc = doc["metrics"]
    _reject_unknown(metrics_doc, _METRICS_KEYS, "metrics")
    _require(metrics_doc, _METRICS_KEYS, "metrics")
    radii = metrics_doc["radii_px"]
    if not isinstance(radii, (list, tuple)) or not radii:
        raise ConfigError("E_CONFIG_INVALID", "metrics.radii_px 必须是非空列表",
                          "例如 [10, 20, 50, 100]")
    metrics = MetricsConfig(
        radii_px=tuple(int(r) for r in radii),
        primary_radius_px=_as_int(metrics_doc["primary_radius_px"], "metrics.primary_radius_px"),
        association=str(metrics_doc["association"]),
    )

    timing_doc = doc["timing"]
    _reject_unknown(timing_doc, _TIMING_KEYS, "timing")
    _require(timing_doc, _TIMING_KEYS, "timing")
    timing = TimingConfig(
        warmup_windows=_as_int(timing_doc["warmup_windows"], "timing.warmup_windows"),
        repeats=_as_int(timing_doc["repeats"], "timing.repeats"),
    )

    return Config(
        schema_version=_as_int(doc["schema_version"], "schema_version"),
        config_name=str(doc["config_name"]),
        sequence=sequence,
        window=window,
        preprocess=preprocess,
        search=search,
        thresholds_file=str(doc["thresholds_file"]),
        cache=cache,
        metrics=metrics,
        timing=timing,
        backend=str(doc["backend"]),
        seed=_as_int(doc["seed"], "seed"),
        source_path=source_path,
    )


def validate(cfg: Config) -> Config:
    """校验全部 §3.1 规则；收集所有违规项后抛出单个 ConfigError（退出码 2）。"""
    errs: list[str] = []

    if cfg.schema_version != SCHEMA_VERSION:
        errs.append(f"schema_version 必须为 {SCHEMA_VERSION}，实际 {cfg.schema_version}")
    if not cfg.config_name:
        errs.append("config_name 不能为空")

    s = cfg.search
    if s.mode not in ALLOWED_SEARCH_MODES:
        errs.append(f"search.mode 必须是 {ALLOWED_SEARCH_MODES} 之一，实际 {s.mode!r}")

    vmax, dc = s.coarse.vmax, s.coarse.delta_c
    if vmax <= 0 or dc <= 0:
        errs.append(f"search.global 必须为正：vmax={vmax}, delta_c={dc}")
    else:
        if not _is_integer_multiple(2.0 * vmax, dc):
            errs.append(f"粗网格不对称：2*vmax={2*vmax} 不能被 delta_c={dc} 整除")
        elif _radius_steps(2.0 * vmax, dc) < 2:
            errs.append(f"粗网格点过少：2*vmax/delta_c={_radius_steps(2.0*vmax, dc)} < 2")

    rf, df = s.fine.rf, s.fine.delta_f
    if rf <= 0 or df <= 0:
        errs.append(f"search.fine 必须为正：rf={rf}, delta_f={df}")
    elif not _is_integer_multiple(rf, df):
        errs.append(f"fine 邻域非整除：rf={rf} 不能被 delta_f={df} 整除")

    rl, dl = s.local.rl, s.local.delta_l
    if rl <= 0 or dl <= 0:
        errs.append(f"search.local 必须为正：rl={rl}, delta_l={dl}")
    elif _radius_steps(rl, dl) < 1:
        errs.append(f"local 邻域步数过少：floor(rl/delta_l)={_radius_steps(rl, dl)} < 1")

    if s.top_k < 1:
        errs.append(f"search.top_k 必须 ≥ 1，实际 {s.top_k}")
    if s.batch_logical < 1:
        errs.append(f"search.batch_logical 必须 ≥ 1，实际 {s.batch_logical}")
    if s.gpu_chunk_bytes < 1:
        errs.append(f"search.gpu_chunk_bytes 必须 ≥ 1，实际 {s.gpu_chunk_bytes}")

    if not (0 < cfg.window.stride <= cfg.window.N):
        errs.append(f"窗口约束要求 0 < stride <= N，实际 N={cfg.window.N}, stride={cfg.window.stride}")
    if cfg.window.N < 2:
        errs.append(f"window.N 必须 ≥ 2，实际 {cfg.window.N}")

    ks = cfg.preprocess.kernel_size
    if ks not in ALLOWED_KERNEL_SIZES:
        errs.append(f"kernel_size 必须是 {ALLOWED_KERNEL_SIZES} 之一，实际 {ks}")
    if cfg.preprocess.kernel_shape not in ALLOWED_KERNEL_SHAPES:
        errs.append(f"kernel_shape 必须是 {ALLOWED_KERNEL_SHAPES} 之一，实际 {cfg.preprocess.kernel_shape!r}")
    if cfg.preprocess.normalize not in ALLOWED_NORMALIZE:
        errs.append(f"normalize 必须是 {ALLOWED_NORMALIZE} 之一，实际 {cfg.preprocess.normalize!r}")

    if cfg.metrics.primary_radius_px not in cfg.metrics.radii_px:
        errs.append(f"primary_radius_px={cfg.metrics.primary_radius_px} 不在 radii_px={cfg.metrics.radii_px} 中")
    if cfg.metrics.association not in ALLOWED_ASSOCIATION:
        errs.append(f"association 必须是 {ALLOWED_ASSOCIATION} 之一，实际 {cfg.metrics.association!r}")

    if cfg.backend not in ALLOWED_BACKENDS:
        errs.append(f"backend 必须是 {ALLOWED_BACKENDS} 之一，实际 {cfg.backend!r}")

    if cfg.sequence.fps <= 0:
        errs.append(f"sequence.fps 必须为正，实际 {cfg.sequence.fps}")
    if cfg.sequence.color_order not in ("BGR", "RGB", "GRAY"):
        errs.append(f"sequence.color_order 必须是 BGR/RGB/GRAY 之一，实际 {cfg.sequence.color_order!r}")
    if cfg.cache.host_max_bytes < 1:
        errs.append(f"cache.host_max_bytes 必须 ≥ 1，实际 {cfg.cache.host_max_bytes}")
    if cfg.timing.warmup_windows < 0:
        errs.append(f"timing.warmup_windows 必须 ≥ 0，实际 {cfg.timing.warmup_windows}")
    if cfg.timing.repeats < 1:
        errs.append(f"timing.repeats 必须 ≥ 1，实际 {cfg.timing.repeats}")

    if errs:
        raise ConfigError("E_CONFIG_INVALID",
                          f"{cfg.config_name} 校验失败（{len(errs)} 项）: " + "; ".join(errs),
                          "修正 configs/*.yaml 后重试")
    return cfg


def load_config(path: str | Path) -> Config:
    """加载并校验单个 config YAML，返回冻结的 Config。"""
    p = Path(path)
    if not p.exists():
        raise ConfigError("E_CONFIG_INVALID", f"配置文件不存在: {p}",
                          "检查路径，或在 configs/ 下创建对应 YAML")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError("E_CONFIG_INVALID", f"YAML 解析失败: {p}: {exc}",
                          "修正 YAML 语法") from exc
    cfg = _build_config(doc, source_path=str(p))
    return validate(cfg)


def load_frozen(path: str | Path) -> FrozenThresholds:
    """加载冻结阈值并校验一致性，计算文件 sha256（SPEC §3.5）。"""
    p = Path(path)
    if not p.exists():
        raise ConfigError("E_FROZEN_MISSING", f"冻结阈值文件不存在: {p}",
                          "先运行 scripts/calibrate.py 生成 configs/thresholds_frozen.yaml",
                          EXIT_FROZEN)
    raw_bytes = p.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    try:
        doc = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ConfigError("E_FROZEN_TAMPER", f"冻结阈值文件解析失败: {exc}",
                          "从标定产物重新生成", EXIT_FROZEN) from exc
    if not isinstance(doc, dict):
        raise ConfigError("E_FROZEN_TAMPER", "冻结阈值顶层必须是映射",
                          "从标定产物重新生成", EXIT_FROZEN)

    th = doc.get("thresholds")
    if not isinstance(th, dict):
        raise ConfigError("E_FROZEN_TAMPER", "缺少 thresholds 段",
                          "从标定产物重新生成", EXIT_FROZEN)
    required = ("theta_high_pnr", "theta_low_pnr", "theta_margin", "theta_edge_pnr",
                "v_stat", "m_lost", "reacq_max", "gap_prior_windows")
    missing = [k for k in required if k not in th]
    if missing:
        raise ConfigError("E_FROZEN_TAMPER", f"thresholds 缺少字段: {missing}",
                          "从标定产物重新生成", EXIT_FROZEN)

    if not math.isclose(float(th["theta_edge_pnr"]), float(th["theta_high_pnr"]) + 2.0, abs_tol=1e-9):
        raise ConfigError("E_FROZEN_TAMPER",
                          f"theta_edge_pnr={th['theta_edge_pnr']} != theta_high_pnr+2={float(th['theta_high_pnr'])+2.0}",
                          "按固定规则 theta_edge_pnr = theta_high_pnr + 2 重新生成", EXIT_FROZEN)
    if float(th["theta_low_pnr"]) > float(th["theta_high_pnr"]) + 1e-9:
        raise ConfigError("E_FROZEN_TAMPER",
                          f"theta_low_pnr={th['theta_low_pnr']} > theta_high_pnr={th['theta_high_pnr']}",
                          "重新标定", EXIT_FROZEN)

    width_ref = doc.get("width_ref")
    if not isinstance(width_ref, dict) or not width_ref:
        raise ConfigError("E_FROZEN_TAMPER", "width_ref 必须是非空映射",
                          "重新标定", EXIT_FROZEN)
    norm_width_ref: dict[str, float | None] = {}
    for k, v in width_ref.items():
        norm_width_ref[str(k)] = None if v is None else float(v)

    pre = doc.get("preprocess") or {}
    kernel_size = pre.get("kernel_size")
    kernel_size = None if kernel_size is None else int(kernel_size)
    matched_sigma = pre.get("matched_sigma")
    matched_sigma = None if matched_sigma is None else float(matched_sigma)

    return FrozenThresholds(
        schema_version=int(doc.get("schema_version", SCHEMA_VERSION)),
        sha256=sha,
        frozen_at=str(doc.get("frozen_at", "")),
        kernel_size=kernel_size,
        matched_sigma=matched_sigma,
        theta_high_pnr=float(th["theta_high_pnr"]),
        theta_low_pnr=float(th["theta_low_pnr"]),
        theta_margin=float(th["theta_margin"]),
        theta_edge_pnr=float(th["theta_edge_pnr"]),
        v_stat=float(th["v_stat"]),
        m_lost=int(th["m_lost"]),
        reacq_max=int(th["reacq_max"]),
        gap_prior_windows=int(th["gap_prior_windows"]),
        width_ref=norm_width_ref,
        raw=doc,
    )


def effective_thresholds(cfg: Config, frozen: FrozenThresholds,
                         width_ref: float | None = None) -> Thresholds:
    """融合 config 与冻结阈值，派生 ρ / r_cont_high / r_cont_low（SPEC §5.5）。"""
    wr = 8.0 if width_ref is None else float(width_ref)
    if wr <= 0:
        raise ConfigError("E_CONFIG_INVALID", f"width_ref 必须为正，实际 {wr}",
                          "检查 GT 宽度统计")

    kernel_size = cfg.preprocess.kernel_size
    if frozen.kernel_size is not None and frozen.kernel_size != kernel_size:
        raise ConfigError("E_FROZEN_MISSING",
                          f"冻结 kernel_size={frozen.kernel_size} 与 config kernel_size={kernel_size} 冲突",
                          "同步 configs/*.yaml 的 kernel_size 或重新标定", EXIT_FROZEN)

    return Thresholds(
        theta_high_pnr=frozen.theta_high_pnr,
        theta_low_pnr=frozen.theta_low_pnr,
        theta_margin=frozen.theta_margin,
        theta_edge_pnr=frozen.theta_edge_pnr,
        v_stat=frozen.v_stat,
        m_lost=frozen.m_lost,
        reacq_max=frozen.reacq_max,
        gap_prior_windows=frozen.gap_prior_windows,
        rho=rho_for(wr),
        r_cont_high=cont_radii(wr)[0],
        r_cont_low=cont_radii(wr)[1],
        width_ref=wr,
        kernel_size=kernel_size,
    )


def _main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="spa2d.config", description="校验 config YAML（SPEC §3.1），非法时退出码 2")
    parser.add_argument("paths", nargs="+", help="一个或多个 config YAML 路径")
    args = parser.parse_args(argv)
    failed = False
    for path in args.paths:
        try:
            cfg = load_config(path)
            print(f"OK {path}: {cfg.config_name} (N={cfg.window.N}, stride={cfg.window.stride})")
        except ConfigError as exc:
            failed = True
            print(exc.format(), file=sys.stderr)
    return EXIT_CONFIG_INVALID if failed else EXIT_OK


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
