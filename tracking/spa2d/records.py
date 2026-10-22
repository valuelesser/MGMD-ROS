"""spa2d.records — 全流程数据结构与 JSONL 序列化（SPEC §3.2/§3.3/§5.8）。

所有记录类型均提供 ``to_dict`` / ``from_dict``，``WindowRecord`` 额外提供
``to_json`` / ``from_json`` 与 ``write_jsonl`` / ``read_jsonl``，保证 JSONL
往返（round-trip）逐字段一致。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

STATES = ("INIT", "TRACKED", "TRACKED_LOW", "REACQ", "LOST", "OUT_OF_FRAME", "ABSENT")
REASONS = ("OK", "EDGE_REJECT", "LOW_PNR", "LOW_MARGIN", "DISCONTINUOUS",
           "NO_PRIOR_GLOBAL", "LOST", "OUT_OF_FRAME", "ABSENT")
SEARCH_MODES = ("local", "global")

GT_COLUMNS = ("frame", "timestamp_s", "visible", "out_of_frame", "cx", "cy",
              "bbox_x", "bbox_y", "bbox_w", "bbox_h", "area_px", "width_px",
              "width_source", "gt_source", "projection_rmse_px")

T = TypeVar("T")


def _pair(value) -> tuple:
    return tuple(value) if value is not None else None


@dataclass
class Estimate:
    x: float
    y: float
    vx: float
    vy: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy}

    @classmethod
    def from_dict(cls, d: dict) -> "Estimate":
        return cls(x=float(d["x"]), y=float(d["y"]), vx=float(d["vx"]), vy=float(d["vy"]))


@dataclass
class PeakResult:
    value: float
    xy_int: tuple[int, int]
    xy_sub: tuple[float, float] | None
    subpixel_used: bool
    valid_count: int
    edge_candidate: bool

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "xy_int": list(self.xy_int),
            "xy_sub": None if self.xy_sub is None else list(self.xy_sub),
            "subpixel_used": self.subpixel_used,
            "valid_count": self.valid_count,
            "edge_candidate": self.edge_candidate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PeakResult":
        return cls(
            value=float(d["value"]),
            xy_int=(int(d["xy_int"][0]), int(d["xy_int"][1])),
            xy_sub=None if d["xy_sub"] is None else (float(d["xy_sub"][0]), float(d["xy_sub"][1])),
            subpixel_used=bool(d["subpixel_used"]),
            valid_count=int(d["valid_count"]),
            edge_candidate=bool(d["edge_candidate"]),
        )


@dataclass
class Confidence:
    pnr: float
    mu_bg: float
    sigma_bg: float
    margin: float
    single_candidate: bool = False

    def to_dict(self) -> dict:
        return {
            "pnr": self.pnr, "mu_bg": self.mu_bg, "sigma_bg": self.sigma_bg,
            "margin": self.margin, "single_candidate": self.single_candidate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Confidence":
        return cls(
            pnr=float(d["pnr"]), mu_bg=float(d["mu_bg"]),
            sigma_bg=float(d["sigma_bg"]), margin=float(d["margin"]),
            single_candidate=bool(d.get("single_candidate", False)),
        )


@dataclass
class Continuity:
    r_cont: float | None
    r_cont_high: float
    r_cont_low: float
    exempt: bool

    def to_dict(self) -> dict:
        return {
            "r_cont": self.r_cont, "r_cont_high": self.r_cont_high,
            "r_cont_low": self.r_cont_low, "exempt": self.exempt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Continuity":
        return cls(
            r_cont=None if d["r_cont"] is None else float(d["r_cont"]),
            r_cont_high=float(d["r_cont_high"]),
            r_cont_low=float(d["r_cont_low"]),
            exempt=bool(d["exempt"]),
        )


@dataclass
class Flags:
    stat: bool = False
    edge: bool = False

    def to_dict(self) -> dict:
        return {"stat": self.stat, "edge": self.edge}

    @classmethod
    def from_dict(cls, d: dict) -> "Flags":
        return cls(stat=bool(d.get("stat", False)), edge=bool(d.get("edge", False)))


@dataclass
class TimingMs:
    t_pre: float
    t_search: float
    t_state: float
    t_total: float

    def to_dict(self) -> dict:
        return {"t_pre": self.t_pre, "t_search": self.t_search,
                "t_state": self.t_state, "t_total": self.t_total}

    @classmethod
    def from_dict(cls, d: dict) -> "TimingMs":
        return cls(t_pre=float(d["t_pre"]), t_search=float(d["t_search"]),
                   t_state=float(d["t_state"]), t_total=float(d["t_total"]))


@dataclass
class WindowRecord:
    window_index: int
    t_start: int
    t_center: float
    assoc_frame: int
    search_mode: str
    state_in: str
    state_out: str
    accepted: bool
    target_present: bool
    reason: str
    est: Estimate | None = None
    peak: PeakResult | None = None
    confidence: Confidence | None = None
    continuity: Continuity | None = None
    flags: Flags = field(default_factory=Flags)
    miss_count: int = 0
    reacq_count: int = 0
    skipped_by_period: bool = False
    timing_ms: TimingMs | None = None

    def to_dict(self) -> dict:
        return {
            "window_index": self.window_index,
            "t_start": self.t_start,
            "t_center": self.t_center,
            "assoc_frame": self.assoc_frame,
            "search_mode": self.search_mode,
            "state_in": self.state_in,
            "state_out": self.state_out,
            "accepted": self.accepted,
            "target_present": self.target_present,
            "reason": self.reason,
            "est": None if self.est is None else self.est.to_dict(),
            "peak": None if self.peak is None else self.peak.to_dict(),
            "confidence": None if self.confidence is None else self.confidence.to_dict(),
            "continuity": None if self.continuity is None else self.continuity.to_dict(),
            "flags": self.flags.to_dict(),
            "miss_count": self.miss_count,
            "reacq_count": self.reacq_count,
            "skipped_by_period": self.skipped_by_period,
            "timing_ms": None if self.timing_ms is None else self.timing_ms.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WindowRecord":
        return cls(
            window_index=int(d["window_index"]),
            t_start=int(d["t_start"]),
            t_center=float(d["t_center"]),
            assoc_frame=int(d["assoc_frame"]),
            search_mode=str(d["search_mode"]),
            state_in=str(d["state_in"]),
            state_out=str(d["state_out"]),
            accepted=bool(d["accepted"]),
            target_present=bool(d["target_present"]),
            reason=str(d["reason"]),
            est=None if d["est"] is None else Estimate.from_dict(d["est"]),
            peak=None if d["peak"] is None else PeakResult.from_dict(d["peak"]),
            confidence=None if d["confidence"] is None else Confidence.from_dict(d["confidence"]),
            continuity=None if d["continuity"] is None else Continuity.from_dict(d["continuity"]),
            flags=Flags.from_dict(d["flags"]),
            miss_count=int(d["miss_count"]),
            reacq_count=int(d["reacq_count"]),
            skipped_by_period=bool(d.get("skipped_by_period", False)),
            timing_ms=None if d["timing_ms"] is None else TimingMs.from_dict(d["timing_ms"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=False)

    @classmethod
    def from_json(cls, s: str) -> "WindowRecord":
        return cls.from_dict(json.loads(s))


@dataclass
class SearchResult:
    cands: list[tuple[float, float]]
    scores: list[float]
    v_best: tuple[float, float]
    second_best: float | None
    chunk_count: int

    def to_dict(self) -> dict:
        return {
            "cands": [list(c) for c in self.cands],
            "scores": list(self.scores),
            "v_best": list(self.v_best),
            "second_best": self.second_best,
            "chunk_count": self.chunk_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SearchResult":
        return cls(
            cands=[(float(c[0]), float(c[1])) for c in d["cands"]],
            scores=[float(s) for s in d["scores"]],
            v_best=(float(d["v_best"][0]), float(d["v_best"][1])),
            second_best=None if d["second_best"] is None else float(d["second_best"]),
            chunk_count=int(d["chunk_count"]),
        )


@dataclass
class TimingStats:
    layer: str
    count: int
    mean: float
    median: float
    p95: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict:
        return {
            "layer": self.layer, "count": self.count, "mean": self.mean,
            "median": self.median, "p95": self.p95,
            "minimum": self.minimum, "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TimingStats":
        return cls(layer=str(d["layer"]), count=int(d["count"]), mean=float(d["mean"]),
                   median=float(d["median"]), p95=float(d["p95"]),
                   minimum=float(d["minimum"]), maximum=float(d["maximum"]))


@dataclass
class MetricsResult:
    config_name: str
    association: str
    radii_px: tuple[int, ...]
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    correct: dict[int, int] = field(default_factory=dict)
    fp: dict[int, int] = field(default_factory=dict)
    valid_outcome: float = 0.0
    rmse_p: float | None = None
    rmse_v: float | None = None
    visible_den: int = 0
    valid_absent: int = 0
    total_window_outputs: int = 0
    diagnostic_rmse_all_visible: float | None = None
    reason_counts: dict[str, int] = field(default_factory=dict)
    fp_sources: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "association": self.association,
            "radii_px": list(self.radii_px),
            "recall": {str(k): v for k, v in self.recall.items()},
            "precision": {str(k): v for k, v in self.precision.items()},
            "correct": {str(k): v for k, v in self.correct.items()},
            "fp": {str(k): v for k, v in self.fp.items()},
            "valid_outcome": self.valid_outcome,
            "rmse_p": self.rmse_p,
            "rmse_v": self.rmse_v,
            "visible_den": self.visible_den,
            "valid_absent": self.valid_absent,
            "total_window_outputs": self.total_window_outputs,
            "diagnostic_rmse_all_visible": self.diagnostic_rmse_all_visible,
            "reason_counts": dict(self.reason_counts),
            "fp_sources": dict(self.fp_sources),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetricsResult":
        return cls(
            config_name=str(d["config_name"]),
            association=str(d["association"]),
            radii_px=tuple(int(r) for r in d["radii_px"]),
            recall={int(k): float(v) for k, v in d["recall"].items()},
            precision={int(k): float(v) for k, v in d["precision"].items()},
            correct={int(k): int(v) for k, v in d["correct"].items()},
            fp={int(k): int(v) for k, v in d["fp"].items()},
            valid_outcome=float(d["valid_outcome"]),
            rmse_p=None if d["rmse_p"] is None else float(d["rmse_p"]),
            rmse_v=None if d["rmse_v"] is None else float(d["rmse_v"]),
            visible_den=int(d["visible_den"]),
            valid_absent=int(d["valid_absent"]),
            total_window_outputs=int(d["total_window_outputs"]),
            diagnostic_rmse_all_visible=(None if d["diagnostic_rmse_all_visible"] is None
                                         else float(d["diagnostic_rmse_all_visible"])),
            reason_counts={str(k): int(v) for k, v in d["reason_counts"].items()},
            fp_sources={str(k): int(v) for k, v in d.get("fp_sources", {}).items()},
        )


@dataclass
class GTFrame:
    frame: int
    timestamp_s: float
    visible: int
    out_of_frame: int
    cx: float | None = None
    cy: float | None = None
    bbox_x: float | None = None
    bbox_y: float | None = None
    bbox_w: float | None = None
    bbox_h: float | None = None
    area_px: float | None = None
    width_px: float | None = None
    width_source: str | None = None
    gt_source: str | None = None
    projection_rmse_px: float | None = None

    def to_dict(self) -> dict:
        return {c: getattr(self, c) for c in GT_COLUMNS}

    @classmethod
    def from_dict(cls, d: dict) -> "GTFrame":
        def num(key):
            v = d.get(key)
            return None if v is None or v == "" else float(v)
        return cls(
            frame=int(d["frame"]),
            timestamp_s=float(d["timestamp_s"]),
            visible=int(d["visible"]),
            out_of_frame=int(d["out_of_frame"]),
            cx=num("cx"), cy=num("cy"),
            bbox_x=num("bbox_x"), bbox_y=num("bbox_y"),
            bbox_w=num("bbox_w"), bbox_h=num("bbox_h"),
            area_px=num("area_px"), width_px=num("width_px"),
            width_source=None if d.get("width_source") in (None, "") else str(d["width_source"]),
            gt_source=None if d.get("gt_source") in (None, "") else str(d["gt_source"]),
            projection_rmse_px=num("projection_rmse_px"),
        )


def write_jsonl(records: Iterable[WindowRecord], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json())
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[WindowRecord]:
    p = Path(path)
    out: list[WindowRecord] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(WindowRecord.from_json(line))
    return out


def iter_jsonl(path: str | Path) -> Iterator[WindowRecord]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield WindowRecord.from_json(line)


def _csv_num(value) -> str:
    return "" if value is None else f"{value:.6f}"


def gt_frame_to_row(gt: GTFrame) -> dict[str, str]:
    """GTFrame → CSV 行（None 写空串，SPEC §3.2 NaN 语义）。"""
    return {
        "frame": str(gt.frame),
        "timestamp_s": f"{gt.timestamp_s:.6f}",
        "visible": str(gt.visible),
        "out_of_frame": str(gt.out_of_frame),
        "cx": _csv_num(gt.cx),
        "cy": _csv_num(gt.cy),
        "bbox_x": _csv_num(gt.bbox_x),
        "bbox_y": _csv_num(gt.bbox_y),
        "bbox_w": _csv_num(gt.bbox_w),
        "bbox_h": _csv_num(gt.bbox_h),
        "area_px": _csv_num(gt.area_px),
        "width_px": _csv_num(gt.width_px),
        "width_source": "" if gt.width_source is None else str(gt.width_source),
        "gt_source": "" if gt.gt_source is None else str(gt.gt_source),
        "projection_rmse_px": _csv_num(gt.projection_rmse_px),
    }


def write_gt_csv(frames: Iterable[GTFrame], path: str | Path) -> int:
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(GT_COLUMNS))
        writer.writeheader()
        for gt in frames:
            writer.writerow(gt_frame_to_row(gt))
            n += 1
    return n


def read_gt_csv(path: str | Path) -> list[GTFrame]:
    import csv

    out: list[GTFrame] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or list(reader.fieldnames) != list(GT_COLUMNS):
            raise ValueError(f"GT CSV 列不匹配: {reader.fieldnames}")
        for row in reader:
            out.append(GTFrame.from_dict(row))
    return out
