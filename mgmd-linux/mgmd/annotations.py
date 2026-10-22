"""mgmd.annotations — Pascal VOC XML → 统一 GT 记录（SPEC §4.3）。

只读 NTFS 上的 60 视频标注集；不使用 100 视频扩展集（SPEC §4.1）。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from mgmd.config import ARD_ANNOTATIONS, MgmdError, EXIT_DATA


@dataclass(frozen=True)
class GtObject:
    """单个目标框（像素坐标，x1<x2, y1<y2）。"""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass(frozen=True)
class GtFrame:
    """单帧标注。frame 为 1 基帧号。"""
    frame: int
    objects: tuple[GtObject, ...]

    @property
    def visible(self) -> bool:
        return len(self.objects) > 0

    @property
    def n_obj(self) -> int:
        return len(self.objects)


def _floats(elem, keys) -> tuple[float, ...]:
    out = []
    for k in keys:
        child = elem.find(k)
        if child is None or child.text is None:
            raise ValueError(f"缺少字段 {k}")
        out.append(float(child.text))
    return tuple(out)


def parse_xml(path: Path) -> GtFrame:
    """解析单个 VOC XML。frame 从文件名尾部的 4 位数字取（1 基）。"""
    stem = path.stem  # phantom09_0100
    tail = stem.rsplit("_", 1)[-1]
    if not tail.isdigit():
        raise ValueError(f"文件名无法解析帧号: {path.name}")
    frame = int(tail)

    root = ET.parse(path).getroot()
    objs: list[GtObject] = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1, y1, x2, y2 = _floats(bb, ("xmin", "ymin", "xmax", "ymax"))
        # 规范化坐标顺序（标注里偶见反序）
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 <= x1 or y2 <= y1:
            continue  # 退化框丢弃
        objs.append(GtObject(x1, y1, x2, y2))
    return GtFrame(frame=frame, objects=tuple(objs))


def load_video_gt(video: str) -> dict[int, GtFrame]:
    """加载某视频全部帧标注，返回 {frame: GtFrame}。"""
    vdir = ARD_ANNOTATIONS / video
    if not vdir.is_dir():
        raise MgmdError("E_DATA_MISSING", f"标注目录不存在: {vdir}",
                        "确认视频名正确且在 60 视频集内（SPEC §4.1）", EXIT_DATA)
    out: dict[int, GtFrame] = {}
    for xml in sorted(vdir.glob("*.xml")):
        gt = parse_xml(xml)
        out[gt.frame] = gt
    if not out:
        raise MgmdError("E_DATA_MISSING", f"标注为空: {vdir}",
                        "检查标注是否完整解压", EXIT_DATA)
    return out


def video_stats(video: str) -> dict:
    """单视频统计（供 T02 校验与报告用）。"""
    gt = load_video_gt(video)
    frames = sorted(gt)
    n_obj = sum(gt[f].n_obj for f in frames)
    empty = sum(1 for f in frames if not gt[f].visible)
    multi = sum(1 for f in frames if gt[f].n_obj > 1)
    areas = [o.area for f in frames for o in gt[f].objects]
    widths = [o.w for f in frames for o in gt[f].objects]
    heights = [o.h for f in frames for o in gt[f].objects]

    def med(xs):
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[len(s) // 2]

    return {
        "video": video,
        "frames_annotated": len(frames),
        "frame_first": frames[0],
        "frame_last": frames[-1],
        "frame_span": frames[-1] - frames[0] + 1,
        "frame_gaps": (frames[-1] - frames[0] + 1) - len(frames),
        "objects": n_obj,
        "empty_frames": empty,
        "multi_obj_frames": multi,
        "median_w": med(widths),
        "median_h": med(heights),
        "median_area": med(areas),
    }
