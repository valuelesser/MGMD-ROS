"""mgmd.viz — MFE/TF 可视化（SPEC §3）。

每个阶段单独落图，便于判断"哪一步出的问题"：
    1. 原图 + GT 框（绿）
    2. 运动掩膜 E_t（阈值+OR 聚合后）
    3. 形态学后掩膜
    4. 候选框（黄）+ GT（绿）叠加
    5. 检测/轨迹框（红）+ GT（绿）叠加

关键：中文路径下 cv2.imwrite 会静默失败，统一用 imencode + Path.write_bytes。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_image(path: Path, img: np.ndarray) -> bool:
    """安全写图（规避中文路径 imwrite 静默失败）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True


def _label(img: np.ndarray, text: str, org=(8, 22), color=(255, 255, 255),
           scale: float = 0.6) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness=1, lineType=cv2.LINE_AA)


def draw_boxes(img: np.ndarray, boxes, color, thickness: int = 2,
               label: str | None = None) -> np.ndarray:
    """在图上画框。boxes 为 (x1,y1,x2,y2) 序列。"""
    out = img.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (int(round(v)) for v in b[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if label and i == 0:
            cv2.putText(out, label, (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    """单通道掩膜转三通道便于拼接。"""
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def make_panel(tiles: list[tuple[str, np.ndarray]], cols: int = 2,
               tile_w: int = 640, pad: int = 6) -> np.ndarray:
    """把多张图拼成网格，每张等宽缩放并加标题。"""
    scaled = []
    for title, img in tiles:
        img = mask_to_bgr(img)
        h, w = img.shape[:2]
        th = int(round(h * tile_w / w))
        small = cv2.resize(img, (tile_w, th), interpolation=cv2.INTER_AREA)
        # 顶部留白写标题
        canvas = np.zeros((th + 34, tile_w, 3), np.uint8)
        canvas[34:] = small
        _label(canvas, title, org=(10, 24), color=(255, 255, 0), scale=0.7)
        scaled.append(canvas)

    if not scaled:
        return np.zeros((10, 10, 3), np.uint8)

    n = len(scaled)
    rows = (n + cols - 1) // cols
    # 各格高度可能不同（宽高比差异），取最大值统一
    th = max(s.shape[0] for s in scaled)
    tw = max(s.shape[1] for s in scaled)
    grid = np.full((rows * th + (rows + 1) * pad, cols * tw + (cols + 1) * pad, 3),
                   32, np.uint8)
    for i, s in enumerate(scaled):
        r, c = divmod(i, cols)
        y = pad + r * (th + pad)
        x = pad + c * (tw + pad)
        sh, sw = s.shape[:2]
        grid[y:y + sh, x:x + sw] = s
    return grid


def crop_zoom(frame: np.ndarray, center, size: int = 200,
              zoom: int = 3) -> np.ndarray:
    """在目标附近裁一块并放大，用来看清像素级目标。"""
    h, w = frame.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    x1 = max(0, min(cx - size // 2, w - size))
    y1 = max(0, min(cy - size // 2, h - size))
    x2, y2 = min(w, x1 + size), min(h, y1 + size)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return frame
    return cv2.resize(patch, ((x2 - x1) * zoom, (y2 - y1) * zoom),
                      interpolation=cv2.INTER_NEAREST)


def side_by_side(a: np.ndarray, b: np.ndarray, gap: int = 8) -> np.ndarray:
    """等高并排。"""
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = max(ha, hb)
    def pad(img):
        ih, iw = img.shape[:2]
        if ih == h:
            return img
        return cv2.copyMakeBorder(img, 0, h - ih, 0, 0,
                                  cv2.BORDER_CONSTANT, value=(32, 32, 32))
    return np.hstack([pad(mask_to_bgr(a)), np.full((h, gap, 3), 32, np.uint8), pad(mask_to_bgr(b))])
