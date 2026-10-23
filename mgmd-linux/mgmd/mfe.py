"""mgmd.mfe — 运动特征增强（论文 III-A，SPEC §8 T04）。

论文三段：
    III-A1 帧对齐   : 网格关键点 → 金字塔 LK → RANSAC 单应
    III-A2 运动提取 : E12=|I_t−Î_{t−k}|, E23=|I_t−Î_{t+k}|, 阈值 5, OR 聚合
    III-A3 后处理   : 3×3 开/闭, 连通域, 面积 < 15 px 丢弃

**与论文/旧工程的关键差异（SPEC §6 偏差表）**：论文没提但常见的四项额外约束
全部做成开关且**默认关闭**，以便区分"论文方法的功劳"与"额外约束的功劳"：
    M4  preprocess_blur   高斯模糊（旧工程写死 11，论文未提）
    M5  disp_filter       关键点大位移过滤（旧工程写死 0.25·max(w,h)，论文未提）
    M11 max_area          连通域面积上限（论文只给下限 15）
    M12 aspect_filter     长宽比过滤（论文未提）

输出候选框列表，不做单目标截断（交给 TF）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class MFEConfig:
    # ---- 论文明确给出 ----
    k: int = 1                     # 帧间隔（论文未给值，SPEC D-03 待扫描）
    threshold: int = 5             # 二值化阈值（论文 5）
    grid_cols: int = 30            # 网格 30×20（论文）
    grid_rows: int = 20
    min_area: int = 15             # 连通域下限（论文 15）
    mode: str = "3frame_or"        # 2frame | 3frame_and | 3frame_or（论文用 OR）

    # ---- 论文未给、必须自选的实现参数（SPEC §6 推断项）----
    ransac_thresh: float = 3.0
    min_keypoints: int = 9
    morph_size: int = 3
    open_iters: int = 1
    close_iters: int = 3

    # ---- 官方 GLAD 口径（SPEC §8.21，默认开启以对齐官方）----
    adaptive_thresh: bool = True   # 缺口1: T = base + int(mean(diff))
    median_blur: int = 5           # 缺口2: 二值化后中值滤波，0=关闭
    official_area_min: int = 0     # 缺口3: >0 时用该值覆盖 min_area（官方30）

    # ---- 四项额外约束（缺口4: 官方默认开启，SPEC D-04~D-07 已修订）----
    preprocess_blur: int = 0       # M4: 官方=11；默认0保留论文"未提"口径，official=True 时启用
    disp_filter: bool = False      # M5: 关键点位移离群过滤（官方有，见 official）
    disp_ratio: float = 0.25       # M5 参数
    max_area: int = 3000           # M11: 官方 3000（缺口4，改为默认开启）
    aspect_filter: bool = True     # M12: 官方默认开启（缺口4）
    min_aspect: float = 0.6        # 官方 0.6（缺口4，此前误用 0.2）
    max_aspect: float = 3.0        # 官方 3.0（缺口4，此前误用 5.0）
    official: bool = False         # True = 一键对齐官方 GLAD MOD2_global

    def apply_official(self) -> None:
        """一键对齐官方 GLAD MOD2_global（SPEC §8.21）。"""
        self.adaptive_thresh = True
        self.median_blur = 5
        self.official_area_min = 30
        self.max_area = 3000
        self.aspect_filter = True
        self.min_aspect = 0.6
        self.max_aspect = 3.0
        self.preprocess_blur = 11
        self.disp_filter = True

    def validate(self) -> None:
        if self.mode not in ("2frame", "3frame_and", "3frame_or"):
            raise ValueError(f"非法 mode: {self.mode}")
        if self.k < 1:
            raise ValueError(f"k 必须 >= 1，实际 {self.k}")
        if self.grid_cols < 2 or self.grid_rows < 2:
            raise ValueError(f"网格过小: {self.grid_cols}x{self.grid_rows}")
        if self.min_area < 0:
            raise ValueError(f"min_area 必须 >= 0")
        if self.max_area and self.max_area <= self.min_area:
            raise ValueError(f"max_area({self.max_area}) 必须 > min_area({self.min_area})")
        if self.preprocess_blur and (self.preprocess_blur < 3 or self.preprocess_blur % 2 == 0):
            raise ValueError(f"preprocess_blur 必须为 >=3 的奇数，实际 {self.preprocess_blur}")

    @property
    def needs_next(self) -> bool:
        return self.mode in ("3frame_and", "3frame_or")


# --------------------------------------------------------------------------- #
# III-A1 帧对齐
# --------------------------------------------------------------------------- #

def grid_keypoints(width: int, height: int, cols: int, rows: int) -> np.ndarray:
    """均匀网格关键点（格心），float32 (N,1,2)。"""
    xs = (np.arange(cols) + 0.5) * (width / cols)
    ys = (np.arange(rows) + 0.5) * (height / rows)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)


def _valid_region(H: np.ndarray, width: int, height: int) -> np.ndarray:
    """参考帧中"能映射回源帧内部"的像素掩膜（255=有效）。"""
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                       dtype=np.float32).reshape(-1, 1, 2)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full((height, width), 255, np.uint8)
    mapped = cv2.perspectiveTransform(corners, H_inv)
    if not np.isfinite(mapped).all():
        return np.full((height, width), 255, np.uint8)
    valid = np.zeros((height, width), np.uint8)
    cv2.fillPoly(valid, [mapped.astype(np.int32).reshape(1, 4, 2)], 255)
    return valid


def align_to_reference(src_gray: np.ndarray, ref_gray: np.ndarray,
                       cfg: MFEConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """把 src_gray 变换到 ref_gray 坐标系。返回 (aligned, valid_mask, H)。"""
    h, w = ref_gray.shape
    pts = grid_keypoints(w, h, cfg.grid_cols, cfg.grid_rows)
    lk = dict(winSize=(15, 15), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(src_gray, ref_gray, pts, None, **lk)
    if tracked is None or status is None:
        return src_gray.copy(), np.full((h, w), 255, np.uint8), None

    ok = status.ravel().astype(bool)
    src_pts = pts.reshape(-1, 2)[ok]
    ref_pts = tracked.reshape(-1, 2)[ok]
    if len(src_pts) < cfg.min_keypoints:
        return src_gray.copy(), np.full((h, w), 255, np.uint8), None

    # M5：位移离群过滤（默认关闭，SPEC D-05）
    if cfg.disp_filter:
        disp = np.linalg.norm(ref_pts - src_pts, axis=1)
        keep = disp < cfg.disp_ratio * max(w, h)
        if keep.sum() >= cfg.min_keypoints:
            src_pts, ref_pts = src_pts[keep], ref_pts[keep]

    H, _inliers = cv2.findHomography(ref_pts, src_pts, cv2.RANSAC, cfg.ransac_thresh)
    if H is None:
        return src_gray.copy(), np.full((h, w), 255, np.uint8), None

    aligned = cv2.warpPerspective(src_gray, H, (w, h),
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aligned, _valid_region(H, w, h), H


def preprocess(frame_bgr: np.ndarray, cfg: MFEConfig) -> np.ndarray:
    """灰度化；M4 高斯模糊默认关闭（SPEC D-04）。"""
    if cfg.preprocess_blur:
        frame_bgr = cv2.GaussianBlur(frame_bgr, (cfg.preprocess_blur, cfg.preprocess_blur), 0)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- #
# III-A2 运动提取
# --------------------------------------------------------------------------- #

def _binarize(diff: np.ndarray, threshold: int, valid: np.ndarray,
              cfg: "MFEConfig | None" = None) -> np.ndarray:
    """阈值二值化 + 有效区掩膜 + 可选中值滤波。

    缺口1（SPEC §8.21）：cfg.adaptive_thresh 时用 T = base + int(mean(diff))，
                        对齐官方 MOD2_global 的 `T_1 = 5 + fix_coef`。
    缺口2：cfg.median_blur > 0 时补 medianBlur，官方为 5。
    掩膜用 bitwise_and（避免 uint8 下溢，SPEC D-10）。
    """
    t = threshold
    if cfg is not None and cfg.adaptive_thresh:
        t = threshold + int(np.mean(diff))
    mask = (diff > t).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, valid)
    if cfg is not None and cfg.median_blur and cfg.median_blur >= 3:
        mask = cv2.medianBlur(mask, cfg.median_blur)
        # 中值滤波后补回有效区（滤波可能把边界外像素拉进来）
        mask = cv2.bitwise_and(mask, valid)
    return mask


def motion_mask(prev_gray, cur_gray, next_gray, cfg: MFEConfig) -> tuple[np.ndarray, dict]:
    """构建聚合运动掩膜 E_t。prev/next 须已对齐到 cur 坐标系（或 None）。"""
    cfg.validate()
    h, w = cur_gray.shape
    prev_al, prev_valid, prev_H = align_to_reference(prev_gray, cur_gray, cfg)
    d12 = _binarize(cv2.absdiff(cur_gray, prev_al), cfg.threshold, prev_valid, cfg)

    info = {"d12": d12, "prev_H": prev_H, "next_H": None, "d23": None}
    d23 = None
    if cfg.needs_next and next_gray is not None:
        next_al, next_valid, next_H = align_to_reference(next_gray, cur_gray, cfg)
        d23 = _binarize(cv2.absdiff(cur_gray, next_al), cfg.threshold, next_valid, cfg)
        info.update({"d23": d23, "next_H": next_H, "next_aligned": next_al})

    if cfg.mode == "2frame" or d23 is None:
        motion = d12
    elif cfg.mode == "3frame_and":
        motion = cv2.bitwise_and(d12, d23)
    else:
        motion = cv2.bitwise_or(d12, d23)

    info["motion_raw"] = motion
    return motion, info


# --------------------------------------------------------------------------- #
# III-A3 后处理
# --------------------------------------------------------------------------- #

def postprocess(motion: np.ndarray, cfg: MFEConfig) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """形态学 + 连通域 → 候选框 (x,y,w,h)。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_size, cfg.morph_size))
    opened = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=cfg.open_iters)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=cfg.close_iters)

    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(closed, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    # 缺口3：官方用 area > 30（开区间），论文写 15
    amin = cfg.official_area_min if cfg.official_area_min > 0 else cfg.min_area
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w == 0 or h == 0:
            continue
        if area < amin:
            continue
        if cfg.max_area and area > cfg.max_area:      # M11
            continue
        if cfg.aspect_filter:                          # M12
            aspect = w / h
            if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
                continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return closed, boxes


# --------------------------------------------------------------------------- #
# 封装
# --------------------------------------------------------------------------- #

@dataclass
class MFEResult:
    frame_index: int
    boxes: list[tuple[int, int, int, int]]
    motion: np.ndarray
    closed: np.ndarray
    info: dict = field(default_factory=dict)


def enhance(prev_bgr, cur_bgr, next_bgr, cfg: MFEConfig, frame_index: int = 0) -> MFEResult:
    """单帧三元组跑完整 MFE（BGR 输入）。"""
    cfg.validate()
    cur_gray = preprocess(cur_bgr, cfg)
    prev_gray = preprocess(prev_bgr, cfg) if prev_bgr is not None else None
    next_gray = preprocess(next_bgr, cfg) if next_bgr is not None else None

    if prev_gray is None:
        # 首帧无前帧：全无运动
        z = np.zeros_like(cur_gray)
        return MFEResult(frame_index, [], z, z, {"note": "no_prev"})

    motion, info = motion_mask(prev_gray, cur_gray, next_gray, cfg)
    closed, boxes = postprocess(motion, cfg)
    return MFEResult(frame_index, boxes, motion, closed, info)
