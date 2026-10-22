"""mgmd.metrics — 框 IoU 0.25 + 匈牙利一对一评测（SPEC §5）。

关键约束（SPEC §5.4）：**同一 GT 被两个检测框覆盖时只能算 1 个 TP**。
贪心匹配会重复吃同一 GT，虚高 precision，因此必须用一对一分配。

本模块无可复用实现（旧工程与 tracking/ 都是点 GT + 半径匹配，口径不同）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mgmd import config as C
from mgmd.annotations import GtFrame, GtObject


@dataclass(frozen=True)
class Detection:
    """单帧内的一个检测框。conf 用于 AP 排序。"""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    frame: int = 0


def box_iou(a, b) -> float:
    """两个 (x1,y1,x2,y2) 的 IoU。任一框退化 → 0。"""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def match_frame(dets: list[Detection], gts: list[GtObject],
                iou_thresh: float = C.IOU_THRESH) -> tuple[list[int], list[int], list[int]]:
    """单帧一对一匹配。

    返回 (tp_det_idx, fp_det_idx, fn_gt_idx)。
    使用匈牙利算法做最大 IoU 总数分配；分配后仍要求 IoU >= thresh。
    """
    if not dets:
        return [], [], list(range(len(gts)))
    if not gts:
        return [], list(range(len(dets))), []

    from scipy.optimize import linear_sum_assignment

    dboxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float64)
    gboxes = np.array([[g.x1, g.y1, g.x2, g.y2] for g in gts], dtype=np.float64)

    # 收益矩阵（IoU）
    gain = np.zeros((len(dets), len(gts)), dtype=np.float64)
    for i in range(len(dets)):
        for j in range(len(gts)):
            gain[i, j] = box_iou(dboxes[i], gboxes[j])

    # 匈牙利最大化总 IoU：对负收益求最小
    row, col = linear_sum_assignment(-gain)

    tp_det, matched_gt = [], set()
    for i, j in zip(row, col):
        if gain[i, j] >= iou_thresh:
            tp_det.append(int(i))
            matched_gt.add(int(j))

    tp_set = set(tp_det)
    fp_det = [i for i in range(len(dets)) if i not in tp_set]
    fn_gt = [j for j in range(len(gts)) if j not in matched_gt]
    return tp_det, fp_det, fn_gt


@dataclass
class FrameOutcome:
    frame: int
    n_det: int
    n_gt: int
    tp: int
    fp: int
    fn: int


@dataclass
class EvalResult:
    """评测结果（SPEC §5.2/§5.3）。"""
    iou_thresh: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    frames: list[FrameOutcome] = field(default_factory=list)
    det_count_hist: dict = field(default_factory=dict)
    gt_count_hist: dict = field(default_factory=dict)
    # AP 用：(conf, is_tp) 全量列表
    ranked: list = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def ap(self, mode: str = "allpoint") -> float:
        """插值 AP。mode: '11point' | 'allpoint'（SPEC §5.2）。"""
        if not self.ranked:
            return 0.0
        n_pos = self.tp + self.fn
        if n_pos == 0:
            return 0.0
        items = sorted(self.ranked, key=lambda t: -t[0])
        tp = fp = 0
        recalls, precisions = [], []
        for _conf, is_tp in items:
            if is_tp:
                tp += 1
            else:
                fp += 1
            recalls.append(tp / n_pos)
            precisions.append(tp / (tp + fp))

        # 单调化：从右往左取最大 precision
        for k in range(len(precisions) - 2, -1, -1):
            precisions[k] = max(precisions[k], precisions[k + 1])

        if mode == "11point":
            total = 0.0
            for k in range(11):
                thr = k / 10.0
                cand = [p for r, p in zip(recalls, precisions) if r >= thr]
                total += max(cand) if cand else 0.0
            return total / 11.0
        if mode == "allpoint":
            area = 0.0
            prev_r = 0.0
            for r, p in zip(recalls, precisions):
                area += (r - prev_r) * p
                prev_r = r
            return area
        raise ValueError(f"未知 AP 模式: {mode}")


def _bucket(n: int) -> str:
    for lo, hi in C.GT_COUNT_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if lo != hi else f"{lo}"
    return "4+"


def evaluate(dets_by_frame: dict[int, list[Detection]],
             gt_by_frame: dict[int, GtFrame],
             iou_thresh: float = C.IOU_THRESH) -> EvalResult:
    """逐帧评测。帧集合 = GT 帧 ∪ 检测帧（两边都要计入，空帧检测算 FP）。"""
    res = EvalResult(iou_thresh=iou_thresh)
    all_frames = sorted(set(gt_by_frame) | set(dets_by_frame))
    det_hist: dict[int, int] = {}
    gt_hist: dict[int, int] = {}

    for f in all_frames:
        dets = dets_by_frame.get(f, [])
        gt = gt_by_frame.get(f)
        gts = list(gt.objects) if gt else []

        tp_det, fp_det, fn_gt = match_frame(dets, gts, iou_thresh)
        res.tp += len(tp_det)
        res.fp += len(fp_det)
        res.fn += len(fn_gt)
        res.frames.append(FrameOutcome(f, len(dets), len(gts),
                                       len(tp_det), len(fp_det), len(fn_gt)))

        det_hist[len(dets)] = det_hist.get(len(dets), 0) + 1
        gt_hist[len(gts)] = gt_hist.get(len(gts), 0) + 1

        tp_set = set(tp_det)
        for i, d in enumerate(dets):
            res.ranked.append((d.conf, i in tp_set))

    res.det_count_hist = dict(sorted(det_hist.items()))
    res.gt_count_hist = dict(sorted(gt_hist.items()))
    return res


def grouped_by_gt_count(res: EvalResult) -> dict[str, dict]:
    """按帧内 GT 数分组统计（SPEC §5.3），并给出样本量。"""
    groups: dict[str, dict] = {}
    for fo in res.frames:
        key = _bucket(fo.n_gt)
        g = groups.setdefault(key, {"frames": 0, "tp": 0, "fp": 0, "fn": 0})
        g["frames"] += 1
        g["tp"] += fo.tp
        g["fp"] += fo.fp
        g["fn"] += fo.fn

    for key, g in groups.items():
        p = g["tp"] / (g["tp"] + g["fp"]) if (g["tp"] + g["fp"]) else 0.0
        r = g["tp"] / (g["tp"] + g["fn"]) if (g["tp"] + g["fn"]) else 0.0
        g["precision"] = p
        g["recall"] = r
        g["f1"] = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def summary(res: EvalResult) -> dict:
    """对外报告用的扁平字典（写入 metrics.json）。"""
    return {
        "iou_thresh": res.iou_thresh,
        "match_algo": C.MATCH_ALGO,
        "tp": res.tp, "fp": res.fp, "fn": res.fn,
        "precision": res.precision,
        "recall": res.recall,
        "f1": res.f1,
        "ap_11point": res.ap("11point"),
        "ap_allpoint": res.ap("allpoint"),
        "det_count_hist": res.det_count_hist,
        "gt_count_hist": res.gt_count_hist,
        "by_gt_count": grouped_by_gt_count(res),
    }
