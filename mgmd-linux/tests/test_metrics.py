"""tests.test_metrics — 评测器验收（SPEC §5.4）。

最关键用例：同一 GT 被两个检测覆盖时只能算 1 TP（防贪心重复计）。
"""
from __future__ import annotations

import math

from mgmd.annotations import GtFrame, GtObject
from mgmd.metrics import (Detection, box_iou, evaluate, grouped_by_gt_count,
                          match_frame, summary)


def _gt(frame, *boxes):
    return GtFrame(frame=frame, objects=tuple(GtObject(*b) for b in boxes))


def test_iou_basic():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # 半重叠：交 50，并 150
    assert abs(box_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9
    # 退化框
    assert box_iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


def test_one_gt_two_dets_only_one_tp():
    """核心反作弊用例：同一 GT 被两个检测覆盖 → 1 TP + 1 FP，不是 2 TP。"""
    gts = [GtObject(100, 100, 140, 130)]
    dets = [Detection(98, 99, 141, 131, conf=0.9),
            Detection(99, 100, 140, 129, conf=0.8)]
    tp, fp, fn = match_frame(dets, gts)
    assert len(tp) == 1, f"应只算 1 TP，实际 {len(tp)}"
    assert len(fp) == 1, f"重复检测应算 FP，实际 {len(fp)}"
    assert len(fn) == 0


def test_two_gt_two_dets_correct_pairing():
    """两个 GT、两个检测：应各自配对，不能交叉错配。"""
    gts = [GtObject(0, 0, 20, 20), GtObject(100, 100, 120, 120)]
    dets = [Detection(101, 101, 121, 121, conf=0.9),
            Detection(1, 1, 21, 21, conf=0.8)]
    tp, fp, fn = match_frame(dets, gts)
    assert len(tp) == 2 and len(fp) == 0 and len(fn) == 0


def test_below_threshold_counts_as_miss():
    """IoU < 0.25 不匹配。"""
    gts = [GtObject(0, 0, 10, 10)]
    dets = [Detection(8, 0, 18, 10, conf=0.9)]  # IoU = 20/180 ≈ 0.111
    tp, fp, fn = match_frame(dets, gts)
    assert tp == [] and len(fp) == 1 and len(fn) == 1


def test_detection_on_empty_frame_is_fp():
    """空标注帧上的检测算 FP。"""
    gt = {1: _gt(1), 2: _gt(2, (100, 100, 140, 130))}
    dets = {1: [Detection(10, 10, 50, 40, conf=0.9)], 2: []}
    res = evaluate(dets, gt)
    assert res.fp == 1 and res.tp == 0 and res.fn == 1


def test_miss_entirely():
    gt = {1: _gt(1, (100, 100, 140, 130))}
    res = evaluate({1: []}, gt)
    assert res.tp == 0 and res.fp == 0 and res.fn == 1
    assert res.recall == 0.0


def test_perfect_detection():
    gt = {f: _gt(f, (100, 100, 140, 130)) for f in range(1, 11)}
    dets = {f: [Detection(100, 100, 140, 130, conf=1.0)] for f in range(1, 11)}
    res = evaluate(dets, gt)
    assert res.precision == 1.0 and res.recall == 1.0 and res.f1 == 1.0
    assert res.ap("allpoint") == 1.0
    assert res.ap("11point") == 1.0


def test_ap_monotonic_and_bounds():
    gt = {1: _gt(1, (0, 0, 20, 20))}
    dets = {1: [Detection(0, 0, 20, 20, conf=0.5),
                Detection(200, 200, 220, 220, conf=0.9)]}
    res = evaluate(dets, gt)
    for mode in ("11point", "allpoint"):
        a = res.ap(mode)
        assert 0.0 <= a <= 1.0, f"{mode} AP 越界: {a}"


def test_count_histograms():
    gt = {1: _gt(1, (0, 0, 10, 10)), 2: _gt(2), 3: _gt(3, (0, 0, 10, 10), (50, 50, 60, 60))}
    dets = {1: [Detection(0, 0, 10, 10), Detection(100, 100, 110, 110)],
            2: [], 3: [Detection(0, 0, 10, 10)]}
    res = evaluate(dets, gt)
    assert res.gt_count_hist == {0: 1, 1: 1, 2: 1}
    assert res.det_count_hist == {0: 1, 1: 1, 2: 1}


def test_grouped_by_gt_count_reports_sample_size():
    gt = {1: _gt(1, (0, 0, 10, 10)),
          2: _gt(2, (0, 0, 10, 10), (50, 50, 70, 70))}
    dets = {1: [Detection(0, 0, 10, 10)], 2: [Detection(0, 0, 10, 10)]}
    res = evaluate(dets, gt)
    g = grouped_by_gt_count(res)
    assert g["1"]["frames"] == 1
    assert g["1"]["recall"] == 1.0
    assert g["2-3"]["frames"] == 1
    assert abs(g["2-3"]["recall"] - 0.5) < 1e-9


def test_summary_keys_present():
    gt = {1: _gt(1, (0, 0, 10, 10))}
    res = evaluate({1: [Detection(0, 0, 10, 10)]}, gt)
    s = summary(res)
    for k in ("precision", "recall", "f1", "ap_11point", "ap_allpoint",
              "det_count_hist", "gt_count_hist", "by_gt_count", "iou_thresh"):
        assert k in s, f"缺字段 {k}"
    assert s["iou_thresh"] == 0.25
    assert s["match_algo"] == "hungarian"


def test_no_gt_no_det_not_nan():
    res = evaluate({}, {})
    assert res.precision == 0.0 and res.recall == 0.0 and res.f1 == 0.0
    assert not math.isnan(res.ap("allpoint"))
