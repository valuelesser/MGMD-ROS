"""test_peak_subpixel — 峰值/亚像素/边缘保护（SPEC §5.4/§5.3，I-05 验收）。"""
from __future__ import annotations

import numpy as np
import pytest

from spa2d import peak as PK
from spa2d import records as R


def _parabola(shape, center, scale=1.0):
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    return -(scale * ((x - center[0]) ** 2 + (y - center[1]) ** 2))


def test_subpixel_refine_parabola():
    A = _parabola((16, 20), (10.27, 7.6))
    offs = PK.subpixel_refine(A, (10, 8))
    assert offs is not None
    assert offs == pytest.approx((0.27, -0.4), abs=1e-3)


def test_extract_peak_parabola_recovery():
    A = _parabola((16, 20), (10.27, 7.6))
    M = np.full(A.shape, 5.0)
    res = PK.extract_peak(A, M, N=5, edge_margin=2)
    assert res.subpixel_used is True
    assert res.xy_int == (10, 8)
    assert res.xy_sub == pytest.approx((10.27, 7.6), abs=1e-3)
    assert res.valid_count == 5
    assert res.edge_candidate is False


def test_extract_peak_tie_lexicographic():
    A = np.zeros((6, 7))
    A[1, 2] = A[3, 4] = 5.0
    res = PK.extract_peak(A, np.full(A.shape, 1.0), N=5, edge_margin=1)
    assert res.xy_int == (2, 1)  # (x,y)：先比 y，再比 x
    B = np.zeros((6, 7))
    B[2, 2] = B[2, 4] = 7.0
    res2 = PK.extract_peak(B, np.full(B.shape, 1.0), N=5, edge_margin=1)
    assert res2.xy_int == (2, 2)


def test_subpixel_refine_border_returns_none():
    A = _parabola((16, 20), (0.0, 0.0))
    assert PK.subpixel_refine(A, (0, 0)) is None
    assert PK.subpixel_refine(A, (19, 0)) is None
    res = PK.extract_peak(A, np.full(A.shape, 5.0), N=5, edge_margin=2)
    assert res.subpixel_used is False
    assert res.xy_sub == res.xy_int


def test_subpixel_refine_degenerate_axes():
    A = np.zeros((5, 5))
    A[2, 1], A[2, 2], A[2, 3] = 1.0, 3.0, 5.0   # 线性 → denom≈0 → fx=0
    A[1, 2], A[3, 2] = 1.0, 5.0                  # 也线性 → fy=0
    assert PK.subpixel_refine(A, (2, 2)) == (0.0, 0.0)

    B = np.zeros((5, 5))
    B[2, 1], B[2, 2], B[2, 3] = 10.0, 9.0, 0.0   # offset 超出 0.5 → 回退 0
    B[1, 2], B[3, 2] = 1.0, 5.0
    fx, fy = PK.subpixel_refine(B, (2, 2))
    assert fx == 0.0


def test_extract_peak_requires_all_valid_mask():
    A = _parabola((16, 20), (10.27, 7.6))
    M = np.full(A.shape, 5.0)
    M[8, 9] = 0.0  # 3×3 含 invalid
    res = PK.extract_peak(A, M, N=5, edge_margin=2)
    assert res.subpixel_used is False
    assert res.xy_sub == res.xy_int


def test_extract_peak_requires_strict_max():
    A = _parabola((16, 20), (10.0, 8.0))
    A[8, 11] = A[8, 10]  # 邻居与中心并列 → 非严格最大
    res = PK.extract_peak(A, np.full(A.shape, 5.0), N=5, edge_margin=2)
    assert res.subpixel_used is False


def test_edge_candidate_band_and_mask():
    A = _parabola((16, 20), (2.0, 8.0))
    res = PK.extract_peak(A, np.full(A.shape, 5.0), N=5, edge_margin=5)
    assert res.edge_candidate is True

    A2 = _parabola((16, 20), (10.0, 8.0))
    M2 = np.full(A2.shape, 5.0)
    M2[8, 10] = 4.0  # valid_count < N
    res2 = PK.extract_peak(A2, M2, N=5, edge_margin=2)
    assert res2.edge_candidate is True
    assert res2.valid_count == 4


@pytest.mark.parametrize("N,v,expected", [
    (5, (0.0, 0.0), 2),
    (5, (1.0, 0.0), 5),
    (5, (0.0, -1.0), 5),
    (5, (0.2, 0.1), 2),
    (8, (2.0, 0.0), 15),
    (8, (0.2, 0.0), 3),
    (8, (15.0, 0.0), 106),
])
def test_edge_margin_formula(N, v, expected):
    assert PK.edge_margin_for(N, v) == expected


@pytest.mark.parametrize("N,expected", [(5, (11.4, 19.6)), (8, (12.45, 19.3))])
def test_center_frame_position(N, expected):
    got = PK.center_frame_position((10.0, 20.0), (0.7, -0.2), N)
    assert got == pytest.approx(expected, abs=1e-9)


def test_peak_result_fields_and_roundtrip():
    A = _parabola((16, 20), (10.27, 7.6))
    res = PK.extract_peak(A, np.full(A.shape, 5.0), N=5, edge_margin=2)
    assert isinstance(res, R.PeakResult)
    assert R.PeakResult.from_dict(res.to_dict()) == res
    assert set(res.to_dict()) == {"value", "xy_int", "xy_sub", "subpixel_used",
                                  "valid_count", "edge_candidate"}


def test_extract_peak_shape_mismatch():
    with pytest.raises(ValueError):
        PK.extract_peak(np.zeros((4, 4)), np.zeros((4, 5)), N=5, edge_margin=1)
