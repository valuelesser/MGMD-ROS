"""test_confidence — PNR/margin/连续性（SPEC §5.5，I-09 验收）。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spa2d import confidence as C
from spa2d.config import effective_thresholds, load_config, load_frozen
from spa2d.records import Continuity

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_rho_and_cont_radii_formulas():
    assert C.rho_for(8) == 16
    assert C.rho_for(4) == 8
    assert C.rho_for(3) == 8
    assert C.rho_for(0) == 8
    assert C.rho_for(4.1) == math.ceil(8.2)
    assert C.cont_radii(8) == (25.0, 12.0)
    assert C.cont_radii(20) == (60.0, 30.0)
    assert C.cont_radii(3) == (25.0, 10.0)


def test_margin_exclusion_radius():
    assert C.margin_exclusion_radius(0.1) == pytest.approx(0.2)
    assert C.margin_exclusion_radius(0.25) == pytest.approx(0.5)
    assert C.margin_exclusion_radius(0.0) == pytest.approx(0.2)


def test_background_stats_hand_computed():
    A = np.arange(400, dtype=np.float32).reshape(20, 20)
    M = np.ones((20, 20), dtype=np.float32)
    mu, sigma, count = C.background_stats(A, M, (10, 10), rho=3)
    vals = []
    for y in range(20):
        for x in range(20):
            if (x - 10) ** 2 + (y - 10) ** 2 <= 9:
                continue
            vals.append(float(A[y, x]))
    exp_mu = sum(vals) / len(vals)
    exp_sigma = math.sqrt(sum((v - exp_mu) ** 2 for v in vals) / len(vals))
    assert count == len(vals) >= C.INSUFFICIENT_BG
    assert mu == pytest.approx(exp_mu, abs=1e-9)
    assert sigma == pytest.approx(exp_sigma, abs=1e-9)


def test_background_stats_respects_mask():
    A = np.ones((24, 24), dtype=np.float32)
    M = np.zeros((24, 24), dtype=np.float32)
    M[4:20, 4:20] = 1.0  # 16x16 = 256 有效像素
    mu, sigma, count = C.background_stats(A, M, (12, 12), rho=3)
    disk = sum(1 for y in range(24) for x in range(24)
               if M[y, x] >= 1 and (x - 12) ** 2 + (y - 12) ** 2 <= 9)
    assert count == 256 - disk
    assert count >= C.INSUFFICIENT_BG
    assert mu == pytest.approx(1.0)
    assert sigma == pytest.approx(0.0)


def test_background_stats_insufficient_warns():
    A = np.zeros((8, 8), dtype=np.float32)
    M = np.ones((8, 8), dtype=np.float32)
    with pytest.warns(UserWarning):
        mu, sigma, count = C.background_stats(A, M, (4, 4), rho=2)
    assert count < C.INSUFFICIENT_BG
    assert (mu, sigma) == (0.0, 0.0)
    with pytest.warns(UserWarning):
        assert C.peak_pnr(A, M, (4, 4), rho=2) == 0.0


def test_pnr_formula():
    assert C.pnr(5.0, 1.0, 0.5) == pytest.approx(8.0)
    assert C.pnr(5.0, 1.0, 0.0) == pytest.approx(4.0 / C.SIGMA_EPS)


def test_peak_pnr_uniform_background():
    A = np.zeros((40, 40), dtype=np.float32)
    A[20, 20] = 5.0
    M = np.ones((40, 40), dtype=np.float32)
    val = C.peak_pnr(A, M, (20, 20), rho=5)
    assert val == pytest.approx(5.0 / C.SIGMA_EPS)


def test_velocity_margin_isolated():
    cands = [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5)]
    scores = [5.0, 3.0, 4.0, 2.0]
    margin, single = C.velocity_margin(scores, cands, (0.0, 0.0), delta_used=0.1)
    assert single is False
    assert margin == pytest.approx((5.0 - 4.0) / 5.0)


def test_velocity_margin_single_candidate():
    margin, single = C.velocity_margin([5.0], [(0.0, 0.0)], (0.0, 0.0), delta_used=0.1)
    assert (margin, single) == (1.0, True)


def test_velocity_margin_all_near():
    cands = [(0.0, 0.0), (0.1, 0.0), (-0.1, 0.1)]  # 均在 radius=0.2 内
    scores = [5.0, 4.0, 3.0]
    margin, single = C.velocity_margin(scores, cands, (0.0, 0.0), delta_used=0.1)
    assert (margin, single) == (1.0, True)


def test_velocity_margin_exclusion_radius_scales():
    cands = [(0.0, 0.0), (0.4, 0.0), (0.6, 0.0)]
    scores = [5.0, 4.9, 4.0]
    # δ=0.1 → radius 0.2：0.4 保留 → S2=4.9
    m1, s1 = C.velocity_margin(scores, cands, (0.0, 0.0), delta_used=0.1)
    assert s1 is False and m1 == pytest.approx((5.0 - 4.9) / 5.0)
    # δ=0.25 → radius 0.5：0.4 被排除，0.6 保留 → S2=4.0
    m2, s2 = C.velocity_margin(scores, cands, (0.0, 0.0), delta_used=0.25)
    assert s2 is False and m2 == pytest.approx((5.0 - 4.0) / 5.0)


def test_continuity_no_prior():
    c = C.continuity((10.0, 20.0), c=2.0, width_ref=8.0, prior=None)
    assert isinstance(c, Continuity)
    assert c.exempt is True
    assert c.r_cont is None
    assert (c.r_cont_high, c.r_cont_low) == (25.0, 12.0)


def test_continuity_with_prior():
    prior = C.Prior(c_prev=0.0, p_prev=np.array([10.0, 20.0]), v_prev=np.array([0.7, -0.3]))
    on = C.continuity((11.4, 19.4), c=2.0, width_ref=8.0, prior=prior)
    assert on.exempt is False
    assert on.r_cont == pytest.approx(0.0, abs=1e-12)
    off = C.continuity((12.0, 20.0), c=2.0, width_ref=20.0, prior=prior)
    assert off.r_cont == pytest.approx(math.sqrt(0.6 ** 2 + 0.6 ** 2))
    assert (off.r_cont_high, off.r_cont_low) == (60.0, 30.0)


def _frozen_doc():
    return {
        "schema_version": 1, "frozen_at": "2026-10-10T00:00:00Z",
        "calibration": {}, 
        "thresholds": {"theta_high_pnr": 8.0, "theta_low_pnr": 5.0, "theta_margin": 0.02,
                       "theta_edge_pnr": 10.0, "v_stat": 0.05, "m_lost": 3,
                       "reacq_max": 5, "gap_prior_windows": 2},
        "width_ref": {"D1/cam2": 12.0}, "preprocess": {"kernel_size": 5},
    }


def test_config_reuses_centralized_formulas(write_yaml):
    frozen_path = write_yaml(_frozen_doc(), "thresholds_frozen.yaml")
    frozen = load_frozen(frozen_path)
    cfg = load_config(REPO / "configs" / "D1_no_overlap.yaml")
    th = effective_thresholds(cfg, frozen, width_ref=12.0)
    assert th.rho == C.rho_for(12.0)
    assert (th.r_cont_high, th.r_cont_low) == C.cont_radii(12.0)
