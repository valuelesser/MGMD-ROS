"""test_state_machine — 门控/转换表/reason 全组合（SPEC §5.6，I-10 验收）。"""
from __future__ import annotations

import numpy as np
import pytest

from spa2d import state_machine as SM
from spa2d.config import Thresholds
from spa2d.records import Continuity


def _th(**kw):
    base = dict(theta_high_pnr=8.0, theta_low_pnr=5.0, theta_margin=0.02,
                theta_edge_pnr=10.0, v_stat=0.05, m_lost=3, reacq_max=5,
                gap_prior_windows=2, rho=16, r_cont_high=25.0, r_cont_low=10.0,
                width_ref=8.0, kernel_size=5)
    base.update(kw)
    return Thresholds(**base)


def _obs(pnr=9.0, margin=0.03, edge=False, r_cont=3.0, exempt=False,
         v=(0.7, 0.0), p=(10.0, 20.0), single=False):
    cont = Continuity(r_cont=None if exempt else r_cont, r_cont_high=25.0,
                      r_cont_low=10.0, exempt=exempt)
    return SM.WindowObservation(pnr=pnr, margin=margin, edge_candidate=edge,
                                continuity=cont, p_center=p, v_best=v,
                                single_candidate=single)


def _last(p_prev=(10.0, 20.0), v_prev=(0.7, 0.0), t0=0):
    return SM.LastAccepted(t_start_prev=t0, c_prev=t0 + 2.0,
                           p_prev=np.array(p_prev, np.float64),
                           v_prev=np.array(v_prev, np.float64))


def _sm(state="INIT", ever=None, last=None, th=None, **kw):
    return SM.StateMachine(th or _th(), stride=kw.pop("stride", 5), N=kw.pop("N", 5),
                           image_size=kw.pop("image_size", (100, 100)),
                           initial_state=state, last_accepted=last,
                           miss_count=kw.pop("miss_count", 0),
                           reacq_count=kw.pop("reacq_count", 0), ever_accepted=ever)


# ---------------- prepare / prior ----------------

def test_prepare_init():
    sm = _sm("INIT", ever=False)
    plan = sm.prepare(0)
    assert plan.search_mode == "global" and plan.disposition == "INIT"
    assert plan.prior_ok is False and plan.t_center == 2.0


def test_prepare_prior_local_and_gap_reacq():
    sm = _sm("TRACKED", last=_last(t0=0))
    assert sm.prepare(5).search_mode == "local"          # gap=1
    assert sm.prepare(10).search_mode == "local"         # gap=2
    p = sm.prepare(20)                                    # gap=4 > 2
    assert p.search_mode == "global" and p.disposition == "REACQ"


def test_oof_periodic_skip():
    sm = _sm("OUT_OF_FRAME", ever=True, last=_last(), th=_th(reacq_max=5))
    assert sm.prepare(0).skip_search is False            # widx0 % 5 == 0 执行
    p1 = sm.prepare(5)
    assert p1.skip_search is True
    out = sm.step(_obs(), p1)
    assert out.reason == "OUT_OF_FRAME" and out.skipped_by_period
    assert out.state_out == "OUT_OF_FRAME" and out.target_present is False


# ---------------- acceptance / reason ----------------

def test_edge_accept_and_reject():
    sm = _sm("TRACKED", last=_last())
    out = sm.process(5, _obs(pnr=11.0, margin=0.0, edge=True, r_cont=3.0))
    assert out.accepted and out.state_out == "TRACKED" and out.reason == "OK"
    assert out.flags.edge is True

    sm2 = _sm("TRACKED", last=_last())
    out2 = sm2.process(5, _obs(pnr=9.0, margin=0.0, edge=True, r_cont=3.0))
    assert not out2.accepted and out2.reason == "EDGE_REJECT"


def test_high_confidence_paths():
    sm = _sm("TRACKED", last=_last())
    assert sm.process(5, _obs(pnr=9.0, margin=0.03)).accepted
    sm2 = _sm("TRACKED", last=_last())
    assert sm2.process(5, _obs(pnr=9.0, margin=0.01)).reason == "LOW_MARGIN"
    sm3 = _sm("TRACKED", last=_last())
    assert sm3.process(5, _obs(pnr=9.0, margin=0.03, r_cont=30.0)).reason == "DISCONTINUOUS"


def test_low_confidence_accept_and_reject():
    sm = _sm("TRACKED", last=_last())
    out = sm.process(5, _obs(pnr=6.0, margin=0.03, r_cont=5.0))
    assert out.accepted and out.state_out == "TRACKED_LOW" and out.reason == "OK"

    sm2 = _sm("INIT", ever=False)
    out2 = sm2.process(0, _obs(pnr=6.0, margin=0.03, exempt=True))
    assert not out2.accepted and out2.reason == "NO_PRIOR_GLOBAL" and out2.state_out == "INIT"

    sm3 = _sm("TRACKED", last=_last())
    assert sm3.process(5, _obs(pnr=6.0, margin=0.03, r_cont=12.0)).reason == "DISCONTINUOUS"


def test_low_pnr_reason_and_priority():
    sm = _sm("TRACKED", last=_last())
    assert sm.process(5, _obs(pnr=4.0, exempt=False, r_cont=3.0)).reason == "LOW_PNR"
    # 优先级：EDGE_REJECT 先于 LOW_MARGIN
    sm2 = _sm("TRACKED", last=_last())
    assert sm2.process(5, _obs(pnr=9.0, margin=0.0, edge=True, r_cont=3.0)).reason == "EDGE_REJECT"


def test_stat_flag():
    sm = _sm("TRACKED", last=_last())
    out = sm.process(5, _obs(pnr=9.0, margin=0.03, v=(0.03, 0.0)))
    assert out.accepted and out.flags.stat is True
    sm2 = _sm("TRACKED", last=_last())
    assert sm2.process(5, _obs(pnr=9.0, margin=0.03, v=(0.1, 0.0))).flags.stat is False


# ---------------- transitions ----------------

def test_tracked_miss_to_lost():
    sm = _sm("TRACKED", last=_last(t0=0))
    rej = _obs(pnr=4.0, exempt=False, r_cont=3.0)
    o1 = sm.process(5, rej)
    assert o1.state_out == "TRACKED" and sm.miss_count == 1
    sm.process(10, rej)
    assert sm.miss_count == 2
    o3 = sm.process(15, rej)
    assert o3.state_out == "LOST" and o3.reason == "LOW_PNR"


def test_accept_clears_counters():
    sm = _sm("TRACKED", last=_last(), miss_count=2, reacq_count=4)
    out = sm.process(5, _obs(pnr=9.0, margin=0.03))
    assert out.accepted and sm.miss_count == 0 and sm.reacq_count == 0
    assert sm.last_accepted is not None and sm.last_accepted.t_start_prev == 5


def test_reacq_row_to_lost_and_oof():
    th = _th(reacq_max=3)
    inside = _sm("REACQ", ever=True, th=th, last=_last(t0=-1000, p_prev=(50.0, 50.0), v_prev=(0.0, 0.0)))
    rej = _obs(pnr=4.0, exempt=False, r_cont=3.0)
    s = [inside.process(0, rej).state_out, inside.process(5, rej).state_out,
         inside.process(10, rej).state_out]
    assert s == ["REACQ", "REACQ", "LOST"]

    outside = _sm("REACQ", ever=True, th=th,
                  last=_last(t0=-1000, p_prev=(500.0, 500.0), v_prev=(0.0, 0.0)))
    outs = [outside.process(0, rej).state_out for _ in range(3)]
    assert outs[-1] == "OUT_OF_FRAME"


def test_lost_row_pred_in_and_out_of_frame():
    sm = _sm("LOST", ever=True, last=_last(t0=-1000, p_prev=(50.0, 50.0), v_prev=(0.0, 0.0)))
    assert sm.process(0, _obs(pnr=4.0, exempt=False, r_cont=3.0)).state_out == "LOST"
    sm2 = _sm("LOST", ever=True, last=_last(t0=-1000, p_prev=(-500.0, 50.0), v_prev=(0.0, 0.0)))
    assert sm2.process(0, _obs(pnr=4.0, exempt=False, r_cont=3.0)).state_out == "OUT_OF_FRAME"


def test_oof_execute_reject_and_accept():
    th = _th(reacq_max=1)  # 每窗都执行
    sm = _sm("OUT_OF_FRAME", ever=True, th=th,
             last=_last(t0=-1000, p_prev=(50.0, 50.0), v_prev=(0.0, 0.0)))
    assert sm.process(0, _obs(pnr=4.0, exempt=False, r_cont=3.0)).state_out == "LOST"

    sm2 = _sm("OUT_OF_FRAME", ever=True, th=th, last=_last())
    out = sm2.process(0, _obs(pnr=9.0, margin=0.03, r_cont=3.0))
    assert out.accepted and out.state_out == "TRACKED" and sm2.miss_count == 0


def test_chain_lost_oof_tracked():
    th = _th(reacq_max=1)
    sm = _sm("LOST", ever=True, th=th,
             last=_last(t0=-1000, p_prev=(-500.0, 50.0), v_prev=(0.0, 0.0)))
    assert sm.process(0, _obs(pnr=4.0, exempt=False, r_cont=3.0)).state_out == "OUT_OF_FRAME"
    out = sm.process(5, _obs(pnr=11.0, margin=0.0, edge=True, r_cont=3.0))
    assert out.state_in == "OUT_OF_FRAME" and out.state_out == "TRACKED"


def test_low_then_high_accept():
    sm = _sm("TRACKED", last=_last())
    assert sm.process(5, _obs(pnr=6.0, margin=0.03, r_cont=5.0)).state_out == "TRACKED_LOW"
    assert sm.process(10, _obs(pnr=9.0, margin=0.03, r_cont=3.0)).state_out == "TRACKED"


# ---------------- 结构/常量 ----------------

def test_reason_codes_membership_and_context():
    sm = _sm("TRACKED", last=_last())
    out = sm.process(5, _obs(pnr=4.0, exempt=False, r_cont=3.0))
    assert out.reason in SM.REASONS
    for key in ("pnr", "margin", "r_cont", "theta_high_pnr", "theta_margin",
                "prior_ok", "search_mode", "miss_count", "reacq_count", "exempt"):
        assert key in out.context
    assert set(SM.STATES) == {"INIT", "TRACKED", "TRACKED_LOW", "REACQ", "LOST",
                              "OUT_OF_FRAME", "ABSENT"}


def test_exempt_only_first_global_window():
    sm = _sm("INIT", ever=False)
    out = sm.process(0, _obs(pnr=9.0, margin=0.03, exempt=True))  # INIT 首窗豁免
    assert out.accepted and out.continuity_exempt is True and out.state_out == "TRACKED"
    sm2 = _sm("TRACKED", last=_last())
    out2 = sm2.process(5, _obs(pnr=9.0, margin=0.03, r_cont=30.0, exempt=False))
    assert out2.continuity_exempt is False and out2.reason == "DISCONTINUOUS"
