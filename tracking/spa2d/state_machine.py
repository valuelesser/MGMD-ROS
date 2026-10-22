"""spa2d.state_machine — 门控、状态转换与 reason code（SPEC §5.6 / §5.7）。

规范状态（7）：INIT, TRACKED, TRACKED_LOW, REACQ, LOST, OUT_OF_FRAME, ABSENT。
布尔属性：stat（接受且 max(|vx|,|vy|) ≤ v_stat）、edge（接受且 edge_candidate）。

接受判定按序：edge → 高置信 → 低置信；reason code 决策序见 `_reason`。
OUT_OF_FRAME 周期：每 reacq_max 个窗口执行一次 global，其余窗口跳过搜索。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spa2d.records import Flags

STATES = ("INIT", "TRACKED", "TRACKED_LOW", "REACQ", "LOST", "OUT_OF_FRAME", "ABSENT")
REASONS = ("OK", "EDGE_REJECT", "LOW_PNR", "LOW_MARGIN", "DISCONTINUOUS",
           "NO_PRIOR_GLOBAL", "LOST", "OUT_OF_FRAME", "ABSENT")
_SEARCH_MODES = ("local", "global")


@dataclass(frozen=True)
class LastAccepted:
    t_start_prev: int
    c_prev: float
    p_prev: np.ndarray
    v_prev: np.ndarray


@dataclass
class WindowPlan:
    window_index: int
    t_start: int
    t_center: float
    search_mode: str | None
    disposition: str
    prior_ok: bool
    skip_search: bool = False


@dataclass
class WindowObservation:
    pnr: float
    margin: float
    edge_candidate: bool
    continuity: Any  # records.Continuity
    p_center: tuple[float, float]
    v_best: tuple[float, float]
    single_candidate: bool = False
    peak_value: float | None = None


@dataclass
class WindowOutcome:
    window_index: int
    t_start: int
    t_center: float
    state_in: str
    state_out: str
    search_mode: str | None
    disposition: str
    accepted: bool
    target_present: bool
    reason: str
    flags: Flags
    miss_count: int
    reacq_count: int
    continuity_exempt: bool
    skipped_by_period: bool
    context: dict = field(default_factory=dict)


class StateMachine:
    def __init__(self, thresholds, stride: int, N: int, image_size: tuple[int, int],
                 initial_state: str = "INIT", last_accepted: LastAccepted | None = None,
                 miss_count: int = 0, reacq_count: int = 0,
                 ever_accepted: bool | None = None) -> None:
        if initial_state not in STATES:
            raise ValueError(f"未知状态 {initial_state!r}")
        self.th = thresholds
        self.stride = int(stride)
        self.N = int(N)
        self.width, self.height = int(image_size[0]), int(image_size[1])
        self.state = initial_state
        self.last_accepted = last_accepted
        self.miss_count = int(miss_count)
        self.reacq_count = int(reacq_count)
        self.ever_accepted = bool(ever_accepted) if ever_accepted is not None \
            else (last_accepted is not None)
        self._widx = -1
        self._plan: WindowPlan | None = None

    # ---------- prior / plan ----------
    def _prior_ok(self, t_start: int) -> bool:
        if self.last_accepted is None:
            return False
        if self.state in ("LOST", "OUT_OF_FRAME", "ABSENT"):
            return False
        gap = (t_start - self.last_accepted.t_start_prev) / self.stride
        return gap <= self.th.gap_prior_windows

    def prepare(self, t_start: int) -> WindowPlan:
        self._widx += 1
        t_center = t_start + (self.N - 1) / 2.0
        prior_ok = self._prior_ok(t_start)
        skip = self.state == "OUT_OF_FRAME" and (self._widx % self.th.reacq_max != 0)
        if skip:
            plan = WindowPlan(self._widx, t_start, t_center, None, "OUT_OF_FRAME",
                              prior_ok, skip_search=True)
        elif prior_ok:
            plan = WindowPlan(self._widx, t_start, t_center, "local", self.state, True)
        else:
            disposition = "REACQ" if self.ever_accepted else "INIT"
            plan = WindowPlan(self._widx, t_start, t_center, "global", disposition, False)
        self._plan = plan
        return plan

    # ---------- decisions ----------
    def _accept(self, obs: WindowObservation, plan: WindowPlan) -> tuple[str | None, bool]:
        th = self.th
        exempt = bool(obs.continuity.exempt)
        cont_high = True if exempt else obs.continuity.r_cont <= th.r_cont_high
        cont_low = True if exempt else obs.continuity.r_cont <= th.r_cont_low
        if obs.edge_candidate:
            return ("edge", True) if (obs.pnr >= th.theta_edge_pnr and cont_high) else (None, False)
        if obs.pnr >= th.theta_high_pnr:
            return ("high", obs.margin >= th.theta_margin and cont_high)
        if obs.pnr >= th.theta_low_pnr:
            ok = obs.margin >= th.theta_margin and plan.prior_ok and cont_low
            return ("low", ok)
        return (None, False)

    def _reason(self, obs: WindowObservation, plan: WindowPlan, accepted: bool) -> str:
        if accepted:
            return "OK"
        th = self.th
        exempt = bool(obs.continuity.exempt)
        r_lim = th.r_cont_high if obs.pnr >= th.theta_high_pnr else th.r_cont_low
        if obs.edge_candidate and obs.pnr < th.theta_edge_pnr:
            return "EDGE_REJECT"
        if obs.pnr < th.theta_low_pnr:
            return "LOW_PNR"
        if obs.margin < th.theta_margin:
            return "LOW_MARGIN"
        if (not exempt) and obs.continuity.r_cont is not None and obs.continuity.r_cont > r_lim:
            return "DISCONTINUOUS"
        if not plan.prior_ok:
            return "NO_PRIOR_GLOBAL"
        if self.state == "LOST":
            return "LOST"
        if self.state == "OUT_OF_FRAME":
            return "OUT_OF_FRAME"
        return "ABSENT"

    def _pred_out_of_frame(self, plan: WindowPlan) -> bool:
        if self.last_accepted is None:
            return False
        p = self.last_accepted.p_prev + self.last_accepted.v_prev * (
            plan.t_center - self.last_accepted.c_prev)
        return not (0.0 <= p[0] < self.width and 0.0 <= p[1] < self.height)

    # ---------- step ----------
    def step(self, obs: WindowObservation, plan: WindowPlan | None = None) -> WindowOutcome:
        plan = plan if plan is not None else self._plan
        if plan is None:
            raise RuntimeError("step 之前必须调用 prepare")
        state_in = self.state
        th = self.th

        if plan.skip_search:
            return WindowOutcome(
                plan.window_index, plan.t_start, plan.t_center, state_in, "OUT_OF_FRAME",
                None, "OUT_OF_FRAME", False, False, "OUT_OF_FRAME", Flags(),
                self.miss_count, self.reacq_count, False, True,
                context={"match": "OUT_OF_FRAME periodic skip"})

        kind, accepted = self._accept(obs, plan)
        reason = self._reason(obs, plan, accepted)
        stat = bool(accepted and max(abs(obs.v_best[0]), abs(obs.v_best[1])) <= th.v_stat)
        flags = Flags(stat=stat, edge=bool(accepted and obs.edge_candidate))

        if accepted:
            state_out = "TRACKED" if kind in ("high", "edge") else "TRACKED_LOW"
            self.miss_count = 0
            self.reacq_count = 0
            self.ever_accepted = True
            self.last_accepted = LastAccepted(
                t_start_prev=plan.t_start, c_prev=plan.t_center,
                p_prev=np.asarray(obs.p_center, np.float64),
                v_prev=np.asarray(obs.v_best, np.float64))
        else:
            state_out = self._on_reject(state_in, plan)

        self.state = state_out
        context = {
            "pnr": obs.pnr, "margin": obs.margin,
            "r_cont": obs.continuity.r_cont, "exempt": obs.continuity.exempt,
            "r_cont_high": obs.continuity.r_cont_high, "r_cont_low": obs.continuity.r_cont_low,
            "theta_high_pnr": th.theta_high_pnr, "theta_low_pnr": th.theta_low_pnr,
            "theta_margin": th.theta_margin, "theta_edge_pnr": th.theta_edge_pnr,
            "edge_candidate": obs.edge_candidate, "prior_ok": plan.prior_ok,
            "search_mode": plan.search_mode, "disposition": plan.disposition,
            "single_candidate": obs.single_candidate,
            "miss_count": self.miss_count, "reacq_count": self.reacq_count,
        }
        return WindowOutcome(
            plan.window_index, plan.t_start, plan.t_center, state_in, state_out,
            plan.search_mode, plan.disposition, accepted, accepted, reason, flags,
            self.miss_count, self.reacq_count, bool(obs.continuity.exempt), False, context)

    def _on_reject(self, state_in: str, plan: WindowPlan) -> str:
        th = self.th
        if state_in in ("TRACKED", "TRACKED_LOW"):
            self.miss_count += 1
            if self.miss_count >= th.m_lost:
                return "LOST"
            if plan.search_mode == "global" and not plan.prior_ok:
                self.reacq_count = 1
                return "REACQ"
            return state_in
        if state_in == "INIT":
            return "INIT"
        if state_in == "REACQ":
            self.reacq_count += 1
            if self.reacq_count < th.reacq_max:
                return "REACQ"
            return "OUT_OF_FRAME" if self._pred_out_of_frame(plan) else "LOST"
        if state_in == "LOST":
            return "OUT_OF_FRAME" if self._pred_out_of_frame(plan) else "LOST"
        if state_in == "OUT_OF_FRAME":
            return "LOST" if not self._pred_out_of_frame(plan) else "OUT_OF_FRAME"
        return "ABSENT"

    # ---------- convenience ----------
    def process(self, t_start: int, obs: WindowObservation) -> WindowOutcome:
        return self.step(obs, self.prepare(t_start))
