"""Candidate hysteresis (plan phase D1): decisions, not safety."""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_autopilot.planning.hysteresis import (
    REASON_DEGRADED,
    REASON_DWELL,
    REASON_EMERGENCY,
    REASON_FIRST,
    REASON_GONE,
    REASON_IMPROVED,
    REASON_MARGIN,
    CandidateHysteresis,
    candidate_key,
)
from beamng_autopilot.planning.trajectory import Candidate


def _cand(kind: str, **meta) -> Candidate:
    path = np.column_stack([np.linspace(0.0, 10.0, 11), np.zeros(11)])
    return Candidate(path=path, meta={"kind": kind, **meta})


def test_first_choice_switches_from_nothing() -> None:
    h = CandidateHysteresis()
    cost, cand, info = h.choose([(2.0, _cand("arc", steer=0.0))], now_s=1.0)
    assert cost == pytest.approx(2.0)
    assert info["reason"] == REASON_FIRST
    assert info["switch"] == 1
    assert info["n_switch"] == 1


def test_dwell_holds_against_a_cheaper_candidate() -> None:
    """Inside the dwell window the cost advantage is ignored."""
    h = CandidateHysteresis(min_dwell_s=0.7, cost_margin=0.1)
    h.choose([(2.0, _cand("arc", steer=0.0))], now_s=10.0)
    cost, cand, info = h.choose(
        [(1.0, _cand("arc", steer=0.1)), (2.0, _cand("arc", steer=0.0))],
        now_s=10.3)
    assert cand.meta["steer"] == 0.0          # held
    assert info["reason"] == REASON_DWELL
    assert info["switch"] == 0
    assert info["n_switch"] == 1


def test_switch_after_dwell_when_margin_exceeded() -> None:
    h = CandidateHysteresis(min_dwell_s=0.5, cost_margin=0.3)
    h.choose([(2.0, _cand("arc", steer=0.0))], now_s=10.0)
    cost, cand, info = h.choose(
        [(1.0, _cand("arc", steer=0.1)), (2.0, _cand("arc", steer=0.0))],
        now_s=10.6)
    assert cand.meta["steer"] == 0.1
    assert info["reason"] == REASON_IMPROVED
    assert info["n_switch"] == 2


def test_within_margin_keeps_the_current_candidate() -> None:
    """A marginally cheaper candidate does not win - that is the flapping."""
    h = CandidateHysteresis(min_dwell_s=0.1, cost_margin=0.6)
    h.choose([(2.0, _cand("arc", steer=0.0))], now_s=10.0)
    cost, cand, info = h.choose(
        [(1.7, _cand("arc", steer=0.1)), (2.0, _cand("arc", steer=0.0))],
        now_s=11.0)
    assert cand.meta["steer"] == 0.0
    assert info["reason"] == REASON_MARGIN
    assert info["n_switch"] == 1


def test_infeasible_previous_switches_immediately() -> None:
    """A candidate the scorer declined this tick is never held."""
    h = CandidateHysteresis(min_dwell_s=5.0)
    h.choose([(2.0, _cand("arc", steer=0.0))], now_s=10.0)
    cost, cand, info = h.choose([(3.0, _cand("arc", steer=0.2))], now_s=10.1)
    assert cand.meta["steer"] == 0.2
    assert info["reason"] == REASON_GONE
    assert info["n_switch"] == 2


def test_emergency_overrides_the_dwell_window() -> None:
    h = CandidateHysteresis(min_dwell_s=5.0)
    h.choose([(2.0, _cand("arc", steer=0.0))], now_s=10.0)
    cost, cand, info = h.choose(
        [(0.5, _cand("arc", steer=-0.2)), (2.0, _cand("arc", steer=0.0))],
        now_s=10.1, emergency=True)
    assert cand.meta["steer"] == -0.2
    assert info["reason"] == REASON_EMERGENCY


def test_previous_cost_explosion_switches() -> None:
    """Still feasible but much worse than when chosen -> switch."""
    h = CandidateHysteresis(min_dwell_s=0.1, cost_margin=0.6,
                            degrade_ratio=1.5)
    h.choose([(2.0, _cand("arc", steer=0.0))], now_s=10.0)
    cost, cand, info = h.choose(
        [(3.0, _cand("arc", steer=0.1)), (4.0, _cand("arc", steer=0.0))],
        now_s=11.0)
    assert cand.meta["steer"] == 0.1
    assert info["reason"] == REASON_DEGRADED


def test_age_and_switch_counter_report_for_telemetry() -> None:
    h = CandidateHysteresis(min_dwell_s=0.5, cost_margin=0.1)
    _, _, first = h.choose([(2.0, _cand("arc", steer=0.0))], now_s=100.0)
    # nothing was chosen before, so there is no previous age to report
    assert first["last_switch_age_s"] is None
    _, _, info = h.choose([(2.0, _cand("arc", steer=0.0))], now_s=103.0)
    assert info["age_s"] == pytest.approx(3.0)
    assert info["n_switch"] == 1
    assert info["n_hold"] == 1
    # a real switch reports how long the replaced candidate had been held
    _, _, sw = h.choose(
        [(1.0, _cand("arc", steer=0.1)), (2.0, _cand("arc", steer=0.0))],
        now_s=104.0)
    assert sw["reason"] == REASON_IMPROVED
    assert sw["last_switch_age_s"] == pytest.approx(4.0)
    assert sw["n_switch"] == 2


def test_candidate_key_uses_the_generator_parameters() -> None:
    assert candidate_key(_cand("arc", steer=0.1)) == ("arc", 0.1)
    assert candidate_key(_cand("lane_shift", offset=-1.5)) == (
        "lane_shift", -1.5)
    assert candidate_key(_cand("lane_center")) == ("lane_center",)
    assert candidate_key(_cand("arc", steer=0.1)) != candidate_key(
        _cand("arc", steer=0.2))


def test_empty_feasible_list_returns_none() -> None:
    h = CandidateHysteresis()
    assert h.choose([], now_s=1.0) is None


def test_digest_is_json_safe() -> None:
    import json
    h = CandidateHysteresis()
    _, _, info = h.choose([(2.0, _cand("arc", steer=0.0))], now_s=1.0)
    text = json.dumps(info)
    assert "first_choice" in text
    assert "nan" not in text.lower()
    assert math.isfinite(float(info["age_s"]))


def test_selector_holds_and_then_switches() -> None:
    """End to end through ``select_trajectory`` (plan D1 acceptance)."""
    from beamng_autopilot.planning import Scene, select_trajectory
    from beamng_autopilot.planning.trajectory import CandidateSet

    class _Cons:
        def score(self, scene, cand):
            return float(cand.meta["cost"]), True

    def _fan(straight_cost: float, left_cost: float) -> CandidateSet:
        s = CandidateSet(None)
        s.add(_cand("arc", steer=0.0).path, "arc", steer=0.0,
              cost=straight_cost)
        s.add(_cand("arc", steer=0.1).path, "arc", steer=0.1,
              cost=left_cost)
        return s

    scene = Scene(pos=np.array([0.0, 0.0]), heading=0.0)
    h = CandidateHysteresis(min_dwell_s=0.7, cost_margin=0.6)

    best, meta = select_trajectory(scene, _fan(2.0, 2.4), _Cons(),
                                   hysteresis=h, now_s=0.0)
    assert meta["kind"] == "arc"
    assert meta["hysteresis"]["reason"] == REASON_FIRST

    # a marginally cheaper alternative must not win inside the dwell
    _, meta = select_trajectory(scene, _fan(2.4, 1.0), _Cons(),
                                hysteresis=h, now_s=0.2)
    assert meta["hysteresis"]["reason"] == REASON_DWELL
    assert meta["why"] == "hysteresis_hold"
    assert meta["hysteresis"]["switch"] == 0

    # after the dwell a clearly cheaper candidate takes over
    _, meta = select_trajectory(scene, _fan(2.4, 1.0), _Cons(),
                                hysteresis=h, now_s=1.0)
    assert meta["hysteresis"]["reason"] == REASON_IMPROVED
    assert meta["hysteresis"]["switch"] == 1
    assert meta["cost"] == pytest.approx(1.0)
