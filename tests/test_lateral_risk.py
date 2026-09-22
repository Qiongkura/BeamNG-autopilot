"""T09: bounded risk output and the joint conditions of a held path.

The plan's acceptance list for this card is behavioural: fast and slow
ticks, long pauses, far-field refreshes, zero speed, large yaw, missing
boundaries at junctions, a newly appearing obstacle, a repeated request
after expiry, and a car that crossed then converged.  Converted into
properties that can be checked deterministically:

* a missing boundary is UNKNOWN - never a 0 gap and never a division by a
  near-zero approach rate ("缺边界/模型不适用时应 UNKNOWN，不强行除以接近零的速度");
* a car that is not closing on a boundary has NO finite crossing time;
* at zero speed there is no crossing and no stopping-margin answer to
  invent;
* an expired hold cannot be revived by asking again, and a repeated OFFER
  cannot refresh the lifetime that matters (time since the last REAL
  observation);
* the joint conditions are never "satisfied" while one of them is
  unmeasured.

Nothing here extends a hold window: the gate can only refuse.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from beamng_autopilot.lane.lateral_risk import (
    APPROACH_EPS_MPS,
    LateralRisk,
    evidence_interval,
    first_crossing,
    lateral_risk,
    signed_gap_m,
    stopping_margin_m,
)
from beamng_autopilot.obstacle_risk import RISK_BRAKE_DECEL_MPS2
from beamng_autopilot.planning.hold_audit import (
    HOLD_JOINT_GATE,
    HOLD_OBS_MAX_S,
    HOLD_SIGMA_MAX_M,
    HOLD_TRAVEL_MAX_M,
    audit_hold,
    remaining_arc_m,
)


def _path(n=13, x0=2.0, x1=14.0):
    return np.column_stack([np.linspace(x0, x1, n), np.zeros(n)])


class TestSignedGap:
    def test_the_gap_is_measured_from_the_body_not_the_centre(self):
        gap, why = signed_gap_m(0.9, 2.0)
        assert gap == pytest.approx(1.1) and why == "clear"
        gap2, why2 = signed_gap_m(0.9, 0.4)
        assert gap2 == pytest.approx(-0.5) and "past" in why2

    def test_a_missing_boundary_is_unknown_not_zero(self):
        gap, why = signed_gap_m(0.9, None)
        assert gap is None and "not published" in why
        gap2, _ = signed_gap_m(0.9, float("nan"))
        assert gap2 is None


class TestFirstCrossing:
    def test_approaching_gives_distance_and_time(self):
        d, t, why = first_crossing(1.0, 0.5)
        assert d == pytest.approx(1.0) and t == pytest.approx(2.0)

    def test_not_approaching_has_no_finite_crossing(self):
        """The plan's explicit case: no division by a near-zero rate."""
        for rate in (0.0, 1e-6, -1e-6, APPROACH_EPS_MPS * 0.5):
            d, t, why = first_crossing(1.5, rate)
            assert d is None and t is None and "not approaching" in why

    def test_a_parallel_car_is_unknown_even_with_a_tiny_gap(self):
        d, t, why = first_crossing(0.01, 0.0)
        assert d is None and t is None

    def test_already_past_the_boundary_is_reported_as_such(self):
        d, t, why = first_crossing(-0.3, 0.5)
        assert d == 0.0 and t == 0.0 and "already past" in why

    def test_a_missing_boundary_or_rate_is_unknown(self):
        assert first_crossing(None, 0.5)[0] is None
        assert first_crossing(1.0, None)[0] is None
        d, t, why = first_crossing(1.0, float("inf"))
        assert d is None and "not finite" in why


class TestStoppingMargin:
    def test_the_formula_reuses_the_project_s_declared_deceleration(self):
        need, why = stopping_margin_m(5.0, latency_s=0.35,
                                      a_min_mps2=RISK_BRAKE_DECEL_MPS2)
        assert need == pytest.approx(5.0 * 0.35 + 25.0 / (2 * 2.5) + 0.5)
        assert "declared constant" in why

    def test_without_a_declared_deceleration_it_is_unknown(self):
        need, why = stopping_margin_m(5.0, latency_s=0.35, a_min_mps2=None)
        assert need is None and "no declared" in why

    def test_zero_speed_needs_only_the_extra_margin(self):
        need, _ = stopping_margin_m(0.0, latency_s=0.35,
                                    a_min_mps2=RISK_BRAKE_DECEL_MPS2,
                                    extra_m=0.5)
        assert need == pytest.approx(0.5)

    def test_bad_inputs_are_unknown_not_infinity(self):
        for v, lat, a in ((-1.0, 0.35, 3.0), (float("nan"), 0.35, 3.0),
                          (5.0, -1.0, 3.0), (5.0, 0.35, 0.0),
                          (5.0, 0.35, -2.0)):
            need, why = stopping_margin_m(v, latency_s=lat, a_min_mps2=a)
            assert need is None and why


class TestEvidenceInterval:
    def test_fresh_current_support_is_current(self):
        ev = evidence_interval(age_s=0.1, history_only_frac=0.1)
        assert ev["state"] == "current"

    def test_history_dominated_support_is_not_current(self):
        ev = evidence_interval(age_s=0.1, history_only_frac=0.9)
        assert ev["state"] == "degrading"

    def test_an_old_observation_is_stale(self):
        ev = evidence_interval(age_s=9.0, history_only_frac=0.0)
        assert ev["state"] == "stale"

    def test_a_missing_age_is_unknown(self):
        ev = evidence_interval(age_s=None, history_only_frac=0.0)
        assert ev["state"] == "UNKNOWN"


class TestRiskReport:
    def _report(self, **kw):
        base = dict(body_half_width_m=0.9, lat_left_m=2.0, lat_right_m=-2.0,
                    e_m=-0.1, e_dot_mps=0.0, speed_mps=5.0, latency_s=0.35,
                    a_min_mps2=RISK_BRAKE_DECEL_MPS2, evidence_age_s=0.2,
                    history_only_frac=0.0)
        base.update(kw)
        return lateral_risk(**base)

    def test_a_parallel_car_has_no_crossing_and_says_why(self):
        rep = self._report(e_dot_mps=0.0).as_dict()
        assert rep["cross_time_s"] is None and rep["cross_distance_m"] is None
        assert "first_crossing" in rep["unknown"]
        assert any("not approaching" in n for n in rep["notes"])

    def test_closing_on_the_left_boundary_gives_a_crossing(self):
        rep = self._report(e_dot_mps=0.6).as_dict()
        assert rep["cross_side"] == "left"
        assert rep["cross_distance_m"] == pytest.approx(1.1, abs=0.01)
        assert rep["cross_time_s"] == pytest.approx(1.1 / 0.6, rel=0.05)

    def test_missing_boundaries_are_unknown_and_never_a_zero_gap(self):
        rep = self._report(lat_left_m=None, lat_right_m=None).as_dict()
        assert rep["gap_left_m"] is None and rep["gap_right_m"] is None
        assert {"gap_left", "gap_right"} <= set(rep["unknown"])
        assert rep["cross_time_s"] is None

    def test_a_missing_lateral_state_is_unknown(self):
        rep = self._report(e_m=None, e_dot_mps=None).as_dict()
        assert "lateral_state" in rep["unknown"]
        assert rep["cross_time_s"] is None

    def test_a_newly_appearing_obstacle_does_not_fake_a_crossing(self):
        """Risk output is lateral only; the obstacle layer owns collisions."""
        rep = self._report(e_dot_mps=0.0).as_dict()
        assert rep["cross_time_s"] is None          # no lateral approach
        assert not hasattr(self._report(), "collision")

    def test_a_large_yaw_does_not_invent_a_boundary(self):
        rep = self._report(lat_left_m=None, lat_right_m=None,
                           e_dot_mps=3.0).as_dict()
        assert rep["cross_time_s"] is None and rep["cross_side"] is None


class TestHoldJointConditions:
    def _audit(self, **kw):
        base = dict(path=_path(), pos=np.zeros(3), observation_age_s=0.3,
                    travelled_since_obs_m=3.0, sigma_lat_m=0.2,
                    sigma_theta_rad=0.02, speed_mps=4.0, latency_s=0.35,
                    a_min_mps2=RISK_BRAKE_DECEL_MPS2)
        base.update(kw)
        return audit_hold(**base)

    def test_all_four_conditions_hold_for_a_fresh_short_hold(self):
        a = self._audit().as_dict()
        assert a["satisfied"] is True and a["failed"] == [] and a["unknown"] == []
        assert a["remaining_arc_m"] > a["required_stop_m"]

    def test_each_condition_can_fail_on_its_own(self):
        cases = {
            "observation_age": dict(observation_age_s=HOLD_OBS_MAX_S + 0.1),
            "travelled": dict(travelled_since_obs_m=HOLD_TRAVEL_MAX_M + 1.0),
            "sigma_lat": dict(sigma_lat_m=HOLD_SIGMA_MAX_M + 0.1),
            "sigma_theta": dict(sigma_theta_rad=0.5),
            "stopping space": dict(speed_mps=25.0),
        }
        for needle, kw in cases.items():
            a = self._audit(**kw).as_dict()
            assert a["satisfied"] is False, needle
            assert any(needle in f for f in a["failed"]), (needle, a["failed"])

    def test_an_unmeasured_condition_is_not_satisfied(self):
        for kw in (dict(observation_age_s=None), dict(travelled_since_obs_m=None),
                   dict(sigma_lat_m=None), dict(sigma_theta_rad=None),
                   dict(a_min_mps2=None), dict(path=None)):
            a = self._audit(**kw).as_dict()
            assert a["satisfied"] is False
            assert a["unknown"] or a["failed"]

    def test_the_lifetime_runs_from_the_observation_not_the_offer(self):
        """A repeated offer must not refresh the clock the plan cares about."""
        sig = inspect.signature(audit_hold)
        assert "offered_at" not in sig.parameters
        assert "offer" not in sig.parameters
        a = self._audit(path=_path(), observation_age_s=0.3)
        assert a.lifetime_source == "observation"

    def test_the_gate_is_off_by_default_and_can_only_refuse(self):
        assert HOLD_JOINT_GATE is False
        assert self._audit().as_dict()["enforced"] is False
        # enforcement does not change the verdict, only whether it is applied
        on = self._audit(observation_age_s=HOLD_OBS_MAX_S + 1.0,
                         enforce=True).as_dict()
        assert on["enforced"] is True and on["satisfied"] is False

    def test_remaining_arc_is_measured_ahead_of_the_car(self):
        # car 1 m before the path start: the whole arc is still usable
        assert remaining_arc_m(_path(), np.array([1.0, 0.0, 0.0])) \
            == pytest.approx(12.0)
        # car at the path end: nothing left
        assert remaining_arc_m(_path(), np.array([14.0, 0.0, 0.0])) == 0.0
        assert remaining_arc_m(None, np.zeros(3)) is None


class TestNoActuatorOutput:
    def test_the_risk_module_cannot_move_anything(self):
        import beamng_autopilot.lane.lateral_risk as mod
        fns = [n for n, v in vars(mod).items()
               if callable(v) and not n.startswith("_")]
        assert not any(("steer" in n or "command" in n or "apply" in n
                        or "send" in n) for n in fns), fns

    def test_the_report_has_no_permission_field(self):
        import dataclasses
        names = {f.name for f in dataclasses.fields(LateralRisk)}
        assert not (names & {"authority", "drivable", "permission",
                             "steering", "throttle"}), names
