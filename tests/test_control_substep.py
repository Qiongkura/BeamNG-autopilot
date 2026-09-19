"""Control sub-step policy (plan phases A2/A4): pure decision tests."""

from __future__ import annotations

import pytest

from beamng_autopilot.control.substep import ControlSubstep


def _policy(stale_s: float = 1.2) -> ControlSubstep:
    return ControlSubstep(stale_plan_s=stale_s)


def test_fresh_plan_keeps_driving_with_the_cached_target() -> None:
    d = _policy().decide(plan_age_s=0.3, target_speed=6.0)
    assert d.drive is True
    assert d.target_speed == pytest.approx(6.0)
    assert d.reason == ""


def test_stale_plan_brakes_instead_of_driving_bind() -> None:
    """Perception stopped refreshing the plan: the sub-step must stop.

    Without this the decoupled loop would keep re-issuing the last plan
    forever, which is the "drove on a stale plan" failure the tick-side
    STALE_CTRL_S guard exists to prevent.
    """
    d = _policy().decide(plan_age_s=1.5, target_speed=6.0)
    assert d.ok is True
    assert d.stop is True
    assert d.reason == "stale plan"
    assert d.target_speed == 0.0


def test_contact_risk_stops_the_substep() -> None:
    d = _policy().decide(plan_age_s=0.2, target_speed=6.0, risk_stop=True)
    assert d.stop is True
    assert d.reason == "obstacle contact risk"


def test_risk_cap_lowers_the_target_but_keeps_driving() -> None:
    d = _policy().decide(plan_age_s=0.2, target_speed=6.0, risk_cap=3.0)
    assert d.drive is True
    assert d.target_speed == pytest.approx(3.0)


def test_risk_cap_never_raises_the_target() -> None:
    d = _policy().decide(plan_age_s=0.2, target_speed=2.0, risk_cap=9.0)
    assert d.target_speed == pytest.approx(2.0)


def test_body_crossing_defers_to_the_full_tick() -> None:
    """A crossing is the monitor's call - a sub-step must not brake it.

    Braking here would fight the convergence recovery that steers the
    car back inside, reproducing the frozen-across-the-line failure the
    recovery rule exists to fix.
    """
    d = _policy().decide(plan_age_s=0.2, target_speed=6.0,
                         pose_crosses=True)
    assert d.ok is False
    assert d.drive is False
    assert d.stop is False
    assert d.reason == "body crosses lane boundary"


def test_stale_plan_outranks_everything_else() -> None:
    d = _policy().decide(plan_age_s=9.0, target_speed=6.0,
                         pose_crosses=True, risk_stop=True)
    assert d.stop is True
    assert d.reason == "stale plan"


def test_zero_substep_rate_style_bounds_are_configurable() -> None:
    assert _policy(stale_s=5.0).decide(plan_age_s=2.0, target_speed=4.0).drive
    assert not _policy(stale_s=0.5).decide(
        plan_age_s=0.6, target_speed=4.0).drive
