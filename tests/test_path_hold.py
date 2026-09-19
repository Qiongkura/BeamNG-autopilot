"""Bounded PATH_HOLD + current/planned crossing split (plan phases B/C1).

Offline regression for the safety-monitor degradation ladder: a tick that
loses the planner path may re-serve the last VERIFIED trajectory inside a
bounded window (re-checked against the current scene), and a planned body
crossing far ahead degrades instead of stopping the car dead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_autopilot.config import (
    FSD_PATH_HOLD_GRACE_S,
    FSD_PATH_HOLD_MAX_S,
    FSD_PLANNED_CROSS_HARD_M,
)
from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import Scene
from beamng_autopilot.safety_monitor import PathHold, SafetyMonitor

_T0 = 1000.0   # explicit test clock (evaluate now_s / offer now_s)


def _scene(strict: bool = True, left=None, right=None,
           pos=(0.0, 0.0), heading: float = 0.0) -> Scene:
    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    xs = np.linspace(0, 30, 31)
    route = np.column_stack([xs, np.zeros_like(xs)])
    return Scene(pos=np.asarray(pos, dtype=float), heading=heading,
                 grid=grid, route=route,
                 lane_ref=None if strict else route,
                 lane_left=left, lane_right=right,
                 strict_perception=strict)


def _straight(y: float = 0.0):
    return np.column_stack([np.linspace(0, 15, 20), np.full(20, y)])


# ---------------------------------------------------------------------------
# PathHold pure logic
# ---------------------------------------------------------------------------

def test_path_hold_rejects_unusable_offers() -> None:
    hold = PathHold()
    assert hold.offer(None, 0.0, 5.0, now_s=_T0) is False
    assert hold.offer(np.array([[0.0, 0.0], [np.nan, 1.0]]),
                      0.0, 5.0, now_s=_T0) is False
    # too short to steer with
    assert hold.offer(np.array([[0.0, 0.0], [1.0, 0.0]]),
                      0.0, 5.0, now_s=_T0) is False
    assert hold.active is False


def test_path_hold_phase_ladder() -> None:
    hold = PathHold()
    hold.offer(_straight(), 0.0, 5.0, now_s=_T0)
    served = hold.request((0.0, 0.0), _T0 + 0.1)
    assert served is not None
    held, age, phase = served
    assert phase == "grace"
    assert age == pytest.approx(0.1)
    assert np.allclose(held.path, _straight())
    served = hold.request((0.0, 0.0),
                          _T0 + FSD_PATH_HOLD_GRACE_S + 0.1)
    assert served is not None
    assert served[2] == "creep"
    # past the horizon the hold is cleared and cannot be revived
    assert hold.request((0.0, 0.0),
                        _T0 + FSD_PATH_HOLD_MAX_S + 0.1) is None
    assert hold.active is False


def test_path_hold_refuses_ego_drift_and_spent_path() -> None:
    hold = PathHold()
    hold.offer(_straight(), 0.0, 5.0, now_s=_T0)
    # ego drifted 3 m off the held path
    assert hold.request((0.0, 3.0), _T0 + 0.1) is None
    # ego near the path END: almost nothing ahead to steer with
    hold2 = PathHold()
    hold2.offer(np.column_stack([np.linspace(0, 10, 21), np.zeros(21)]),
                0.0, 5.0, now_s=_T0)
    assert hold2.request((9.5, 0.0), _T0 + 0.1) is None
    assert hold2.request((0.0, 0.0), _T0 + 0.1) is not None


# ---------------------------------------------------------------------------
# SafetyMonitor integration
# ---------------------------------------------------------------------------

def test_hold_serves_grace_phase_at_the_offered_target() -> None:
    mon = SafetyMonitor(max_speed=6.0)
    scene = _scene()
    assert mon.offer_verified_path(_straight(), 0.0, 5.0,
                                   now_s=_T0, strict=True)
    v = mon.evaluate(scene, None, now_s=_T0 + 0.1)
    assert v.level == "degraded"
    assert v.drivable
    assert v.path_hold_active is True
    assert v.path_hold_phase == "grace"
    assert v.reason == "path hold (grace)"
    assert v.target_speed == pytest.approx(5.0)
    assert np.allclose(np.asarray(v.held_path), _straight())


def test_hold_creeps_after_the_grace_window() -> None:
    mon = SafetyMonitor(max_speed=6.0)
    mon.offer_verified_path(_straight(), 0.0, 5.0, now_s=_T0, strict=True)
    v = mon.evaluate(_scene(), None,
                     now_s=_T0 + FSD_PATH_HOLD_GRACE_S + 0.1)
    assert v.level == "degraded"
    assert v.path_hold_phase == "creep"
    # the creep cap is the monitor's minimal-risk speed
    assert v.target_speed == pytest.approx(mon.min_risk_speed)


def test_hold_expires_into_a_minimal_risk_stop() -> None:
    mon = SafetyMonitor(max_speed=6.0)
    mon.offer_verified_path(_straight(), 0.0, 5.0, now_s=_T0, strict=True)
    v = mon.evaluate(_scene(), None,
                     now_s=_T0 + FSD_PATH_HOLD_MAX_S + 0.1)
    assert v.level == "minimal_risk"
    assert v.reason == "no drivable path"
    assert v.target_speed == 0.0
    assert v.path_hold_active is False
    assert mon.path_hold.active is False


def test_hold_refuses_when_the_current_body_crosses() -> None:
    left = np.array([[0.0, 0.5], [30.0, 0.5]])
    mon = SafetyMonitor(max_speed=6.0)
    mon.offer_verified_path(_straight(), 0.0, 5.0, now_s=_T0, strict=True)
    # the ego itself sits across the boundary - no replay may drive it
    scene = _scene(left=left, pos=(5.0, 0.4))
    v = mon.evaluate(scene, None, now_s=_T0 + 0.1)
    assert v.level == "minimal_risk"
    assert v.path_hold_active is False


def test_hold_refuses_when_the_held_path_now_crosses() -> None:
    # boundary at 1.5 m keeps the EGO's inflated body (1.15 m) clean, so
    # only the held path's own crossing can refuse the replay
    left = np.array([[0.0, 1.5], [30.0, 1.5]])
    mon = SafetyMonitor(max_speed=6.0)
    # offered path drifts toward the boundary and its body would cross it
    drifting = np.column_stack([np.linspace(0, 15, 20),
                                np.linspace(0.0, 0.9, 20)])
    mon.offer_verified_path(drifting, 0.0, 5.0, now_s=_T0, strict=True)
    scene = _scene(left=left)
    v = mon.evaluate(scene, None, now_s=_T0 + 0.1)
    assert v.level == "minimal_risk"
    assert v.path_hold_active is False


def test_hold_refuses_when_the_held_path_is_blocked() -> None:
    mon = SafetyMonitor(max_speed=6.0)
    mon.offer_verified_path(_straight(), 0.0, 5.0, now_s=_T0, strict=True)
    scene = _scene()
    scene.grid.mark_obstacle_region(6.0, 0.0, 6.0, 0.5)
    v = mon.evaluate(scene, None, now_s=_T0 + 0.1)
    assert v.level == "minimal_risk"
    assert v.path_hold_active is False


def test_hold_refuses_when_the_ego_left_the_held_path() -> None:
    mon = SafetyMonitor(max_speed=6.0)
    mon.offer_verified_path(_straight(), 0.0, 5.0, now_s=_T0, strict=True)
    v = mon.evaluate(_scene(pos=(0.0, 3.0)), None, now_s=_T0 + 0.1)
    assert v.level == "minimal_risk"
    assert v.path_hold_active is False


def test_no_offer_keeps_the_single_frame_stop() -> None:
    mon = SafetyMonitor(max_speed=6.0)
    v = mon.evaluate(_scene(), None, now_s=_T0)
    assert v.level == "minimal_risk"
    assert v.reason == "no drivable path"
    assert v.path_hold_active is False


# ---------------------------------------------------------------------------
# current/planned crossing split
# ---------------------------------------------------------------------------

def test_far_planned_crossing_degrades_instead_of_stopping() -> None:
    """A planned sweep crossing beyond the near/far bound caps the speed
    and lets the next tick re-plan; it must not stand the car dead."""
    # boundary starts 10 m ahead; the ego's INFLATED body (half width
    # 0.9 + monitor margin 0.25 = 1.15) stays inside the 1.5 m boundary
    left = np.array([[10.0, 1.5], [30.0, 1.5]])
    scene = _scene(strict=False, left=left)
    path = _straight(y=0.55)   # body (half width 0.9) clips it out ahead
    v = SafetyMonitor(max_speed=6.0).evaluate(scene, path)
    assert v.body_cross_current is False
    assert v.body_cross_planned is True
    assert v.first_crossing_distance_m is not None
    assert v.first_crossing_distance_m >= FSD_PLANNED_CROSS_HARD_M
    assert v.crossing_boundary_side == "left"
    assert v.crossing_path_index is not None and v.crossing_path_index >= 0
    assert v.level == "degraded"
    assert v.reason == "planned boundary crossing ahead"
    assert v.target_speed == pytest.approx(2.0 * 2.0)


def test_near_planned_crossing_still_stops() -> None:
    left = np.array([[0.0, 1.5], [30.0, 1.5]])
    scene = _scene(strict=False, left=left)
    path = _straight(y=0.55)
    v = SafetyMonitor(max_speed=6.0).evaluate(scene, path)
    assert v.body_cross_current is False
    assert v.body_cross_planned is True
    assert v.first_crossing_distance_m < FSD_PLANNED_CROSS_HARD_M
    assert v.level == "minimal_risk"
    assert v.reason == "planned vehicle body crosses lane boundary"
    assert v.target_speed == 0.0


def test_current_cross_sets_structured_fields() -> None:
    """A current body crossing reports itself as the CURRENT event."""
    left = np.array([[0.0, 4.0], [20.0, 4.0]])
    right = np.array([[0.0, -4.0], [20.0, -4.0]])
    scene = Scene(pos=np.array([10.0, 3.0]), heading=math.radians(18.0),
                  grid=OccupancyGrid(60, 60, 0.5),
                  route=np.array([[0.0, 0.0], [20.0, 0.0]]),
                  lane_ref=np.array([[0.0, 0.0], [20.0, 0.0]]),
                  lane_left=left, lane_right=right, lane_width=8.0)
    scene.lane_ref_src = "sensor"
    verdict = SafetyMonitor(max_speed=6.0).evaluate(
        scene, np.array([[10.0, 3.0], [16.0, 4.5]]))
    assert verdict.body_cross_current is True
    assert verdict.level == "minimal_risk"
    assert "vehicle body" in verdict.reason
