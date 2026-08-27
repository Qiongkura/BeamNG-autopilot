"""Offline tests for the FSD-style safety monitor."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import Scene
from beamng_autopilot.safety_monitor import SafetyMonitor


def _scene(obs_at=None, n=60, obs_half=1.0):
    grid = OccupancyGrid(n, n, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    if obs_at is not None:
        x, y = obs_at
        grid.mark_obstacle_region(x, y, obs_half, obs_half)
    xs = np.linspace(0, 30, 31)
    route = np.column_stack([xs, np.zeros_like(xs)])
    return Scene(pos=np.array([0.0, 0.0]), heading=0.0, grid=grid,
                 route=route, lane_ref=route)


def _straight():
    return np.column_stack([np.linspace(0, 15, 20), np.zeros(20)])


def test_safe_open_road() -> None:
    mon = SafetyMonitor(max_speed=12.0)
    v = mon.evaluate(_scene(), _straight())
    assert v.safe
    assert v.target_speed == pytest.approx(12.0)


def test_minimal_risk_when_blocked() -> None:
    mon = SafetyMonitor()
    # a wall spanning the whole forward corridor locks every path
    scene = _scene(obs_at=(4.0, 0.0), obs_half=50.0)
    v = mon.evaluate(scene, _straight())
    assert v.level == "minimal_risk"
    assert v.target_speed == 0.0
    assert "blocked" in v.reason


def test_scattered_cluster_degrades_not_stops() -> None:
    """A cluster of roadside poles leaves the corridor open: the monitor
    must degrade (slow, find another path) instead of a full stop."""
    mon = SafetyMonitor(occ_fraction_degrade=0.05,
                        occ_fraction_stop=0.5)
    scene = _scene()
    for y in (-4.0, -2.0, 2.0, 4.0):
        scene.grid.mark_obstacle_region(6.0, y, 0.4, 0.4)
    v = mon.evaluate(scene, _straight())
    assert v.level != "minimal_risk"


def test_degraded_when_obstacle_grazed() -> None:
    mon = SafetyMonitor(occ_fraction_degrade=0.02,
                        occ_fraction_stop=0.4)
    scene = _scene(obs_at=(6.0, 0.0))
    path = np.column_stack([np.linspace(0, 15, 30),
                            np.zeros(30)])
    v = mon.evaluate(scene, path)
    # straight path runs through the box -> a high occupied fraction
    assert v.level in ("minimal_risk", "degraded")


def test_stale_sensor_degrades() -> None:
    mon = SafetyMonitor(max_speed=12.0)
    v = mon.evaluate(_scene(), _straight(),
                     snapshot_age_s=2.0)
    assert v.degraded
    assert "stale" in v.reason


def test_no_path_minimal_risk() -> None:
    mon = SafetyMonitor()
    v = mon.evaluate(_scene(), None)
    assert v.level == "minimal_risk"
    assert v.target_speed == 0.0


def test_obstacle_approach_slows() -> None:
    mon = SafetyMonitor(max_speed=15.0)
    scene = _scene(obs_at=(5.0, 2.0))  # off to the side path still clears
    path = _straight()
    v = mon.evaluate(scene, path)
    # A roadside box with the forward corridor open is a lane bound, not
    # a blockage: the monitor stays safe (FSD keeps control) but eases
    # speed as the obstacle approaches (town 2026-08-21).
    assert v.safe
    assert 0.0 < v.target_speed < 15.0


def test_roadside_clutter_beside_lane_does_not_creep() -> None:
    """Continuous roadside trees/curbs beside the lane are lane bounds:
    with the forward corridor open they must NOT pin the target to the
    2 m/s creep (run 2026-08-27: plan 6 m/s, monitor crept all run).
    Only a corridor-intruding obstacle ahead of the ego eases speed."""
    mon = SafetyMonitor(max_speed=15.0)
    scene = _scene()
    # a wall of clutter 2 m beside the straight path (outside the 1.6 m
    # ease corridor), present along the whole approach
    for x in range(3, 28, 3):
        scene.grid.mark_obstacle_region(float(x), 2.5, 0.4, 0.4)
    v = mon.evaluate(scene, _straight())
    assert v.safe
    assert v.target_speed == pytest.approx(15.0)


def test_long_route_past_grid_horizon_not_occupied() -> None:
    """A nav-route reference extends beyond the sensor grid horizon.

    The grid only holds evidence inside its extent (unknown beyond);
    the monitor must not read the far tail of a long route as a wall
    and degrade a perfectly straight path (town stall 2026-08-21).
    """
    mon = SafetyMonitor(max_speed=8.0, occ_fraction_degrade=0.05,
                        occ_fraction_stop=0.4)
    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    xs = np.linspace(0, 120, 100)
    route = np.column_stack([xs, np.zeros_like(xs)])
    scene = Scene(pos=np.array([0.0, 0.0]), heading=0.0, grid=grid,
                  route=route, lane_ref=route)
    v = mon.evaluate(scene, route)
    assert v.safe


def test_path_inside_grid_still_blocked() -> None:
    """Real in-grid walls must still trip the safety monitor."""
    mon = SafetyMonitor(occ_fraction_degrade=0.05,
                        occ_fraction_stop=0.4)
    scene = _scene(obs_at=(5.0, 0.0), obs_half=1.0)
    path = np.column_stack([np.linspace(0, 10, 30), np.zeros(30)])
    v = mon.evaluate(scene, path)
    assert v.level in ("minimal_risk", "degraded")
