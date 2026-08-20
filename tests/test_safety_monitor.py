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
    # path at y=0 a couple of metres from the x=5 box - occupied fraction
    # stays small but the closest obstacle is near -> speed eases
    assert v.safe or v.degraded
    if v.safe:
        assert 0.0 < v.target_speed < 15.0