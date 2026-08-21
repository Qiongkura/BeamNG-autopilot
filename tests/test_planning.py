"""Offline tests for the FSD-style layered planner (planning/)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import (
    Constraints,
    CandidateSet,
    Scene,
    cost_collision,
    cost_curvature,
    cost_lane_align,
    corridor_free_band,
    sample_arc,
    sample_lane_shift,
    select_trajectory,
)
from beamng_autopilot.perception import Obstacle


def _scene(grid_origin=(0.0, 0.0), heading=0.0, route=None):
    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = grid_origin
    grid.heading = heading
    xs = np.linspace(0, 30, 31)
    if route is None:
        route = np.column_stack([xs, np.zeros_like(xs)])
    return Scene(pos=np.array([0.0, 0.0]), heading=heading,
                 grid=grid, route=route)


def test_arc_fan_produces_multiple_candidates() -> None:
    cs = sample_arc([0.0, 0.0], 0.0, speed=10.0, max_steer=0.5)
    assert len(cs) >= 5
    for c in cs.candidates:
        assert c.path.shape[0] >= 3
        assert c.path.shape[1] == 2
    # the middle (near-straight) candidate makes the full horizon
    straight = min(cs.candidates,
                   key=lambda c: abs(float(c.meta.get("steer", 0))))
    assert straight.path.shape[0] >= 20


def test_arc_curvature_signs_differ() -> None:
    cs = sample_arc([0.0, 0.0], 0.0, speed=10.0, max_steer=0.5, n_curv=7)
    ends = {float(c.path[-1, 1]) for c in cs.candidates}  # lateral endpoint
    assert any(e > 1.0 for e in ends) and any(e < -1.0 for e in ends)


def test_lane_shift_reference_rejoins_route() -> None:
    ref = np.column_stack([np.linspace(0, 30, 31), np.zeros(31)])
    cs = sample_lane_shift(ref, offsets=(2.0, -2.0), blend_m=6.0, ahead_m=20.0)
    assert len(cs) >= 2
    for c in cs.candidates:
        # the far end of the shifted path should be back near the route
        assert abs(float(c.path[-1, 1])) < 0.6, c.path[-1]


def test_cost_collision_free_and_blocked() -> None:
    sc = _scene()
    sc.grid.origin = (0.0, 0.0)
    sc.grid.mark_obstacle_region(6.0, 0.0, 1.0, 1.0)
    # clean path runs clear of the box (y = 3 m away)
    clean = np.column_stack([np.linspace(0, 10, 20), np.full(20, 3.0)])
    # blocked path runs straight through the box at (6, 0)
    through = np.column_stack([np.linspace(0, 10, 20), np.zeros(20)])
    assert cost_collision(sc, clean) < 0.5
    assert cost_collision(sc, through) >= 0.5


def test_cost_curvature_straight_low_curvy_high() -> None:
    straight = np.column_stack([np.linspace(0, 10, 20), np.zeros(20)])
    curve = np.column_stack([
        np.cos(np.linspace(0, np.pi / 2, 20)) * 10,
        np.sin(np.linspace(0, np.pi / 2, 20)) * 10])
    assert cost_curvature(straight) < cost_curvature(curve)


def test_cost_lane_align_measures_deviation() -> None:
    sc = _scene()
    on_lane = np.column_stack([np.linspace(0, 20, 20), np.zeros(20)])
    off_lane = np.column_stack([np.linspace(0, 20, 20), np.full(20, 3.0)])
    assert cost_lane_align(sc, on_lane) < 0.5
    assert cost_lane_align(sc, off_lane) >= 2.5


def test_selector_picks_clear_path_over_blocked() -> None:
    sc = _scene()
    sc.grid.origin = (0.0, 0.0)
    # block straight ahead
    sc.grid.mark_obstacle_region(5.0, 0.0, 0.7, 1.5)
    arc = sample_arc([0.0, 0.0], 0.0, speed=8.0, max_steer=0.6, n_curv=9)
    cons = Constraints(w_collision=5.0, w_curvature=0.5, w_lane_align=0.0)
    best, meta = select_trajectory(sc, arc, cons)
    assert best is not None
    # the chosen path must avoid the occupied cell at (5, 0): no sample
    # of the selected path may land inside it
    from beamng_autopilot.planning.constraints import _path_infractions
    bad, _ = _path_infractions(sc, best)
    assert bad == 0, f"selected path hits obstacle: {bad} infractions"


def test_selector_full_blockage_returns_none() -> None:
    sc = _scene()
    sc.grid.origin = (0.0, 0.0)
    # an enormous wall covering the whole forward corridor (wider than the
    # fan's lateral reach and the grid itself) leaves no feasible path
    sc.grid.mark_obstacle_region(3.0, 0.0, 50.0, 50.0)
    arc = sample_arc([0.0, 0.0], 0.0, speed=8.0, max_steer=0.6, n_curv=9)
    cons = Constraints(w_lane_align=0.0)
    best, meta = select_trajectory(sc, arc, cons)
    assert best is None
    assert "no feasible" in meta["why"]


def test_corridor_free_open() -> None:
    sc = _scene()
    sc.grid.origin = (0.0, 0.0)
    assert corridor_free_band(sc) is True


def test_corridor_blocked_full_wall() -> None:
    sc = _scene()
    sc.grid.origin = (0.0, 0.0)
    sc.grid.mark_obstacle_region(3.0, 0.0, 50.0, 50.0)
    assert corridor_free_band(sc, bands=6) is False


def test_corridor_scattered_cluster_still_free() -> None:
    """A row of roadside poles leaves an open lane between them: the
    corridor is NOT fully closed, so candidates stay feasible."""
    sc = _scene()
    sc.grid.origin = (0.0, 0.0)
    # poles at x=6, scattered across the lateral band (not full-span)
    for y in (-4.0, -2.0, 2.0, 4.0):
        sc.grid.mark_obstacle_region(6.0, y, 0.4, 0.4)
    assert corridor_free_band(sc, min_clear_m=2.0) is True


def test_selector_keeps_candidate_on_scattered_cluster() -> None:
    sc = _scene()
    sc.grid.origin = (0.0, 0.0)
    for y in (-3.0, 0.0, 3.0):
        sc.grid.mark_obstacle_region(5.0, y, 1.0, 1.0)
    arc = sample_arc([0.0, 0.0], 0.0, speed=6.0, max_steer=0.4, n_curv=9)
    cons = Constraints(w_collision=5.0, w_curvature=0.2, w_lane_align=0.0)
    # at least one feasible candidate exists (corridor not fully closed)
    best, meta = select_trajectory(sc, arc, cons)
    assert best is not None, meta

def test_out_of_grid_path_not_collision() -> None:
    """Collision cost counts only cells inside the sensor grid; a route
    that continues past the grid horizon is unknown, not a wall."""
    from beamng_autopilot.planning.constraints import cost_collision
    from beamng_autopilot.planning import Scene
    from beamng_autopilot.occupancy import OccupancyGrid
    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    xs = np.linspace(0, 80, 60)
    route = np.column_stack([xs, np.zeros_like(xs)])
    scene = Scene(pos=np.array([0.0, 0.0]), heading=0.0, grid=grid,
                  route=route, lane_ref=route)
    assert cost_collision(scene, route, max_frac=0.15) < 1.0
