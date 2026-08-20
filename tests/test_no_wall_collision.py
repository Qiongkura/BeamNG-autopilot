"""No-wall-collision and keep-lane regression (pure logic, no game).

Goal 3 of the project: the vehicle must never hit a wall and, in normal
driving, must not cross lane markings.  These tests drive several wall
scenarios through the full ``LocalPlanner.plan`` pipeline and assert that
the returned drive path, whatever its mode (follow / detour / blocked),
never intersects any wall's footprint inflated by the car half width.
They also assert normal straight-line driving with a solid centre line
stays on the legal side.
"""
from __future__ import annotations

import numpy as np

from beamng_autopilot.perception import Obstacle
from beamng_autopilot.planner import (
    LocalPlanner,
    _MapLaneBoundary,
    _seg_hits_obstacle,
)
from beamng_autopilot.planner.constants import CAR_HALF_WIDTH


def _route(n=13, step=5.0):
    return np.array([[i * step, 0.0] for i in range(n)], dtype=float)


def _oriented_wall(x=15.0, y=0.0, half_len=2.0, half_thick=0.6,
                   axis=(1.0, 0.0)):
    return Obstacle(
        x=x, y=y, half_w=half_len, half_h=half_thick,
        category="raycast", label="wall",
        axis=np.array(axis, dtype=float), half_len=half_len,
        half_thick=half_thick)


def _axis_wall(x=15.0, y=0.0, half_w=0.9, half_h=0.9):
    return Obstacle(x=x, y=y, half_w=half_w, half_h=half_h,
                    category="raycast", label="wall")


def _collides(drive, walls) -> tuple[int, float, float] | None:
    """First (segment, wx, wy) where the drive path hits a wall."""
    if len(drive) < 2:
        return None
    for k in range(len(drive) - 1):
        ax, ay = float(drive[k, 0]), float(drive[k, 1])
        bx, by = float(drive[k + 1, 0]), float(drive[k + 1, 1])
        for w in walls:
            if _seg_hits_obstacle(ax, ay, bx, by, w, CAR_HALF_WIDTH):
                return k, float(w.x), float(w.y)
    return None


class TestNoWallCollision:
    def _check(self, walls, pos=(0.0, 0.0), heading=0.0):
        planner = LocalPlanner()
        drive, blocked = planner.plan(
            _route(), walls, pos, heading, 0)
        hit = _collides(drive, walls)
        assert hit is None, (
            f"mode={planner.last_mode} blocked={blocked} "
            f"collision at seg {hit[0]} wall=({hit[1]}, {hit[2]})")

    def test_thick_wall_on_right(self):
        self._check([_oriented_wall(y=-1.2, half_thick=0.6)])

    def test_axis_thick_wall_on_right(self):
        self._check([_axis_wall(y=-1.0)])

    def test_wall_crossing_centre(self):
        self._check([_oriented_wall(y=0.6, half_thick=0.6)])

    def test_vertical_wall_across_road(self):
        self._check([_oriented_wall(y=0.0, half_thick=0.3,
                                    axis=(0.0, 1.0))])

    def test_both_sides_walls(self):
        self._check([_oriented_wall(y=-1.6, half_thick=0.6),
                     _oriented_wall(y=2.0, half_thick=0.5)])

    def test_full_width_gate(self):
        self._check([_oriented_wall(y=0.0, half_len=8.0,
                                    half_thick=5.0)])

    def test_wall_row(self):
        self._check([_oriented_wall(x=12.0, y=-0.8, half_thick=0.5),
                     _oriented_wall(x=18.0, y=0.6, half_thick=0.5),
                     _oriented_wall(x=24.0, y=-0.6, half_thick=0.5)])

    def test_far_left_wall_still_clear(self):
        self._check([_oriented_wall(y=3.0, half_thick=0.25)])

    def test_wall_behind_car_ignored(self):
        self._check([_oriented_wall(x=-5.0, y=0.0)],
                    pos=(0.0, 0.0))


class TestKeepLane:
    def test_solid_centre_line_kept_on_legal_side(self):
        """Straight road, solid centre line along the route: the driven
        path must stay on the legal (right) side and never
        cross the marking."""
        planner = LocalPlanner()
        route = _route()
        centre = _MapLaneBoundary(
            np.array([[0.0, 0.0], [60.0, 0.0]]), allowed_side=1.0)
        drive, blocked = planner.plan(
            route, [], (0.0, 0.0), 0.0, 0, solid_lines=[centre])
        assert not blocked
        assert len(drive) >= 2
        # right offset is negative lat (route runs +X, right = -Y): no
        # drive point may sit on the left (positive) side of the line.
        assert np.all(drive[:, 1] <= 0.05), (
            f"crossed centre line, max lat={drive[:, 1].max():.2f}")

    def test_open_straight_stays_smooth(self):
        """No obstacles: the drive path follows the route centre and
        does not weave laterally."""
        planner = LocalPlanner()
        route = _route()
        drive, blocked = planner.plan(
            route, [], (0.0, 0.0), 0.0, 0)
        assert not blocked
        # monotone forward motion along +X, no lateral jumps
        assert np.all(np.diff(drive[:, 0]) > 0.0)
        lat = drive[:, 1]
        assert np.all(np.abs(np.diff(lat)) < 1.0), (
            "path weaves laterally on an open straight")
