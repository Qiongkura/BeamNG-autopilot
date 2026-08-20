"""Wall inner-edge guard regression (pure logic, no game).

The planner now drives the route centre (RIGHT_OFFSET_M=0); a wall
sitting left of the route is clear of the car.
The dangerous case is a wall whose inner edge intrudes into the lane beside
the car's driving corridor: a centred car would press toward it, so
the roadside-wall classifier must NOT drop that wall from the
collision/blocker logic just because its centre is "a bit off the
route".  The inner-edge distance is what matters.
"""
from __future__ import annotations

import numpy as np

from beamng_autopilot.perception import Obstacle
from beamng_autopilot.planner import LocalPlanner


def _route(n=13, step=5.0):
    return np.array([[i * step, 0.0] for i in range(n)], dtype=float)


def _oriented_wall(y=-1.2, half_thick=0.6):
    # oriented wall: 4 m long, 1.2 m thick, centred 1.2 m right of the
    # route -> inner edge is -0.6 m (0.6 m over the lane centre line).
    # It intrudes into the centred driving corridor, so it is
    # a real lane blocker, not a roadside boundary.
    return Obstacle(
        x=15.0, y=y, half_w=2.0, half_h=half_thick,
        category="raycast", label="wall",
        axis=np.array([1.0, 0.0]), half_len=2.0, half_thick=half_thick)


class TestWallInnerEdge:
    def test_oriented_thick_wall_on_right_is_not_roadside(self):
        planner = LocalPlanner()
        route = _route()
        wall = _oriented_wall(y=-1.2, half_thick=0.6)
        drive, blocked = planner.plan(route, [wall], (0.0, 0.0), 0.0, 0)

        # It must not be silently ignored: either the planner stops in
        # front of it (blocked) or it plans a real detour.  Following the
        # straight route at speed would mean the wall was misclassified
        # as a roadside boundary.
        assert blocked or planner.last_mode == "detour", (
            f"mode={planner.last_mode} blocked={blocked}")

    def test_axis_aligned_thick_wall_on_right_is_not_roadside(self):
        planner = LocalPlanner()
        route = _route()
        # axis-aligned box centred 1.0 m right of the route with half
        # extents 0.9 m -> inner edge is 0.1 m right of the lane centre
        # (inside the car's 1 m half width).  This is a real lane
        # blocker, not a roadside boundary.
        wall = Obstacle(x=15.0, y=-1.0, half_w=0.9, half_h=0.9,
                        category="raycast", label="wall")
        drive, blocked = planner.plan(route, [wall], (0.0, 0.0), 0.0, 0)
        assert blocked or planner.last_mode == "detour", (
            f"mode={planner.last_mode} blocked={blocked}")

    def test_far_thin_wall_is_roadside(self):
        # A long thin wall running along the road, centred 3 m left with
        # 0.25 m half-thickness: inner edge 2.75 m away, well outside the
        # car's half width -> roadside boundary, follow.
        planner = LocalPlanner()
        route = _route()
        wall = Obstacle(x=15.0, y=3.0, half_w=2.0, half_h=0.25,
                        category="raycast", label="wall")
        drive, blocked = planner.plan(route, [wall], (0.0, 0.0), 0.0, 0)
        assert not blocked and planner.last_mode == "follow"

    def test_far_right_thin_wall_is_roadside(self):
        # Same long thin wall on the right, centred 4 m right (inner edge
        # -3.75 m): well outside the centred car + half width, so the
        # car stays clear -> roadside boundary, not a lane blocker.
        planner = LocalPlanner()
        route = _route()
        wall = Obstacle(x=15.0, y=-4.0, half_w=2.0, half_h=0.25,
                        category="raycast", label="wall")
        drive, blocked = planner.plan(route, [wall], (0.0, 0.0), 0.0, 0)
        assert not blocked and planner.last_mode == "follow", (
            f"mode={planner.last_mode} blocked={blocked}")
