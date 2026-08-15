"""``merge_obstacles()`` regression: dynamic-vehicle state survives merging.

The same car is usually reported by both the scene vehicle scan (with
velocity / heading / id) and the vision channel (without); after merging
the ACC layer must still see a moving lead.
"""
from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.perception import Obstacle, merge_obstacles


def _vision(x, y=0.0):
    return Obstacle(x=x, y=y, half_w=0.9, half_h=1.8, category="vision")


def _vehicle(x, y=0.0, vx=4.0):
    ob = Obstacle(x=x, y=y, half_w=1.0, half_h=2.2, category="vehicle",
                  heading=0.0, vehicle_id="veh-1")
    ob.velocity = np.array([vx, 0.0], dtype=float)
    return ob


class TestMergeObstacles:
    def test_overlapping_boxes_merge(self):
        out = merge_obstacles([_vision(10.0), _vehicle(11.0)])
        assert len(out) == 1

    def test_velocity_survives_merge_into_vision_box(self):
        # vision box comes first in the list; without the dynamic-state
        # merge the ACC velocity would be lost
        out = merge_obstacles([_vision(10.0), _vehicle(11.0)])
        assert len(out) == 1
        ob = out[0]
        assert ob.velocity is not None
        assert ob.velocity[0] == pytest.approx(4.0)
        assert ob.vehicle_id == "veh-1"

    def test_existing_velocity_is_kept(self):
        out = merge_obstacles([_vehicle(10.0), _vision(11.0)])
        assert out[0].velocity is not None

    def test_raycast_wall_does_not_merge_with_vehicle(self):
        wall = Obstacle(x=10.0, y=0.0, half_w=3.0, half_h=3.0,
                        category="wall")
        out = merge_obstacles([wall, _vehicle(11.0)])
        assert len(out) == 2

    def test_merged_box_covers_both_footprints(self):
        out = merge_obstacles([_vision(10.0), _vehicle(11.0)])
        ob = out[0]
        assert ob.x == pytest.approx(10.5, abs=0.2)