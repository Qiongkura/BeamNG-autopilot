"""Randomised wall-collision fuzz regression (pure logic, no game).

Requirement: the vehicle must never drive through a wall.  These cases
push random wall layouts (single / crossing / both sides / thick / thin /
tilted) through the full ``LocalPlanner.plan`` pipeline and assert the
returned drive path never intersects a wall footprint inflated by the car
half width.  They complement the hand-picked scenarios in
``test_no_wall_collision.py`` with broad, deterministic-random coverage.
"""
from __future__ import annotations

import random

import numpy as np

from beamng_autopilot.perception import Obstacle
from beamng_autopilot.planner import (
    LocalPlanner,
    _seg_hits_obstacle,
)
from beamng_autopilot.planner.constants import CAR_HALF_WIDTH


def _route(n=16, step=5.0):
    return np.array([[i * step, 0.0] for i in range(n)], dtype=float)


def _wall(rng, x, y):
    """Random wall near (x, y): axis-aligned or oriented, thin or thick."""
    if rng.random() < 0.5:
        return Obstacle(x=x, y=y,
                        half_w=rng.uniform(0.4, 2.5),
                        half_h=rng.uniform(0.4, 2.5),
                        category="raycast", label="wall")
    ang = rng.uniform(-0.9, 0.9)  # mostly lengthwise along the road
    ax, ay = np.cos(ang), np.sin(ang)
    return Obstacle(x=x, y=y,
                    half_w=0.0, half_h=0.0,
                    category="raycast", label="wall",
                    axis=np.array([ax, ay], dtype=float),
                    half_len=rng.uniform(1.0, 6.0),
                    half_thick=rng.uniform(0.25, 1.6))


def _collides(drive, walls):
    if len(drive) < 2:
        return None
    for k in range(len(drive) - 1):
        ax, ay = float(drive[k, 0]), float(drive[k, 1])
        bx, by = float(drive[k + 1, 0]), float(drive[k + 1, 1])
        for w in walls:
            if _seg_hits_obstacle(ax, ay, bx, by, w, CAR_HALF_WIDTH):
                return k, (w.x, w.y)
    return None


class TestWallFuzz:
    def test_100_random_scenes(self):
        rng = random.Random(20260819)
        for scene in range(100):
            n_walls = rng.randint(1, 4)
            walls = []
            # A mix of road-crossing walls, roadside walls and both sides,
            # at varying down-track positions.
            for _ in range(n_walls):
                mode = rng.random()
                if mode < 0.45:      # somewhere across the corridor
                    y = rng.uniform(-1.8, 1.8)
                elif mode < 0.75:    # right roadside (still drivable band)
                    y = rng.uniform(-4.0, -1.6)
                else:                # left roadside (still drivable band)
                    y = rng.uniform(1.6, 4.0)
                walls.append(_wall(rng, x=rng.uniform(6.0, 70.0), y=y))
            planner = LocalPlanner()
            drive, blocked = planner.plan(
                _route(), walls, (0.0, 0.0), 0.0, 0)
            hit = _collides(drive, walls)
            assert hit is None, (
                f"scene {scene}: mode={planner.last_mode} "
                f"blocked={blocked} collision at seg {hit[0]} "
                f"wall=({hit[1][0]:.2f}, {hit[1][1]:.2f})")

    def test_stop_before_crossing_wall(self):
        """A wall spanning the full corridor must stop (blocked), not
        detour through or around it into the wall."""
        wall = Obstacle(x=20.0, y=0.0, half_w=6.0, half_h=2.0,
                        category="raycast", label="wall")
        planner = LocalPlanner()
        drive, blocked = planner.plan(
            _route(), [wall], (0.0, 0.0), 0.0, 0)
        # Either blocked before it, or a path that never touches it.
        hit = _collides(drive, [wall])
        assert hit is None, f"drove into full-width gate: {hit}"
        if len(drive) >= 2:
            assert float(drive[0, 0]) < 20.0 or blocked, (
                "must not plan through the gate")
