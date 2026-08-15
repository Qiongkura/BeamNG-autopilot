"""Tech LiDAR obstacle pipeline regression (pure logic, no game).

Covers the ground-removal clustering (``lidar_obstacles``), the fusion
rules in ``merge_obstacles`` (lidar boxes merge with vehicles / walls) and
the centroid velocity tracker (``LidarClusterTracker``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_autopilot.perception import (
    LidarClusterTracker,
    Obstacle,
    downsample_cloud,
    lidar_obstacles,
    merge_obstacles,
)


def _ground_ring(ox=0.0, oy=0.0, r0=4.0, r1=30.0, z=0.0, step=1.0):
    """Dense flat ground ring around the ego at height ``z``."""
    pts = []
    for r in np.arange(r0, r1, step):
        for ang in np.arange(0.0, 360.0, 5.0):
            a = math.radians(ang)
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a), z))
    return np.asarray(pts, dtype=float)


def _box_wall(x0, x1, y, z=1.5, step=0.7):
    pts = []
    for x in np.arange(x0, x1, step):
        pts.append((x, y, z))
        pts.append((x, y, z + 0.8))
    return np.asarray(pts, dtype=float)


class TestDownsample:
    def test_small_cloud_untouched(self):
        c = np.zeros((10, 3))
        assert downsample_cloud(c, max_points=100).shape == (10, 3)

    def test_big_cloud_capped(self):
        rng = np.random.default_rng(0)
        c = rng.uniform(0.0, 50.0, size=(5000, 3))
        out = downsample_cloud(c, max_points=500)
        assert len(out) <= 500


class TestLidarObstacles:
    def test_flat_ground_is_removed(self):
        ground = _ground_ring(z=0.0)
        cloud = np.vstack([ground, _box_wall(10.0, 16.0, 6.0, z=1.5)])
        boxes = lidar_obstacles(cloud, (0.0, 0.0, 0.3), radius=45.0,
                                self_rect=(3.0, 1.5, 0.0))
        # the wall is the only above-ground object
        assert len(boxes) == 1, boxes
        b = boxes[0]
        assert b.category == "lidar"
        assert abs(b.x - 13.0) < 2.0 and abs(b.y - 6.0) < 2.0
        assert b.half_len > 2.0  # wall footprint, not a speck

    def test_self_points_removed(self):
        # points inside the ego footprint must not become obstacles; the
        # far cluster needs ground hits in its bin to survive (the local
        # ground reference comes from the road surface below it)
        rng = np.random.default_rng(1)
        far_ground = np.column_stack([
            rng.uniform(27.0, 33.0, 300), rng.uniform(-2.0, 2.0, 300),
            rng.normal(0.0, 0.03, 300)])
        far_obj = np.array([[30.0, 0.0, 2.0], [30.0, 0.5, 2.0],
                            [30.5, 0.0, 2.0], [30.5, 0.5, 2.0],
                            [30.0, 0.0, 1.5], [30.0, 0.5, 1.5],
                            [30.5, 0.0, 1.5], [30.5, 0.5, 1.5]])
        cloud = np.vstack([far_ground, far_obj])
        boxes = lidar_obstacles(cloud, (0.0, 0.0, 0.3), radius=45.0,
                                self_rect=(3.0, 1.5, 0.0))
        assert len(boxes) == 1
        assert math.hypot(boxes[0].x, boxes[0].y) > 20.0

    def test_empty_and_garbage_inputs(self):
        assert lidar_obstacles(np.empty((0, 3)), (0.0, 0.0, 0.3)) == []
        assert lidar_obstacles(np.full((5, 3), np.nan), (0.0, 0.0, 0.3)) == []


class TestMergeLidar:
    def test_lidar_vehicle_merges_with_registry_vehicle(self):
        lua = Obstacle(x=10.0, y=0.0, half_w=1.0, half_h=2.2,
                       category="vehicle", velocity=np.array([5.0, 0.0]),
                       heading=0.0, vehicle_id="veh-1")
        lid = Obstacle(x=11.0, y=0.2, half_w=1.1, half_h=1.1,
                       category="lidar")
        out = merge_obstacles([lua, lid])
        assert len(out) == 1
        assert out[0].velocity is not None  # registry velocity survives

    def test_lidar_wall_merges_with_raycast_wall(self):
        ray = Obstacle(x=20.0, y=0.0, half_w=4.0, half_h=0.5,
                       category="raycast", label="wall")
        lid = Obstacle(x=21.0, y=0.0, half_w=3.5, half_h=0.5,
                       category="lidar", label="wall")
        out = merge_obstacles([ray, lid])
        assert len(out) == 1

    def test_wall_never_merges_with_vehicle(self):
        wall = Obstacle(x=20.0, y=0.0, half_w=2.0, half_h=2.0,
                        category="wall", label="wall")
        veh = Obstacle(x=21.0, y=0.0, half_w=2.3, half_h=1.1,
                       category="vehicle")
        assert len(merge_obstacles([wall, veh])) == 2


class TestLidarClusterTracker:
    def _boxes(self, xy):
        return [Obstacle(x=x, y=y, half_w=1.0, half_h=1.0,
                         category="lidar") for x, y in xy]

    def test_moving_cluster_gets_velocity(self):
        tr = LidarClusterTracker(min_matches=2)
        t = 100.0
        tr.update(self._boxes([(10.0, 0.0)]), t)
        tr.update(self._boxes([(10.5, 0.0)]), t + 0.1)   # 5 m/s
        tr.update(self._boxes([(11.0, 0.0)]), t + 0.2)   # 5 m/s
        boxes = self._boxes([(11.5, 0.0)])
        tr.update(boxes, t + 0.3)
        assert boxes[0].velocity is not None
        # EMA (k=0.5) converges toward 5 m/s within a few polls
        assert boxes[0].velocity[0] == pytest.approx(5.0, abs=1.5)

    def test_static_cluster_gets_no_velocity(self):
        tr = LidarClusterTracker(min_matches=2)
        t = 100.0
        for i in range(4):
            boxes = self._boxes([(10.0, 0.0)])
            tr.update(boxes, t + i * 0.1)
        assert boxes[0].velocity is None  # jitter stays below min_speed

    def test_slow_jitter_phantom_is_filtered(self):
        # a static object whose centroid wanders ~1 m/s (voxel resampling
        # jitter) must not be reported as a moving vehicle
        tr = LidarClusterTracker(min_matches=2, min_speed=2.0)
        t = 100.0
        seq = [(10.0, 0.0), (10.05, 0.02), (10.08, -0.01), (10.02, 0.03)]
        for i, xy in enumerate(seq):
            boxes = self._boxes([xy])
            tr.update(boxes, t + i * 0.2)
        assert boxes[0].velocity is None

    def test_new_cluster_has_no_velocity_yet(self):
        tr = LidarClusterTracker(min_matches=2)
        boxes = self._boxes([(10.0, 0.0)])
        tr.update(boxes, 100.0)
        assert boxes[0].velocity is None

    def test_teleport_gets_no_phantom_velocity(self):
        tr = LidarClusterTracker(min_matches=1, ttl_s=1.0)
        tr.update(self._boxes([(10.0, 0.0)]), 100.0)
        tr.update(self._boxes([(30.0, 0.0)]), 101.5)  # 20 m jump: new track
        boxes = self._boxes([(30.0, 0.0)])
        tr.update(boxes, 102.0)
        # the jump must not look like a 40 m/s vehicle
        assert boxes[0].velocity is None \
            or math.hypot(*boxes[0].velocity) < 1.0