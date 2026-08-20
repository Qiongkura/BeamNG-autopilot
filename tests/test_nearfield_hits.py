"""Near-field raycast hits must stay visible to the raw-sensor safety
layer (forward_clearance_m) without polluting obstacle clustering.

Root cause this guards: ``scan_obstacles_raycast`` used to drop every hit
closer than ``min_dist`` (2.5 m) entirely, so a wall 1-2 m in front of
the bonnet was invisible to the emergency-stop layer and the car floored
the throttle into it (observed throttle=94.7% at v=0 until the STUCK
watchdog).  Near-field points (>= 1.0 m) are now returned with ``hits``,
but never enter the clustering pool.
"""
from __future__ import annotations

import json

import numpy as np

from beamng_autopilot.perception import scan_obstacles_raycast


class FakeBng:
    """Minimal fake: captures the Lua command and returns canned hits."""

    def __init__(self, near_dist, far_dist, up=None, near_z=0.0, far_z=0.0):
        pos = (0.0, 0.0, 0.0)
        rows = []
        if far_dist is not None:
            rows.append({"x": float(far_dist), "y": 0.0, "z": float(far_z),
                         "fan": "mid", "up": None})
        if near_dist is not None:
            rows.append({"x": float(near_dist), "y": 0.0, "z": float(near_z),
                         "fan": "mid", "up": up})
        self.payload = json.dumps(rows)

    def queue_lua_command(self, _chunk, response=True):
        return self.payload


def _hits_with(near, far, return_hits):
    return scan_obstacles_raycast(
        FakeBng(near, far), (0.0, 0.0, 0.0), radius=55.0,
        return_hits=return_hits)


class TestNearFieldRetention:
    def test_return_hits_keeps_wall_1m5_ahead(self):
        obs, hits = _hits_with(near=1.5, far=5.0, return_hits=True)
        pts = dict((round(x, 2), round(y, 2)) for x, y in hits)
        assert (1.5, 0.0) in hits, f"near wall dropped, hits={hits}"
        assert (5.0, 0.0) in hits
        # clustering pool never sees the sub-min_dist near point: only the
        # 5 m hit (a sparse speck) becomes an obstacle box.
        assert len(obs) >= 1, "far hit should still cluster"

    def test_without_return_hits_near_dropped_as_before(self):
        obs = _hits_with(near=1.5, far=5.0, return_hits=False)
        assert len(obs) >= 1  # far speck still clustered

    def test_only_near_point_still_returned(self):
        obs, hits = _hits_with(near=1.5, far=None, return_hits=True)
        assert (1.5, 0.0) in hits, "solitary near wall must stay visible"
        # no far hit -> no obstacle cluster at all
        assert len(obs) == 0, f"near point leaked into clustering: {obs}"

    def test_cloud_style_near_point_visible(self):
        """forward_clearance on the returned near hit reports ~1.5 m."""
        from beamng_autopilot.planner import forward_clearance_m
        _, hits = _hits_with(near=1.5, far=None, return_hits=True)
        clear = forward_clearance_m(hits, (0.0, 0.0), (1.0, 0.0),
                                    half_width=1.5)
        assert np.isclose(clear, 1.5), f"clearance={clear}"


class TestApproachSpeedLimit:
    def test_zero_at_reserve(self):
        from beamng_autopilot.planner import approach_speed_limit_mps
        assert approach_speed_limit_mps(6.0, 6.0) == 0.0

    def test_grows_with_spare(self):
        from beamng_autopilot.planner import approach_speed_limit_mps
        assert (approach_speed_limit_mps(10.0, 6.0)
                > approach_speed_limit_mps(7.0, 6.0))

    def test_below_reserve_clamped_zero(self):
        from beamng_autopilot.planner import approach_speed_limit_mps
        assert approach_speed_limit_mps(5.0, 6.0) == 0.0

    def test_never_negative(self):
        from beamng_autopilot.planner import approach_speed_limit_mps
        assert approach_speed_limit_mps(0.0, 6.0) == 0.0


class TestNearFieldGroundFilter:
    """Near-field ground/curb hits must not pin the emergency stop forever.

    The Lua fan already returns an ``up`` probe distance so a higher ray
    that clears a low hit identifies the ground.  The near-field split
    (`hdist < min_dist`) used to skip that check entirely, which meant
    flat ground at a 45-degree corner projected onto the forward corridor
    and produced a permanent `EMERGENCY STOP raw clear=0.7m` even at v=0
    on an empty road.
    """
    def test_ground_probe_drops_near_hit(self):
        _, hits = scan_obstacles_raycast(
            FakeBng(near_dist=1.5, far_dist=None, up=30.0), (0.0, 0.0, 0.0),
            radius=55.0, return_hits=True)
        assert (1.5, 0.0) not in hits, f"near ground leak: {hits}"

    def test_wall_probe_keeps_near_hit(self):
        _, hits = scan_obstacles_raycast(
            FakeBng(near_dist=1.5, far_dist=None, up=1.6), (0.0, 0.0, 0.0),
            radius=55.0, return_hits=True)
        assert (1.5, 0.0) in hits, f"near wall dropped: {hits}"

    def test_high_bridge_near_hit_dropped(self):
        _, hits = scan_obstacles_raycast(
            FakeBng(near_dist=1.5, far_dist=None, up=1.6, near_z=50.0),
            (0.0, 0.0, 0.0), radius=55.0, return_hits=True)
        assert (1.5, 0.0) not in hits, f"bridge leaked: {hits}"

    def test_lidar_local_ground_filter_keeps_wall_only(self):
        """The same ground-plane removal used for the Tech 360 near field
        keeps a wall while discarding flat-ground returns."""
        from beamng_autopilot.perception import (
            LIDAR_GROUND_CLEARANCE_M, LIDAR_MAX_HEIGHT_M, _local_ground_z)
        # Flat ground ring 1-2.5 m away at z=0 plus a short wall at x=1.2
        # (z from 0.5 to 1.5).
        cloud = []
        for r in (1.0, 1.4, 1.8, 2.2, 2.5):
            for a in (0.0, 0.9, 1.8, 2.7, 3.6):
                cloud.append((r * np.cos(a), r * np.sin(a), 0.0))
        for z in (0.5, 1.0, 1.5):
            cloud.append((1.2, 0.0, z))
        cloud = np.asarray(cloud, dtype=float)
        gnd = _local_ground_z(cloud, 0.0, 0.0)
        keep = ((cloud[:, 2] - gnd >= LIDAR_GROUND_CLEARANCE_M)
                & (cloud[:, 2] - gnd <= LIDAR_MAX_HEIGHT_M))
        kept = cloud[keep]
        # Only the wall points are above the ground reference.
        assert len(kept) >= 3, f"wall points lost: {kept.tolist()}"
        assert all(k[0] == 1.2 for k in kept),             f"ground leaked through: {kept.tolist()}"
