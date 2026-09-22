"""T05: one geometry baseline, validated against an independent oracle.

The plan's T05 asks for a single definition of the vehicle origin, the
camera mounts and the ground plane - shared by the main view, the fisheye,
the marking and pavement consumers and telemetry - plus an error budget
whose oracle is INDEPENDENT of the code under test (otherwise two
implementations sharing one mistake prove nothing).

The oracle lives in ``scripts/m5_geometry_audit.py`` and is built from
first principles (mount offset/axes -> ray -> plane intersection).  Writing
it found four separate errors in its own inputs, each worth metres:
the mount height used as the camera height; the lateral ray sign flipped;
the ground plane taken as ``z = 0`` in the origin frame instead of
``-EGO_GROUND_GAP_M``; and the returned pair read as (lateral, forward).
That history is why these tests exist as a gate rather than as a note.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import geometry as G  # noqa: E402
from beamng_autopilot.vision.lanes import _back_project_many  # noqa: E402
from beamng_autopilot.vision.ring import (  # noqa: E402
    CAMERA_RING,
    camera_ring_models,
)
from beamng_autopilot.vision.projection import CameraModel  # noqa: E402


def _audit():
    spec = importlib.util.spec_from_file_location(
        "_m5_geom_audit", ROOT / "scripts" / "m5_geometry_audit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m5_geom_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


RING = camera_ring_models(320, 240)
ROWS = (170, 200, 232)
COLS = (20, 160, 300)


class TestOracleAgreement:
    """The oracle must reproduce the pipeline's own geometry."""

    @pytest.mark.parametrize("role", [m.role for m in CAMERA_RING])
    def test_every_ring_mount_matches_the_independent_oracle(self, role):
        audit = _audit()
        cam = RING[role]
        off = np.asarray(cam.offset, dtype=float)
        fwd = np.asarray(cam.fwd_local, dtype=float)
        up = np.asarray(cam.up_local, dtype=float)
        pos = np.array([0.0, 0.0, G.EGO_GROUND_GAP_M])
        gz = G.ego_ground_z(pos)
        checked = 0
        for v in ROWS:
            for u in COLS:
                got = None
                pts, ok = _back_project_many(np.array([u], dtype=float),
                                             np.array([v], dtype=float), cam,
                                             pos, 0.0, gz)
                if bool(ok[0]):
                    got = (float(pts[0][0]), float(pts[0][1]))
                ref = audit.oracle_ground_point(u, v, cam, off, fwd, up)
                if got is None or ref is None:
                    continue
                # both are (forward, lateral) from the vehicle origin
                err = math.hypot(got[0] - ref[0], got[1] - ref[1])
                assert err < 0.02, f"{role} ({u},{v}) off by {err:.4f} m"
                checked += 1
        assert checked >= 4, f"{role}: only {checked} pixels were comparable"

    def test_the_oracle_needs_the_origin_gap_not_zero(self):
        """Regression for the harness bug that treated the origin as ground."""
        audit = _audit()
        cam = RING["front_main"]
        off = np.asarray(cam.offset, dtype=float)
        v, u = 220, 160
        with_gap = audit.oracle_ground_point(u, v, cam, off, cam.fwd_local,
                                             cam.up_local, ground_gap_m=0.17)
        no_gap = audit.oracle_ground_point(u, v, cam, off, cam.fwd_local,
                                           cam.up_local, ground_gap_m=0.0)
        assert with_gap is not None and no_gap is not None
        # 0.17 m of missing plane height is worth tens of centimetres
        assert abs(with_gap[0] - no_gap[0]) > 0.1

    def test_an_extra_down_tilt_moves_the_hit_closer(self):
        audit = _audit()
        cam = RING["front_main"]
        off = np.asarray(cam.offset, dtype=float)
        v = 200
        level = audit.oracle_ground_point(v=v, u=cam.cx, cam=cam, offset=off,
                                          fwd_local=cam.fwd_local,
                                          up_local=cam.up_local)
        down = audit.oracle_ground_point(v=v, u=cam.cx, cam=cam, offset=off,
                                         fwd_local=cam.fwd_local,
                                         up_local=cam.up_local,
                                         extra_pitch_rad=math.radians(2.0))
        assert level is not None and down is not None
        assert down[0] < level[0]


class TestGroundBaseline:
    def test_the_road_plane_is_the_origin_minus_the_gap(self):
        pos = np.array([10.0, 20.0, 5.0])
        assert G.ego_ground_z(pos) == pytest.approx(
            5.0 - G.EGO_GROUND_GAP_M)
        assert G.projection_ground_z(pos, unified=True) == pytest.approx(
            5.0 - G.EGO_GROUND_GAP_M)
        assert G.projection_ground_z(pos, unified=False) == pytest.approx(5.0)

    def test_missing_height_is_zero_not_a_crash(self):
        assert G.ego_ground_z(np.array([1.0, 2.0])) == 0.0
        assert G.ego_ground_z(np.array([np.nan, 0.0, np.nan])) == 0.0

    def test_the_pose_label_names_the_attitude_model(self):
        assert G.pose_label(None) == "yaw_only"
        assert G.pose_label(None, enabled=True) == "yaw_only"
        q = np.array([0.0, 0.0, 0.0, 1.0])
        assert G.pose_label(q, enabled=True) == "quat_6dof"
        assert G.pose_label(q, enabled=False) == "yaw_only"
        assert G.pose_label(np.array([1.0, 2.0])) == "yaw_only"

    def test_the_footprint_matches_the_vehicle_constants(self):
        from beamng_autopilot.config import (EGO_HALF_LENGTH_M,
                                             EGO_HALF_WIDTH_M)
        assert G.FOOTPRINT_HALF_LENGTH_M == float(EGO_HALF_LENGTH_M)
        assert G.FOOTPRINT_HALF_WIDTH_M == float(EGO_HALF_WIDTH_M)


class TestResolutionBudget:
    def test_the_budget_follows_the_pinhole_relation(self):
        # px = fx * w / d  =>  d = fx * w / px
        assert G.resolution_distance_m(200.0, 0.1, 2.0) == pytest.approx(10.0)
        assert G.resolution_distance_m(400.0, 0.1, 2.0) == pytest.approx(20.0)
        assert G.resolution_distance_m(200.0, 0.2, 2.0) == pytest.approx(20.0)
        assert math.isnan(G.resolution_distance_m(0.0, 0.1, 2.0))

    def test_the_live_ring_budget_is_reported_and_bounded(self):
        """A 0.1 m marking at 320 px: the main view resolves it to ~9 m.

        This is the number the plan wants stated instead of assumed: the
        25 m reference horizon is far beyond what a 2-pixel-wide marking
        can support at this resolution, which is why the far field leans on
        temporal evidence and why the near-field channels matter.
        """
        main = RING["front_main"]
        d2 = G.resolution_distance_m(main.fx, 0.1, 2.0)
        assert 5.0 < d2 < 15.0
        fisheye = RING["front_fisheye"]
        assert G.resolution_distance_m(fisheye.fx, 0.1, 2.0) < d2

    def test_the_blind_zone_is_a_property_of_the_mount(self):
        main = RING["front_main"]
        fish = RING["front_fisheye"]
        pos = np.array([0.0, 0.0, G.EGO_GROUND_GAP_M])
        near_main = G.nearest_ground_distance_m(main, pos)
        near_fish = G.nearest_ground_distance_m(fish, pos)
        assert near_main is not None and near_fish is not None
        assert 2.0 < near_fish < near_main < 8.0

    def test_a_camera_pointing_at_the_sky_has_no_ground_hit(self):
        up_cam = CameraModel(offset=np.array([0.0, 0.0, 1.5]),
                             fwd_local=np.array([0.0, 1.0, 0.5]),
                             up_local=np.array([0.0, 0.0, 1.0]),
                             fov_deg=40.0, width=320, height=240)
        pos = np.array([0.0, 0.0, G.EGO_GROUND_GAP_M])
        assert G.nearest_ground_distance_m(up_cam, pos) is None


class TestSlopeAssumption:
    """The flat-ground model's error must be measured, not assumed away."""

    def test_zero_slope_costs_nothing(self):
        audit = _audit()
        rep = audit.audit_slope(RING["front_main"], "front_main",
                                slopes_deg=(0.0,), rows=(200, 220))
        assert rep["rows"][0]["max_m"] == pytest.approx(0.0, abs=1e-9)

    def test_a_slope_costs_metres_at_useful_range(self):
        audit = _audit()
        rep = audit.audit_slope(RING["front_main"], "front_main",
                                slopes_deg=(0.0, 8.0), rows=(200, 220))
        p50 = {r["slope_deg"]: r["p50_m"] for r in rep["rows"]}
        assert p50[8.0] > 0.2, p50
        assert p50[8.0] > p50[0.0]


class TestRotationThreading:
    """The real attitude must reach the projection call, not just be supported."""

    def test_a_rotated_vehicle_projects_differently(self):
        cam = RING["front_main"]
        pos = np.array([0.0, 0.0, G.EGO_GROUND_GAP_M])
        gz = G.ego_ground_z(pos)
        level, ok1 = _back_project_many(np.array([200.0]), np.array([200.0]),
                                        cam, pos, 0.0, gz)
        # a 3 deg roll about the vehicle's forward axis
        q = np.array([math.sin(math.radians(1.5)), 0.0, 0.0,
                      math.cos(math.radians(1.5))])
        rolled, ok2 = _back_project_many(np.array([200.0]), np.array([200.0]),
                                         cam, pos, 0.0, gz, rotation=q)
        assert bool(ok1[0]) and bool(ok2[0])
        assert abs(float(rolled[0][1]) - float(level[0][1])) > 0.05, (
            "a 3 deg roll must move the ground hit by centimetres")

    def test_a_yaw_only_pose_is_unchanged_by_the_switch(self):
        cam = RING["front_main"]
        pos = np.array([0.0, 0.0, G.EGO_GROUND_GAP_M])
        gz = G.ego_ground_z(pos)
        a, _ = _back_project_many(np.array([40.0]), np.array([210.0]), cam,
                                  pos, 0.0, gz)
        b, _ = _back_project_many(np.array([40.0]), np.array([210.0]), cam,
                                  pos, 0.0, gz, rotation=None)
        np.testing.assert_allclose(a, b)


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
