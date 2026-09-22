"""T07: separate boundary evidence, geometric thresholds, offline arms.

The plan's T07 wants real boundary evidence published as THREE separate
kinds (curb / pavement edge / obstacle entity), thresholds derived from the
sensor's own geometry instead of hand-tuned constants, and an offline arm
comparison - with the explicit warning that a coarse road mask must never
become a driving permission, and that a missing detection must stay
UNKNOWN rather than become "drivable".

The capability check is part of the evidence: the LiDAR cloud carries no
per-beam id / vertical angle / scan time (checked in
``beamngpy/sensors/lidar.py``), so the plan's sliding-beam method cannot be
reproduced and the fallback (surface discontinuities) is what these tests
cover.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_autopilot.lane.boundary_evidence import (
    BoundaryEvidence,
    CURB_HEIGHT_M,
    adaptive_step_threshold,
    associate,
    beam_spacing_m,
    curb_candidates,
    obstacle_entities,
    pavement_edges,
)


def _profile(x0=2.0, x1=30.0, step=None, step_at=10.0, dy=0.01, dx=0.05,
             z_step=0.15, seed=0):
    """A dense along-surface sweep (one LiDAR ring, flattened)."""
    rng = np.random.default_rng(seed)
    xs = np.arange(x0, x1, dx)
    ys = rng.normal(0.0, dy, size=xs.size)
    zs = np.zeros_like(xs)
    if step:
        zs[xs >= step_at] = z_step
    return np.column_stack([xs, ys, zs])


class TestGeometricThresholds:
    def test_the_vertical_ring_spacing_grows_with_range(self):
        assert beam_spacing_m(5.0) == pytest.approx(5.0 * math.tan(
            math.radians(2.0 * 26.9 / 16)), rel=1e-6)
        assert beam_spacing_m(20.0) > beam_spacing_m(5.0)
        assert beam_spacing_m(0.0) == 0.0
        assert beam_spacing_m(float("nan")) == 0.0

    def test_the_threshold_follows_the_measured_sampling_and_is_capped(self):
        # denser sampling -> smaller threshold, but never below the floor
        assert adaptive_step_threshold(0.5) > adaptive_step_threshold(0.05)
        assert adaptive_step_threshold(0.0) >= 0.05
        # ...and never above half a kerb: the threshold may not grow past
        # the signal it is looking for
        assert adaptive_step_threshold(100.0) <= 0.6 * CURB_HEIGHT_M + 1e-9

    def test_a_sixteen_beam_ring_cannot_sample_a_kerb_vertically(self):
        """The measured reason the detector is not based on beam pitch."""
        assert beam_spacing_m(5.0) > CURB_HEIGHT_M
        assert beam_spacing_m(20.0) > 4 * CURB_HEIGHT_M


class TestCurbDetection:
    def test_a_kerb_is_found_and_a_flat_road_is_not(self):
        with_kerb = curb_candidates(_profile(step=True), ground_z=0.0,
                                    pos=np.zeros(3))
        flat = curb_candidates(_profile(step=False), ground_z=0.0,
                               pos=np.zeros(3))
        assert with_kerb.as_dict()["n_points"] > 0
        xs = with_kerb.points[:, 0]
        assert xs.min() < 10.0 < xs.max() or \
            np.all(np.abs(xs - 10.0) < 0.5)
        assert flat.as_dict()["n_points"] == 0

    def test_the_published_threshold_is_the_geometric_one(self):
        ev = curb_candidates(_profile(step=True), ground_z=0.0,
                             pos=np.zeros(3))
        meta = ev.as_dict()["meta"]
        assert meta["threshold_p50_m"] is not None
        assert meta["threshold_p50_m"] <= 0.6 * CURB_HEIGHT_M + 1e-9
        assert "capability" in meta          # the recorded sensor gap

    def test_a_grass_edge_is_not_a_kerb(self):
        """No height step -> no candidate, however bright the grass is."""
        ev = curb_candidates(_profile(step=False, dy=0.05), ground_z=0.0,
                             pos=np.zeros(3))
        assert ev.as_dict()["n_points"] == 0

    def test_no_cloud_is_reported_not_invented(self):
        ev = curb_candidates(np.empty((0, 3)), ground_z=0.0, pos=np.zeros(3))
        assert ev.as_dict()["n_points"] == 0
        assert "reason" in ev.as_dict()["meta"]

    def test_points_beyond_the_range_limit_are_not_judged(self):
        far = _profile(x0=60.0, x1=80.0, step=True, step_at=70.0)
        ev = curb_candidates(far, ground_z=0.0, pos=np.zeros(3))
        assert ev.as_dict()["n_points"] == 0


class TestSeparateKinds:
    def test_the_three_kinds_are_published_separately(self):
        cond = BoundaryEvidence(kind="curb",
                                points=np.array([[10.0, 0.0], [10.1, 0.1]]),
                                provenance="lidar_height_step")
        edge = BoundaryEvidence(kind="pavement_edge",
                                points=np.array([[10.05, 0.0]]),
                                provenance="semantic_road_mask")
        obs = obstacle_entities([type("O", (), {"x": 10.0, "y": 0.0,
                                                "category": "lidar"})()])
        assert {e.kind for e in (cond, edge, obs)} == {
            "curb", "pavement_edge", "obstacle"}
        for e in (cond, edge, obs):
            assert e.as_dict()["provenance"]

    def test_association_reports_coincidence_without_merging(self):
        cond = BoundaryEvidence(kind="curb",
                                points=np.array([[10.0, 0.0]]))
        edge = BoundaryEvidence(kind="pavement_edge",
                                points=np.array([[10.1, 0.0]]))
        other = BoundaryEvidence(kind="obstacle",
                                 points=np.array([[30.0, 0.0]]))
        rep = associate([cond, edge, other])
        assert rep["merged"] is False
        assert any(r["a"] == "curb" and r["b"] == "pavement_edge"
                   for r in rep["same_object"])
        assert any("obstacle" in (r["a"], r["b"]) for r in rep["distinct"])
        # and the inputs are untouched
        assert cond.points.shape == (1, 2)
        assert cond.kind == "curb"

    def test_an_obstacle_is_not_promoted_to_a_boundary(self):
        obs = obstacle_entities([type("O", (), {"x": 5.0, "y": 1.0,
                                                "category": "tree"})()])
        assert obs.kind == "obstacle"
        assert obs.as_dict()["provenance"] == "obstacle_boxes"

    def test_the_pavement_publisher_reports_its_absence(self):
        ev = pavement_edges(None, None, np.zeros(3), 0.0, ground_z=0.0)
        assert ev.kind == "pavement_edge"
        assert ev.as_dict()["n_points"] == 0
        assert "reason" in ev.as_dict()["meta"]


class TestNoPermissionFromCoarseMasks:
    """The plan: a coarse on-road mask must never become driving permission."""

    def test_an_unknown_mask_yields_no_edge_rather_than_drivable(self):
        ev = pavement_edges(np.zeros((20, 20), dtype=bool), None,
                            np.zeros(3), 0.0, ground_z=0.0)
        assert ev.as_dict()["n_points"] == 0
        assert "reason" in ev.as_dict()["meta"]

    def test_boundary_evidence_carries_no_permission_field(self):
        for cls in (BoundaryEvidence,):
            fields = set(getattr(cls, "__dataclass_fields__", {}))
            assert not any(f in fields for f in
                           ("drivable", "permission", "authority", "crossable"))


class TestTemporalPersistence:
    """T07's selectivity fix: one-off candidates are not evidence."""

    def test_a_persistent_candidate_survives_and_a_one_off_does_not(self):
        from beamng_autopilot.lane.boundary_evidence import CurbPersistence
        cp = CurbPersistence()
        kerb = np.array([[10.0, 0.0], [10.05, 0.02]])
        noise = np.array([[20.0, 5.0]])
        first = cp.update(np.vstack([kerb, noise]), now_s=0.0)
        assert first.as_dict()["n_points"] == 0, "one sighting is not enough"
        for i in range(1, 4):
            ev = cp.update(kerb, now_s=0.5 * i)
        assert ev.as_dict()["n_points"] == 1          # the kerb cell
        meta = ev.as_dict()["meta"]
        assert meta["min_hits"] == 3 and meta["raw_cells"] >= 1
        # the one-off noise cell never becomes persistent
        assert all(abs(p[0] - 10.0) < 0.6 for p in ev.points)

    def test_the_window_is_time_not_frames(self):
        from beamng_autopilot.lane.boundary_evidence import CurbPersistence
        cp = CurbPersistence(window_s=1.0)
        kerb = np.array([[10.0, 0.0]])
        cp.update(kerb, now_s=0.0)
        cp.update(kerb, now_s=0.4)
        ev = cp.update(kerb, now_s=2.5)   # the first two hits aged out
        assert ev.as_dict()["meta"]["persistent_cells"] == 0

    def test_reset_clears_the_state(self):
        from beamng_autopilot.lane.boundary_evidence import CurbPersistence
        cp = CurbPersistence()
        for i in range(4):
            cp.update(np.array([[1.0, 1.0]]), now_s=float(i))
        cp.reset()
        assert cp.update(np.array([[1.0, 1.0]]), now_s=9.0).as_dict()[
            "n_points"] == 0


def _line_beside(lat: float, fwd0: float = 3.5, fwd1: float = 12.0,
                 n: int = 40, jitter: float = 0.02, seed: int = 3):
    """A thin world-space line beside a car at the origin facing +x."""
    rng = np.random.default_rng(seed)
    f = np.linspace(fwd0, fwd1, n)
    l = lat + jitter * rng.standard_normal(n)
    return np.column_stack([f, l])


class TestLineConsistency:
    """T07 measured on real frames: a boundary is a curve, not a point set.

    The 8-frame real sequence (2026-09-22) showed the height-step arm
    publishing ~4845 candidates per frame spread over 2.04 m laterally and
    persistence keeping 1156 of them; the line filter is the constraint
    that turned that set into one thin curve (10 points, 0.025 m residual)
    with 61% of its projections on the semantic mask edge instead of 16%.
    """

    def test_a_thin_line_beside_the_car_survives_with_its_offset(self):
        from beamng_autopilot.lane.boundary_evidence import (
            line_consistent_candidates)
        ev = line_consistent_candidates(_line_beside(-2.6), np.zeros(3), 0.0)
        assert ev.kind == "curb_boundary_line"
        assert len(ev.points) >= 30
        assert ev.meta["lat_offset_m"] == pytest.approx(-2.6, abs=0.05)
        span = ev.meta["fwd_span_m"]
        assert span[0] >= 3.0 and span[1] >= 11.0
        assert ev.meta["lat_spread_m"] < 0.35
        assert ev.meta["chains"] == 1

    def test_the_offset_is_measured_in_the_ego_frame_not_the_world(self):
        from beamng_autopilot.lane.boundary_evidence import (
            line_consistent_candidates)
        # The same world line seen by a car yawed 10 deg: the published
        # offset follows the CAR (the filter is an ego-frame statement).
        line = _line_beside(-2.6)
        straight = line_consistent_candidates(line, np.zeros(3), 0.0)
        h = math.radians(10)
        yawed = line_consistent_candidates(line, np.zeros(3), h)
        assert straight.meta["lat_offset_m"] == pytest.approx(-2.6, abs=0.05)
        # analytic ego-frame offset of that world line at this yaw
        expect = float(np.median(-math.sin(h) * line[:, 0]
                                 - math.cos(h) * 2.6))
        assert yawed.meta["lat_offset_m"] == pytest.approx(expect, abs=0.1)
        assert abs(yawed.meta["lat_offset_m"]
                   - straight.meta["lat_offset_m"]) > 0.5

    def test_a_line_at_a_large_angle_to_the_path_is_not_this_evidence(self):
        """Measured limit: the chain step caps the relative angle.

        At 1 m forward bins a 0.4 m chain step tolerates ~22 deg of
        relative angle, which is why 4 of the 8 real frames published no
        line where the road curved away from the placed heading.
        """
        from beamng_autopilot.lane.boundary_evidence import (
            line_consistent_candidates)
        line = _line_beside(-2.6)
        ev = line_consistent_candidates(line, np.zeros(3), math.radians(30))
        assert len(ev.points) == 0
        assert ev.meta["chains"] == 0


    def test_a_scattered_candidate_set_is_rejected(self):
        from beamng_autopilot.lane.boundary_evidence import (
            line_consistent_candidates)
        rng = np.random.default_rng(11)
        blob = np.column_stack([rng.uniform(3.0, 25.0, 400),
                                rng.uniform(-8.0, 8.0, 400)])
        ev = line_consistent_candidates(blob, np.zeros(3), 0.0)
        assert ev.meta["chains"] == 0
        assert len(ev.points) == 0

    def test_candidates_under_the_car_are_not_a_boundary(self):
        from beamng_autopilot.lane.boundary_evidence import (
            line_consistent_candidates)
        ev = line_consistent_candidates(_line_beside(0.4), np.zeros(3), 0.0)
        assert len(ev.points) == 0
        assert ev.meta["reason"] in ("no candidate beside the car",
                                     "no thin cluster",
                                     "no chain of thin clusters")

    def test_a_lateral_jump_breaks_the_chain(self):
        from beamng_autopilot.lane.boundary_evidence import (
            line_consistent_candidates)
        rng = np.random.default_rng(5)
        near = _line_beside(-2.6, fwd0=3.5, fwd1=7.0, n=25)
        far = np.column_stack([np.linspace(8.0, 12.0, 20),
                               -5.0 + 0.02 * rng.standard_normal(20)])
        ev = line_consistent_candidates(np.vstack([near, far]),
                                        np.zeros(3), 0.0)
        # only the chain that stayed within LINE_CHAIN_STEP_M is published
        assert len(ev.points) >= 20
        assert all(abs(p[1] + 2.6) < 0.5 for p in ev.points)

    def test_two_parallel_lines_publish_one_and_record_the_other(self):
        from beamng_autopilot.lane.boundary_evidence import (
            line_consistent_candidates)
        left = _line_beside(3.5, fwd0=3.5, fwd1=12.0, n=40)
        right = _line_beside(-3.5, fwd0=3.5, fwd1=7.0, n=20)
        ev = line_consistent_candidates(np.vstack([left, right]),
                                        np.zeros(3), 0.0)
        assert ev.meta["chains"] == 2
        # the LONGER chain is the published line; the other is recorded by
        # identity, never averaged into the published offset
        assert ev.meta["lat_offset_m"] == pytest.approx(3.5, abs=0.1)
        other = ev.meta["other_chains"][0]
        assert other["lat_offset_m"] == pytest.approx(-3.5, abs=0.1)
        # and the second chain keeps its OWN geometry: a two-sided stretch
        # cannot be measured from a summary alone
        assert len(other["points"]) >= 15
        assert ev.meta["lat_offset_m"] * other["lat_offset_m"] < 0.0

    def test_a_long_chain_far_to_the_side_is_not_the_lane_boundary(self):
        """Measured counterexample: the longest chain can be irrelevant.

        On a real straight stretch (2026-09-22) the longest thin chain sat
        18.9-19.7 m to the LEFT while the pavement-edge candidates were
        3.6 px from the mask edge.  Publishing the far structure as "the
        boundary" was wrong, so relevance is distance from the ego path:
        out-of-band chains are recorded with identity, in-band ones are
        published.
        """
        from beamng_autopilot.lane.boundary_evidence import (
            LINE_LAT_MAX_M, line_consistent_candidates)
        far = _line_beside(19.0, fwd0=3.5, fwd1=14.0, n=60)
        near = _line_beside(-2.6, fwd0=3.5, fwd1=8.0, n=25)
        ev = line_consistent_candidates(np.vstack([far, near]),
                                        np.zeros(3), 0.0)
        assert ev.meta["lat_offset_m"] == pytest.approx(-2.6, abs=0.1)
        assert abs(ev.meta["lat_offset_m"]) <= LINE_LAT_MAX_M
        distant = ev.meta["distant_chains"]
        assert len(distant) == 1
        assert distant[0]["lat_offset_m"] == pytest.approx(19.0, abs=0.2)
        assert len(distant[0]["points"]) >= 30, "kept, with its geometry"

    def test_with_only_a_distant_chain_nothing_is_published(self):
        from beamng_autopilot.lane.boundary_evidence import (
            line_consistent_candidates)
        ev = line_consistent_candidates(_line_beside(19.0),
                                        np.zeros(3), 0.0)
        assert len(ev.points) == 0
        assert "no chain within" in ev.meta["reason"]
        assert len(ev.meta["distant_chains"]) == 1

    def test_it_uses_no_map_and_no_world_frame(self):
        from beamng_autopilot.lane import boundary_evidence as be
        import inspect
        src = inspect.getsource(be.line_consistent_candidates)
        assert "findBestRoad" not in src and "nav_route" not in src
        assert "road_rule" not in src

