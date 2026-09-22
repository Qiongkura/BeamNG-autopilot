"""Strict fallback paths must pass the same constraints as sampled paths."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import Constraints, Scene, select_trajectory
from beamng_autopilot.planning.selector import _perception_hold_path
from beamng_autopilot.planning.trajectory import CandidateSet


def _scene():
    path = np.column_stack([np.arange(11.0), np.zeros(11)])
    grid = OccupancyGrid(60, 60, 0.5)
    scene = Scene(pos=np.zeros(2), heading=0.0, grid=grid,
                  lane_ref=path, strict_perception=True)
    return scene, CandidateSet(reference=path)


@pytest.mark.parametrize("missing_grid", [False, True])
def test_strict_empty_road_cannot_be_revived_by_open_corridor(missing_grid):
    scene, candidates = _scene()
    if missing_grid:
        scene.grid = None
    scorer = Constraints()
    assert not scorer.score(scene, candidates.candidates[0])[1]
    path, meta = select_trajectory(scene, candidates, scorer)
    assert path is None
    assert meta["why"] == "hold_heading_constraints_rejected"


def test_strict_fallback_uses_accepted_reference_not_rejected_envelope():
    scene, _ = _scene()
    rejected = scene.lane_ref + np.array([0.0, 1.5])
    scene.lane_envelope = SimpleNamespace(center=rejected)
    path = _perception_hold_path(scene)
    assert np.allclose(path[:, 1], 0.0)


def test_strict_invalid_reference_cannot_fall_back_to_envelope():
    scene, candidates = _scene()
    scene.lane_envelope = SimpleNamespace(center=scene.lane_ref.copy())
    scene.lane_ref = np.full((3, 2), np.nan)
    path, _ = select_trajectory(scene, candidates, Constraints())
    assert path is None


def test_strict_fallback_is_rescored_even_when_corridor_is_clear():
    scene, candidates = _scene()
    scene.grid.drivable[:] = 1
    seen = []

    class Reject:
        def score(self, sc, candidate):
            seen.append(candidate)
            return 0.0, False

    path, meta = select_trajectory(scene, candidates, Reject())
    assert path is None
    assert len(seen) == 2
    assert seen[-1].meta["kind"] == "lane_center"
    assert meta["why"] == "hold_heading_constraints_rejected"


def test_strict_verified_hold_remains_available():
    scene, candidates = _scene()
    scene.grid.drivable[:] = 1

    class RejectOriginal:
        def score(self, sc, candidate):
            if candidate.meta.get("kind") == "reference":
                return 0.0, False
            return Constraints().score(sc, candidate)

    path, meta = select_trajectory(scene, candidates, RejectOriginal())
    assert path is not None
    assert meta["kind"] == "hold_heading"
    assert np.allclose(path[:, 1], 0.0)


def test_legacy_low_occupancy_fallback_remains_unchanged():
    scene, candidates = _scene()
    scene.strict_perception = False

    class Reject:
        def score(self, sc, candidate):
            return 0.0, False

    path, meta = select_trajectory(scene, candidates, Reject())
    assert path is not None
    assert meta["why"].startswith("fallback_hold_heading")


class TestRejectionAttribution:
    """P1-3: "no drivable path" must name the gate that fired.

    The first instrumentation pass mislabelled the gates by one line, so a
    live run reported ``strict_no_drivable_evidence`` while the tick's own
    coverage metric showed 42 drivable cells - the gate that actually
    fired was the strict LANE-reference gate.  These tests pin the mapping.
    """

    def test_missing_candidate_geometry(self):
        from beamng_autopilot.planning import Constraints
        from beamng_autopilot.planning.trajectory import Candidate

        scene = Scene(pos=np.zeros(2), heading=0.0)
        c = Constraints()
        cost, ok = c.score(scene, Candidate(path=np.zeros((1, 2))))
        assert not ok and cost >= 1e9
        assert c.reject_counts == {"no_path_geometry": 1}

    def test_strict_without_a_lane_reference_names_the_lane_gate(self):
        from beamng_autopilot.planning import Constraints
        from beamng_autopilot.planning.trajectory import Candidate

        grid = OccupancyGrid(60, 60, 0.5)
        grid.drivable[:] = 1              # road evidence IS present
        grid.observed[:] = 1
        path = np.column_stack([np.arange(11.0), np.zeros(11)])
        scene = Scene(pos=np.zeros(2), heading=0.0, grid=grid,
                      lane_ref=None, strict_perception=True)
        c = Constraints()
        c.score(scene, Candidate(path=path))
        assert c.reject_counts == {"strict_no_perception_lane": 1}

    def test_strict_with_a_lane_but_no_drivable_evidence_names_that_gate(self):
        from beamng_autopilot.planning import Constraints
        from beamng_autopilot.planning.trajectory import Candidate

        grid = OccupancyGrid(60, 60, 0.5)      # drivable layer all zero
        path = np.column_stack([np.arange(11.0), np.zeros(11)])
        scene = Scene(pos=np.zeros(2), heading=0.0, grid=grid,
                      lane_ref=path, strict_perception=True)
        c = Constraints()
        c.score(scene, Candidate(path=path))
        assert c.reject_counts == {"strict_empty_drivable_layer": 1}

    def test_a_tick_that_accepts_something_has_no_rejects(self):
        from beamng_autopilot.planning import Constraints, select_trajectory

        grid = OccupancyGrid(60, 60, 0.5)
        grid.drivable[:] = 1
        grid.observed[:] = 1
        path = np.column_stack([np.arange(21.0), np.zeros(21)])
        scene = Scene(pos=np.zeros(2), heading=0.0, grid=grid,
                      lane_ref=path, strict_perception=True)
        cons = Constraints()
        best, meta = select_trajectory(scene, CandidateSet(reference=path),
                                       cons)
        assert best is not None
        assert not meta.get("rejects")
