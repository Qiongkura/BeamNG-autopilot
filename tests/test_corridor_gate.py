"""Wiring the feasibility answer into the escape hatch (plan P2.3).

The gate that is being replaced returned True when it could not see.  So
the tests here are mostly about what happens when the evidence is missing:
UNKNOWN has to stay closed, and an exception has to stay closed.
"""

import numpy as np
import pytest

from beamng_autopilot import safety_monitor as sm


class _Grid:
    def __init__(self, obstacle, drivable=None, res=0.5):
        self.obstacle = np.asarray(obstacle, dtype=np.int8)
        self.n_rows, self.n_cols = self.obstacle.shape
        self.res = res
        self.drivable = drivable


class _Scene:
    def __init__(self, grid=None, **kw):
        self.grid = grid
        self.speed_mps = kw.get("speed_mps", 0.0)
        self.closest_obs_m = kw.get("closest_obs_m", None)
        self.bev_age_s = kw.get("bev_age_s", None)


def _open():
    return np.zeros((40, 40), dtype=np.int8)


class TestDefaultOff:
    def test_the_gate_is_off_unless_asked_for(self):
        # Behaviour must not change because a new primitive exists.
        assert sm.CORRIDOR_FEASIBILITY_GATE is False

    def test_the_verdict_starts_unknown_not_open(self):
        v = sm.SafetyVerdict()
        assert v.corridor_state == "unknown"
        assert v.corridor_reason == ""
        assert v.corridor_evidence == {}


class TestCorridorFeasibilityMethod:
    def test_no_grid_is_unknown_and_closed(self):
        mon = sm.SafetyMonitor()
        res = mon._corridor_feasibility(_Scene(None))
        assert res.state == "unknown"
        assert res.feasible is False

    def test_an_open_corridor_is_feasible(self):
        mon = sm.SafetyMonitor()
        res = mon._corridor_feasibility(_Scene(_Grid(_open())))
        assert res.state == "feasible"
        assert res.feasible is True

    def test_the_999_sentinel_is_not_passed_in_as_a_distance(self):
        # 999 would make "required distance" 999 m and the band would
        # always look too short - or, read the other way, it would be
        # treated as an obstacle a kilometre away.
        mon = sm.SafetyMonitor()
        res = mon._corridor_feasibility(
            _Scene(_Grid(_open()), closest_obs_m=999.0))
        assert res.state == "feasible"

    def test_a_real_distance_is_used(self):
        mon = sm.SafetyMonitor()
        res = mon._corridor_feasibility(
            _Scene(_Grid(_open()), closest_obs_m=8.0))
        # 8 m required, 8.5 m clear on a 40-row / 0.5 m grid
        assert res.available_distance_m == 8.0

    def test_stale_evidence_is_unknown(self):
        mon = sm.SafetyMonitor()
        res = mon._corridor_feasibility(
            _Scene(_Grid(_open()), bev_age_s=9.0))
        assert res.state == "unknown"
        assert res.feasible is False

    def test_an_unreadable_grid_is_unknown_not_open(self):
        """A grid object that raises: the old gate's try/except set
        corridor_open = False, but the P2.1 call has to fail the same
        conservative way."""
        class _Bad:
            n_rows = 40
            n_cols = 40
            res = 0.5

            @property
            def obstacle(self):
                raise RuntimeError("grid reader blew up")

            drivable = None

        mon = sm.SafetyMonitor()
        res = mon._corridor_feasibility(_Scene(_Bad()))
        assert res.state == "unknown"
        assert res.feasible is False
        assert "failed" in res.reason

    def test_a_bad_speed_reads_as_zero_not_as_an_exception(self):
        mon = sm.SafetyMonitor()
        res = mon._corridor_feasibility(
            _Scene(_Grid(_open()), speed_mps="not a number"))
        assert res.state in ("feasible", "infeasible")


class TestUnknownDoesNotOpenTheHatch:
    def test_every_unknown_shape_reports_closed(self, monkeypatch):
        mon = sm.SafetyMonitor()
        scenes = [
            _Scene(None),
            _Scene(_Grid(_open()), bev_age_s=99.0),
        ]
        for sc in scenes:
            assert mon._corridor_feasibility(sc).feasible is False

    def test_a_gridless_scene_is_no_longer_treated_as_open(self, monkeypatch):
        """The old code: grid is None -> corridor_free_band returns True.

        That is what let a scene the pipeline could not read authorise
        cruise.  With the gate on, the answer is unknown and stays closed.
        """
        monkeypatch.setattr(sm, "CORRIDOR_FEASIBILITY_GATE", True)
        mon = sm.SafetyMonitor()
        res = mon._corridor_feasibility(_Scene(None))
        assert res.state == "unknown"
        assert res.feasible is False
