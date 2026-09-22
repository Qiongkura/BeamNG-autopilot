"""P1-3: is the car standing on pavement the sensors can see?

`road_off` cannot answer this when no lane boundary is published - it
returns 0 for "inside the boundary" and for "never measured" alike, which
is exactly the state of the crash run.  The drivable mask can answer it,
in three states, and ``unknown`` must never be read as ``on_road``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from beamng_autopilot import safety_monitor as sm
from beamng_autopilot.occupancy import (
    BODY_COV_OFF_ROAD,
    BODY_COV_ON_ROAD,
    BODY_COV_UNKNOWN,
    OccupancyGrid,
    body_drivable_coverage,
)
from beamng_autopilot.planning import Scene
from beamng_autopilot import safety_monitor as _sm


def _grid(**kw):
    g = OccupancyGrid(60, 60, 0.5)
    for key, val in kw.items():
        setattr(g, key, val)
    return g


def _road_grid(n_drivable_cols: int | None = None, observed: bool = True):
    g = _grid()
    if observed:
        g.observed[:] = 1
    if n_drivable_cols is None:
        g.drivable[:] = 1
    elif n_drivable_cols > 0:
        mid = g.n_cols // 2
        half = n_drivable_cols // 2
        g.drivable[:, mid - half:mid + half] = 1
    return g


class TestPrimitive:
    def test_unknown_when_nothing_was_observed(self):
        rep = body_drivable_coverage(_grid(), np.zeros(3), 0.0, 2.2, 0.9)
        assert rep["status"] == BODY_COV_UNKNOWN
        assert rep["coverage"] is None          # not 0, not 1
        assert rep["observed_cells"] == 0
        assert rep["footprint_cells"] > 0

    def test_on_road_under_the_car(self):
        rep = body_drivable_coverage(_road_grid(), np.zeros(3), 0.0, 2.2, 0.9)
        assert rep["status"] == BODY_COV_ON_ROAD
        assert rep["coverage"] >= rep["min_frac"]

    def test_off_road_when_the_footprint_is_observed_but_not_pavement(self):
        rep = body_drivable_coverage(_road_grid(0), np.zeros(3), 0.0, 2.2, 0.9)
        assert rep["status"] == BODY_COV_OFF_ROAD
        assert rep["coverage"] == 0.0

    def test_too_few_observed_cells_stays_unknown(self):
        g = _grid()
        g.observed[:] = 0
        g.observed[30, 30] = 1                 # a single cell
        g.drivable[:] = 0
        rep = body_drivable_coverage(g, np.zeros(3), 0.0, 2.2, 0.9)
        assert rep["status"] == BODY_COV_UNKNOWN

    def test_a_thin_sample_of_the_footprint_is_unknown_not_off_road(self):
        """The field failure: 6 observed cells of 72 read ``off_road``.

        Measured on Tech (logs/goal_20260921/body_cov): the front camera's
        nearest visible ground is 2.4 m ahead of the ego centre, 0 cells in
        0-2 m, and the only footprint cells it stamps are at 2-3 m - the
        fringe of the blind zone, where the road mask reads "not road".
        Eight such cells used to be enough for a verdict; on open pavement
        that verdict was ``off_road``.
        """
        g = _grid()
        g.observed[:] = 0
        g.observed[30:34, 30:32] = 1           # 8 cells, none drivable
        g.drivable[:] = 0
        rep = body_drivable_coverage(g, np.zeros(3), 0.0, 2.2, 0.9)
        assert rep["observed_cells"] >= rep["min_observed"]   # the count passes
        assert rep["observed_frac"] < rep["min_observed_frac"]
        assert rep["status"] == BODY_COV_UNKNOWN
        assert rep["coverage"] is None

    def test_the_observed_fraction_is_always_published(self):
        rep = body_drivable_coverage(_road_grid(), np.zeros(3), 0.0, 2.2, 0.9)
        assert rep["observed_frac"] == pytest.approx(
            rep["observed_cells"] / rep["footprint_cells"], abs=1e-3)
        assert rep["footprint_cells"] > 0

    def test_a_covered_footprint_stays_judgeable(self):
        """The frac floor must not swallow the verdict it exists to protect."""
        rep = body_drivable_coverage(_road_grid(0), np.zeros(3), 0.0, 2.2, 0.9)
        assert rep["observed_frac"] == 1.0
        assert rep["status"] == BODY_COV_OFF_ROAD

    def test_it_reads_no_map_geometry(self):
        """It takes a GRID, so it structurally cannot see the nav route."""
        sig = inspect.signature(body_drivable_coverage)
        assert "grid" in sig.parameters
        assert "scene" not in sig.parameters
        src = inspect.getsource(body_drivable_coverage)
        for banned in ("nav_route", "scene.route", "lane_ref", "DecalRoad",
                       "right_offset"):
            assert banned not in src, banned


def _scene(grid):
    return Scene(pos=np.zeros(2), heading=0.0, grid=grid,
                 lane_ref=np.column_stack([np.arange(21.0), np.zeros(21)]),
                 strict_perception=True)


class TestMonitorGate:
    def _path(self):
        return np.column_stack([np.arange(0.0, 20.0), np.zeros(20)])

    def test_the_gate_is_off_by_default_and_still_measured(self):
        assert sm.BODY_COVERAGE_GATE_ENABLED is False
        mon = sm.SafetyMonitor(max_speed=6.0)
        v = mon.evaluate(_scene(_road_grid(0)), self._path(), now_s=0.0)
        assert v.body_cov_status == BODY_COV_OFF_ROAD
        assert v.body_cov_checked is True
        assert v.body_cov_frac == 0.0
        # ...and with the switch off the car is NOT stopped for it
        assert v.effective_rule != "body_off_pavement"

    def test_off_road_degrades_then_fails_closed(self):
        """Creep after ONE confirming observation, stop after THREE."""
        mon = sm.SafetyMonitor(max_speed=6.0)
        mon.body_coverage_gate = True
        path, grid = self._path(), _road_grid(0)
        v0 = mon.evaluate(_scene(grid), path, now_s=0.0)
        assert v0.level != "minimal_risk"       # first observation
        v1 = mon.evaluate(_scene(grid), path, now_s=1.0)
        assert v1.effective_rule == "body_off_pavement"
        assert v1.level == "degraded"           # 1.0 s of confirmed evidence
        assert v1.target_speed <= sm.BODY_COV_CREEP_MPS
        mon.evaluate(_scene(grid), path, now_s=2.0)
        v3 = mon.evaluate(_scene(grid), path, now_s=3.0)
        assert v3.body_cov_low_s >= sm.BODY_COV_STOP_S
        assert v3.level == "minimal_risk"
        assert v3.target_speed == 0.0

    def test_returning_to_pavement_clears_the_clock(self):
        mon = sm.SafetyMonitor(max_speed=6.0)
        mon.body_coverage_gate = True
        path = self._path()
        mon.evaluate(_scene(_road_grid(0)), path, now_s=0.0)
        mon.evaluate(_scene(_road_grid(0)), path, now_s=2.0)
        v = mon.evaluate(_scene(_road_grid()), path, now_s=2.5)
        assert v.body_cov_status == BODY_COV_ON_ROAD
        assert v.body_cov_low_s == 0.0
        # the clock restarted, so a single fresh off-road tick does not stop
        v2 = mon.evaluate(_scene(_road_grid(0)), path, now_s=2.6)
        assert v2.level != "minimal_risk"
        assert v2.body_cov_low_s == pytest.approx(0.1)

    def test_unknown_neither_confirms_nor_clears(self):
        """A blind stretch pauses the clock; it is not proof of anything."""
        mon = sm.SafetyMonitor(max_speed=6.0)
        mon.body_coverage_gate = True
        path = self._path()
        mon.evaluate(_scene(_road_grid(0)), path, now_s=0.0)
        v = mon.evaluate(_scene(_road_grid(0)), path, now_s=1.0)
        assert v.body_cov_low_s >= 1.0
        # blind for a long time: the clock is PAUSED, the car is not stopped
        for t in (2.0, 10.0, 30.0):
            v = mon.evaluate(_scene(_grid()), path, now_s=t)
            assert v.body_cov_status == BODY_COV_UNKNOWN
            assert v.level != "minimal_risk"
            assert v.body_cov_low_s == pytest.approx(1.0)
        # ...and the evidence resumes: the paused clock continues from 1 s
        # the tick that resumes charges its own (capped) step only
        v = mon.evaluate(_scene(_road_grid(0)), path, now_s=31.5)
        assert v.body_cov_low_s == pytest.approx(1.0 + sm.BODY_COV_MAX_STEP_S)

    def test_one_tick_cannot_fabricate_seconds_of_evidence(self):
        """A slow tick charges at most BODY_COV_MAX_STEP_S to the clock."""
        mon = sm.SafetyMonitor(max_speed=6.0)
        mon.body_coverage_gate = True
        path = self._path()
        mon.evaluate(_scene(_road_grid(0)), path, now_s=0.0)
        v = mon.evaluate(_scene(_road_grid(0)), path, now_s=6.0)
        assert v.body_cov_low_s == pytest.approx(sm.BODY_COV_MAX_STEP_S)
        assert v.level != "minimal_risk"      # one observation, not six seconds

    def test_a_thin_sample_never_fires_the_gate(self):
        """The live false positive, at monitor level: on pavement, thin
        sample, gate on - the car must NOT be slowed or stopped."""
        g = _grid()
        g.observed[:] = 0
        g.observed[30:34, 30:32] = 1
        g.drivable[:] = 0
        mon = sm.SafetyMonitor(max_speed=6.0)
        mon.body_coverage_gate = True
        path = self._path()
        for t in (0.0, 1.0, 5.0, 20.0):
            v = mon.evaluate(_scene(g), path, now_s=t)
            assert v.body_cov_status == BODY_COV_UNKNOWN
            assert v.level != "minimal_risk"
            assert v.target_speed > sm.BODY_COV_CREEP_MPS
            assert v.effective_rule != "body_off_pavement"
        assert v.body_cov_observed_frac < 0.3

    def test_a_stale_tick_does_not_advance_the_clock(self):
        mon = sm.SafetyMonitor(max_speed=6.0)
        mon.body_coverage_gate = True
        path = self._path()
        mon.evaluate(_scene(_road_grid(0)), path, now_s=0.0)
        v = mon.evaluate(_scene(_road_grid(0)), path, now_s=5.0,
                         snapshot_age_s=99.0)
        assert v.body_cov_checked is False
        assert v.level != "minimal_risk"


# --------------------------------------------------------------------------
# T05 反例: different vehicle footprints
# --------------------------------------------------------------------------
class TestFootprintSensitivity:
    """The body check must follow the VEHICLE, not one hard-coded size."""

    def _grid_with_road(self, half_width_cells: int):
        g = _grid()
        g.observed[:] = 1
        mid = g.n_cols // 2
        g.drivable[:, mid - half_width_cells:mid + half_width_cells] = 1
        return g

    def test_a_wider_vehicle_reads_less_coverage_on_the_same_road(self):
        """Measured on a 1.0 m half-width drivable band (0.5 m cells):

        a 1.8 m-wide car reads 0.50 coverage (on_road, exactly at the
        threshold) while a 2.6 m one reads 0.33 (off_road) - the same road,
        the same sensors, a different vehicle.  The check must therefore
        never use one hard-coded footprint.
        """
        g = self._grid_with_road(1)          # 1 cell = 0.5 m half-width
        narrow = body_drivable_coverage(g, np.zeros(3), 0.0, 2.2, 0.9)
        wide = body_drivable_coverage(g, np.zeros(3), 0.0, 2.6, 1.3)
        assert narrow["status"] == BODY_COV_ON_ROAD
        assert wide["status"] == BODY_COV_OFF_ROAD
        assert wide["coverage"] < narrow["coverage"]

    def test_the_production_footprint_comes_from_the_config(self):
        from beamng_autopilot import geometry as G
        from beamng_autopilot.config import (EGO_HALF_LENGTH_M,
                                             EGO_HALF_WIDTH_M)
        assert G.FOOTPRINT_HALF_LENGTH_M == float(EGO_HALF_LENGTH_M)
        assert G.FOOTPRINT_HALF_WIDTH_M == float(EGO_HALF_WIDTH_M)
        # the monitor's own check must use the same numbers
        rep = body_drivable_coverage(self._grid_with_road(4), np.zeros(3),
                                     0.0, G.FOOTPRINT_HALF_LENGTH_M,
                                     G.FOOTPRINT_HALF_WIDTH_M)
        assert rep["footprint_cells"] > 0
