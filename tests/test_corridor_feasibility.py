"""Positive and negative cases for the feasibility primitive (P2.2).

The old gate's failure was not that it was wrong on the case it was
written for - it was that "there is a gap somewhere" got used as "I can
drive through".  So the negative cases here are not exotic: each is a
shape a town run actually produces, and each has to come back INFEASIBLE
or UNKNOWN rather than True.
"""

import numpy as np
import pytest

from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.vehicle_body import HALF_LENGTH_M
from beamng_autopilot.planning.corridor_feasibility import (
    FEASIBLE,
    INFEASIBLE,
    UNKNOWN,
    CorridorFeasibility,
    corridor_feasibility,
    free_intervals,
    intervals_connected,
    max_lateral_shift_m,
)

RES = 0.5
N_ROWS = 40            # ego row = 20
N_COLS = 40            # 20 m wide


class _Grid:
    def __init__(self, obstacle, drivable=None, res=RES):
        self.obstacle = np.asarray(obstacle, dtype=np.int8)
        self.n_rows, self.n_cols = self.obstacle.shape
        self.res = res
        self.drivable = (np.ones_like(self.obstacle) if drivable is None
                         else np.asarray(drivable, dtype=np.int8))
        self.observed = np.ones_like(self.obstacle)


class _Scene:
    def __init__(self, grid):
        self.grid = grid


def _empty():
    return np.zeros((N_ROWS, N_COLS), dtype=np.int8)


class TestIntervals:
    def test_a_wide_gap_is_one_interval(self):
        row = np.zeros(20, dtype=bool)
        row[3:12] = True
        assert free_intervals(row, 5) == [(3, 11)]

    def test_two_narrow_gaps_are_not_one_wide_one(self):
        """The counted-cells error: 3 + 3 free cells scored as a gap the
        car fits through.  Neither half is wide enough, so neither is a
        band."""
        row = np.zeros(20, dtype=bool)
        row[1:4] = True
        row[10:13] = True
        assert free_intervals(row, 5) == []

    def test_a_gap_at_the_row_edge_counts(self):
        row = np.zeros(20, dtype=bool)
        row[15:] = True
        assert free_intervals(row, 5) == [(15, 19)]

    def test_intervals_that_overlap_connect(self):
        assert intervals_connected((0, 9), (5, 14)) is True

    def test_intervals_on_opposite_sides_do_not(self):
        # A band that jumps from the left edge to the right edge between
        # two adjacent rows is two different bands.
        assert intervals_connected((0, 9), (25, 34)) is False


class TestLateralReach:
    def test_standstill_is_bounded_by_geometry_not_by_time(self):
        # At v=0 a time-based bound would say "unlimited, you have all
        # day", which is the same error as "a gap exists".
        assert max_lateral_shift_m(1.0, 0.0) == pytest.approx(1.0 / 11.0)

    def test_shifting_further_takes_quadratically_more_room(self):
        assert max_lateral_shift_m(2.0, 0.0) == pytest.approx(
            4 * max_lateral_shift_m(1.0, 0.0))

    def test_speed_caps_it_below_the_geometric_ceiling(self):
        # 10 m at 10 m/s is one second; the lateral rate binds.
        assert max_lateral_shift_m(10.0, 10.0) == pytest.approx(2.0)

    def test_zero_distance_is_zero_shift(self):
        assert max_lateral_shift_m(0.0, 5.0) == 0.0


class TestPositive:
    def test_an_open_corridor_is_feasible(self):
        res = corridor_feasibility(_Scene(_Grid(_empty())))
        assert res.state == FEASIBLE
        assert res.feasible is True
        assert res.clear_distance_m > 0.0

    def test_a_band_wide_enough_for_the_body_only(self):
        """A centred 3 m gap admits the 1.9 m body plus 0.5 m margin."""
        occ = _empty()
        occ[:, :17] = 1
        occ[:, 23:] = 1          # leaves cols 17..22 = 3.0 m
        res = corridor_feasibility(_Scene(_Grid(occ)))
        assert res.state == FEASIBLE
        assert res.width_m == pytest.approx(3.0, abs=0.01)
        assert res.lateral_shift_m == 0.0

    def test_the_evidence_is_carried_on_the_answer(self):
        res = corridor_feasibility(
            _Scene(_Grid(_empty())), candidate_id="cand_3",
            evidence={"source": "bev", "version": 12, "age_s": 0.1})
        assert res.candidate_id == "cand_3"
        assert res.evidence["version"] == 12


class TestNegativeGeometry:
    def test_a_gap_that_alternates_sides_is_not_a_band(self):
        """Fragmented free space: every row has a wide gap, but they are
        on opposite sides, so no line leads through.  The old gate counted
        free cells per row and called this open."""
        occ = _empty()
        for r in range(N_ROWS):
            if r % 2 == 0:
                occ[r, 20:] = 1     # gap on the left
            else:
                occ[r, :20] = 1     # gap on the right
        res = corridor_feasibility(_Scene(_Grid(occ)),
                                   required_distance_m=5.0)
        assert res.state == INFEASIBLE
        assert res.feasible is False

    def test_a_thin_wall_between_sampled_rows_closes_it(self):
        """A one-row barrier the sampler can step over: rows either side
        are wide open, so counting per-row free cells never sees it."""
        occ = _empty()
        occ[8, :] = 1
        res = corridor_feasibility(_Scene(_Grid(occ)),
                                   required_distance_m=7.0)
        assert res.state == INFEASIBLE
        assert res.clear_distance_m < 7.0

    def test_a_band_that_ends_before_the_obstacle_is_not_enough(self):
        occ = _empty()
        occ[:3, :] = 1              # open nearby, walled off further out
        res = corridor_feasibility(_Scene(_Grid(occ)),
                                   required_distance_m=9.0)
        assert res.state == INFEASIBLE
        assert res.reason == "band ends before the obstacle"

    def test_space_exists_but_the_car_cannot_get_there(self):
        """The 2026-09-20 shape in miniature: a gap exists, the car is
        nearly stopped, and the shift needs more room than it has."""
        occ = _empty()
        for r in range(N_ROWS):
            occ[r, :30] = 1         # the only band is far right
        res = corridor_feasibility(_Scene(_Grid(occ)),
                                   ego_speed_mps=0.5,
                                   required_distance_m=3.0)
        assert res.state == INFEASIBLE
        assert res.reason == "band not reachable in the distance left"
        assert res.lateral_shift_m > res.reachable_shift_m


class TestNegativeDrivability:
    def test_a_gap_off_the_pavement_is_not_a_gap(self):
        occ = _empty()
        occ[:, :30] = 1             # only the right side is obstacle-free
        drivable = np.ones((N_ROWS, N_COLS), dtype=np.int8)
        drivable[:, 30:] = 0        # but it is not drivable
        res = corridor_feasibility(_Scene(_Grid(occ, drivable)))
        assert res.state == INFEASIBLE

    def test_a_mismatched_drivable_mask_is_unknown_not_ignored(self):
        # Ignoring the mask would silently drop the pavement check and
        # put "not checked" back where it was.
        occ = _empty()
        res = corridor_feasibility(
            _Scene(_Grid(occ, np.ones((3, 3), dtype=np.int8))))
        assert res.state == UNKNOWN
        assert res.reason == "drivable mask shape mismatch"


class TestUnknown:
    def test_no_grid(self):
        res = corridor_feasibility(_Scene(None))
        assert res.state == UNKNOWN
        assert res.reason == "no grid"

    def test_no_obstacle_layer(self):
        res = corridor_feasibility(_Scene(_Grid(np.zeros((0, 0)))))
        assert res.state == UNKNOWN
        assert res.reason == "no obstacle layer"

    def test_evidence_too_old(self):
        res = corridor_feasibility(
            _Scene(_Grid(_empty())),
            evidence={"source": "bev", "age_s": 5.0},
            max_evidence_age_s=2.0)
        assert res.state == UNKNOWN
        assert res.reason == "evidence too old"

    def test_unknown_never_opens_the_escape_hatch(self):
        """The whole point of the three-state answer.

        The old bool returned True when the grid was missing, so a scene
        that could not be read at all was the one that authorised cruise.
        """
        for rec in (corridor_feasibility(_Scene(None)),
                    corridor_feasibility(_Scene(_Grid(np.zeros((0, 0))))),
                    corridor_feasibility(_Scene(_Grid(_empty())),
                                         evidence={"age_s": 9.0})):
            assert rec.state == UNKNOWN
            assert rec.feasible is False

    def test_a_default_result_is_unknown(self):
        assert CorridorFeasibility().state == UNKNOWN
        assert CorridorFeasibility().feasible is False

    def test_the_answer_serialises_for_the_shadow_replay(self):
        d = corridor_feasibility(_Scene(_Grid(_empty()))).as_dict()
        assert set(d) >= {"state", "reason", "clear_distance_m",
                          "lateral_shift_m", "reachable_shift_m",
                          "evidence", "notes"}


class TestPhysicalCorridor:
    @staticmethod
    def scene(n=60, cell=0.5):
        grid = OccupancyGrid(n, n, cell)
        grid.drivable[:] = 1.0
        grid.observed[:] = 1.0
        return _Scene(grid)

    @pytest.mark.parametrize("n,cell", [(40, 0.5), (60, 0.5), (120, 0.25)])
    def test_wall_just_beyond_nose_is_never_skipped(self, n, cell):
        scene = self.scene(n, cell)
        grid = scene.grid
        row, _ = grid.ego_to_cell(HALF_LENGTH_M + cell, 0.0)
        grid.obstacle[row, :] = 1
        distance = grid.max_x - (row + 0.5) * cell
        result = corridor_feasibility(scene, required_distance_m=distance)
        assert result.state == INFEASIBLE
        assert result.clear_distance_m <= distance

    def test_single_cell_overlap_cannot_connect_vehicle_width_gaps(self):
        scene = self.scene()
        grid = scene.grid
        for row in range(8):
            grid.obstacle[row, :] = 1
            first = 27 if row % 2 else 31
            grid.obstacle[row, first:first + 5] = 0
        result = corridor_feasibility(scene, required_distance_m=14.0)
        assert result.state == INFEASIBLE
        assert result.reason == "vehicle-centre bands disconnected"

    def test_near_entry_cannot_borrow_the_distant_horizon(self):
        scene = self.scene()
        scene.grid.obstacle[:, :34] = 1
        result = corridor_feasibility(scene, required_distance_m=12.0)
        assert result.state == INFEASIBLE
        assert result.reason == "band not reachable in the distance left"
        assert result.reachable_shift_m == 0.0

    def test_lateral_sign_and_cell_centres_match_occupancy_grid(self):
        scene = self.scene()
        scene.grid.obstacle[:, :] = 1
        scene.grid.obstacle[:, 30:36] = 0
        result = corridor_feasibility(scene, ego_lateral_m=-1.5,
                                       required_distance_m=8.0)
        assert result.state == FEASIBLE
        assert result.centre_lateral_m == -1.5
        assert result.lateral_shift_m == 0.0
        assert result.clear_distance_m == 8.0

    def test_ego_inside_wide_band_does_not_need_to_reach_its_midpoint(self):
        scene = self.scene()
        scene.grid.obstacle[:, :25] = 1
        result = corridor_feasibility(scene, required_distance_m=3.0)
        assert result.state == FEASIBLE
        assert result.lateral_shift_m == 0.0

    def test_obstacle_beyond_requested_distance_does_not_shorten_known_band(self):
        scene = self.scene()
        scene.grid.obstacle[5, :] = 1
        result = corridor_feasibility(scene, required_distance_m=6.0)
        assert result.state == FEASIBLE
        assert result.clear_distance_m == 6.0

    def test_required_distance_beyond_grid_is_unknown(self):
        result = corridor_feasibility(self.scene(), required_distance_m=20.0)
        assert result.state == UNKNOWN
        assert "horizon" in result.reason

    def test_geometry_answer_does_not_claim_a_validated_maneuver(self):
        result = corridor_feasibility(self.scene())
        assert any("geometry only" in note for note in result.notes)


class TestInvalidOrMissingEvidence:
    @pytest.mark.parametrize("layer", ["drivable", "observed"])
    def test_unobserved_road_does_not_open_a_corridor(self, layer):
        scene = TestPhysicalCorridor.scene()
        getattr(scene.grid, layer)[:] = 0
        if layer == "drivable":
            scene.grid.observed[:] = 0
        result = corridor_feasibility(scene)
        assert result.state == UNKNOWN
        assert not result.feasible

    def test_observed_nonroad_is_infeasible(self):
        scene = TestPhysicalCorridor.scene()
        scene.grid.drivable[:] = 0
        assert corridor_feasibility(scene).state == INFEASIBLE

    def test_missing_drivable_layer_is_unknown(self):
        grid = _Grid(_empty())
        grid.drivable = None
        assert corridor_feasibility(_Scene(grid)).state == UNKNOWN

    @pytest.mark.parametrize("age", [float("nan"), float("inf"), -1.0, "bad"])
    def test_invalid_age_is_unknown(self, age):
        result = corridor_feasibility(TestPhysicalCorridor.scene(),
                                       evidence={"age_s": age})
        assert result.state == UNKNOWN
        assert result.reason == "evidence age unreadable"

    @pytest.mark.parametrize("key,value", [
        ("ego_speed_mps", float("nan")),
        ("ego_speed_mps", -1.0),
        ("ego_lateral_m", float("inf")),
        ("vehicle_width_m", 0.0),
        ("vehicle_width_m", 1e308),
        ("margin_m", -0.5),
        ("min_turn_radius_m", 0.0),
        ("max_lateral_speed_mps", float("nan")),
        ("required_distance_m", -2.0),
        ("required_distance_m", float("inf")),
        ("max_evidence_age_s", float("nan")),
    ])
    def test_invalid_parameters_are_unknown(self, key, value):
        result = corridor_feasibility(TestPhysicalCorridor.scene(),
                                       **{key: value})
        assert result.state == UNKNOWN
        assert not result.feasible

    @pytest.mark.parametrize("layer", ["obstacle", "drivable", "observed"])
    def test_nonfinite_layers_are_unknown(self, layer):
        scene = TestPhysicalCorridor.scene()
        values = np.asarray(getattr(scene.grid, layer), dtype=float)
        values[10, 30] = np.nan
        setattr(scene.grid, layer, values)
        assert corridor_feasibility(scene).state == UNKNOWN

    def test_obstacle_shape_mismatch_is_unknown(self):
        scene = TestPhysicalCorridor.scene()
        scene.grid.obstacle = np.zeros((5, 5))
        assert corridor_feasibility(scene).state == UNKNOWN

    def test_ragged_obstacle_layer_is_unknown(self):
        scene = TestPhysicalCorridor.scene()
        scene.grid.obstacle = [[0, 0], [0]]
        assert corridor_feasibility(scene).state == UNKNOWN

    def test_reachability_overflow_is_unknown(self):
        result = corridor_feasibility(TestPhysicalCorridor.scene(),
                                       min_turn_radius_m=1e-320)
        assert result.state == UNKNOWN

    def test_nonfinite_resolution_is_unknown(self):
        scene = TestPhysicalCorridor.scene()
        scene.grid.res = float("nan")
        assert corridor_feasibility(scene).state == UNKNOWN

    @pytest.mark.parametrize("distance,speed,radius,rate", [
        (float("nan"), 1.0, 5.5, 2.0),
        (1.0, float("inf"), 5.5, 2.0),
        (1.0, 1.0, 0.0, 2.0),
        (1.0, 1.0, 5.5, -2.0),
    ])
    def test_reachability_rejects_nonfinite_or_invalid_limits(
            self, distance, speed, radius, rate):
        with pytest.raises(ValueError):
            max_lateral_shift_m(distance, speed, radius, rate)
