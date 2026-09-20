"""Positive and negative cases for the feasibility primitive (P2.2).

The old gate's failure was not that it was wrong on the case it was
written for - it was that "there is a gap somewhere" got used as "I can
drive through".  So the negative cases here are not exotic: each is a
shape a town run actually produces, and each has to come back INFEASIBLE
or UNKNOWN rather than True.
"""

import numpy as np
import pytest

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
        self.drivable = (None if drivable is None
                         else np.asarray(drivable, dtype=np.int8))


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
        """Narrow but sufficient: a 3 m gap with a 1.9 m body and 0.5 m
        margin is 2.4 m of requirement, so it passes."""
        occ = _empty()
        occ[:, :12] = 1
        occ[:, 24:] = 1          # leaves cols 12..23 = 6.0 m
        res = corridor_feasibility(_Scene(_Grid(occ)))
        assert res.state == FEASIBLE
        assert res.width_m == pytest.approx(6.0, abs=0.01)

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
                                   required_distance_m=8.0)
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
