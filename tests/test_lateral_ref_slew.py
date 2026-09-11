"""Bounded tick-to-tick slew of the accepted own-lane reference."""

from __future__ import annotations

import numpy as np

from beamng_autopilot.planning.lateral_ref import (
    LANE_REF_SLEW_HOLD_MAX_S, LANE_REF_SLEW_MAX_M, limit_reference_slew,
    near_lat,
)

POS = np.array([0.0, 0.0, 0.0])
HEADING = 0.0


def _ref(lat: float, n: int = 4):
    """Straight reference beside the ego at lateral offset ``lat`` (left +)."""
    x = np.linspace(0.0, 12.0, n)
    return np.column_stack([x, np.full(n, lat)])


def test_near_lat_reads_the_side_correctly():
    assert near_lat(_ref(1.0), POS, HEADING) > 0.9
    assert near_lat(_ref(-1.0), POS, HEADING) < -0.9
    assert near_lat(None, POS, HEADING) is None


def test_first_tick_accepts_the_reference():
    ref, hold = limit_reference_slew(None, 0.0, _ref(1.0), 0.0, POS, HEADING)
    assert ref is not None and hold == 0.0


def test_small_move_is_accepted():
    prev = _ref(1.0)
    new = _ref(1.4)                      # 0.4 m step, under the 0.8 bound
    ref, hold = limit_reference_slew(prev, 0.0, new, 10.0, POS, HEADING)
    assert np.allclose(ref, new)
    assert hold == 0.0


def test_sideways_teleport_is_rejected_and_starts_a_hold():
    prev, new = _ref(1.0), _ref(-1.2)    # 2.2 m step, the measured worst case
    ref, hold = limit_reference_slew(prev, 0.0, new, 10.0, POS, HEADING)
    assert np.allclose(ref, prev)
    assert hold == 10.0


def test_hold_expires_so_a_real_change_is_not_blocked():
    prev, new = _ref(1.0), _ref(-1.2)
    held_start = 10.0
    # still inside the window -> keep holding
    ref, hold = limit_reference_slew(
        prev, held_start, new, held_start + 0.5, POS, HEADING)
    assert np.allclose(ref, prev) and hold == held_start
    # past the window -> accept the new reference (junction / new lane)
    ref, hold = limit_reference_slew(
        prev, held_start, new,
        held_start + LANE_REF_SLEW_HOLD_MAX_S + 0.1, POS, HEADING)
    assert np.allclose(ref, new) and hold == 0.0


def test_absent_reference_publishes_none():
    ref, hold = limit_reference_slew(_ref(1.0), 5.0, None, 6.0, POS,
                                    HEADING)
    assert ref is None and hold == 0.0


def test_bound_is_the_documented_measured_scale():
    """The bound must sit above the stable reference and below the jumps."""
    from beamng_autopilot.planning.lateral_ref import LANE_REF_SLEW_MAX_M
    assert 0.41 <= LANE_REF_SLEW_MAX_M < 2.20
