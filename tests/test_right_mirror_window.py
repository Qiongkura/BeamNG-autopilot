"""The right-mirror near window, and why it must stay strict.

2026-09-11, v13b: on 410 unpaired frames that carry a right edge the gate
passed 0.  Every failure was ``len(near) < 2`` - with a 3 m window not one
right-line point sat beside the car, because the town dashed line starts a
median 3.76 m ahead - and the riding-line branch never fired.

Widening the window to 8 m was implemented and then reverted the same
hour: measured under the required in-lane constraint (centre within 1.2 m
of the ego-lane centre) it raises lane availability from 81.1% to 82.2%
while the laterally-correct share stays at ~11%.  A single-edge mirror
infers the centre from an assumed width, so these guards are the lateral
correctness layer, not a redundant second owner.

These tests pin that decision.  If one of them is changed to expect a
pass, re-run the in-lane constraint first (see LANE_RIGHT_MIRROR_NEAR_M).
"""

from __future__ import annotations

import numpy as np

from beamng_autopilot.lane.constants import (
    LANE_RIDING_LINE_MAX_M, LANE_RIGHT_MIRROR_NEAR_M)
from beamng_autopilot.lane.pairing import LaneFrame
from beamng_autopilot.lane.tracking import _mirror_right_ok

POS = np.array([0.0, 0.0, 0.0])
HEADING = 0.0


def _frame_with_right(start_lon: float, lat: float, end_lon: float = 18.0,
                      n: int = 14) -> LaneFrame:
    lon = np.linspace(start_lon, end_lon, n)
    right = np.column_stack([lon, np.full(n, lat)])
    return LaneFrame(center=np.column_stack([lon, np.zeros(n)]),
                     right=right, paired=False)


def test_the_measured_median_start_is_rejected_on_purpose():
    """The town right line starts 3.76 m ahead; the 3 m rule rejects it.

    That rejection is the point: the frame has no line beside the car, and
    a mirror built from paint several metres ahead lands the centre off
    the ego lane (~11% in-lane measured either way).
    """
    assert not _mirror_right_ok(_frame_with_right(3.76, -2.0), POS, HEADING)


def test_paint_beside_the_car_clearly_to_the_right_passes():
    assert _mirror_right_ok(_frame_with_right(0.5, -2.0), POS, HEADING)


def test_paint_beside_the_car_but_riding_it_fails():
    riding = -LANE_RIDING_LINE_MAX_M / 2.0
    assert not _mirror_right_ok(_frame_with_right(0.5, riding), POS, HEADING)


def test_far_ahead_paint_never_steers_an_unpaired_mirror():
    """Run 188: a line only ahead must not define the boundary."""
    assert not _mirror_right_ok(_frame_with_right(12.0, -2.0), POS, HEADING)


def test_no_right_edge_is_never_a_blocker():
    frame = LaneFrame(center=np.array([[0.0, 0.0], [5.0, 0.0]]), paired=False)
    assert _mirror_right_ok(frame, POS, HEADING)


def test_window_is_still_the_legacy_three_metres():
    assert LANE_RIGHT_MIRROR_NEAR_M == 3.0
