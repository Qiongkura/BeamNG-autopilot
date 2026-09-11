"""The right-mirror near window, and why it was widened.

2026-09-11, v13b: of 410 live-fusion ticks whose vision frame carried a
right edge but was not paired, `_mirror_right_ok` passed 0.  Every failure
was the ``len(near) < 2`` branch - with a 3 m window not one right-line
point sat beside the car - while the riding-line branch never fired.  The
nearest right-line point is a median 6.64 m ahead (p90 10.79 m), because
the town line class is short dashed blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.lane import tracking
from beamng_autopilot.lane.constants import (
    LANE_RIGHT_MIRROR_NEAR_LEGACY_M, LANE_RIGHT_MIRROR_NEAR_M,
    LANE_RIDING_LINE_MAX_M)
from beamng_autopilot.lane.pairing import LaneFrame
from beamng_autopilot.lane.tracking import _mirror_right_ok

POS = np.array([0.0, 0.0, 0.0])
HEADING = 0.0


def _frame_with_right(start_lon: float, lat: float, n: int = 6) -> LaneFrame:
    """A single-edge frame whose right line starts ``start_lon`` ahead."""
    lon = np.linspace(start_lon, start_lon + 10.0, n)
    right = np.column_stack([lon, np.full(n, lat)])
    center = np.column_stack([lon, np.zeros(n)])
    return LaneFrame(center=center, right=right, paired=False)


@pytest.fixture(autouse=True)
def _wide_default(monkeypatch):
    monkeypatch.setattr(tracking, "_WIDE_RIGHT_MIRROR_NEAR", True)


def test_line_starting_inside_the_window_passes():
    """With an 8 m window a line starting 4 m ahead has >=2 points in it."""
    assert _mirror_right_ok(_frame_with_right(4.0, -2.0), POS, HEADING)


def test_the_measured_median_start_passes_with_the_real_polyline():
    """The measured median start is 3.76 m, and real polylines are dense.

    The full pipeline gives right polylines a median of 14 points, so a
    line starting at the measured median puts well over the gate's two
    required points inside an 8 m window.
    """
    lon = np.linspace(3.76, 18.0, 14)
    right = np.column_stack([lon, np.full(len(lon), -2.0)])
    frame = LaneFrame(center=np.column_stack([lon, np.zeros(len(lon))]),
                      right=right, paired=False)
    assert _mirror_right_ok(frame, POS, HEADING)


def test_line_starting_beyond_the_window_still_fails():
    """Run 188: paint only far ahead must not steer an unpaired mirror."""
    assert not _mirror_right_ok(_frame_with_right(15.0, -2.0), POS, HEADING)


def test_line_beside_the_car_but_riding_it_still_fails():
    """The lateral test is unchanged - that branch never fired live."""
    riding = -LANE_RIDING_LINE_MAX_M / 2.0      # inside the riding band
    assert not _mirror_right_ok(_frame_with_right(0.5, riding), POS, HEADING)


def test_legacy_window_still_rejects_the_same_frame(monkeypatch):
    monkeypatch.setattr(tracking, "_WIDE_RIGHT_MIRROR_NEAR", False)
    assert not _mirror_right_ok(_frame_with_right(4.0, -2.0), POS, HEADING)


def test_no_right_edge_is_never_a_blocker():
    frame = LaneFrame(center=np.array([[0.0, 0.0], [5.0, 0.0]]), paired=False)
    assert _mirror_right_ok(frame, POS, HEADING)


def test_window_is_the_repo_near_far_start_scale():
    from beamng_autopilot.lane.constants import LANE_ONE_NEAR_FAR_START_MAX_M
    assert LANE_RIGHT_MIRROR_NEAR_M == LANE_ONE_NEAR_FAR_START_MAX_M
    assert LANE_RIGHT_MIRROR_NEAR_LEGACY_M == 3.0
