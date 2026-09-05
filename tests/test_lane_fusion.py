"""Regression tests for the lane-fusion anti-flicker hold."""

from __future__ import annotations

import numpy as np
import pytest



def _frame(paired=True, src=("vision",), yoff=0.0):
    from beamng_autopilot.lane.pairing import LaneFrame
    ys = np.full(6, yoff)
    xs = np.linspace(0.0, 30.0, 6)
    center = np.column_stack([xs, 1.75 + ys])
    left = np.column_stack([xs, 3.5 + ys])
    right = np.column_stack([xs, 0.0 + ys])
    return LaneFrame(center=center, left=left if paired else None,
                     right=right, width=3.5, confidence=0.9,
                     span_m=30.0, sources=src, paired=paired)


def test_fusion_holds_lane_through_one_frame_glitch() -> None:
    """A one-frame glitch to a different source must NOT adopt instantly
    (the old counter tested the ACTIVE source's tenure, which in steady
    state always exceeded the hold - bug audit 2026-09-06)."""
    from beamng_autopilot.lane.fusion import choose_sensor_lane
    pos = np.array([5.0, 0.0])
    state = {}
    good = _frame(paired=True, src=("vision", "lidar"))
    # establish the active paired lane
    for _ in range(5):
        out = choose_sensor_lane(good, None, pos, 0.0, state=state)
    assert out.sources == ("vision", "lidar")
    # one-frame glitch to an unpaired fallback: must be held off
    glitch = _frame(paired=False, src=("lidar",), yoff=3.0)
    out = choose_sensor_lane(None, glitch, pos, 0.0, state=state)
    assert out.sources == ("vision", "lidar"), \
        "a single-frame glitch must not replace the stable lane"
    # back to good: still stable
    out = choose_sensor_lane(good, None, pos, 0.0, state=state)
    assert out.sources == ("vision", "lidar")


def test_fusion_adopts_persistent_new_source() -> None:
    from beamng_autopilot.lane.fusion import choose_sensor_lane
    from beamng_autopilot.lane.constants import LANE_FUSION_HOLD_FRAMES
    pos = np.array([5.0, 0.0])
    state = {}
    good = _frame(paired=True, src=("vision", "lidar"))
    for _ in range(5):
        choose_sensor_lane(good, None, pos, 0.0, state=state)
    other = _frame(paired=True, src=("lidar",), yoff=0.4)
    out = None
    for _ in range(LANE_FUSION_HOLD_FRAMES + 1):
        out = choose_sensor_lane(None, other, pos, 0.0, state=state)
    assert out.sources == ("lidar",), \
        "a source that persists past the hold window must adopt"
