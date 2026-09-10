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


def _usable(paired=True, src=("vision",), span=3.5, width=3.5,
            conf=0.5):
    from beamng_autopilot.lane.pairing import LaneFrame
    xs = np.linspace(0.0, 30.0, 6)
    return LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, -1.75)]),
        left=np.column_stack([xs, np.zeros_like(xs)]),
        right=np.column_stack([xs, np.full_like(xs, -3.5)]),
        width=width, confidence=conf, span_m=span, sources=src,
        paired=paired)


def test_short_pair_frame_is_not_usable() -> None:
    """The span floor is strict even for a two-sided painted pair.

    A 3.2 m pair can have its centre metres off the nav route while every
    point is "near" the car; that dragged the car sideways in runs
    42/47/52.  Measured over 654 shadow frames this floor rejects almost
    every vision pair on the 2026-09-07 town runs, which is intended -
    the fix is a longer detected line, not a lower floor.  Nothing else
    in the suite covers ``lane_frame_usable``, so pin it here.
    """
    from beamng_autopilot.lane.constants import (
        LANE_PAIRED_VISION_MIN_SPAN_M)
    from beamng_autopilot.lane.tracking import lane_frame_usable
    assert not lane_frame_usable(_usable(span=3.5))
    assert not lane_frame_usable(_usable(span=3.0))
    assert not lane_frame_usable(
        _usable(span=LANE_PAIRED_VISION_MIN_SPAN_M - 0.1))
    assert lane_frame_usable(_usable(span=LANE_PAIRED_VISION_MIN_SPAN_M))


def test_fused_overlap_uses_the_same_span_floor() -> None:
    """A vision+LiDAR overlap is held to the same floor as a painted pair."""
    from beamng_autopilot.lane.tracking import lane_frame_usable
    assert not lane_frame_usable(
        _usable(src=("vision", "lidar"), span=4.5))
    assert lane_frame_usable(
        _usable(src=("vision", "lidar"), span=6.0))
    # a mirror frame is judged by the weaker single-edge floor
    assert lane_frame_usable(_usable(paired=False, span=3.5))


def test_usable_gate_still_bounds_width_and_confidence() -> None:
    from beamng_autopilot.lane.constants import LANE_MIN_CONF
    from beamng_autopilot.lane.tracking import lane_frame_usable
    assert not lane_frame_usable(_usable(width=8.0))     # whole road
    assert not lane_frame_usable(_usable(width=1.2))     # straddled line
    assert not lane_frame_usable(_usable(conf=LANE_MIN_CONF - 0.01))
    assert not lane_frame_usable(None)
