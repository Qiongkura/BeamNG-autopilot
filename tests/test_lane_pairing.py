"""Pairing regressions for US single-edge lane recovery."""

from __future__ import annotations

import numpy as np

from beamng_autopilot.lane import pair_lane_markings
from beamng_autopilot.vision.lanes import LaneMarking


def _line(y: float, x0: float = 4.0, x1: float = 9.0,
          kind: str = "solid", color: str = "white") -> LaneMarking:
    world = np.column_stack([
        np.linspace(x0, x1, 8), np.full(8, y)])
    return LaneMarking(world=world, pixels=world.copy(),
                       color=color, kind=kind, confidence=0.75)


def test_near_left_painted_edge_is_trusted_in_strict_mode():
    debug = {}
    frame = pair_lane_markings([_line(2.3)], np.zeros(3), 0.0,
                               debug=debug)
    assert frame is not None
    assert frame.paired is False
    assert frame.confidence >= 0.50
    assert debug["mode"] == "mirror_left"


def test_far_yellow_center_line_builds_own_lane():
    world = np.column_stack([
        np.linspace(20.0, 32.0, 12), np.full(12, 1.8)])
    frame = pair_lane_markings([
        LaneMarking(world=world, pixels=world.copy(),
                    color="yellow", kind="solid", confidence=0.9)],
        np.zeros(3), 0.0)
    assert frame is not None
    assert frame.paired is True
    assert frame.span_m >= 6.0


def test_far_left_edge_stays_low_trust():
    debug = {}
    frame = pair_lane_markings([_line(4.0)], np.zeros(3), 0.0,
                               debug=debug)
    assert frame is not None
    assert frame.confidence < 0.50



    debug = {}
    frame = pair_lane_markings([_line(4.0)], np.zeros(3), 0.0,
                               debug=debug)
    assert frame is not None
    assert frame.confidence < 0.50
