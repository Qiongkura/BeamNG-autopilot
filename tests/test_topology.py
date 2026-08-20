"""Offline tests for the lane-topology (lane-graph) head."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.lane import LaneFrame
from beamng_autopilot.vision.heads.topology import (
    build_lane_graph,
)
from beamng_autopilot.vision.hydra import FrameContext


def _frame(right_kind="dashed", left_kind="dashed", width=3.5,
           left_none=False, right_none=False):
    xs = np.arange(0.0, 12.0, 1.5)
    centre = np.column_stack([xs, np.zeros_like(xs)])
    left = np.column_stack([xs, np.full_like(xs, -1.75)])
    right = np.column_stack([xs, np.full_like(xs, 1.75)])
    return LaneFrame(
        center=centre,
        left=None if left_none else left,
        right=None if right_none else right,
        width=width, confidence=0.8, span_m=12.0,
        sources=("vision",), paired=True,
        left_kind=left_kind, right_kind=right_kind)


def test_graph_painted_sides_both_crossable() -> None:
    g = build_lane_graph(_frame())
    assert g.has_lane
    assert g.left.exists and g.left.crossable
    assert g.right.exists and g.right.crossable
    assert g.labels["change_left_ok"] and g.labels["change_right_ok"]


def test_graph_solid_right_not_crossable() -> None:
    g = build_lane_graph(_frame(right_kind="solid", left_kind="dashed"))
    assert not g.right.crossable
    assert g.left.crossable
    assert not g.labels["change_right_ok"]
    assert g.labels["change_left_ok"]


def test_graph_guardrail_not_crossable() -> None:
    g = build_lane_graph(_frame(left_kind="guardrail"))
    assert not g.left.crossable


def test_graph_single_sided_mirrors() -> None:
    g = build_lane_graph(_frame(right_none=True))
    assert g.right.exists and g.right.crossable  # mirrored assumption
    assert g.right.offset_m == pytest.approx(1.75, abs=0.01)


def test_graph_none_frame_has_no_lane() -> None:
    g = build_lane_graph(None)
    assert not g.has_lane
    assert not g.labels["lane_ok"]


def test_topology_in_hydranet() -> None:
    from beamng_autopilot.vision.heads.topology import LaneTopologyHead
    from beamng_autopilot.vision.projection import CameraModel
    head = LaneTopologyHead()
    ctx = FrameContext(frame_rgb=np.zeros((60, 80, 3), dtype=np.uint8),
                       cam=CameraModel(np.zeros(3), np.array([0., 1., 0.]),
                                       np.array([0., 0., 1.]), 65., 80, 60),
                       pos=np.array([0., 0., 0.]), heading=0.0)
    out = head.run(ctx, sensor_lane=_frame())
    assert out.meta["lane_graph"]["has_lane"]
    assert out.meta["change_left"] and out.meta["change_right"]