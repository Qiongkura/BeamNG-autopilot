"""Tests for the canonical sensor lane envelope."""

from __future__ import annotations

import time

import numpy as np
import pytest

from beamng_autopilot.lane.envelope import SensorLaneEnvelope
from beamng_autopilot.lane.pairing import LaneFrame


def _lane(paired=True):
    x = np.linspace(0.0, 10.0, 6)
    return LaneFrame(
        center=np.column_stack([x, np.full(6, 1.75)]),
        left=np.column_stack([x, np.full(6, 3.5)]) if paired else None,
        right=np.column_stack([x, np.zeros(6)]) if paired else None,
        width=3.5, confidence=0.8, span_m=10.0,
        sources=("vision",), paired=paired)


def test_envelope_preserves_geometry_and_provenance():
    e = SensorLaneEnvelope.from_lane_frame(_lane(), captured_at=time.time())
    assert e.valid
    assert e.paired is True
    assert e.left_real and e.right_real
    assert e.virtual_boundary is False
    assert e.source == "vision"
    assert e.width_m == pytest.approx(3.5)
    assert e.as_meta()["paired"] == 1


def test_single_lane_marks_virtual_boundary_and_age():
    e = SensorLaneEnvelope.from_lane_frame(
        _lane(False), captured_at=time.time() - 0.2,
        uncertainty_m=0.4)
    assert e.valid
    assert e.paired is False
    assert e.virtual_boundary is True
    assert e.right_real is False
    assert e.age_s >= 0.15
    assert e.as_meta()["uncertainty_m"] == pytest.approx(0.4)


def test_none_frame_returns_none():
    assert SensorLaneEnvelope.from_lane_frame(None) is None
