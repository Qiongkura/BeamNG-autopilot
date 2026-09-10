"""Tests for the canonical per-tick perception snapshot."""

from __future__ import annotations

import time

import numpy as np

from beamng_autopilot.perception_snapshot import PerceptionSnapshot


def test_snapshot_validity_and_meta() -> None:
    s = PerceptionSnapshot(
        captured_at=time.time() - 0.1,
        tick_id=7,
        pos=np.array([1.0, 2.0, 0.0]),
        heading=0.3,
        bev=np.zeros((4, 4), dtype=np.float32),
        head_age_s={"semantic": 0.2},
        range_age_s=0.4)
    assert s.valid
    assert s.age_s(time.time()) >= 0.09
    meta = s.meta()
    assert meta["tick_id"] == 7
    assert meta["head_age_s"]["semantic"] == 0.2
    assert meta["range_age_s"] == 0.4
    assert meta["has_bev"] == 1


def test_snapshot_without_sensor_is_invalid() -> None:
    s = PerceptionSnapshot(
        captured_at=time.time(), tick_id=0,
        pos=np.array([0.0, 0.0]), heading=0.0)
    assert not s.valid
