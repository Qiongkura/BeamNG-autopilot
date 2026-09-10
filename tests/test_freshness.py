"""Freshness and perception-loss regression tests."""

from __future__ import annotations

import time

import numpy as np
import pytest

from beamng_autopilot.planning import Scene
from beamng_autopilot.safety_monitor import SafetyMonitor, _perception_freshness
from beamng_autopilot.perception_snapshot import PerceptionSnapshot


def test_missing_head_age_is_stale_not_fresh():
    scene = Scene(
        pos=np.array([0.0, 0.0]), heading=0.0,
        meta={"head_age_s": {"semantic": None},
              "bev_age_s": 0.0, "range_age_s": 0.0})
    f = _perception_freshness(scene, snapshot_age_s=0.0)
    assert f["max_s"] == float("inf")
    verdict = SafetyMonitor(max_speed=6.0).evaluate(
        scene, np.array([[0.0, 0.0], [5.0, 0.0]]))
    assert verdict.stale_sensor is True
    assert verdict.level == "degraded"


def test_bev_and_lane_ages_participate_in_max_age():
    scene = Scene(
        pos=np.array([0.0, 0.0]), heading=0.0,
        meta={"head_age_s": {"semantic": 0.1},
              "bev_age_s": 1.2, "range_age_s": 0.2})
    f = _perception_freshness(scene, snapshot_age_s=0.0)
    assert f["bev_age_s"] == 1.2
    assert f["max_s"] == 1.2
    assert SafetyMonitor(max_speed=6.0).evaluate(
        scene, np.array([[0.0, 0.0], [5.0, 0.0]])).stale_sensor


def test_safety_prefers_canonical_snapshot_over_stale_scene_metadata():
    snapshot = PerceptionSnapshot(
        captured_at=time.time(), tick_id=3,
        pos=np.array([0.0, 0.0]), heading=0.0,
        bev=np.zeros((4, 4)),
        head_age_s={"semantic": 1.1}, range_age_s=0.1,
        bev_age_s=0.1)
    scene = Scene(
        pos=np.array([0.0, 0.0]), heading=0.0,
        perception_snapshot=snapshot,
        meta={"head_age_s": {"semantic": 0.0}, "bev_age_s": 0.0})
    f = _perception_freshness(scene, snapshot_age_s=0.0)
    assert f["head_max_s"] == pytest.approx(1.1)
    verdict = SafetyMonitor(max_speed=6.0).evaluate(
        scene, np.array([[0.0, 0.0], [5.0, 0.0]]))
    assert verdict.stale_sensor is True

    from beamng_autopilot.lane.envelope import SensorLaneEnvelope
    e = SensorLaneEnvelope(
        center=np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]),
        left=None, right=None, width_m=3.5, source="vision",
        paired=False, confidence=0.8,
        captured_at=time.time() - 1.5)
    s = PerceptionSnapshot(
        captured_at=time.time(), tick_id=1,
        pos=np.array([0.0, 0.0]), heading=0.0,
        bev=np.zeros((4, 4)), lane_envelope=e,
        head_age_s={"semantic": 0.1}, range_age_s=0.1,
        bev_age_s=0.1)
    assert s.max_sensor_age_s >= 1.4
    assert s.freshness()["lane_s"] >= 1.4
