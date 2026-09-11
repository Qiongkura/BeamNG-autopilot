"""Freshness and perception-loss regression tests."""

from __future__ import annotations

import time

import numpy as np
import pytest

from beamng_autopilot.planning import Scene
from beamng_autopilot.safety_monitor import (
    STALE_PIPELINE_S, STALE_RANGE_S, SafetyMonitor, _perception_freshness)
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


def _range_scene(range_age_s: float):
    """Live modalities fresh, only the (reusable) range aged."""
    return Scene(
        pos=np.array([0.0, 0.0]), heading=0.0,
        meta={"head_age_s": {"semantic": 0.1},
              "bev_age_s": 0.1, "range_age_s": range_age_s})


def test_reused_range_scan_is_not_a_stale_sensor():
    """A re-served, motion-compensated range scan is normal operation.

    2026-09-11 town runs: the range age was the dominant term in 188/653
    fail-closed ticks while no perception head exceeded 0.77 s, so the
    range must not be judged against the live-modality threshold.
    """
    scene = _range_scene(1.5)
    f = _perception_freshness(scene, snapshot_age_s=0.1)
    assert f["range_age_s"] == 1.5
    assert f["max_s"] == 1.5          # telemetry keeps the honest max
    verdict = SafetyMonitor(max_speed=6.0).evaluate(
        scene, np.array([[0.0, 0.0], [5.0, 0.0]]), snapshot_age_s=0.1)
    assert verdict.stale_sensor is False
    assert verdict.level != "degraded"


def test_range_past_its_reuse_horizon_is_stale():
    scene = _range_scene(STALE_RANGE_S + 0.5)
    verdict = SafetyMonitor(max_speed=6.0).evaluate(
        scene, np.array([[0.0, 0.0], [5.0, 0.0]]), snapshot_age_s=0.1)
    assert verdict.stale_sensor is True
    assert verdict.level == "degraded"
    assert "stale" in verdict.reason


def test_live_modalities_keep_the_short_bound_with_range_fresh():
    """A fresh range must not relax the live-modality staleness bound."""
    scene = Scene(
        pos=np.array([0.0, 0.0]), heading=0.0,
        meta={"head_age_s": {"semantic": 1.4},
              "bev_age_s": 0.1, "range_age_s": 0.1})
    verdict = SafetyMonitor(max_speed=6.0).evaluate(
        scene, np.array([[0.0, 0.0], [5.0, 0.0]]), snapshot_age_s=0.1)
    assert verdict.stale_sensor is True


def test_slow_but_healthy_tick_is_not_a_stale_sensor():
    """A tick slower than the modality bound is latency, not a dead sensor.

    2026-09-11 town runs: every modality was fresher than 0.7 s while the
    composite snapshot age (control period) reached 0.92 s, so the old
    composite comparison declared 217/808 ticks stale and fail-closed the
    car into a crawl (stall 150-190 frames).
    """
    scene = _range_scene(0.3)
    verdict = SafetyMonitor(max_speed=6.0).evaluate(
        scene, np.array([[0.0, 0.0], [5.0, 0.0]]),
        snapshot_age_s=0.92)
    assert verdict.stale_sensor is False
    assert verdict.level != "degraded"


def test_pipeline_latency_past_its_bound_degrades():
    """The composite age keeps a backstop below the watchdog's abort."""
    scene = _range_scene(0.3)
    verdict = SafetyMonitor(max_speed=6.0).evaluate(
        scene, np.array([[0.0, 0.0], [5.0, 0.0]]),
        snapshot_age_s=STALE_PIPELINE_S + 0.1)
    assert verdict.stale_sensor is True
    assert verdict.level == "degraded"


def test_range_bound_covers_the_stack_reuse_horizon():
    """The range bound must not be tighter than the stack's reuse cap.

    Otherwise a scan the planner is still allowed to consume would be
    declared stale, which is the mismatch this bound exists to remove.
    """
    from beamng_autopilot.fsd_stack import RANGE_REUSE_MAX_DT_S
    assert STALE_RANGE_S >= RANGE_REUSE_MAX_DT_S
