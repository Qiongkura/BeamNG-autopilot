"""Offline tests for the FSD realism invariants (bottom-layer requirements)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.fsd_realism import (
    FSD_INVARIANTS,
    NO_MAP_GUARDED_FILES,
    SRC_MAP,
    SRC_SENSOR,
    SRC_UNAVAILABLE,
    assert_realistic_lane,
    check_no_map_imports,
    lane_source_ok,
)


def test_lane_source_ok_permissive() -> None:
    assert lane_source_ok(SRC_SENSOR, strict=False)
    assert lane_source_ok(SRC_UNAVAILABLE, strict=False)
    assert lane_source_ok(SRC_MAP, strict=False)  # old-rule fallback allowed
    assert not lane_source_ok(None, strict=False)


def test_lane_source_ok_strict() -> None:
    assert lane_source_ok(SRC_SENSOR, strict=True)
    assert lane_source_ok(SRC_UNAVAILABLE, strict=True)
    assert not lane_source_ok(SRC_MAP, strict=True)   # FSD: never map lane
    assert not lane_source_ok("bev/route", strict=True)


def test_assert_realistic_lane_raises_on_map_in_strict() -> None:
    assert_realistic_lane(SRC_SENSOR, strict=True)       # no raise
    assert_realistic_lane(SRC_UNAVAILABLE, strict=True)  # no raise
    with pytest.raises(ValueError):
        assert_realistic_lane(SRC_MAP, strict=True)


def test_invariants_registry_complete() -> None:
    ids = [r[0] for r in FSD_INVARIANTS]
    assert len(ids) == 5
    assert "lane-perception-only" in ids
    assert "perception-unavailable-degrades" in ids
    for _id, rule, _by in FSD_INVARIANTS:
        assert len(rule) > 20
        assert len(_by) > 5


def test_perception_guard_has_no_map_imports() -> None:
    # Perception / safety / planning / control pure-logic modules must
    # never pull in the road graph / map lane code - the hard rule from
    # docs/fsd_realism.md.
    assert check_no_map_imports() == []


def test_no_map_guard_covers_whole_perception_safety_planning_stack() -> None:
    # The machine-checkable guard must not silently shrink: these are the
    # modules that are required to stay sensor-only by the FSD invariants.
    required = [
        "beamng_autopilot/lane/perception_guard.py",
        "beamng_autopilot/lane/fusion.py",
        "beamng_autopilot/lane/lidar.py",
        "beamng_autopilot/lane/pairing.py",
        "beamng_autopilot/lane/tracking.py",
        "beamng_autopilot/vision/lanes.py",
        "beamng_autopilot/vision/hydra.py",
        "beamng_autopilot/vision/segmentation.py",
        "beamng_autopilot/occupancy.py",
        "beamng_autopilot/bev_fusion.py",
        "beamng_autopilot/temporal.py",
        "beamng_autopilot/safety_monitor.py",
        "beamng_autopilot/planning/constraints.py",
        "beamng_autopilot/planning/selector.py",
        "beamng_autopilot/planning/trajectory.py",
        "beamng_autopilot/planning/scene.py",
        "beamng_autopilot/control/reverse_guard.py",
        "beamng_autopilot/control/reverse_maneuver.py",
        "beamng_autopilot/control/speed.py",
    ]
    assert set(required) <= set(NO_MAP_GUARDED_FILES)


def test_fsd_stack_is_not_guarded_but_still_split_from_lateral_logic() -> None:
    # fsd_stack.py is the integration point: it legitimately consumes the
    # nav route as *destination intent* only.  It must stay out of the
    # never-map list (that list is for sensor-only modules).
    assert "beamng_autopilot/fsd_stack.py" not in NO_MAP_GUARDED_FILES


def test_fsd_stack_accepts_strict_param() -> None:
    import inspect
    from beamng_autopilot.fsd_stack import FSDStack
    sig = inspect.signature(FSDStack.__init__)
    assert "strict_sensor" in sig.parameters
    assert sig.parameters["strict_sensor"].default is False
