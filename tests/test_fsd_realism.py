"""Offline tests for the FSD realism invariants (bottom-layer requirements)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.fsd_realism import (
    FSD_INVARIANTS,
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
    # The perception-only lateral guard must never pull in the road
    # graph / map lane code - the hard rule from docs/fsd_realism.md.
    assert check_no_map_imports() == []


def test_fsd_stack_accepts_strict_param() -> None:
    import inspect
    from beamng_autopilot.fsd_stack import FSDStack
    sig = inspect.signature(FSDStack.__init__)
    assert "strict_sensor" in sig.parameters
    assert sig.parameters["strict_sensor"].default is False
