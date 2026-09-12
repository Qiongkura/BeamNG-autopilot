"""Offline tests for planner arbitration (FSD trajectory vs rule fallback)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.planning import (
    anchored_rule_ref, arbitrate, arbitrate_fsd_tick,
    strict_lane_unavailable,
)


def _path():
    return np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])


def test_fsd_wins_when_safe() -> None:
    fsd = _path()
    rule = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    out = arbitrate(fsd, rule, fsd_safe=True)
    assert out.source == "fsd"
    assert out.path is fsd


def test_rule_fallback_when_fsd_unavailable() -> None:
    rule = _path()
    out = arbitrate(None, rule, fsd_safe=False)
    assert out.source == "rule"
    assert out.path is rule


def test_rule_kept_when_fsd_unsafe() -> None:
    fsd = _path()
    rule = np.array([[0.0, 0.0], [0.0, 1.0]])
    out = arbitrate(fsd, rule, fsd_safe=False)
    assert out.source == "rule"
    assert "fsd unavailable" in out.why


def test_e2e_wins_when_fsd_unavailable() -> None:
    e2e = _path()
    rule = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    out = arbitrate(None, rule, fsd_safe=False,
                    e2e_path=e2e, e2e_safe=True)
    assert out.source == "e2e"
    assert out.path is not None
    assert np.allclose(out.path, e2e)


def test_fsd_still_wins_over_e2e() -> None:
    fsd = _path()
    e2e = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
    out = arbitrate(fsd, None, fsd_safe=True,
                    e2e_path=e2e, e2e_safe=True)
    assert out.source == "fsd"


def test_rule_fallback_when_e2e_unsafe() -> None:
    e2e = _path()
    rule = np.array([[0.0, 0.0], [0.0, 1.0]])
    out = arbitrate(None, rule, fsd_safe=False,
                    e2e_path=e2e, e2e_safe=False)
    assert out.source == "rule"


def test_e2e_ignored_when_unsafe_or_empty() -> None:
    e2e = _path()
    out = arbitrate(None, None, fsd_safe=False,
                    e2e_path=e2e, e2e_safe=False)
    assert out.source == "none"
    out = arbitrate(None, None, fsd_safe=False,
                    e2e_path=None, e2e_safe=True)
    assert out.source == "none"


def test_prefer_rule_forces_rule() -> None:
    fsd = _path()
    rule = np.array([[3.0, 3.0], [4.0, 4.0]])
    out = arbitrate(fsd, rule, fsd_safe=True, prefer_rule=True)
    assert out.source == "rule"


def test_minimal_risk_still_uses_rule_when_available() -> None:
    """FSD declared minimal risk (path blocked) but the rule reference
    exists: the car must NOT stop dead - it degrades to the rule path."""
    rule = _path()
    out = arbitrate(None, rule, fsd_safe=False)
    assert out.source == "rule"
    assert out.path is rule
    assert out.why  # explains the fallback


def test_none_when_everything_missing() -> None:
    out = arbitrate(None, None, fsd_safe=False)
    assert out.source == "none"
    assert out.path is None

def test_anchored_rule_ref_keeps_forward_reference() -> None:
    ref = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
    out = anchored_rule_ref(np.array([0.0, 0.0]), 0.0, ref)
    assert out is ref


def test_anchored_rule_ref_rejects_far_start() -> None:
    ref = np.array([[5.0, 0.0], [8.0, 0.0]])
    out = anchored_rule_ref(np.array([0.0, 0.0]), 0.0, ref)
    assert out is None


def test_anchored_rule_ref_rejects_backward_path() -> None:
    back = np.array([[0.0, 0.0], [0.0, -3.0]])
    assert anchored_rule_ref(np.array([0.0, 0.0]), 0.0, back) is None
    back2 = np.array([[0.0, 0.0], [-3.0, 0.0]])
    assert anchored_rule_ref(np.array([0.0, 0.0]), 0.0, back2) is None


def test_anchored_rule_ref_none_for_empty() -> None:
    assert anchored_rule_ref(np.array([0.0, 0.0]), 0.0, None) is None
    assert anchored_rule_ref(np.array([0.0, 0.0]), 0.0,
                             np.zeros((1, 2))) is None


# --- strict fail-closed contract consumed by the runtime ---------------
def test_strict_lane_unavailable_reads_the_planner_decision() -> None:
    # The planner's block flag is the primary signal.
    assert strict_lane_unavailable(True, "no_perception_lane", "sensor")
    # A missing sensor lane LOCK is the second, independent read: a lost
    # lock must not unblock motion even without the block flag.
    assert strict_lane_unavailable(True, "", "map")
    assert strict_lane_unavailable(True, None, "bev/route")
    assert strict_lane_unavailable(True, "", "")
    # Locked onto a perception lane and nothing declared blocked: usable.
    assert not strict_lane_unavailable(True, "", "sensor")
    assert not strict_lane_unavailable(True, None, "sensor")


def test_strict_lane_unavailable_never_fires_outside_strict_mode() -> None:
    # Legacy rule-compatibility mode keeps its map fallback.
    for blocked, src in (("no_perception_lane", "sensor"),
                         ("", "map"), ("", ""), (None, None)):
        assert not strict_lane_unavailable(False, blocked, src), (blocked, src)


def test_arbitrate_fsd_tick_blocks_rule_fallback_without_lane() -> None:
    rule = _path()
    strict = dict(strict=True, plan_blocked="no_perception_lane",
                  lane_src_sel="sensor")
    out = arbitrate_fsd_tick(None, rule, fsd_safe=False, **strict)
    # The rule path is map/nav geometry: a strict tick without a perception
    # lane must stop, not be steered by it.
    assert out.source == "none"
    assert out.path is None
    # Same inputs, legacy mode: the fallback is still available.
    out = arbitrate_fsd_tick(None, rule, fsd_safe=False, strict=False,
                             plan_blocked="no_perception_lane",
                             lane_src_sel="map")
    assert out.source == "rule"
    assert out.path is rule


def test_arbitrate_fsd_tick_blocks_rule_when_sensor_lock_is_lost() -> None:
    rule = _path()
    out = arbitrate_fsd_tick(None, rule, fsd_safe=False, strict=True,
                             plan_blocked="", lane_src_sel="map")
    assert out.source == "none"
    assert out.path is None


def test_arbitrate_fsd_tick_fail_closed_beats_forced_rule() -> None:
    # Even "rule mode" cannot override the strict no-lane gate.
    rule = _path()
    out = arbitrate_fsd_tick(None, rule, fsd_safe=False, strict=True,
                             plan_blocked="no_perception_lane",
                             lane_src_sel="sensor", prefer_rule=True)
    assert out.source == "none"
    assert out.path is None


def test_arbitrate_fsd_tick_keeps_neural_candidates_eligible() -> None:
    # E2E / BC are perception-derived (they do not need a lane polyline),
    # so they stay inside the legal degradation set.
    rule = _path()
    e2e = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    out = arbitrate_fsd_tick(None, rule, fsd_safe=False,
                             e2e_path=e2e, e2e_safe=True, strict=True,
                             plan_blocked="no_perception_lane",
                             lane_src_sel="sensor")
    assert out.source == "e2e"
    bc = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    out = arbitrate_fsd_tick(None, rule, fsd_safe=False,
                             bc_path=bc, bc_safe=True, strict=True,
                             plan_blocked="no_perception_lane",
                             lane_src_sel="sensor")
    assert out.source == "bc"


def test_arbitrate_fsd_tick_normal_ranking_with_lane_lock() -> None:
    fsd = _path()
    rule = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    out = arbitrate_fsd_tick(fsd, rule, fsd_safe=True, strict=True,
                             plan_blocked="", lane_src_sel="sensor")
    assert out.source == "fsd"
    # A strict tick does NOT keep the rule fallback even with a lane lock.
    # The rule path is planned from the nav route with no sensor lane, so
    # steering by it is map-geometry lateral control, which the iron rule
    # forbids the FSD stack (AGENTS.md).  Gating this on "no perception
    # lane" alone let the map-route path drive whenever the layered planner
    # declined WITH a lane present: on the 2026-09-11 town run that was
    # every body-crossing (17) and off-road (12) frame, all source=rule
    # with lane_sel=sensor.  The legal degradations are stop / hold heading
    # / a safe point; neural (perception-derived) candidates stay eligible.
    out = arbitrate_fsd_tick(None, rule, fsd_safe=False, strict=True,
                             plan_blocked="", lane_src_sel="sensor")
    assert out.source == "none"
    assert out.path is None
    # The same inputs in legacy non-strict mode keep the map fallback,
    # which is what it exists for.
    out = arbitrate_fsd_tick(None, rule, fsd_safe=False, strict=False,
                             plan_blocked="", lane_src_sel="sensor")
    assert out.source == "rule"
