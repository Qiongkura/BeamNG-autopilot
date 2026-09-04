"""Offline tests for planner arbitration (FSD trajectory vs rule fallback)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.planning import anchored_rule_ref, arbitrate


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
