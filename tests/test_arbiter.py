"""Offline tests for planner arbitration (FSD trajectory vs rule fallback)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.planning import arbitrate


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