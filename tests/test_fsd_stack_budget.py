"""The tick-budget governor must never defer the strict semantic lane."""

from __future__ import annotations

from beamng_autopilot.fsd_stack import _budget_defers


def test_strict_semantic_is_never_deferred():
    assert not _budget_defers("semantic", strict=True, every_n=1,
                              budget=0.3, elapsed=9.9)


def test_object_head_still_defers_in_strict_mode():
    assert _budget_defers("object", strict=True, every_n=2,
                          budget=0.3, elapsed=9.9)


def test_non_strict_semantic_still_yields_to_the_budget():
    assert _budget_defers("semantic", strict=False, every_n=2,
                          budget=0.3, elapsed=9.9)


def test_no_budget_or_within_budget_never_defers():
    assert not _budget_defers("object", strict=True, every_n=2,
                              budget=None, elapsed=9.9)
    assert not _budget_defers("object", strict=True, every_n=2,
                              budget=0.3, elapsed=0.1)
