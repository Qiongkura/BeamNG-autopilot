"""Deterministic tests for the scheduler keep-alive floor (plan P1, item 1).

The 8-run A/B could not say whether the floor works, because its forced branch
never fired.  These tests reach that cell with a fake clock instead of by
driving town: (a) the budget already blown at the range decision point AND
(b) the reused scan past its bound.

They also reproduce the recorded runs' ACTUAL readings, so the extraction is
pinned to behaviour that was already observed - if the refactor changed
anything, these fail first.
"""

from __future__ import annotations

import pytest

from beamng_autopilot.fsd_stack import (
    RANGE_KEEPALIVE_S,
    _budget_defers,
    _keepalive_expired,
    range_schedule,
)

BUDGET = 0.45          # budget_s used by the recorded town runs
BOUND = RANGE_KEEPALIVE_S   # 1.0 s for range


# ------------------------------------------------------ the deferred cells

def test_under_budget_always_scans():
    assert range_schedule(budget=BUDGET, elapsed=0.30, age_s=0.2,
                          has_prev=True, keepalive_s=BOUND) == (
        "scan", "scanned")


def test_without_a_there_is_no_budget_so_no_deferral():
    assert range_schedule(budget=None, elapsed=9.9, age_s=5.0,
                          has_prev=True, keepalive_s=BOUND) == (
        "scan", "scanned")


def test_floor_off_defers_unconditionally():
    """The pre-floor behaviour, kept exactly."""
    assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=1.5,
                          has_prev=True, keepalive_s=None) == (
        "defer", "budget_deferred")


def test_over_budget_but_inside_the_bound_still_defers():
    """Under the bound, serving the compensated cache is legal."""
    assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=0.5,
                          has_prev=True, keepalive_s=BOUND) == (
        "defer", "budget_deferred")


# ------------------------------------------------------- the forced cell

def test_over_budget_and_past_the_bound_forces_a_scan():
    """THE cell the A/B never reached.

    This is the answer to "does the keep-alive floor do anything?" - a
    controlled over-budget test, not 61 more town runs.
    """
    assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=1.5,
                          has_prev=True, keepalive_s=BOUND) == (
        "scan", "keepalive_forced")


def test_the_bound_is_inclusive():
    """age >= bound, exactly as `_keepalive_expired` compares."""
    assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=BOUND,
                          has_prev=True, keepalive_s=BOUND)[1] == \
        "keepalive_forced"


def test_just_under_the_bound_does_not_force():
    assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=0.999,
                          has_prev=True, keepalive_s=BOUND)[1] == \
        "budget_deferred"


def test_a_forced_refresh_resets_the_age():
    """After a forced scan the sample is NEW, so reuse becomes legal again.

    This is "the new source result is actually adopted", which is the other
    half of what the A/B could not show.
    """
    age = 1.5
    assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=age,
                          has_prev=True, keepalive_s=BOUND) == (
        "scan", "keepalive_forced")
    age = 0.0                                  # the scan ran
    assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=age,
                          has_prev=True, keepalive_s=BOUND) == (
        "defer", "budget_deferred")


def test_without_the_floor_a_starved_scan_keeps_being_deferred():
    """The 2026-09-20 starvation shape in four lines.

    Once deferred, the cache is served again, so the age only grows and the
    scan is never re-run: 151/151 and 120/120 frames on the town baselines.
    """
    age = 1.5
    for _ in range(5):
        assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=age,
                              has_prev=True, keepalive_s=None) == (
            "defer", "budget_deferred")
        age += 0.7                             # reuse does not refresh


def test_with_the_floor_the_same_sequence_recovers():
    age = 1.5
    states = []
    for _ in range(5):
        action, state = range_schedule(
            budget=BUDGET, elapsed=0.60, age_s=age,
            has_prev=True, keepalive_s=BOUND)
        states.append(state)
        age = 0.0 if action == "scan" else age + 0.7
    assert states == ["keepalive_forced", "budget_deferred",
                      "budget_deferred", "keepalive_forced",
                      "budget_deferred"]


# ---------------------------------------------------------- no cache yet

def test_nothing_to_reuse_is_not_forced():
    """age None means no sample exists; forcing a scan is just a scan."""
    assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=None,
                          has_prev=False, keepalive_s=BOUND) == (
        "scan", "scanned")


def test_no_previous_sample_is_never_deferred():
    assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=0.1,
                          has_prev=False, keepalive_s=None) == (
        "scan", "scanned")


# ------------------------------------------- reproduce the recorded runs

def test_the_on_arm_frames_that_passed_the_bound_still_read_scanned():
    """town_1789888192 idx=160: age 1.322 s, ring 416 ms of a 450 ms budget.

    The budget was NOT blown at the decision point, so the scan ran normally
    and the state recorded was 'scanned' - which is what the log says.  This
    pins why the floor never fired on those frames.
    """
    assert range_schedule(budget=BUDGET, elapsed=0.416, age_s=1.322,
                          has_prev=True, keepalive_s=BOUND) == (
        "scan", "scanned")


def test_the_off_arm_worst_defer_reproduces():
    """town_1789888682 idx=102: age 1.552 s, deferred with the floor OFF.

    That defer pushed the next frame to 2.232 s, past STALE_RANGE_S=2.0.
    """
    assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=1.552,
                          has_prev=True, keepalive_s=None) == (
        "defer", "budget_deferred")


def test_the_same_frame_with_the_floor_on_would_have_been_forced():
    """The counterfactual: with the floor on, that 1.552 s defer is refused."""
    assert range_schedule(budget=BUDGET, elapsed=0.60, age_s=1.552,
                          has_prev=True, keepalive_s=BOUND) == (
        "scan", "keepalive_forced")


# -------------------------------------------------------- _budget_defers

def test_defers_when_the_floor_is_off():
    assert _budget_defers("range", strict=False, every_n=2, budget=BUDGET,
                          elapsed=0.60, age_s=9.9, keepalive_s=None) is True


def test_refuses_to_defer_past_the_bound():
    assert _budget_defers("range", strict=False, every_n=2, budget=BUDGET,
                          elapsed=0.60, age_s=1.5, keepalive_s=BOUND) is False


def test_defers_inside_the_bound():
    assert _budget_defers("range", strict=False, every_n=2, budget=BUDGET,
                          elapsed=0.60, age_s=0.5, keepalive_s=BOUND) is True


def test_never_defers_when_there_is_nothing_to_serve():
    """age None: deferring leaves the modality absent for the whole run."""
    assert _budget_defers("range", strict=False, every_n=2, budget=BUDGET,
                          elapsed=0.60, age_s=None, keepalive_s=BOUND) is False


def test_semantic_is_never_deferred_in_strict_mode():
    assert _budget_defers("semantic", strict=True, every_n=2, budget=BUDGET,
                          elapsed=0.60, age_s=0.1, keepalive_s=BOUND) is False


# --------------------------------------------------- injected keepalive

def test_the_injected_bound_overrides_the_module_switch():
    assert _keepalive_expired("range", 1.5, keepalive_s=None) is False
    assert _keepalive_expired("range", 1.5, keepalive_s=1.0) is True


def test_an_injected_none_means_no_floor():
    assert _keepalive_expired("range", 99.0, keepalive_s=None) is False


def test_a_missing_age_is_never_expired():
    assert _keepalive_expired("range", None, keepalive_s=1.0) is False


@pytest.mark.parametrize("age", [0.0, 0.5, 0.999])
def test_ages_under_the_bound_are_not_expired(age):
    assert _keepalive_expired("range", age, keepalive_s=1.0) is False
