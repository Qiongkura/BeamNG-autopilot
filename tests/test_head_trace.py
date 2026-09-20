"""Tests for the per-head traceability record the tick now emits (plan P1).

The point is not that the fields exist - it is that each failure lands on the
right STAGE.  A bare age says a head is 113 s old; this says whether the head
never produced, was never dispatched, is still in flight, finished but never
published, or was published and the control tick never read it.
"""

from __future__ import annotations

import pytest

from beamng_autopilot.fsd_stack import sched_record
from beamng_autopilot.telemetry_contract import (
    STAGE_CONSUMED,
    STAGE_FINISHED_NOT_PUBLISHED,
    STAGE_IN_FLIGHT,
    STAGE_NOT_DISPATCHED,
    STAGE_NOT_PRODUCED,
    STAGE_PUBLISHED_NOT_CONSUMED,
    check_trace,
    missing_trace_fields,
    span_ms,
    stage_of,
)

T0 = 1000.0


def _rec(state, **over):
    rec = sched_record("object", state, source_seq=7, result_seq=3,
                       eligible_t=T0, source_t=T0)
    rec.update(over)
    return rec


# ------------------------------------------------------------ the fields

def test_a_fresh_run_carries_the_whole_chain():
    rec = _rec("ran", dispatch_t=T0 + 0.01, finish_t=T0 + 0.30,
               publish_t=T0 + 0.30)
    assert rec["head"] == "object"
    assert rec["source_seq"] == 7
    assert rec["result_seq"] == 3
    assert rec["decision_state"] == "ran"
    assert rec["eligible_t"] == T0
    assert rec["source_t"] == T0


def test_the_original_keys_keep_their_meaning():
    """Existing consumers read state / age_s / compute_ms / reason."""
    rec = _rec("ran", age_s=0.4, compute_ms=12.5, reason="")
    assert rec["state"] == "ran"
    assert rec["age_s"] == 0.4
    assert rec["compute_ms"] == 12.5
    assert rec["reason"] == ""


def test_a_step_that_did_not_happen_is_none_not_zero():
    rec = _rec("async_in_flight")
    assert rec["dispatch_t"] is None
    assert rec["finish_t"] is None
    assert rec["publish_t"] is None


def test_the_times_are_one_clock_so_any_pair_can_be_differenced():
    rec = _rec("ran", dispatch_t=T0 + 0.01, finish_t=T0 + 0.35,
               publish_t=T0 + 0.36)
    assert span_ms(rec, "eligible_t", "dispatch_t") == pytest.approx(10.0)
    assert span_ms(rec, "dispatch_t", "finish_t") == pytest.approx(340.0)
    assert span_ms(rec, "source_t", "publish_t") == pytest.approx(360.0)


# ------------------------------------------------------------ the stages

def test_a_fresh_result_waits_for_the_control_tick():
    """fsd_stack publishes; only fsd_drive can say it was consumed.

    So the honest stage here is "published, consumption not yet recorded".
    """
    rec = _rec("ran", dispatch_t=T0 + 0.01, finish_t=T0 + 0.30,
               publish_t=T0 + 0.30)
    assert stage_of(rec) == STAGE_PUBLISHED_NOT_CONSUMED


def test_the_same_record_reads_consumed_once_the_tick_marks_it():
    rec = _rec("ran", dispatch_t=T0 + 0.01, finish_t=T0 + 0.30,
               publish_t=T0 + 0.30, consumed_result_seq=3,
               consumed_age_s=0.1)
    assert stage_of(rec, stale_age_s=1.0) == STAGE_CONSUMED


def test_a_deferral_is_a_reuse_not_a_dispatch():
    """Deferring publishes the CACHED result; no new work was dispatched."""
    rec = _rec("budget_deferred", publish_t=T0 + 0.02)
    assert stage_of(rec) == STAGE_NOT_DISPATCHED


def test_an_in_flight_head_is_in_flight():
    rec = _rec("async_in_flight", dispatch_t=T0 + 0.01)
    assert stage_of(rec) == STAGE_IN_FLIGHT


def test_a_failed_head_finished_but_published_nothing():
    rec = _rec("async_failed", dispatch_t=T0 + 0.01, finish_t=T0 + 0.50)
    assert stage_of(rec) == STAGE_FINISHED_NOT_PUBLISHED


def test_a_run_that_threw_is_also_finished_not_published():
    """An exception is not 'still coming' - the work ended, it just failed."""
    rec = _rec("error", dispatch_t=T0 + 0.01, finish_t=T0 + 0.05,
               reason="boom")
    assert stage_of(rec) == STAGE_FINISHED_NOT_PUBLISHED
    assert rec["reason"] == "boom"


def test_an_idle_head_with_no_output_was_never_dispatched():
    """No result because no work was submitted - the source frame exists,

    so this is NOT "not produced".  Calling it not-produced would blame the
    head for something the scheduler decided.
    """
    rec = _rec("async_idle_no_output", result_seq=None)
    assert stage_of(rec) == STAGE_NOT_DISPATCHED


def test_not_produced_needs_both_identities_absent():
    """Neither a source frame nor a result: the head produced nothing."""
    rec = _rec("async_idle_no_output", source_seq=None, result_seq=None)
    assert stage_of(rec) == STAGE_NOT_PRODUCED


def test_a_submitted_head_has_dispatched_but_not_finished():
    rec = _rec("async_submitted", dispatch_t=T0 + 0.01)
    assert stage_of(rec) == STAGE_IN_FLIGHT


# ----------------------------------------------------------- completeness

def test_a_complete_record_only_lacks_the_consumer_side():
    """Missing keys are the ones only fsd_drive can fill - not gaps here."""
    rec = _rec("ran", dispatch_t=T0 + 0.01, finish_t=T0 + 0.30,
               publish_t=T0 + 0.30)
    missing = set(missing_trace_fields(rec))
    # Everything the stack can know is here.  What is left is the consumer
    # side (fsd_drive) and the arbitration side - not gaps in this record.
    assert missing == {"consumed_t", "cmd_t", "sim_t",
                       "consumed_result_seq", "consumed_source_seq",
                       "cmd_seq", "reason", "effective_rule"}
    assert check_trace(rec)["ok"] is False      # until the consumer writes


def test_a_deferred_record_reports_what_it_did_not_do():
    rec = _rec("budget_deferred", publish_t=T0 + 0.02)
    missing = missing_trace_fields(rec)
    assert "dispatch_t" in missing
    assert "finish_t" in missing
    assert "publish_t" not in missing


@pytest.mark.parametrize("state", [
    "ran", "not_due", "budget_deferred", "keepalive_forced",
    "async_adopted", "async_submitted", "async_in_flight",
    "async_failed", "async_idle_no_output", "error",
])
def test_every_state_carries_the_identity_fields(state):
    rec = _rec(state)
    assert rec["head"] == "object"
    assert rec["source_seq"] == 7
    assert rec["decision_state"] == state
    assert rec["eligible_t"] == T0
    assert rec["source_t"] == T0
