"""Tests for the end-to-end traceability contract (plan P1).

The contract exists so an anomaly lands on a STAGE, not on a bare age, and so
a cross-clock subtraction fails loudly instead of producing a plausible number.
"""

from __future__ import annotations

import pytest

from beamng_autopilot.telemetry_contract import (
    ALL_STAGES,
    CLOCK_SIM,
    CLOCK_WALL,
    ClockMismatch,
    STAGE_CONSUMED,
    STAGE_CONSUMED_STALE,
    STAGE_FINISHED_NOT_PUBLISHED,
    STAGE_IN_FLIGHT,
    STAGE_NOT_DISPATCHED,
    STAGE_NOT_PRODUCED,
    STAGE_PUBLISHED_NOT_CONSUMED,
    STAGE_UNKNOWN,
    TelemetryBuffer,
    check_trace,
    missing_trace_fields,
    span_ms,
    stage_of,
    stage_summary,
)


def _full(**over):
    """A record that reaches CONSUMED, with overrides."""
    rec = {
        "head": "range",
        "source_seq": 10,
        "result_seq": 10,
        "consumed_result_seq": 10,
        "consumed_source_seq": 10,
        "cmd_seq": 99,
        "source_t": 100.0,
        "eligible_t": 100.1,
        "dispatch_t": 100.2,
        "finish_t": 100.5,
        "publish_t": 100.5,
        "consumed_t": 100.6,
        "cmd_t": 100.7,
        "sim_t": 5.0,
        "decision_state": "ok",
        "reason": "clear",
        "effective_rule": "corridor_open",
    }
    rec.update(over)
    return rec


# ------------------------------------------------------------------ clocks

def test_span_within_one_clock_returns_ms():
    assert span_ms(_full(), "dispatch_t", "finish_t") == pytest.approx(300.0)


def test_span_refuses_a_cross_clock_pair():
    """Sim time minus wall time computes, and means nothing.  Refuse it."""
    with pytest.raises(ClockMismatch):
        span_ms(_full(), "source_t", "sim_t")


def test_span_is_none_when_an_end_is_missing():
    """An unmeasured span is not zero."""
    assert span_ms(_full(finish_t=None), "dispatch_t", "finish_t") is None


def test_span_rejects_an_unknown_field():
    with pytest.raises(KeyError):
        span_ms(_full(), "source_t", "not_a_field")


def test_clock_constants_differ():
    assert CLOCK_WALL != CLOCK_SIM


# ------------------------------------------------------------ field checks

def test_a_complete_record_has_nothing_missing():
    res = check_trace(_full())
    assert res["ok"] is True
    assert res["missing"] == []


def test_a_present_but_none_field_counts_as_missing():
    """Writing the key and measuring the value are different statements."""
    rec = _full(dispatch_t=None)
    assert "dispatch_t" in missing_trace_fields(rec)
    assert check_trace(rec)["ok"] is False


def test_an_empty_reason_counts_as_missing():
    assert "reason" in missing_trace_fields(_full(reason=""))


def test_missing_fields_lists_every_group():
    rec = {"source_t": 1.0}          # no ids, no state, most times absent
    missing = missing_trace_fields(rec)
    assert "head" in missing and "result_seq" in missing
    assert "decision_state" in missing
    assert "dispatch_t" in missing
    assert "source_t" not in missing


# --------------------------------------------------------------- diagnosis

def test_no_source_means_not_produced():
    assert stage_of({"head": "range"}) == STAGE_NOT_PRODUCED


def test_a_result_that_never_dispatched():
    assert stage_of(_full(dispatch_t=None)) == STAGE_NOT_DISPATCHED


def test_a_result_still_in_flight():
    assert stage_of(_full(finish_t=None)) == STAGE_IN_FLIGHT


def test_finished_but_not_published():
    assert stage_of(_full(publish_t=None)) == STAGE_FINISHED_NOT_PUBLISHED


def test_published_but_the_tick_never_read_it():
    assert stage_of(_full(consumed_result_seq=None)) \
        == STAGE_PUBLISHED_NOT_CONSUMED


def test_the_tick_consumed_a_different_version():
    """Consuming an older result is not consuming this one."""
    assert stage_of(_full(consumed_result_seq=9)) \
        == STAGE_PUBLISHED_NOT_CONSUMED


def test_consumed_but_already_stale():
    rec = _full(consumed_age_s=1.4)
    assert stage_of(rec, stale_age_s=1.0) == STAGE_CONSUMED_STALE


def test_consumed_in_time():
    rec = _full(consumed_age_s=0.2)
    assert stage_of(rec, stale_age_s=1.0) == STAGE_CONSUMED


def test_stale_without_a_bound_is_not_judged():
    """No bound given means we cannot call it stale - so it reads CONSUMED."""
    assert stage_of(_full(consumed_age_s=99.0)) == STAGE_CONSUMED


def test_consumed_without_a_published_id_is_unknown():
    """A match we cannot identify must not be reported as a match."""
    assert stage_of(_full(result_seq=None)) == STAGE_UNKNOWN


def test_the_stage_order_puts_production_first():
    """A record missing everything reports the EARLIEST gap, not the last."""
    assert stage_of({}) == STAGE_NOT_PRODUCED


def test_stage_summary_keeps_zero_stages_visible():
    out = stage_summary([_full(), {"head": "range"}])
    assert out["total"] == 2
    assert set(out["stages"]) == set(ALL_STAGES)
    assert out["stages"][STAGE_CONSUMED] == 1
    assert out["stages"][STAGE_NOT_PRODUCED] == 1
    assert out["stages"][STAGE_IN_FLIGHT] == 0


# --------------------------------------------------------- bounded buffer

def test_unbounded_buffer_keeps_everything():
    buf = TelemetryBuffer()
    for i in range(50):
        buf.append({"t": i})
    assert len(buf) == 50 and buf.dropped == 0 and buf.total == 50


def test_bounded_buffer_keeps_the_newest_frames():
    """A bound that kept the OLDEST would hide the end of the run."""
    buf = TelemetryBuffer(limit=3)
    for i in range(10):
        buf.append({"t": i})
    assert [f["t"] for f in buf.frames()] == [7, 8, 9]
    assert buf.dropped == 7
    assert buf.total == 10


def test_drops_are_counted_and_reported():
    buf = TelemetryBuffer(limit=2)
    for i in range(5):
        buf.append({"t": i})
    s = buf.summary()
    assert s == {"kept": 2, "dropped": 3, "total": 5, "limit": 2}


def test_a_limit_of_one_still_works():
    buf = TelemetryBuffer(limit=1)
    buf.append({"t": 0})
    buf.append({"t": 1})
    assert [f["t"] for f in buf.frames()] == [1]
    assert buf.dropped == 1


def test_frames_returns_a_copy():
    """Callers must not be able to mutate the buffer through the export."""
    buf = TelemetryBuffer(limit=3)
    buf.append({"t": 0})
    out = buf.frames()
    out.append({"t": 999})
    assert len(buf) == 1
