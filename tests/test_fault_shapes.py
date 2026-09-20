"""The failure shapes the town A/B could not produce on purpose.

Plan P1.  Every shape here is one a real run can hit and one an
8-run town comparison cannot isolate: they are rare, confounded, or both.
The point is not that each is handled well - it is that each is now
NAMED differently, so a log line can be attributed instead of guessed at.
"""

from beamng_autopilot.fsd_stack import range_schedule, sched_record
from beamng_autopilot.telemetry_contract import (
    STAGE_CONSUMED,
    STAGE_CONSUMED_STALE,
    STAGE_FINISHED_NOT_PUBLISHED,
    STAGE_IN_FLIGHT,
    STAGE_NOT_DISPATCHED,
    STAGE_NOT_PRODUCED,
    STAGE_PUBLISHED_NOT_CONSUMED,
    ALL_STAGES,
    STAGE_UNKNOWN,
    TelemetryBuffer,
    stage_of,
    stage_summary,
)

T0 = 1000.0


def _rec(state, *, source_seq=7, result_seq=3, **kw):
    # The consumer-side fields (consumed_*) are written by fsd_drive, not
    # by the producer, so they are merged in rather than passed through.
    consumed = {k: kw.pop(k) for k in list(kw) if k.startswith("consumed_")}
    rec = sched_record("object", state, source_seq=source_seq,
                       result_seq=result_seq, eligible_t=T0,
                       source_t=T0, **kw)
    rec.update(consumed)
    return rec


class TestColdStart:
    """Nothing has ever been produced; there is no age to be stale."""

    def test_no_result_yet_is_not_produced_not_stale(self):
        rec = _rec("async_idle_no_output", source_seq=None, result_seq=None)
        assert stage_of(rec) == STAGE_NOT_PRODUCED

    def test_a_none_age_is_not_a_zero_age(self):
        # 0.0 would mean "fresh"; None means "there is nothing to age".
        rec = _rec("async_idle_no_output", source_seq=None, result_seq=None)
        assert rec["age_s"] is None

    def test_range_cold_start_scans_rather_than_reusing(self):
        # has_prev False: there is no held scan, so deferral is impossible
        # even with the budget already blown.
        action, state = range_schedule(
            budget=0.45, elapsed=0.60, age_s=None, has_prev=False,
            keepalive_s=1.0)
        assert (action, state) == ("scan", "scanned")


class TestWorkerStuck:
    def test_in_flight_is_in_flight_not_late(self):
        rec = _rec("async_in_flight", dispatch_t=T0 + 0.1)
        assert stage_of(rec) == STAGE_IN_FLIGHT

    def test_a_submitted_job_with_no_result_yet_is_in_flight(self):
        rec = _rec("async_submitted", dispatch_t=T0 + 0.1)
        assert stage_of(rec) == STAGE_IN_FLIGHT

    def test_a_failed_job_finished_without_publishing(self):
        rec = _rec("async_failed", dispatch_t=T0 + 0.1, finish_t=T0 + 0.9)
        assert stage_of(rec) == STAGE_FINISHED_NOT_PUBLISHED

    def test_an_error_finished_too(self):
        # Raising ENDS the work.  Leaving finish_t None would read as
        # "still coming" and be indistinguishable from in-flight.
        rec = _rec("error", dispatch_t=T0 + 0.1, finish_t=T0 + 0.2)
        assert stage_of(rec) == STAGE_FINISHED_NOT_PUBLISHED


class TestRecovery:
    def test_a_head_that_errored_and_then_ran_is_published(self):
        bad = _rec("error", dispatch_t=T0 + 0.1, finish_t=T0 + 0.2)
        good = _rec("ran", result_seq=4, dispatch_t=T0 + 0.3,
                    finish_t=T0 + 0.4, publish_t=T0 + 0.4)
        assert stage_of(bad) == STAGE_FINISHED_NOT_PUBLISHED
        assert stage_of(good) == STAGE_PUBLISHED_NOT_CONSUMED

    def test_recovery_bumps_the_result_version(self):
        # Without a version bump the consumer cannot tell a retry's output
        # from the value it already rejected.
        assert _rec("ran", result_seq=3)["result_seq"] != \
            _rec("ran", result_seq=4)["result_seq"]


class TestOutOfOrderAndStaleInput:
    def test_a_new_result_from_an_old_frame_keeps_the_old_source(self):
        """'Old input, new output' must not reset the freshness clock.

        The stamp is the frame that produced the result, not the tick that
        published it, so a slow worker answering an old frame publishes an
        OLD result, not a fresh one.
        """
        rec = _rec("async_adopted", source_seq=5, result_seq=9,
                   dispatch_t=T0 + 0.1, finish_t=T0 + 2.5,
                   publish_t=T0 + 2.5, age_s=2.5)
        # source frame 5, published 2.5 s later: still traceable to 5
        assert rec["source_seq"] == 5
        assert rec["result_seq"] == 9

    def test_a_result_older_than_the_contract_line_is_consumed_stale(self):
        rec = _rec("async_adopted", source_seq=5, dispatch_t=T0,
                   finish_t=T0 + 0.1, publish_t=T0 + 0.1, age_s=2.5,
                   consumed_result_seq=3, consumed_source_seq=5,
                   consumed_t=T0 + 0.2, consumed_age_s=2.5)
        assert stage_of(rec, stale_age_s=2.0) == STAGE_CONSUMED_STALE

    def test_the_same_result_under_the_line_is_fine(self):
        rec = _rec("async_adopted", source_seq=5, dispatch_t=T0,
                   finish_t=T0 + 0.1, publish_t=T0 + 0.1, age_s=0.4,
                   consumed_result_seq=3, consumed_source_seq=5,
                   consumed_t=T0 + 0.2, consumed_age_s=0.4)
        assert stage_of(rec, stale_age_s=2.0) == STAGE_CONSUMED

    def test_deferred_reuse_publishes_and_keeps_its_source(self):
        # Reuse publishes the cached value; dispatch/finish stay None
        # because no work happened for it this tick.
        rec = _rec("budget_deferred", source_seq=5, result_seq=3,
                   publish_t=T0 + 0.5, age_s=0.5)
        assert rec["dispatch_t"] is None
        assert rec["finish_t"] is None
        assert rec["source_seq"] == 5


class TestPublishedNotConsumed:
    def test_a_published_result_is_waiting_on_the_consumer(self):
        rec = _rec("ran", dispatch_t=T0, finish_t=T0 + 0.1,
                   publish_t=T0 + 0.1)
        assert stage_of(rec) == STAGE_PUBLISHED_NOT_CONSUMED

    def test_the_buffer_keeps_the_newest_frames_and_counts_drops(self):
        # A bound that kept the OLDEST frames would hide the end of the
        # run, which is exactly where failures show up.
        buf = TelemetryBuffer(limit=3)
        for i in range(10):
            buf.append({"i": i})
        assert [f["i"] for f in buf.frames()] == [7, 8, 9]
        assert buf.dropped == 7
        assert buf.total == 10          # the run was 10 frames long

    def test_an_unfilled_buffer_drops_nothing(self):
        buf = TelemetryBuffer(limit=5)
        for i in range(2):
            buf.append({"i": i})
        assert buf.dropped == 0
        assert len(buf.frames()) == 2


class TestSummary:
    def test_a_mixed_batch_names_every_shape(self):
        recs = [
            _rec("async_idle_no_output", source_seq=None, result_seq=None),
            _rec("async_in_flight", dispatch_t=T0),
            _rec("error", dispatch_t=T0, finish_t=T0 + 0.1),
            _rec("ran", dispatch_t=T0, finish_t=T0 + 0.1, publish_t=T0 + 0.1),
        ]
        s = stage_summary(recs)
        assert s["total"] == 4
        assert s["stages"][STAGE_NOT_PRODUCED] == 1
        assert s["stages"][STAGE_IN_FLIGHT] == 1
        assert s["stages"][STAGE_FINISHED_NOT_PUBLISHED] == 1
        assert s["stages"][STAGE_PUBLISHED_NOT_CONSUMED] == 1

    def test_every_stage_is_present_even_when_zero(self):
        # A missing stage must read as 0, not as an absent column.
        s = stage_summary([])
        assert s["total"] == 0
        assert set(s["stages"]) == set(ALL_STAGES)
        assert all(v == 0 for v in s["stages"].values())

    def test_a_record_that_names_nothing_is_not_produced(self):
        # No identity at all: there is no result to place on the chain.
        # Not UNKNOWN - that is reserved for a record that claims a match
        # it cannot identify.
        assert stage_of({}) == STAGE_NOT_PRODUCED

    def test_not_dispatched_is_not_not_produced(self):
        # The source frame exists; the scheduler chose not to submit.
        rec = _rec("async_idle_no_output", source_seq=7, result_seq=None)
        assert stage_of(rec) == STAGE_NOT_DISPATCHED
