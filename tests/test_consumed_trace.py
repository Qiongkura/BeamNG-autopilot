"""Tests for the consumption side of the traceability contract (plan P1).

fsd_stack publishes; this is what the control tick actually acted on.  Without
it, "the head produced a fresh result" and "the car used that result" are the
same sentence - and a result that was published but never consumed still looks
fresh in the telemetry.
"""

from __future__ import annotations

from beamng_autopilot.fsd_drive import consumed_from_head_sched
from beamng_autopilot.fsd_stack import sched_record
from beamng_autopilot.telemetry_contract import stage_of

T0 = 100.0


def _pub(name, result_seq, publish_t, source_seq=5):
    return sched_record(name, "ran", source_seq=source_seq,
                        result_seq=result_seq, eligible_t=T0, source_t=T0,
                        dispatch_t=T0 + 0.01, finish_t=T0 + 0.2,
                        publish_t=publish_t)


# ------------------------------------------------------------- the record

def test_a_consumed_result_keeps_its_identity():
    sched = {"object": _pub("object", 7, T0 + 0.2)}
    got = consumed_from_head_sched(sched, T0 + 0.25)
    assert got["object"]["result_seq"] == 7
    assert got["object"]["source_seq"] == 5


def test_the_age_is_measured_at_command_time_not_publish_time():
    """The age that mattered is how old it was when the command went out."""
    got = consumed_from_head_sched(
        {"object": _pub("object", 7, T0 + 0.2)}, T0 + 0.45)
    assert got["object"]["age_s"] == 0.25


def test_a_missing_publish_time_gives_none_not_a_plausible_age():
    rec = sched_record("object", "async_in_flight", source_seq=5,
                       result_seq=None, eligible_t=T0, source_t=T0)
    got = consumed_from_head_sched({"object": rec}, T0 + 0.5)
    assert got["object"]["age_s"] is None


def test_no_command_time_gives_none_too():
    got = consumed_from_head_sched({"object": _pub("object", 7, T0)}, None)
    assert got["object"]["age_s"] is None


def test_every_published_head_appears():
    sched = {"object": _pub("object", 7, T0 + 0.2),
             "semantic": _pub("semantic", 3, T0 + 0.1)}
    got = consumed_from_head_sched(sched, T0 + 0.25)
    assert set(got) == {"object", "semantic"}


def test_empty_sched_is_empty_consumed():
    assert consumed_from_head_sched({}, T0) == {}
    assert consumed_from_head_sched(None, T0) == {}


def test_a_non_dict_entry_is_skipped_not_crashed():
    assert consumed_from_head_sched({"object": "junk"}, T0) == {}


def test_a_head_that_never_ran_still_reports_its_missing_result():
    """No result_seq, no publish time - the record says so, twice."""
    rec = sched_record("object", "async_idle_no_output", source_seq=5,
                       result_seq=None, eligible_t=T0, source_t=T0)
    got = consumed_from_head_sched({"object": rec}, T0 + 0.5)
    assert got["object"]["result_seq"] is None
    assert got["object"]["age_s"] is None


# ------------------------------------------ the two failure shapes

def test_a_stale_consumption_is_detectable_after_the_fact():
    """Published result_seq 9, acted on result_seq 7 -> one version behind."""
    sched = {"object": _pub("object", 9, T0 + 0.2)}
    got = consumed_from_head_sched(sched, T0 + 0.25)
    # the tick consumed an older version: 7 < 9
    merged = dict(sched["object"])
    merged["consumed_result_seq"] = 7
    merged["consumed_age_s"] = got["object"]["age_s"]
    assert stage_of(merged) == "published_not_consumed"


def test_the_same_version_reads_as_consumed():
    sched = {"object": _pub("object", 9, T0 + 0.2)}
    merged = dict(sched["object"])
    merged["consumed_result_seq"] = 9
    merged["consumed_age_s"] = 0.05
    assert stage_of(merged, stale_age_s=1.0) == "consumed"


def test_an_old_but_matching_version_reads_consumed_stale():
    sched = {"object": _pub("object", 9, T0 + 0.2)}
    merged = dict(sched["object"])
    merged["consumed_result_seq"] = 9
    merged["consumed_age_s"] = 1.4
    assert stage_of(merged, stale_age_s=1.0) == "consumed_stale"
