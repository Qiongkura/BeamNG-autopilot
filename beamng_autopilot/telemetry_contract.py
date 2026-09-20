"""End-to-end traceability contract for one perception result (plan P1).

Today an anomaly shows up as a single age, and the age cannot say WHERE the
result stopped.  It could be that the head never produced it, produced it but
did not finish, finished but never published, published but the control tick
never consumed it, consumed but already old, or consumed in time and the
command went out too late.  Those are six different bugs with six different
fixes, and "age 1.4 s" does not tell them apart.

This module is the vocabulary for that.  It collects nothing - it defines the
field names, the clock types and the diagnosis, so the producer (``fsd_stack``)
and the consumer (``fsd_drive``) can agree on one record.  Existing
``errors`` / ``worker`` / ``age`` telemetry stays as it is; this does not
replace it.

Two rules the review calls out explicitly:

* **Clock types are not interchangeable.**  Scheduling latency is measured on
  the monotonic wall clock; scene state comes from the sim clock.  Subtracting
  one from the other produces a number that looks plausible and means nothing,
  so :func:`span` refuses the pair instead of returning it.
* **A receipt is not an execution.**  If the command side cannot confirm the
  vehicle acted, the stage is UNKNOWN - never "executed".  Writing "sent" and
  reading it as "done" is how a control failure gets recorded as success.
"""
from __future__ import annotations

from collections import deque

# ------------------------------------------------------------------ clocks

CLOCK_WALL = "monotonic_wall"
CLOCK_SIM = "sim"

# Every timestamp the record may carry, with the clock it belongs to.  A field
# that is not here cannot be used with :func:`span`.
TRACE_TIME_FIELDS: tuple[tuple[str, str], ...] = (
    ("source_t", CLOCK_WALL),      # when the frame was captured
    ("eligible_t", CLOCK_WALL),    # when the head became due
    ("dispatch_t", CLOCK_WALL),    # when work was submitted
    ("finish_t", CLOCK_WALL),      # when the worker finished
    ("publish_t", CLOCK_WALL),     # when the result became readable
    ("consumed_t", CLOCK_WALL),    # when the control tick read it
    ("cmd_t", CLOCK_WALL),         # when the command went out
    ("sim_t", CLOCK_SIM),          # scene time (NOT comparable to the above)
)

TRACE_ID_FIELDS: tuple[str, ...] = (
    "head",
    "source_seq",
    "result_seq",
    "consumed_result_seq",
    "consumed_source_seq",
    "cmd_seq",
)

TRACE_STATE_FIELDS: tuple[str, ...] = (
    "decision_state",
    "reason",
    "effective_rule",
)

# The stages a result can stop at, in order.  These are the review's release
# criterion: every anomaly must land on one of these, not on a bare age.
STAGE_NOT_PRODUCED = "not_produced"
STAGE_NOT_DISPATCHED = "not_dispatched"
STAGE_IN_FLIGHT = "in_flight"
STAGE_FINISHED_NOT_PUBLISHED = "finished_not_published"
STAGE_PUBLISHED_NOT_CONSUMED = "published_not_consumed"
STAGE_CONSUMED_STALE = "consumed_stale"
STAGE_CONSUMED = "consumed"
STAGE_UNKNOWN = "unknown"

ALL_STAGES: tuple[str, ...] = (
    STAGE_NOT_PRODUCED,
    STAGE_NOT_DISPATCHED,
    STAGE_IN_FLIGHT,
    STAGE_FINISHED_NOT_PUBLISHED,
    STAGE_PUBLISHED_NOT_CONSUMED,
    STAGE_CONSUMED_STALE,
    STAGE_CONSUMED,
    STAGE_UNKNOWN,
)


class ClockMismatch(ValueError):
    """Raised when two timestamps from different clocks are subtracted."""


def _clock_of(field: str) -> str:
    for name, clock in TRACE_TIME_FIELDS:
        if name == field:
            return clock
    raise KeyError(f"{field!r} is not a trace time field")


def span_ms(record: dict, start_field: str, end_field: str) -> float | None:
    """Elapsed ms between two trace fields of the SAME clock.

    ``None`` when either end is missing - an unmeasured span is not zero.
    Raises :class:`ClockMismatch` when the two fields belong to different
    clocks, because that subtraction is meaningless even though it computes.
    """
    cs = _clock_of(start_field)
    ce = _clock_of(end_field)
    if cs != ce:
        raise ClockMismatch(
            f"{start_field} is {cs} but {end_field} is {ce}; "
            f"cross-clock subtraction is meaningless")
    a = record.get(start_field)
    b = record.get(end_field)
    if a is None or b is None:
        return None
    return (float(b) - float(a)) * 1000.0


# ------------------------------------------------------------ field checks

def missing_trace_fields(record: dict) -> list[str]:
    """Which required traceability fields this record does not carry.

    A field present but ``None`` counts as missing: "we wrote the key" and
    "we measured the value" are different statements, and only the second one
    supports a diagnosis.
    """
    missing = [n for n, _ in TRACE_TIME_FIELDS if record.get(n) is None]
    missing += [n for n in TRACE_ID_FIELDS if record.get(n) is None]
    missing += [n for n in TRACE_STATE_FIELDS if not record.get(n)]
    return missing


def check_trace(record: dict) -> dict:
    """``{"ok", "missing", "present"}`` for one record."""
    missing = missing_trace_fields(record)
    all_fields = ([n for n, _ in TRACE_TIME_FIELDS]
                  + list(TRACE_ID_FIELDS) + list(TRACE_STATE_FIELDS))
    return {
        "ok": not missing,
        "missing": missing,
        "present": [f for f in all_fields if f not in missing],
    }


# --------------------------------------------------------------- diagnosis

def stage_of(record: dict, *, stale_age_s: float | None = None) -> str:
    """Where the result stopped, as one of :data:`ALL_STAGES`.

    The order matters: the first missing stage wins, because a later field
    cannot be filled in honestly when an earlier one is absent.  In
    particular a record with a ``publish_t`` but no ``result_seq`` is
    UNKNOWN, not "consumed" - we cannot claim a match we cannot identify.
    """
    if record.get("source_seq") is None and record.get("result_seq") is None:
        return STAGE_NOT_PRODUCED
    if record.get("dispatch_t") is None:
        return STAGE_NOT_DISPATCHED
    if record.get("finish_t") is None:
        return STAGE_IN_FLIGHT
    if record.get("publish_t") is None:
        return STAGE_FINISHED_NOT_PUBLISHED

    consumed = record.get("consumed_result_seq")
    published = record.get("result_seq")
    if consumed is None:
        return STAGE_PUBLISHED_NOT_CONSUMED
    if published is None:
        # Something was consumed but the record does not say what was
        # published, so a match cannot be checked either way.
        return STAGE_UNKNOWN
    if consumed != published:
        return STAGE_PUBLISHED_NOT_CONSUMED

    age = record.get("consumed_age_s")
    if stale_age_s is not None and age is not None \
            and float(age) > float(stale_age_s):
        return STAGE_CONSUMED_STALE
    return STAGE_CONSUMED


def stage_summary(records, *, stale_age_s: float | None = None) -> dict:
    """Count how many records stopped at each stage.

    Returns ``{"stages": {...}, "total": n}`` with every stage present (zero
    included), so a caller can print a fixed set of columns and a missing
    stage is visible as 0 rather than as an absent key.
    """
    counts = {s: 0 for s in ALL_STAGES}
    total = 0
    for rec in records:
        counts[stage_of(rec, stale_age_s=stale_age_s)] += 1
        total += 1
    return {"stages": counts, "total": total}


# -------------------------------------------------------- bounded buffer

class TelemetryBuffer:
    """Bounded in-memory frame buffer for the control hot path.

    The drive already accumulates frames in memory and writes the JSON once at
    the end, so nothing syncs to disk mid-tick.  What was missing is the
    BOUND: the list grew without limit, so a long run could exhaust memory and
    there was no count of what a bound would have dropped.

    Two deliberate choices:

    * **The newest frames are kept.**  A bound that kept the oldest would hide
      the end of a run, which is where failures show up.
    * **Drops are counted, never silent.**  ``dropped`` is part of the record,
      so a truncated log cannot be mistaken for a short run.
    """

    def __init__(self, limit: int | None = None):
        self.limit = None if limit is None else max(1, int(limit))
        self._frames: deque = deque(maxlen=self.limit)
        self.dropped = 0
        self.total = 0

    def append(self, frame: dict) -> None:
        if self.limit is not None and len(self._frames) == self.limit:
            self.dropped += 1
        self._frames.append(frame)
        self.total += 1

    def frames(self) -> list[dict]:
        return list(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def summary(self) -> dict:
        return {
            "kept": len(self._frames),
            "dropped": self.dropped,
            "total": self.total,
            "limit": self.limit,
        }
