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

* **Clock types are not interchangeable.**  Production trace timestamps use
  ``time.time()``; command deadlines use a separate monotonic clock and scene
  state uses the sim clock.  :func:`span_ms` rejects cross-clock pairs and
  returns unknown for invalid or reversed timestamps.
* **A receipt is not an execution.**  If the command side cannot confirm the
  vehicle acted, the stage is UNKNOWN - never "executed".  Writing "sent" and
  reading it as "done" is how a control failure gets recorded as success.
"""
from __future__ import annotations

from collections import deque
import math

# ------------------------------------------------------------------ clocks

CLOCK_WALL = "wall_time"
CLOCK_MONOTONIC = "monotonic_wall"
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

# ---------------------------------------------------------- lateral fields
# Review handoff P0-2: every lateral field the telemetry can publish is
# declared here as ``reference + frame + sign + unit + coverage``, so one
# frame of telemetry plus this contract answers "which side of the line is
# the car on" without reading the code.  ``zero`` and ``none`` are part of
# the contract because "0" and "no measurement" were being confused:
# ``road_off = 0`` reads as "on the road" in any report, while it also
# means "no boundary was seen".
LATERAL_FIELD_SPECS: dict[str, dict] = {
    "line_lat": {
        "reference": "painted line(s) - semantic line mask, back-projected "
                     "to world; MEAN over every marking within 25 m",
        "frame": "car", "sign": "+ = the paint is to the car's LEFT "
                              "(the car sits RIGHT of it)",
        "unit": "m",
        "coverage": "frames where at least one marking back-projected",
        "zero": "the paint runs through the car's centre line",
        "none": "no marking projected this frame - NOT 'centred'",
    },
    "ego_lat_route_m": {
        "reference": "nav route polyline (road centreline; METRIC ONLY)",
        "frame": "route", "sign": "+ = the car is LEFT of the route "
                                  "direction of travel",
        "unit": "m", "coverage": "diagnostic probe frames with a route",
        "zero": "the car sits on the route centreline",
        "none": "no nav route available",
    },
    "lat_route_m": {
        "reference": "nav route polyline (metric only)", "frame": "route",
        "sign": "+ = left of the route direction", "unit": "m",
        "coverage": "diagnostic probe, per detected marking",
        "zero": "the point is on the route centreline",
        "none": "no route / point not projectable",
    },
    "lat_car_m": {
        "reference": "the car's own centre line", "frame": "car",
        "sign": "+ = left of the car's heading", "unit": "m",
        "coverage": "diagnostic probe, per detected marking",
        "zero": "the point is beside the car's centre",
        "none": "point further than 25 m from the car",
    },
    "lane_side_off_m": {
        "reference": "nav route polyline (used as the road-centre metric "
                     "for the SIDE gate, never as a lateral setpoint)",
        "frame": "route",
        "sign": "+ = the sensor lane CENTRE is LEFT of the route",
        "unit": "m",
        "coverage": "frames where the perception lane produced a centre",
        "zero": "the lane centre sits on the road centreline; strict mode "
                "rejects it (limit -0.2 m)",
        "none": "no measurable overlap between lane centre and route",
    },
    "lane_dev_m": {
        "reference": "the accepted lateral reference (perception lane)",
        "frame": "magnitude - no side information",
        "sign": "UNSIGNED: median absolute distance from the planned path "
                "to the reference",
        "unit": "m",
        "coverage": "frames with both a reference and a path",
        "zero": "the path coincides with the reference",
        "none": "no lateral reference this tick -> UNKNOWN; a 0 used to be "
                "emitted here and read as 'perfectly aligned'",
    },
    "lat_left": {
        "reference": "the LEFT lane boundary published this tick",
        # Measured 2026-09-22: the producer (fsd_drive) projects the
        # vehicle CENTRE point onto the boundary; the four-corner worst
        # case is body_lat_left/right.  Saying "body corners" here made the
        # two fields look interchangeable when they are different
        # measurements (plan §2.5-1 / T01).
        "frame": "car (vehicle centre point)",
        "sign": "+ = the centre is past the boundary towards the oncoming "
                "lane", "unit": "m",
        "coverage": "ONLY frames with a paired (two-sided) boundary "
                    "published - a single-edge mirror publishes none",
        "zero": "the centre exactly on the boundary",
        "none": "no boundary published -> UNKNOWN, never 'on road'",
        "corner_metric": "body_lat_left (worst of the four corners)",
    },
    "lat_right": {
        "reference": "the RIGHT lane boundary published this tick",
        "frame": "car (vehicle centre point)",
        "sign": "- = the centre is past the boundary away from the road",
        "unit": "m", "coverage": "same rule as lat_left",
        "zero": "the centre exactly on the boundary",
        "none": "no boundary published -> UNKNOWN, never 'on road'",
        "corner_metric": "body_lat_right (worst of the four corners)",
    },
    "body_lat_left": {
        "reference": "LEFT boundary (worst of the four body corners)",
        "frame": "car",
        "sign": "+ = the body is past the boundary towards the oncoming "
                "lane (same convention as lat_left)",
        "unit": "m", "coverage": "same rule as lat_left",
        "zero": "a corner exactly on the boundary",
        "none": "no boundary published -> UNKNOWN, never 'on road'",
    },
    "body_lat_right": {
        "reference": "RIGHT boundary (worst of the four body corners)",
        "frame": "car",
        "sign": "- = the body is past the boundary away from the road "
                "(same convention as lat_right)",
        "unit": "m", "coverage": "same rule as lat_right",
        "zero": "a corner exactly on the boundary",
        "none": "no boundary published -> UNKNOWN, never 'on road'",
    },
    "road_off": {
        "reference": "detected lane boundary (perception only)",
        "frame": "car (body corners)",
        "sign": ">= 0 magnitude of the worst overshoot past a boundary",
        "unit": "m",
        "coverage": "frames where a boundary covered the body position",
        "zero": "EITHER the body is inside the boundary OR no boundary was "
                "detected - the two are NOT distinguishable from this field "
                "alone; read lat_left/lat_right first",
        "none": "never emitted as None; use lat_left/lat_right to tell "
                "'inside' from 'unmeasured'",
    },
    "edge_over": {
        "reference": "map DecalRoad edge polylines - METRIC ONLY, never a "
                     "lateral control input",
        "frame": "car", "sign": ">= 0 distance beyond the road's own edge",
        "unit": "m", "coverage": "frames with a route and both map edges",
        "zero": "the car is between the map road edges",
        "none": "no route / no map edges",
    },
}


def lateral_digest(hist, fields=None) -> dict:
    """Per-run coverage and spread of every lateral field (handoff P0-2).

    Reports, per field: frames carrying a number, frames with the column
    missing, and median/min/max.  A column that is absent from the whole
    run is reported as ``None`` with ``status="UNKNOWN"`` - never as 0,
    because "we did not measure a crossing" and "we measured no crossing"
    are different statements (the round-5 report was wrong exactly there).
    """
    rows = list(hist or [])
    names = list(fields or LATERAL_FIELD_SPECS)
    out: dict = {"frames": len(rows), "fields": {}}
    for name in names:
        vals: list[float] = []
        missing = 0
        for row in rows:
            v = row.get(name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                missing += 1
                continue
            if not math.isfinite(float(v)):
                missing += 1
                continue
            vals.append(float(v))
        spec = LATERAL_FIELD_SPECS.get(name, {})
        if not vals:
            out["fields"][name] = {
                "status": "UNKNOWN", "measured_frames": 0,
                "missing_frames": missing, "median": None, "min": None,
                "max": None, "spec": spec}
            continue
        vals_sorted = sorted(vals)
        out["fields"][name] = {
            "status": "measured", "measured_frames": len(vals),
            "missing_frames": missing,
            "median": round(vals_sorted[len(vals_sorted) // 2], 3),
            "min": round(vals_sorted[0], 3),
            "max": round(vals_sorted[-1], 3),
            "spec": spec}
    return out

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
    if field == "cmd_monotonic_t":
        return CLOCK_MONOTONIC
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
    try:
        elapsed = (float(b) - float(a)) * 1000.0
    except (TypeError, ValueError):
        return None
    return elapsed if math.isfinite(elapsed) and elapsed >= 0.0 else None


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
    if stale_age_s is not None:
        try:
            age, bound = float(age), float(stale_age_s)
        except (TypeError, ValueError):
            return STAGE_UNKNOWN
        if not (math.isfinite(age) and age >= 0.0
                and math.isfinite(bound) and bound >= 0.0):
            return STAGE_UNKNOWN
        if age > bound:
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
