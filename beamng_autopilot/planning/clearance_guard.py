"""Temporal guard around a forward-clearance safety reading.

The last-line clearance layer answers one question per tick: "how far can
the car still travel before something occupies the corridor?"  Two
sources feed it (the grid path clearance along the chosen trajectory, and
the raw sensor heading corridor), and both are *instantaneous* readings:
whatever this tick's sample says replaces last tick's verdict outright.

That is the wrong semantics for a safety reserve, because the reading is
not continuous.  Town run 2026-09-20 (``logs/fsd_benchmark/
town_1789886413.json``, collision at t=28.27 s, damage 0 -> 119.7) shows
the failure: ``fwd_clear`` read 3.44 m, then 1.20 m (already inside the
3.5 m/s braking reserve of 2.0 m) while the car was still accelerating,
then -0.25 m at the frame the damage sensor fired.  Across the recorded
town runs, 31 single-frame steps take the reading from below 2 m to above
8 m (e.g. -0.075 -> 16.36, 1.65 -> 15.56, 1.82 -> 14.02).  A jump like
that is not a measurement of "the wall left"; it is the corridor
geometry changing (a lateral obstacle swinging out of the 3.2 m wide
heading corridor as the car yaws), a different path being selected, or
the sample simply not covering the near field.  Trusting it immediately
clears a collision risk that was real one tick ago.

The same gap shows up as "no measurement reads as clear":
``forward_clearance_m`` returns ``inf`` for an *empty* hit list, and
``path_grid_clearance_m`` returns ``inf`` when the grid is missing - both
mean "unknown", not "clear".

This guard makes the reserve conservative in one direction only:

* a **decrease** is trusted immediately (fail-safe: new evidence of a
  closer obstruction always wins);
* an **increase** may only raise the enforced value to the minimum seen
  in the last ``hold_s`` seconds, so a single anomalous frame cannot
  erase the previous risk;
* an increase larger than ``jump_m`` additionally *latches* the low value
  until ``confirm_n`` consecutive non-jump frames corroborate the new
  reading - a genuine "the obstacle is gone" needs agreement, not one
  frame;
* an **unmeasured or stale** source never reads as clear: the window is
  frozen and its last known minimum keeps the braking authority, so a
  corridor that read 1.2 m before the sensor went quiet can never come
  back reading "clear" on no evidence.  Only a source that has *never*
  produced a usable reading fails closed to ``0.0`` (stop).

That last rule is deliberately narrow.  Deciding whether stale sensors
permit driving at all belongs to ``safety_monitor`` (it already owns the
stale contract and the degraded speed cap); duplicating it here would
turn a scheduling problem into a parking problem, because the 2026-09-20
town runs had a stale range sample on a quarter of the frames.  The guard
answers exactly one question: "may this tick's reading raise the
clearance the car is allowed to act on?"

It is pure logic (the caller supplies ``now_s``) and never raises a value
above what the sensor reported, so it can only ever brake earlier.  Every
decision is exposed in ``ClearanceReading.digest()`` for telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# How long a low reading keeps braking authority after it was measured.
# The control loop ticks at ~0.7 s wall time, so this is the "last few
# frames" window rather than a long memory.
CLEARANCE_HOLD_S = 1.2
# An upward step larger than this (m) is treated as an anomaly that needs
# corroboration instead of being believed outright.  It sits well above
# the per-tick variation a smooth approach produces (~0.1-0.5 m) and well
# below the observed 8-16 m jumps.
CLEARANCE_JUMP_M = 4.0
# Consecutive non-jump readings required before a latched low value is
# released.  Three ticks is ~2 s: long enough that a one-frame flicker
# cannot clear the reserve, short enough that a real departure resumes.
CLEARANCE_CONFIRM_N = 3
# A sample older than this is not usable evidence.  Matches
# ``safety_monitor.STALE_RANGE_S``: past that age the pipeline already
# declares the sensor stale, so the clearance derived from it must not be
# allowed to read as fresh clearance.
CLEARANCE_MAX_AGE_S = 2.0

REASON_FIRST = "first"
REASON_DECREASE = "decrease"
REASON_INCREASE = "increase"
REASON_WINDOW = "window_min"
REASON_JUMP_HELD = "jump_held"
REASON_UNMEASURED = "unmeasured"
REASON_STALE = "stale"

# Fail-closed reading when nothing has ever been measured: the car stops.
UNMEASURED_CLEARANCE_M = 0.0


@dataclass
class ClearanceReading:
    """One guarded clearance decision (telemetry + decision record)."""

    raw: float
    value: float
    valid: bool
    age_s: float | None
    min_recent: float
    jumped: bool
    held: bool
    reason: str
    n_hold: int = 0
    n_jump: int = 0

    def digest(self) -> dict:
        return {
            "raw": _finite_or_str(self.raw),
            "value": _finite_or_str(self.value),
            "valid": bool(self.valid),
            "age_s": (None if self.age_s is None
                      else round(float(self.age_s), 3)),
            "min_recent": _finite_or_str(self.min_recent),
            "jumped": bool(self.jumped),
            "held": bool(self.held),
            "reason": self.reason,
            "n_hold": int(self.n_hold),
            "n_jump": int(self.n_jump),
        }


def _finite_or_str(x) -> float | str:
    """JSON-safe clearance value: ``inf`` is not representable as a float."""
    x = float(x)
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    return round(x, 3)


@dataclass
class ClearanceGuard:
    """Window-minimum + jump latch around a clearance reading.

    Feed it one reading per tick with the source's validity and age; it
    returns the value the safety layer is allowed to act on.  Call
    ``update`` for EVERY tick, including the ones with no reading at all
    (``valid=False``), because "not measured" is exactly the state that
    must not read as clear.
    """

    hold_s: float = CLEARANCE_HOLD_S
    jump_m: float = CLEARANCE_JUMP_M
    confirm_n: int = CLEARANCE_CONFIRM_N
    max_age_s: float = CLEARANCE_MAX_AGE_S
    # (t, value) samples inside the hold window, oldest first
    _recent: list = field(default_factory=list)
    # latched (t, value) from the last anomalous jump, released only after
    # ``confirm_n`` corroborating frames or when the reading itself drops
    # back to/below it
    _latch: tuple | None = None
    _confirm: int = 0
    _last_raw: float | None = None
    _n_hold: int = 0
    _n_jump: int = 0

    def reset(self) -> None:
        self._recent = []
        self._latch = None
        self._confirm = 0
        self._last_raw = None
        self._n_hold = 0
        self._n_jump = 0

    # ------------------------------------------------------------------
    def update(self, value, now_s, *, valid: bool = True,
               age_s: float | None = None) -> ClearanceReading:
        """Guard one clearance reading.

        ``value`` is the raw reading in metres (``inf`` = measured clear,
        a negative value = the car already overlaps a hit).  ``valid``
        says whether the source actually produced a measurement this
        tick; ``age_s`` how old that measurement is.  Either being
        unusable makes the reading unusable as *fresh* evidence.
        """
        raw = float(value)
        now = float(now_s)
        stale = bool(age_s is not None
                     and float(age_s) > float(self.max_age_s))
        usable = bool(valid) and not stale
        prev_raw = self._last_raw

        jumped = False
        if usable:
            if prev_raw is not None:
                jumped = (raw - float(prev_raw)) > float(self.jump_m)
            self._recent.append((now, raw))
            self._last_raw = raw
            # Only a usable tick may age the window out.  An unusable tick
            # freezes it: "the sensor stopped reporting" must not be a way
            # for a low reading to expire into a clear corridor.
            cutoff = now - float(self.hold_s)
            self._recent = [(t, v) for (t, v) in self._recent
                            if t >= cutoff]
        window_min = min((v for _, v in self._recent), default=None)

        if usable and jumped:
            # Everything that was known before the jump - the previous
            # reading and the rest of the window - is what must survive
            # it.  The latched value is the most pessimistic of those.
            base = [raw] + [v for (t, v) in self._recent if t < now]
            if prev_raw is not None:
                base.append(float(prev_raw))
            self._latch = (now, float(min(base)))
            self._confirm = 0
            self._n_jump += 1
        elif usable:
            self._confirm += 1
            if self._latch is not None:
                latch_v = float(self._latch[1])
                if raw <= latch_v or self._confirm >= int(self.confirm_n):
                    self._latch = None

        floor = window_min
        if self._latch is not None:
            latch_v = float(self._latch[1])
            floor = latch_v if floor is None else min(float(floor), latch_v)

        if usable:
            value_out = raw if floor is None else min(raw, float(floor))
            if jumped:
                reason = REASON_JUMP_HELD
            elif prev_raw is None:
                reason = REASON_FIRST
            elif raw < float(prev_raw):
                reason = REASON_DECREASE
            elif value_out < raw:
                reason = REASON_WINDOW
            else:
                reason = REASON_INCREASE
        else:
            value_out = (UNMEASURED_CLEARANCE_M if floor is None
                         else float(floor))
            reason = REASON_STALE if stale else REASON_UNMEASURED

        held = bool(not usable or value_out < raw)
        if held:
            self._n_hold += 1

        min_recent = (float(floor) if floor is not None
                      else (raw if usable else UNMEASURED_CLEARANCE_M))
        return ClearanceReading(
            raw=raw,
            value=float(value_out),
            valid=usable,
            age_s=(None if age_s is None else float(age_s)),
            min_recent=float(min_recent),
            jumped=bool(jumped),
            held=held,
            reason=str(reason),
            n_hold=int(self._n_hold),
            n_jump=int(self._n_jump),
        )
