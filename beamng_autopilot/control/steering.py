"""Steering shaping: rate AND jerk limits, reversal suppression (plan D3).

``autopilot.smooth_steer`` is a first-order limiter: it bounds how far the
wheel may move per second, nothing else.  A control loop can still reach
the same wheel angle through a path no human would drive - slamming to
full rate instantly (unbounded jerk) and, worse, oscillating: with a
perception reference that flickers a few centimetres, the loop commands
left-right-left a few times per second at tiny amplitude, which is the
"continuous small corrections" wobble the improvement plan's phase D3
asks to remove.

``SteeringShaper`` adds the three missing constraints:

* rate limit - the same first-order bound the existing helper applies;
* JERK limit - the applied rate itself may only change so fast, so the
  wheel ramps into and out of a correction instead of snapping;
* reversal guard - the commanded direction may flip at most N times per
  window; beyond that, and only while the command stays inside a small
  trim band (both the request and the applied angle are tiny), the
  shaper holds the previous angle and bleeds the rate to zero instead of
  following the wiggle.  Large commands are never suppressed: a real
  correction (or a safety manoeuvre) always goes through.

``force=True`` bypasses every limit for one call, because a limiter must
never be able to delay a safety action - callers that own a hard stop
decide, not this module.

``cap`` is the OTHER direction: an upper bound on the command, i.e. the
authority the car currently has (mode gain, reference-stability limit,
post-veto steering budget).  The shaper is a comfort filter and used to
override it - measured 2026-09-22: with the wheel at 0.55 and a new
authority of 0.15, the next shaped command was still 0.49, so a
"small corrections only" reference was driving at three times its
permission for several hundred milliseconds.  When ``cap`` is given, the
request AND the shaped output are clamped to it, and the internal
value/rate are synchronised with the clamp so the state cannot keep
pushing outward.  Safety outranks comfort: reaching the cap immediately
is intended, and callers that also need to reduce speed do that in their
own layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Normalized steering input units (BeamNG vehicle input, -1..1).
STEER_MAX_RATE_PER_S = 1.2      # |d steer| / dt at the wheel
STEER_MAX_JERK_PER_S2 = 6.0     # |d rate| / dt
STEER_REVERSAL_WINDOW_S = 1.0
STEER_MAX_REVERSALS = 3
# Below this magnitude (both request and applied angle) a reversal is
# "trim wiggle" and may be suppressed; above it the command is a real
# correction and always passes.
STEER_TRIM_BAND = 0.05


@dataclass
class SteeringShaper:
    """Rate/jerk-limited steering with a small-amplitude reversal guard."""

    max_rate_per_s: float = STEER_MAX_RATE_PER_S
    max_jerk_per_s2: float = STEER_MAX_JERK_PER_S2
    reversal_window_s: float = STEER_REVERSAL_WINDOW_S
    max_reversals: int = STEER_MAX_REVERSALS
    trim_band: float = STEER_TRIM_BAND
    # state
    value: float = 0.0
    rate: float = 0.0
    started: bool = False
    reversals: int = 0
    suppressed: int = 0
    capped: int = 0          # steps whose OUTPUT the authority cap clipped
    window_t: float | None = None
    _last_sign: int = 0     # sign of the last non-zero REQUESTED direction
    history: list = field(default_factory=list)

    # ------------------------------------------------------------------
    def force_state(self, value: float) -> None:
        """Adopt ``value`` as the current command, rate reset to 0.

        Used after a constraint that had to be applied OUTSIDE the shaper
        (a directional veto on the shaped output): the shaper must start
        the next step from what was actually sent, not from the value it
        wanted to send, or it keeps re-applying the vetoed command.
        """
        self.value = float(np.clip(float(value), -1.0, 1.0))
        self.rate = 0.0
        self.started = True

    def update(self, desired: float, dt: float, *, now: float | None = None,
               force: bool = False, cap: float | None = None) -> float:
        """Shape ``desired`` into the commanded steering for this step.

        ``cap`` (>= 0) is the current steering authority; ``None`` means
        unbounded.  It is applied to the request and to the shaped output,
        including the case where only the shaper's own state is outside it.
        """
        desired = float(np.clip(float(desired), -1.0, 1.0))
        cap_v: float | None = None
        if cap is not None:
            try:
                cap_v = abs(float(cap))
            except (TypeError, ValueError):
                cap_v = None
            if cap_v is not None and not np.isfinite(cap_v):
                cap_v = None
            if cap_v is not None:
                desired = float(np.clip(desired, -cap_v, cap_v))
                if self.started and abs(self.value) > cap_v + 1e-9:
                    # The authority shrank below what the wheel is doing:
                    # come inside at once and re-base the rate, or every
                    # following step starts outside the cap again.
                    self.capped += 1
                    self.force_state(float(np.clip(self.value, -cap_v,
                                                   cap_v)))
        if force:
            # A safety action owns this tick: pass it through untouched,
            # keep the limiter's state consistent for the next call.
            rate = ((desired - self.value) / max(1e-3, float(dt))
                    if self.started else 0.0)
            self.value = desired
            self.rate = float(np.clip(rate, -self.max_rate_per_s,
                                      self.max_rate_per_s))
            self.started = True
            self.history.append((float(now) if now is not None else None,
                                 self.value, self.rate))
            return self.value
        dt = float(dt)
        if not np.isfinite(dt):
            # a bad dt must not turn the command into NaN; fall back to a
            # typical control period and let the limits do the rest
            dt = 0.05
        dt = float(np.clip(dt, 1e-3, 0.5))
        if not self.started:
            self.value = 0.0
            self.rate = 0.0
            self.started = True
        _prev_value = float(self.value)
        # 1) what rate would reach the request, bounded
        want_rate = (desired - self.value) / dt
        want_rate = float(np.clip(want_rate, -self.max_rate_per_s,
                                  self.max_rate_per_s))
        # 2) the rate itself may only change so fast (the jerk bound)
        max_drate = self.max_jerk_per_s2 * dt
        rate = self.rate + float(np.clip(want_rate - self.rate,
                                         -max_drate, max_drate))
        # 3) reversal guard: count flips of the REQUESTED direction.
        # Tracking the rate's sign instead is far too noisy near zero (a
        # bounded rate sits at 0.000 between small commands, so a real
        # left-right-left wiggle took 20 ticks to register two flips and
        # the guard never engaged); the request direction IS the wiggle.
        req_sign = 0 if abs(desired) < 1e-6 else (1 if desired > 0 else -1)
        if (req_sign != 0 and self._last_sign != 0
                and req_sign != self._last_sign):
            t = float(now) if now is not None else None
            if (self.window_t is None or t is None
                    or t - self.window_t > self.reversal_window_s):
                self.window_t = t
                self.reversals = 0
            self.reversals += 1
            if (self.reversals > self.max_reversals
                    and abs(desired) <= self.trim_band
                    and abs(self.value) <= self.trim_band):
                # hold the wheel, bleed the rate out - the wiggle dies
                self.reversals = self.max_reversals
                self.rate = 0.0
                self._last_sign = req_sign
                self.suppressed += 1
                return self.value
        if req_sign != 0:
            self._last_sign = req_sign
        self.rate = rate
        self.value = float(np.clip(self.value + rate * dt, -1.0, 1.0))
        if cap_v is not None and abs(self.value) > cap_v + 1e-9:
            # The shaped command still left the authority (the wheel was
            # already outside it, or the limit changed under us).  Clip the
            # OUTPUT and re-base the state so the next step starts inside.
            self.capped += 1
            self.value = float(np.clip(self.value, -cap_v, cap_v))
            achieved = (self.value - _prev_value) / max(1e-3, float(dt))
            self.rate = float(np.clip(achieved, -self.max_rate_per_s,
                                      self.max_rate_per_s))
        self.history.append((float(now) if now is not None else None,
                             self.value, self.rate))
        if len(self.history) > 4096:
            del self.history[:2048]
        return self.value

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.value = 0.0
        self.rate = 0.0
        self.started = False
        self.reversals = 0
        self.suppressed = 0
        self.capped = 0
        self.window_t = None
        self._last_sign = 0
        self.history.clear()

    def digest(self) -> dict:
        """JSON-safe state summary for telemetry."""
        return {
            "steer": round(float(self.value), 4),
            "rate": round(float(self.rate), 4),
            "reversals": int(self.reversals),
            "suppressed": int(self.suppressed),
            "capped": int(self.capped),
        }
