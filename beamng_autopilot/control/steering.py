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
    window_t: float | None = None
    _last_sign: int = 0     # sign of the last non-zero REQUESTED direction
    history: list = field(default_factory=list)

    # ------------------------------------------------------------------
    def update(self, desired: float, dt: float, *, now: float | None = None,
               force: bool = False) -> float:
        """Shape ``desired`` into the commanded steering for this step."""
        desired = float(np.clip(float(desired), -1.0, 1.0))
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
        }
