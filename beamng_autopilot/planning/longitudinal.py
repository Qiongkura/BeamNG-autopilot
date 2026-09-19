"""Longitudinal target composition (plan phase D4), opt-in and reversible.

The plan lists what the longitudinal plan must consider together - road
curvature, lateral acceleration, obstacle TTC, road-boundary confidence,
lane-marking confidence - plus the shape of the resulting speed: bounded
acceleration AND bounded jerk, so the speed profile is as driveable as
the steering one.

Pieces of that already exist and are NOT re-implemented here: the
curvature profile and its look-ahead brake band live in
``planning.speed_profile`` (this module reuses its ``COMFORT_LAT``
constant), the obstacle TTC bound lives in ``obstacle_risk`` (reused via
``ttc_speed_cap``), and the per-tick target intersection (monitor cap,
plan, cruise) stays in the drive loop.  What was missing is the layer
that composes them and shapes the result:

* ``compose()`` - pure: the tightest of the plan / curvature / obstacle
  caps, scaled DOWN by a perception-confidence factor (never up: low
  confidence is a reason to go slower, never faster);
* ``update()`` - stateful: moves a reference speed toward that target
  with bounded acceleration, bounded deceleration and a bounded JERK
  (the rate at which the acceleration itself may change), so the pedal
  demand ramps instead of stepping.

Unknown inputs are treated as UNRESTRICTING, not as untrusted: a missing
radius, gap or confidence means "no evidence to slow for", because
inventing a penalty for absent data would park the car on every frame
that lacks a lane envelope.  Nothing here reads a map, a route or an
offset constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from beamng_autopilot.obstacle_risk import ttc_speed_cap
from beamng_autopilot.planning.speed_profile import COMFORT_LAT

LONG_MAX_ACCEL_MPS2 = 1.2
LONG_MAX_DECEL_MPS2 = 3.0
# The rate at which the acceleration may change (m/s^3).  The plan asks
# for a longitudinal jerk bound; 2.5 reaches full braking in ~1.2 s, which
# is the "ramp the pedal, don't step it" behaviour the comfort metrics
# measure.
LONG_MAX_JERK_MPS3 = 2.5
# Response time of the reference toward its target (a first-order pull).
LONG_RESPONSE_TAU_S = 0.6
# Speed scale at ZERO confidence (low confidence slows, it never stops on
# its own - the safety monitor owns stopping).
LONG_CONFIDENCE_FLOOR = 0.6


@dataclass
class LongitudinalTerms:
    """Every term of one longitudinal step (telemetry + tests)."""

    plan: float = 0.0
    curvature: float = float("inf")
    obstacle: float = float("inf")
    confidence: float = 1.0
    target: float = 0.0
    reference: float = 0.0
    accel: float = 0.0
    jerk: float = 0.0

    def digest(self) -> dict:
        """JSON-safe summary (inf is reported as None)."""
        def _f(v: float):
            return (None if not math.isfinite(float(v))
                    else round(float(v), 3))
        return {
            "plan": _f(self.plan),
            "curve": _f(self.curvature),
            "obs": _f(self.obstacle),
            "conf": round(float(self.confidence), 3),
            "target": _f(self.target),
            "ref": _f(self.reference),
            "accel": round(float(self.accel), 3),
            "jerk": round(float(self.jerk), 3),
        }


@dataclass
class LongitudinalPlanner:
    """Compose the longitudinal target, then shape the reference speed."""

    max_accel_mps2: float = LONG_MAX_ACCEL_MPS2
    max_decel_mps2: float = LONG_MAX_DECEL_MPS2
    max_jerk_mps3: float = LONG_MAX_JERK_MPS3
    tau_s: float = LONG_RESPONSE_TAU_S
    confidence_floor: float = LONG_CONFIDENCE_FLOOR
    comfort_lat_mps2: float = COMFORT_LAT
    accel: float = 0.0
    started: bool = False

    # ------------------------------------------------------------------
    def compose(self, *, plan_speed: float, radius_m: float | None = None,
                gap_m: float | None = None, ttc_s: float | None = None,
                road_conf: float | None = None,
                line_conf: float | None = None) -> LongitudinalTerms:
        """The tightest cap, scaled down by perception confidence."""
        terms = LongitudinalTerms(plan=max(0.0, float(plan_speed)))
        if radius_m is not None:
            r = float(radius_m)
            if math.isfinite(r) and r > 0.0:
                terms.curvature = math.sqrt(
                    max(0.0, float(self.comfort_lat_mps2)) * r)
        if gap_m is not None:
            gap = float(gap_m)
            closing = 0.0
            if ttc_s is not None and float(ttc_s) > 1e-6 and gap > 0.0:
                closing = gap / float(ttc_s)
            terms.obstacle = ttc_speed_cap(gap, closing)
        confs = [float(c) for c in (road_conf, line_conf) if c is not None]
        if confs:
            conf = float(np.clip(min(confs), 0.0, 1.0))
            terms.confidence = (self.confidence_floor
                                + (1.0 - self.confidence_floor) * conf)
        terms.target = (min(terms.plan, terms.curvature, terms.obstacle)
                        * terms.confidence)
        return terms

    # ------------------------------------------------------------------
    def update(self, *, speed_mps: float, dt: float, plan_speed: float,
               radius_m: float | None = None, gap_m: float | None = None,
               ttc_s: float | None = None, road_conf: float | None = None,
               line_conf: float | None = None) -> LongitudinalTerms:
        """One limited step of the reference speed toward the target.

        The reference never overshoots the target and stays within the
        acceleration, deceleration and jerk bounds; ``dt`` is sanitized
        so a bad period cannot emit NaN into a speed command.
        """
        terms = self.compose(plan_speed=plan_speed, radius_m=radius_m,
                             gap_m=gap_m, ttc_s=ttc_s,
                             road_conf=road_conf, line_conf=line_conf)
        dt = float(dt)
        if not np.isfinite(dt):
            dt = 0.05
        dt = float(np.clip(dt, 1e-3, 0.5))
        v = max(0.0, float(speed_mps))
        err = terms.target - v
        want_a = float(np.clip(err / max(1e-3, float(self.tau_s)),
                               -self.max_decel_mps2, self.max_accel_mps2))
        prev_a = float(self.accel) if self.started else 0.0
        max_da = self.max_jerk_mps3 * dt
        a = prev_a + float(np.clip(want_a - prev_a, -max_da, max_da))
        a = float(np.clip(a, -self.max_decel_mps2, self.max_accel_mps2))
        terms.accel = a
        terms.jerk = (a - prev_a) / dt
        # approach the target without overshooting it
        lo, hi = min(v, terms.target), max(v, terms.target)
        terms.reference = max(0.0, float(np.clip(v + a * dt, lo, hi)))
        self.accel = a
        self.started = True
        return terms

    def reset(self) -> None:
        self.accel = 0.0
        self.started = False
