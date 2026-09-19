"""Speed-weighted steering blend (plan phase D2), opt-in and reversible.

The FSD loop steers with Pure Pursuit plus a curvature feed-forward
(``fsd_drive._path_curvature_ff``).  The improvement plan's phase D2 asks
for the missing terms - a lateral-error and a heading-error feedback -
with the weights scheduled by SPEED (at low speed a lateral offset is
cheap to correct and matters most; at high speed the heading error and
the curvature feed-forward dominate, because a large late lateral
correction is exactly what makes a car weave), and it is explicit that
the existing Pure Pursuit must stay: "不要直接废弃现有 Pure Pursuit，
应先采用可配置的混合控制，便于回退和对比".

So this module is the *policy* for that blend, and it is OFF by default:
``enabled=False`` returns the Pure Pursuit value untouched, and the
weights are the lever for a live A/B.  Nothing here reads a map, a route
or a lane centre - the error terms are measured against the trajectory
the planner already published.

Pure logic: inputs are scalars, output is one scalar plus a digest for
telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SteeringBlendWeights:
    """Weights at the low- and high-speed ends of the schedule.

    ``enabled=False`` keeps the loop on Pure Pursuit + feed-forward, so
    the blend only ever runs when a live A/B turns it on; the gain
    defaults below are the starting point that A/B sweeps, not a tuned
    result (nothing here has been calibrated on the car yet).
    """

    enabled: bool = False
    # schedule ends (m/s) between which the weights interpolate
    speed_low_mps: float = 2.0
    speed_high_mps: float = 12.0
    # curvature feed-forward gain is shared with the existing term
    ff_gain: float = 1.0
    # lateral-error feedback (steer per metre the ego sits off the path):
    # a low-speed offset is cheap to correct, a late high-speed pull is
    # what makes the car weave, so the gain FALLS with speed
    lat_gain_low: float = 0.12
    lat_gain_high: float = 0.05
    # heading-error feedback (steer per radian the nose is off the path
    # direction): a high-speed car needs the nose fixed early, so the
    # gain RISES with speed
    head_gain_low: float = 0.35
    head_gain_high: float = 0.60


@dataclass
class SteeringBlend:
    """One tick's blend result, with the terms for telemetry."""

    steer: float = 0.0
    pp: float = 0.0
    ff: float = 0.0
    lat_term: float = 0.0
    head_term: float = 0.0
    lat_gain: float = 0.0
    head_gain: float = 0.0
    enabled: bool = False
    clamped: bool = False
    rate: float = 0.0     # applied lat/head share of the total demand

    def digest(self) -> dict:
        return {
            "on": int(bool(self.enabled)),
            "pp": round(float(self.pp), 4),
            "ff": round(float(self.ff), 4),
            "lat": round(float(self.lat_term), 4),
            "head": round(float(self.head_term), 4),
            "lat_w": round(float(self.lat_gain), 4),
            "head_w": round(float(self.head_gain), 4),
            "share": round(float(self.rate), 3),
            "clamped": int(bool(self.clamped)),
        }


def _schedule(low: float, high: float, speed_mps: float,
              lo_mps: float, hi_mps: float) -> float:
    """Linear ramp of a weight between the two speed ends."""
    lo = float(min(lo_mps, hi_mps))
    hi = float(max(lo_mps, hi_mps))
    v = float(np.clip(float(speed_mps), lo, hi))
    if hi - lo <= 1e-9:
        return float(high)
    t = (v - lo) / (hi - lo)
    return float(low) * (1.0 - t) + float(high) * t


def blend_steering(pp_steer: float, curvature_ff: float,
                   lateral_error_m: float, heading_error_rad: float,
                   speed_mps: float,
                   weights: SteeringBlendWeights | None = None,
                   ) -> SteeringBlend:
    """Combine Pure Pursuit with feed-forward and error feedback.

    Sign convention (the caller's): a POSITIVE steering value steers
    right, and both error terms are expressed as "what the wheel should
    do about it" - so ``lateral_error_m`` is positive when the ego sits
    LEFT of the path (steer right to close it) and ``heading_error_rad``
    is positive when the nose must turn right.  ``pp_steer`` is the Pure
    Pursuit command already sign- and scale-converted by the caller and
    ``curvature_ff`` the existing feed-forward term.  With
    ``weights.enabled`` False the output is exactly
    ``pp_steer + curvature_ff * ff_gain`` - the behaviour the loop has
    today - so enabling is the only thing that changes anything.
    """
    w = weights or SteeringBlendWeights()
    out = SteeringBlend(pp=float(pp_steer), ff=float(curvature_ff),
                        enabled=bool(w.enabled))
    base = float(pp_steer) + float(curvature_ff) * float(w.ff_gain)
    if not w.enabled:
        out.steer = float(np.clip(base, -1.0, 1.0))
        out.clamped = abs(base) > 1.0
        return out
    out.lat_gain = _schedule(w.lat_gain_low, w.lat_gain_high,
                             speed_mps, w.speed_low_mps, w.speed_high_mps)
    out.head_gain = _schedule(w.head_gain_low, w.head_gain_high,
                              speed_mps, w.speed_low_mps, w.speed_high_mps)
    out.lat_term = float(out.lat_gain) * float(lateral_error_m)
    out.head_term = float(out.head_gain) * float(heading_error_rad)
    total = base + out.lat_term + out.head_term
    out.steer = float(np.clip(total, -1.0, 1.0))
    out.clamped = abs(total) > 1.0
    extra = abs(out.lat_term) + abs(out.head_term)
    out.rate = (extra / (abs(base) + extra)) if (abs(base) + extra) > 1e-9 \
        else 0.0
    return out
