"""Joint conditions for a held path: time, travel, uncertainty, stopping space.

T09 of the round-5 plan.  ``SafetyMonitor.PathHold`` bounds a reused path by
TIME (grace 0.30 s, max 0.80 s) plus two geometry checks that the plan
explicitly warns must not be over-read:

* ``PATH_HOLD_MAX_LAT_M = 2.5`` is the distance from the ego to the NEAREST
  POINT of the held path - not accumulated travel, not a dead-reckoning
  error bound;
* ``PATH_HOLD_MIN_AHEAD_M = 4.0`` is the REMAINING ARC LENGTH of the held
  path - not a bound on how far the car may have moved.

So a hold can currently be kept alive by an offer while three things the
plan asks for are unmeasured: how long since the last REAL observation,
how far the car has actually driven since then, how uncertain the lateral
state is, and whether the remaining path still covers the braking
distance.  This module computes those four conditions and reports them.

Enforcement is **contraction only** and switch-gated (default off): when the
switch is on, a hold whose joint conditions fail is refused - a window is
never extended, and the grace/max numbers are untouched.  Repeated offers
cannot refresh the lifetime either: the clock here runs from the last real
OBSERVATION, which the caller supplies, not from ``offered_at``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

# The stopping-margin model lives in ``obstacle_risk``, with the RISK_*
# constants.  It must not be reached through the lane package: a planner
# module importing ``lane`` is a layering inversion, and it showed up as a
# real ImportError when this round's commits were checked one by one.
from beamng_autopilot.obstacle_risk import stopping_margin_m

#: Contract-only enforcement of the joint conditions.  OFF by default: it
#: changes when the car may reuse a path, so it is a behaviour change that
#: has to be A/B'd (plan §6/§8.2), exactly like every other gate here.
HOLD_JOINT_GATE = os.environ.get("BEAMNG_HOLD_JOINT_GATE", "0") != "0"

#: How long after the last real observation a held path is still tolerable.
HOLD_OBS_MAX_S = 0.80
#: How far the car may drive on a held path before it must stop asking.
HOLD_TRAVEL_MAX_M = 12.0
#: Lateral uncertainty beyond which the held path is not trustworthy.
HOLD_SIGMA_MAX_M = 0.60
#: Heading uncertainty, radians.
HOLD_SIGMA_THETA_MAX_RAD = 0.10


@dataclass
class HoldAudit:
    """The four joint conditions plus their verdict (JSON-safe)."""

    observation_age_s: float | None = None
    travelled_since_obs_m: float | None = None
    sigma_lat_m: float | None = None
    sigma_theta_rad: float | None = None
    required_stop_m: float | None = None
    remaining_arc_m: float | None = None
    satisfied: bool = False
    failed: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    lifetime_source: str = "observation"
    enforced: bool = False

    def as_dict(self) -> dict:
        out = {
            "observation_age_s": (None if self.observation_age_s is None
                                  else round(float(self.observation_age_s), 3)),
            "travelled_since_obs_m": (
                None if self.travelled_since_obs_m is None
                else round(float(self.travelled_since_obs_m), 3)),
            "sigma_lat_m": (None if self.sigma_lat_m is None
                            else round(float(self.sigma_lat_m), 3)),
            "sigma_theta_rad": (None if self.sigma_theta_rad is None
                                else round(float(self.sigma_theta_rad), 4)),
            "required_stop_m": (None if self.required_stop_m is None
                                else round(float(self.required_stop_m), 3)),
            "remaining_arc_m": (None if self.remaining_arc_m is None
                                else round(float(self.remaining_arc_m), 3)),
            "satisfied": bool(self.satisfied),
            "failed": list(self.failed),
            "unknown": sorted(set(self.unknown)),
            "lifetime_source": self.lifetime_source,
            "enforced": bool(self.enforced),
        }
        return out


def remaining_arc_m(path, pos) -> float | None:
    """Arc length of ``path`` from the nearest point ahead of ``pos``."""
    if path is None:
        return None
    pts = np.asarray(path, dtype=float)
    if pts.ndim != 2 or len(pts) < 2:
        return None
    p = np.asarray(pos, dtype=float).ravel()[:2]
    d = np.linalg.norm(pts[:, :2] - p[None, :], axis=1)
    j = int(np.argmin(d))
    if j >= len(pts) - 1:
        return 0.0
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    return float(seg[j:].sum())


def audit_hold(*, path, pos, observation_age_s: float | None,
               travelled_since_obs_m: float | None,
               sigma_lat_m: float | None, sigma_theta_rad: float | None,
               speed_mps: float, latency_s: float,
               a_min_mps2: float | None, extra_margin_m: float = 0.5,
               enforce: bool | None = None) -> HoldAudit:
    """Evaluate the four joint conditions for one held path.

    ``satisfied`` is False when any condition FAILS, and also when a
    condition cannot be measured (``unknown`` lists those): an unmeasured
    joint condition must not read as "fine", which is the same discipline
    the rest of the project applies to UNKNOWN.
    """
    out = HoldAudit(observation_age_s=observation_age_s,
                    travelled_since_obs_m=travelled_since_obs_m,
                    sigma_lat_m=sigma_lat_m,
                    sigma_theta_rad=sigma_theta_rad)
    out.enforced = bool(HOLD_JOINT_GATE if enforce is None else enforce)
    # 1) time since the last REAL observation
    if observation_age_s is None:
        out.unknown.append("observation_age")
    elif float(observation_age_s) > HOLD_OBS_MAX_S:
        out.failed.append(
            f"observation_age: {float(observation_age_s):.2f}s > "
            f"{HOLD_OBS_MAX_S:.2f}s")
    # 2) accumulated travel since that observation
    if travelled_since_obs_m is None:
        out.unknown.append("travelled_since_obs")
    elif float(travelled_since_obs_m) > HOLD_TRAVEL_MAX_M:
        out.failed.append(
            f"travelled: {float(travelled_since_obs_m):.1f}m > "
            f"{HOLD_TRAVEL_MAX_M:.1f}m")
    # 3) lateral / heading uncertainty
    if sigma_lat_m is None:
        out.unknown.append("sigma_lat")
    elif float(sigma_lat_m) > HOLD_SIGMA_MAX_M:
        out.failed.append(
            f"sigma_lat: {float(sigma_lat_m):.2f}m > {HOLD_SIGMA_MAX_M:.2f}m")
    if sigma_theta_rad is None:
        out.unknown.append("sigma_theta")
    elif abs(float(sigma_theta_rad)) > HOLD_SIGMA_THETA_MAX_RAD:
        out.failed.append(
            f"sigma_theta: {float(sigma_theta_rad):.3f}rad > "
            f"{HOLD_SIGMA_THETA_MAX_RAD:.3f}rad")
    # 4) stopping space on what is left of the path
    arc = remaining_arc_m(path, pos)
    out.remaining_arc_m = arc
    need, why = stopping_margin_m(speed_mps, latency_s=latency_s,
                                  a_min_mps2=a_min_mps2, extra_m=extra_margin_m)
    out.required_stop_m = need
    if need is None or arc is None:
        out.unknown.append("stopping_space")
        if need is None:
            out.failed.append(f"stopping distance unknown ({why})")
        if arc is None:
            out.failed.append("remaining path unknown")
    elif arc + 1e-9 < need:
        out.failed.append(
            f"stopping space: {arc:.1f}m left < {need:.1f}m needed")
    out.satisfied = not out.failed and not out.unknown
    return out
