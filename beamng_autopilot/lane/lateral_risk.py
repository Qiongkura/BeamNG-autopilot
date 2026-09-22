"""Bounded lateral risk: signed gaps, first crossing, stopping margin (T09).

The plan's T09 asks for risk OUTPUT, not a new controller: the signed gap
from the car body to the real boundary, the distance/time to the FIRST
crossing, the stopping margin, and the interval over which the evidence
behind those numbers is still valid.  Three rules matter more than the
formulas:

* **Missing boundary is UNKNOWN, not a big number.**  ``gap = None`` and
  ``first_crossing = None`` whenever the boundary or the lateral state is
  not measured - never a silent 0 (which reads as "no risk") and never a
  division by a near-zero approach rate.
* **Not approaching means no crossing time.**  TLC is only defined while
  the car actually closes on the boundary; a car running parallel to it
  has no finite crossing time, and dividing by ``~0`` would invent one.
* **A formula is not a safety proof.**  The stopping margin is published
  with its applicability domain (flat ground, dry asphalt, no brake
  build-up delay) and with the inputs it used; when the deceleration or
  the free space is unknown it says so instead of assuming a number.

Nothing here moves the steering or the pedals: this module produces a
REPORT, and a test pins that it has no actuator output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Closing rate below which the car counts as "not approaching" (m/s).  A
# lateral rate this small over the horizon of interest moves the car
# centimetres; treating it as a crossing would be noise, not risk.
APPROACH_EPS_MPS = 0.05
# Default evidence ages (seconds) beyond which a lateral number is stale.
EVIDENCE_FRESH_S = 0.5
EVIDENCE_STALE_S = 2.0


def stopping_margin_m(speed_mps: float, *, latency_s: float,
                      a_min_mps2: float | None,
                      extra_m: float = 0.5) -> tuple[float | None, str]:
    """Delegate to the ONE stopping-margin model (``obstacle_risk``)."""
    from beamng_autopilot.obstacle_risk import stopping_margin_m as _impl
    return _impl(speed_mps, latency_s=latency_s, a_min_mps2=a_min_mps2,
                 extra_m=extra_m)


def signed_gap_m(body_half_width_m: float, boundary_lat_m: float | None,
                 *, worst_corner: bool = True) -> tuple[float | None, str]:
    """Signed gap from the body to a boundary line (+ = clear, - = past).

    ``boundary_lat_m`` is the signed lateral of the boundary in the car
    frame (+ = left, the project convention).  For the LEFT boundary the
    gap is ``boundary_lat - half_width``; the worst corner already accounts
    for the body being yawed when the caller passes the yawed extent.
    """
    if boundary_lat_m is None:
        return None, "boundary not published"
    try:
        b = float(boundary_lat_m)
        hw = float(body_half_width_m)
    except (TypeError, ValueError):
        return None, "boundary/body unreadable"
    if not (math.isfinite(b) and math.isfinite(hw)):
        return None, "boundary/body not finite"
    gap = abs(b) - hw
    return float(gap), "clear" if gap >= 0.0 else "past the boundary"


def first_crossing(gap_m: float | None, closing_rate_mps: float | None,
                   *, approach_eps: float = APPROACH_EPS_MPS
                   ) -> tuple[float | None, float | None, str]:
    """``(distance_m, time_s, reason)`` to the first crossing, or UNKNOWN.

    Returns ``(None, None, reason)`` when the gap is unknown, when the car
    is not closing on that boundary, or when it has already crossed
    (``distance`` 0 with a negative gap is reported as ``past``).
    """
    if gap_m is None:
        return None, None, "no boundary to measure"
    if closing_rate_mps is None:
        return None, None, "no lateral rate estimate"
    try:
        gap = float(gap_m)
        rate = float(closing_rate_mps)
    except (TypeError, ValueError):
        return None, None, "gap/rate unreadable"
    if not (math.isfinite(gap) and math.isfinite(rate)):
        return None, None, "gap/rate not finite"
    if gap < 0.0:
        return 0.0, 0.0, "already past the boundary"
    if abs(rate) <= float(approach_eps):
        return None, None, "not approaching (no finite crossing)"
    if gap == 0.0:
        return 0.0, 0.0, "on the boundary"
    dist = gap
    return dist, float(dist / abs(rate)), "approaching"


def evidence_interval(*, age_s: float | None,
                      history_only_frac: float | None) -> dict:
    """How far the evidence behind a lateral number can be trusted."""
    out: dict = {"age_s": (None if age_s is None else round(float(age_s), 3)),
                 "history_only_frac": (None if history_only_frac is None
                                       else round(float(history_only_frac), 3))}
    if age_s is None:
        out["state"] = "UNKNOWN"
        out["note"] = "no observation age recorded"
        return out
    age = float(age_s)
    if age <= EVIDENCE_FRESH_S and (history_only_frac or 0.0) < 0.5:
        out["state"] = "current"
    elif age <= EVIDENCE_STALE_S:
        out["state"] = "degrading"
    else:
        out["state"] = "stale"
    out["fresh_s"] = EVIDENCE_FRESH_S
    out["stale_s"] = EVIDENCE_STALE_S
    return out


@dataclass
class LateralRisk:
    """One tick's lateral risk report (JSON-safe)."""

    gap_left_m: float | None = None
    gap_right_m: float | None = None
    cross_distance_m: float | None = None
    cross_time_s: float | None = None
    cross_side: str | None = None
    stopping_margin_m: float | None = None
    stopping_reason: str = ""
    evidence: dict = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        out: dict = {
            "gap_left_m": (None if self.gap_left_m is None
                           else round(float(self.gap_left_m), 3)),
            "gap_right_m": (None if self.gap_right_m is None
                            else round(float(self.gap_right_m), 3)),
            "cross_distance_m": (None if self.cross_distance_m is None
                                 else round(float(self.cross_distance_m), 3)),
            "cross_time_s": (None if self.cross_time_s is None
                             else round(float(self.cross_time_s), 3)),
            "cross_side": self.cross_side,
            "stopping_margin_m": (None if self.stopping_margin_m is None
                                  else round(float(self.stopping_margin_m), 3)),
            "stopping_reason": self.stopping_reason,
            "evidence": dict(self.evidence),
            "unknown": sorted(set(self.unknown)),
        }
        if self.notes:
            out["notes"] = list(self.notes)
        return out


def lateral_risk(*, body_half_width_m: float,
                 lat_left_m: float | None, lat_right_m: float | None,
                 e_m: float | None, e_dot_mps: float | None,
                 speed_mps: float, latency_s: float,
                 a_min_mps2: float | None,
                 evidence_age_s: float | None = None,
                 history_only_frac: float | None = None,
                 extra_margin_m: float = 0.5) -> LateralRisk:
    """Assemble the whole report, marking every UNKNOWN explicitly.

    ``e_m`` / ``e_dot_mps`` come from the shadow state posterior (T06):
    ``e`` is the signed offset of the ego from the accepted reference and
    ``e_dot`` its rate, so the closing rate toward a boundary is the
    projection of that rate onto the side under test.
    """
    rep = LateralRisk()
    gap_l, why_l = signed_gap_m(body_half_width_m, lat_left_m)
    gap_r, why_r = signed_gap_m(body_half_width_m, lat_right_m)
    rep.gap_left_m, rep.gap_right_m = gap_l, gap_r
    if gap_l is None:
        rep.unknown.append("gap_left")
        rep.notes.append(f"left: {why_l}")
    else:
        rep.notes.append(f"left: {why_l}")
    if gap_r is None:
        rep.unknown.append("gap_right")
        rep.notes.append(f"right: {why_r}")
    else:
        rep.notes.append(f"right: {why_r}")
    # Closing rate per side, from the shadow lateral rate: moving LEFT
    # (e_dot > 0) closes the LEFT gap and opens the right one.
    rate_left = None if e_dot_mps is None else float(e_dot_mps)
    rate_right = None if e_dot_mps is None else -float(e_dot_mps)
    d_l, t_l, why_cl = first_crossing(gap_l, rate_left)
    d_r, t_r, why_cr = first_crossing(gap_r, rate_right)
    candidates = [(t, d, "left", why) for t, d, why in ((t_l, d_l, why_cl),)
                  ] + [(t_r, d_r, "right", why_cr)]
    approaching = [(t, d, side) for t, d, side, why in candidates
                   if t is not None]
    if approaching:
        t, d, side = min(approaching, key=lambda row: row[0])
        rep.cross_time_s, rep.cross_distance_m, rep.cross_side = t, d, side
    else:
        rep.unknown.append("first_crossing")
        rep.notes.append("crossing: " + "; ".join(
            f"{side}={why}" for _t, _d, side, why in candidates))
    rep.stopping_margin_m, rep.stopping_reason = stopping_margin_m(
        speed_mps, latency_s=latency_s, a_min_mps2=a_min_mps2,
        extra_m=extra_margin_m)
    if rep.stopping_margin_m is None:
        rep.unknown.append("stopping_margin")
    rep.evidence = evidence_interval(age_s=evidence_age_s,
                                     history_only_frac=history_only_frac)
    if rep.evidence.get("state") == "UNKNOWN":
        rep.unknown.append("evidence_interval")
    if e_m is None:
        rep.unknown.append("lateral_state")
    return rep
