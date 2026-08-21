"""Planner arbitration: FSD stack trajectory with a rule fallback.

Tesla FSD keeps a conservative rule/kinematic layer underneath the
neural planner: when the learned planner returns nothing feasible or is
stale, the vehicle does not stop dead - it degrades to a kinematic
backup or a minimal-risk manoeuvre.  This module puts that *arbitration*
into the planning package as pure, testable logic:

* ``ArbiterOutcome``: the final path + source + why.
* ``arbitrate``: given the FSD stack's chosen trajectory and a rule
  reference path, pick which to actually steer by:

    1. FSD path when it is feasible (not empty) and the safety monitor
       marks it safe/degraded-but-drivable.
    2. Otherwise the rule reference (the proven route planner output).
    3. Else None -> the caller executes a minimal-risk stop.

The source labels feed telemetry so you can see whether the car was on
the FSD trajectory or on the rule fallback at any moment - the same
forensics FSD exposes between its neural and rule planners.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ArbiterOutcome:
    path: np.ndarray | None
    source: str          # "fsd" | "rule" | "none"
    why: str = ""


def arbitrate(fsd_path, rule_path, fsd_safe: bool = True,
              prefer_rule: bool = False) -> ArbiterOutcome:
    """Choose the path to steer.

    ``fsd_path`` is the layered planner's chosen trajectory (None when
    it produced nothing feasible).  ``rule_path`` is the rule autopilot's
    reference (route/drive path).  ``fsd_safe`` is the safety monitor's
    green light for the FSD path.  ``prefer_rule`` forces the rule path
    (used by "rule mode" / shadow tests).
    """
    if prefer_rule:
        if rule_path is not None and len(rule_path) >= 2:
            return ArbiterOutcome(rule_path, "rule", "forced")
        return ArbiterOutcome(None, "none", "forced rule empty")

    # FSD path wins when feasible and green-lit.
    if fsd_path is not None and len(fsd_path) >= 2 and fsd_safe:
        return ArbiterOutcome(fsd_path, "fsd", "fsd feasible+safe")

    # Rule fallback so the car does not stop dead when FSD declined.
    if rule_path is not None and len(rule_path) >= 2:
        return ArbiterOutcome(rule_path, "rule", "fsd unavailable")

    return ArbiterOutcome(None, "none", "no fsd and no rule path")


def anchored_rule_ref(pos, heading, ref, near_m: float = 4.0,
                      forward_m: float = 1.0):
    """Return ``ref`` only when it is a path the car can actually drive
    from ``pos``.

    A rule/route fallback is only usable when it is anchored at the ego
    (start near the car, endpoint forward of it).  A mis-anchored map
    prior whose start sits metres away - or a path that leads backward /
    sideways - is not a drivable fallback; when the layered planner
    declines, the correct FSD behaviour is a minimal-risk stop, not
    steering at a wall under a distant reference (town runs 2026-08-21).
    """
    if ref is None or len(ref) < 2:
        return None
    r = np.asarray(ref, dtype=float)[:, :2]
    pos = np.asarray(pos[:2], dtype=float)
    d0 = float(np.hypot(r[0, 0] - pos[0], r[0, 1] - pos[1]))
    fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
    fwd_m = float(np.dot(r[-1] - pos, fwd))
    if d0 > near_m or fwd_m < forward_m:
        return None
    return ref