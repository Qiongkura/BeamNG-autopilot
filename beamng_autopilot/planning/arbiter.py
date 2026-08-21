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


def _ref_blocked_fraction(ref, pos, heading, grid,
                          lo_m: float = 3.0, hi_m: float = 25.0,
                          fwd_min_m: float = 0.5) -> float:
    """Fraction of a reference line's near samples inside occupied cells.

    ``ref`` is a world-space polyline, ``pos``/``heading`` the ego pose and
    ``grid`` the current occupancy grid.  Only samples a sensible 3-25 m
    window ahead of the ego are counted; a long map-route tail beyond the
    sensor horizon is unknown space, not a wall (so it does not punish a
    route that simply extends past the grid).  Used to decide whether the
    map/nav route should keep governing planning when the sensor lane is
    clearly the drivable space ahead.
    """
    if ref is None or len(ref) < 2 or grid is None:
        return 0.0
    r = np.asarray(ref[:, :2], dtype=float)
    p = np.asarray(pos[:2], dtype=float)
    ch, sh = math.cos(float(heading)), math.sin(float(heading))
    extent = float(getattr(grid, "extent", 0.0) or 0.0)
    bad = 0.0
    tot = 0.0
    for x, y in r:
        dx, dy = x - p[0], y - p[1]
        d = math.hypot(dx, dy)
        if d < lo_m or d > hi_m:
            continue
        if extent > 0.0 and d > extent:
            continue
        if (dx * ch + dy * sh) < fwd_min_m:
            continue
        tot += 1.0
        cell = grid.world_to_cell(x, y)
        if cell is not None and grid.obstacle[cell] > 0:
            bad += 1.0
    return (bad / tot) if tot > 0 else 0.0


def choose_plan_route(route, lane_ref, pos, heading, grid,
                      route_blocked_frac: float = 0.20,
                      lane_clear_frac: float = 0.15):
    """Pick the navigational reference the planner should follow.

    Real FSD plans in vector space: the map/nav route carries the long
    intent, but the *sensor lane* (drivable-space centreline) is the
    roadway actually observed ahead.  Normally plan along the ego-anchored
    map route, but if that route's near-ahead corridor is occupied while
    the sensor lane ahead is clearly free, choose the sensor lane instead -
    a route that cuts through a building wall just parks the car against
    the wall if planning insists on it (town corner run 2026-08-21: the
    single-marker ``setPath`` route was a straight line through a wall,
    while the BEV lane actually turned away).
    """
    if route is None or len(route) < 2:
        return lane_ref if (lane_ref is not None and len(lane_ref) >= 2) else route
    if lane_ref is None or len(lane_ref) < 2:
        return route
    r_b = _ref_blocked_fraction(route, pos, heading, grid)
    l_b = _ref_blocked_fraction(lane_ref, pos, heading, grid)
    if float(r_b) >= float(route_blocked_frac) and float(l_b) <= float(lane_clear_frac):
        return lane_ref
    return route


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