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



def polyline_bearing(ref, pos, min_m: float = 1.5, max_m: float = 20.0):
    """Direction (rad) of ``ref``'s near-ahead part seen from ``pos``.

    The ego-anchored polyline starts at (or near) the car, so the zero-
    distance first vertex is skipped and the bearing is measured from
    the first vertex at least ``min_m`` ahead to the last vertex inside
    ``max_m``.  None when the reference has no measurable forward
    extent.
    """
    if ref is None or len(ref) < 2:
        return None
    r = np.asarray(ref[:, :2], dtype=float)
    p = np.asarray(pos[:2], dtype=float)
    d = np.linalg.norm(r - p, axis=1)
    sel = np.flatnonzero((d >= min_m) & (d <= max_m))
    if len(sel) < 2:
        sel = np.flatnonzero(d >= min_m)
    if len(sel) < 2:
        return None
    i = int(sel[0])
    j = int(sel[-1])
    v = r[j] - r[i]
    L = float(np.linalg.norm(v))
    if L < 1e-9:
        return None
    return math.atan2(v[1], v[0])


def bearing_diff_deg(a, b):
    """Signed angle difference (deg) normalized to (-180, 180]."""
    d = (float(a) - float(b)) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return math.degrees(d)


def lane_heading_ok(route, lane_ref, pos, heading,
                    max_yaw_deg: float = 60.0) -> bool:
    """True when the sensor lane points along the same roadway as route.

    The lane may only replace the route (or steer lateral alignment)
    when its near-ahead direction agrees with the route / ego heading.
    At a junction the vision/LiDAR lane can pair onto a DIFFERENT road
    (side branch or oncoming lane) whose free corridor looks clear; a
    heading check keeps the car on the navigational route instead of
    driving into that road (town run 2026-08-22: the paired lane headed
    into the side road and the car left the road and wedged).  When the
    lane bearing cannot be measured the decision falls back to True so
    the blocked-fraction logic keeps its old behaviour.
    """
    l_b = polyline_bearing(lane_ref, pos)
    if l_b is None:
        return True
    r_b = polyline_bearing(route, pos)
    ref = r_b if r_b is not None else float(heading)
    return abs(bearing_diff_deg(l_b, ref)) <= float(max_yaw_deg)


def lane_side_offset_m(lane, route, pos, max_m: float = 20.0):
    """Median signed lateral offset (m) of the lane from the route.

    Positive = LEFT of the route's travel direction, negative = RIGHT.
    The map nav route is the ROAD CENTRE line; the ego's own lane (right-
    hand traffic) sits on the RIGHT of it (~ -half lane width).  A sensor
    lane centre that lies LEFT of (or on) the road centreline is the
    ONCOMING lane and must never steer the car (town runs 2026-08-22:
    the paired vision/LiDAR lane locked onto the oncoming lane, passed
    the bearing gate - same direction - and the car rode the centre line
    / oncoming lane end to end).  None when no measurable overlap.
    """
    if route is None or len(route) < 2 or lane is None or len(lane) < 2:
        return None
    r = np.asarray(route[:, :2], dtype=float)
    l = np.asarray(lane[:, :2], dtype=float)
    p = np.asarray(pos[:2], dtype=float)
    dl = np.linalg.norm(l - p, axis=1)
    near = l[(dl >= 0.5) & (dl <= max_m)]
    if len(near) < 2:
        near = l[dl <= max_m]
    if len(near) < 2:
        return None
    vals = []
    for px, py in near:
        best = None
        for k in range(len(r) - 1):
            ax, ay = r[k]
            bx, by = r[k + 1]
            tx, ty = bx - ax, by - ay
            l2 = tx * tx + ty * ty
            if l2 < 1e-12:
                continue
            t = max(0.0, min(1.0, ((px - ax) * tx + (py - ay) * ty) / l2))
            cx, cy = ax + t * tx, ay + t * ty
            sx, sy = px - cx, py - cy
            d = math.hypot(sx, sy)
            if d > 12.0:
                continue
            cross = tx * sy - ty * sx   # >0 = left of travel
            sign = 1.0 if cross > 0 else -1.0
            val = sign * d
            if best is None or d < best[0]:
                best = (d, val)
        if best is not None:
            vals.append(best[1])
    return float(np.median(vals)) if vals else None


def lane_side_ok(lane, route, pos, left_max_m: float = 0.0) -> bool:
    """True when the lane sits on the ego (RIGHT) side of the route.

    A sensor lane whose centre is LEFT of (or on) the road centreline is
    the oncoming lane, never the ego lane - reject it (the map-prior own
    lane / navigational route takes over).  When the offset cannot be
    measured the decision falls back to True (old behaviour).
    """
    off = lane_side_offset_m(lane, route, pos)
    if off is None:
        return True
    return float(off) <= float(left_max_m)


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

    The sensor lane must also HEAD the same way as the route: at a
    junction the lane can pair onto a different road whose corridor
    reads clear, and following it drives the car off the navigational
    route (town run 2026-08-22 - the lane pointed into the side road
    and the car wedged).  ``lane_heading_ok`` gates that decision.
    """
    if route is None or len(route) < 2:
        return lane_ref if (lane_ref is not None and len(lane_ref) >= 2) else route
    if lane_ref is None or len(lane_ref) < 2:
        return route
    r_b = _ref_blocked_fraction(route, pos, heading, grid)
    l_b = _ref_blocked_fraction(lane_ref, pos, heading, grid)
    if float(r_b) >= float(route_blocked_frac) and \
            float(l_b) <= float(lane_clear_frac) and \
            lane_heading_ok(route, lane_ref, pos, heading) and \
            lane_side_ok(lane_ref, route, pos):
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
    # Forward progress over the FIRST ~8 m of the reference, never the
    # endpoint.  At a hairpin the reference turns ~90-180 deg and its END
    # sits BEHIND the ego heading while the near part still drives
    # forward along the road; the old endpoint gate rejected that fallback
    # and the car stopped dead at the apex with src=none (mountain run
    # 2026-08-26, run_fix8: stall at (724.8, 753.2) after the car drifted
    # off-route at the first hairpin).  Only the near part is evidence
    # the reference is drivable from here.
    k = max(2, min(len(r), 8))
    rel = r[:k] - pos
    fwd_m = float(np.max(rel[:, 0] * fwd[0] + rel[:, 1] * fwd[1]))
    if d0 > near_m or fwd_m < forward_m:
        return None
    return ref
