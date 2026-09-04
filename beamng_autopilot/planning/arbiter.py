"""Planner arbitration: FSD stack trajectory with a rule fallback.

Tesla FSD keeps a conservative rule/kinematic layer underneath the
neural planner: when the learned planner returns nothing feasible or is
stale, the vehicle does not stop dead - it degrades to a kinematic
backup or a minimal-risk manoeuvre.  This module puts that *arbitration*
into the planning package as pure, testable logic:

* ``ArbiterOutcome``: the final path + source + why.
* ``arbitrate``: given the layered FSD stack's chosen trajectory, a
  trained E2E neural trajectory and a rule reference path, pick which to
  actually steer by:

    1. the layered FSD path when it is feasible (not empty) and the
       safety monitor marks it safe/degraded-but-drivable;
    2. otherwise the E2E neural path when it is feasible AND safe (the
       trained end-to-end planner ranks above the kinematic backup,
       exactly like FSD's own neural-vs-rule ordering);
    3. otherwise the rule reference (the proven route planner output);
    4. else None -> the caller executes a minimal-risk stop.

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
    source: str          # "fsd" | "e2e" | "rule" | "none"
    why: str = ""


def arbitrate(fsd_path, rule_path, fsd_safe: bool = True,
              e2e_path=None, e2e_safe: bool = False,
              prefer_rule: bool = False) -> ArbiterOutcome:
    """Choose the path to steer.

    ``fsd_path`` is the layered planner's chosen trajectory (None when
    it produced nothing feasible).  ``e2e_path`` is the trained neural
    planner's trajectory (None when the network is not loaded or its
    frame was unusable); ``e2e_safe`` is the safety monitor's green
    light for it.  ``rule_path`` is the rule autopilot's reference
    (route/drive path).  ``fsd_safe`` is the safety monitor's green
    light for the layered FSD path.  ``prefer_rule`` forces the rule
    path (used by "rule mode" / shadow tests).
    """
    if prefer_rule:
        if rule_path is not None and len(rule_path) >= 2:
            return ArbiterOutcome(rule_path, "rule", "forced")
        return ArbiterOutcome(None, "none", "forced rule empty")

    # Layered FSD path wins when feasible and green-lit.
    if fsd_path is not None and len(fsd_path) >= 2 and fsd_safe:
        return ArbiterOutcome(fsd_path, "fsd", "fsd feasible+safe")

    # Neural (E2E) planner next: a trained end-to-end trajectory is a
    # perception-driven candidate ranked above the map/rule backup - the
    # same neural-above-kinematic ordering FSD exposes in its telemetry.
    if e2e_path is not None and len(e2e_path) >= 2 and e2e_safe:
        return ArbiterOutcome(np.asarray(e2e_path, dtype=float), "e2e",
                              "fsd unavailable; e2e feasible+safe")

    # Rule fallback so the car does not stop dead when FSD declined.
    if rule_path is not None and len(rule_path) >= 2:
        return ArbiterOutcome(rule_path, "rule",
                              "fsd unavailable; e2e declined")

    return ArbiterOutcome(None, "none", "no fsd/e2e/rule path")


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


def lane_route_turn_ok(route, pos, look_m: float = 12.0,
                      max_turn_deg: float = 25.0) -> bool:
    """True when the nav route does NOT turn hard in the near-ahead window.

    A hard turn ahead means the car is at a real corner; there the
    map-prior own lane (a rounded copy of the same route) is the reliable
    reference and the sensor lane must not override it - vision/LiDAR
    pairing reads corner geometry wide (town run 2026-08-28 run11: the
    paired sensor lane read -48.6 deg vs route -24.6 deg, only 24 deg
    off so the 35 deg heading gate passed; the car S-curved 2.7 m left
    of the centreline then 5.2 m right past the edge).  Falls back to
    True when the turn cannot be measured.
    """
    if route is None or len(route) < 4:
        return True
    r = np.asarray(route[:, :2], dtype=float)
    p = np.asarray(pos[:2], dtype=float)
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))])
    i_near = int(np.argmin(np.linalg.norm(r - p, axis=1)))
    if float(arc[i_near]) >= float(arc[-1]):
        return True
    # Include a few samples BEHIND the nearest vertex: at a corner the
    # nearest route point snaps to the post-turn road, so a pure forward
    # window measures only the straight and misses the turn being driven.
    i0 = max(0, i_near - 5)
    i1 = int(np.searchsorted(arc, float(arc[i_near]) + float(look_m)))
    i1 = min(i1, len(r) - 1)
    if i1 - i0 < 2:
        return True
    seg = np.diff(r[i0:i1 + 1], axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    d = np.diff(ang)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.degrees(np.abs(d).sum())) <= float(max_turn_deg)


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

    Vectorised point-to-polyline projection (same math as the scalar
    loop it replaces: nearest route segment per lane point, segments
    farther than 12 m skipped, first-minimum tie-breaking preserved).
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
    seg = r[1:] - r[:-1]                                  # (M-1, 2)
    raw_l2 = (seg * seg).sum(axis=1)
    l2 = np.maximum(raw_l2, 1e-12)
    rel = near[:, None, :] - r[None, :-1, :]              # (K, M-1, 2)
    t = np.clip(np.einsum("kmi,mi->km", rel, seg) / l2[None, :], 0.0, 1.0)
    proj = r[None, :-1, :] + t[..., None] * seg[None, :, :]
    d = np.linalg.norm(proj - near[:, None, :], axis=2)   # (K, M-1)
    # zero-length segments were skipped entirely by the scalar loop
    d[:, raw_l2 < 1e-12] = np.inf
    bi = np.argmin(d, axis=1)
    rows = np.arange(len(near))
    dmin = d[rows, bi]
    # a lane point whose WHOLE route is > 12 m away contributes nothing
    # (the scalar loop skipped every such segment)
    keep = dmin <= 12.0
    if not keep.any():
        return None
    sy = near[keep, 1] - proj[rows[keep], bi[keep], 1]
    sx = near[keep, 0] - proj[rows[keep], bi[keep], 0]
    cross = seg[bi[keep], 0] * sy - seg[bi[keep], 1] * sx  # >0 = left
    val = np.where(cross > 0, 1.0, -1.0) * dmin[keep]
    return float(np.median(val))


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
