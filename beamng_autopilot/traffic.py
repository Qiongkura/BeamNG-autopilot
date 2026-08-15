"""Road-speed and traffic-signal rules shared by Steam and BeamNG.tech.

The module stays pure Python on purpose: it only turns the map / signal
snapshots read from Lua into speed decisions, so the same code can run on
both runtimes and in offline validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


SIGNAL_ACTION_GO = 0
SIGNAL_ACTION_ALERT = 1
SIGNAL_ACTION_STOP = 2
SIGNAL_ACTION_BRIEF_STOP = 3
SIGNAL_ACTION_SLOW = 4
SIGNAL_ACTION_YIELD = 5

SIGNAL_MAX_LOOKAHEAD_M = 60.0
SIGNAL_MIN_FACING_DOT = 0.2
SIGNAL_STOP_MARGIN_M = 4.0
SIGNAL_DECEL_MPS2 = 3.0
SIGNAL_SLOW_CAP_MPS = 30.0 / 3.6
SIGNAL_ALERT_FLOOR_MPS = 3.0

# ACC / car-following defaults.  These are deliberately conservative so a
# slow lead vehicle is followed smoothly instead of being treated as a
# static wall until the planner finds a detour.
ACC_TIME_GAP_S = 1.6
ACC_MIN_GAP_M = 3.0
ACC_MAX_DIST_M = 60.0
ACC_LANE_HALF_WIDTH_M = 1.8
ACC_OVERTAKE_SPEED_RATIO = 0.85   # overtake when lead speed < cruise * this
ACC_OVERTAKE_MIN_SPEED_MPS = 3.0

# Used only when a map link has lane strings but no node radii: the offset
# math still works with a reasonable road width, and live Tech maps always
# override it with real link geometry.
FALLBACK_LANE_WIDTH_M = 3.5


@dataclass
class RoadRuleView:
    """Parsed view of the road map data under the ego vehicle.

    ``lanes`` is the BeamNG lane string: ``+`` means the lane runs from the
    link's ``inNode`` to its ``outNode``, ``-`` means the opposite direction.
    """

    speed_limit_mps: float | None = None
    one_way: bool | None = None
    lane_direction: str = ""
    drivability: float | None = None
    road_type: str | None = None
    right_hand_drive: bool | None = None
    turn_on_red: bool | None = None
    n1: str | None = None
    n2: str | None = None
    lanes: str | None = None
    in_pos: tuple[float, float, float] | None = None
    out_pos: tuple[float, float, float] | None = None
    in_radius: float | None = None
    out_radius: float | None = None
    right_vec: tuple[float, float, float] | None = None

    @classmethod
    def from_lua_dict(cls, data: Any) -> "RoadRuleView | None":
        if not isinstance(data, dict):
            return None
        speed = data.get("speedLimit")
        try:
            speed = float(speed) if speed is not None else None
        except (TypeError, ValueError):
            speed = None
        if speed is not None and (not math.isfinite(speed) or speed <= 0.0):
            speed = None
        lanes = data.get("lanes")
        lanes = lanes if isinstance(lanes, str) else None
        one_way = data.get("oneWay")
        if one_way is None and lanes:
            one_way = one_way_from_lanes(lanes)
        elif one_way is not None:
            one_way = bool(one_way)
        drivability = data.get("drivability")
        try:
            drivability = float(drivability) if drivability is not None \
                else None
        except (TypeError, ValueError):
            drivability = None
        if drivability is not None and not math.isfinite(drivability):
            drivability = None
        rhd = data.get("rightHandDrive")
        turn_on_red = data.get("turnOnRed")
        return cls(
            speed_limit_mps=speed,
            one_way=one_way,
            lane_direction=classify_lane_direction(lanes),
            drivability=drivability,
            road_type=data.get("type"),
            right_hand_drive=bool(rhd) if rhd is not None else None,
            turn_on_red=bool(turn_on_red) if turn_on_red is not None else None,
            n1=data.get("n1"),
            n2=data.get("n2"),
            lanes=lanes,
            in_pos=_as_vec3(data.get("inPos")),
            out_pos=_as_vec3(data.get("outPos")),
            in_radius=_as_float(data.get("inRadius")),
            out_radius=_as_float(data.get("outRadius")),
            right_vec=_as_vec3(data.get("rightVec")),
        )


@dataclass
class LegalLaneView:
    """Map-derived lane range the ego may legally drive in.

    BeamNG lane strings are ordered left-to-right as traversed from the
    link's ``inNode`` to its ``outNode``; ``+`` lanes run in that same
    direction.  ``rightHandDrive`` here means a right-hand-drive car, so
    the legal direction is the left side of the road (leftmost ``+``
    lanes); ``False`` (the common LHD case) means the right side.

    ``start`` / ``end`` are inclusive lane indices.  ``boundaries`` holds
    ``(offset_m, allowed_side)`` pairs for every edge of the legal run
    that borders an opposing lane: ``allowed_side`` is +1 when the legal
    lanes lie to the right of that boundary and -1 when they lie to the
    left, matching the planner's positive-right sign convention.
    """

    legal: bool
    lane_count: int
    lane_width_m: float
    start: int = -1
    end: int = -1
    preferred_index: int = -1
    preferred_offset_m: float = 0.0
    boundaries: tuple[tuple[float, float], ...] = ()
    right_hand_drive: bool | None = None
    lanes: str = ""


@dataclass
class SignalRule:
    """One traffic signal snapshot and the ego's placement relative to it."""

    name: str = ""
    action: int = SIGNAL_ACTION_GO
    state: str | None = None
    rel_dist: float | None = None
    dist: float | None = None
    dot: float | None = None
    pos: tuple[float, float, float] | None = None
    use_lane: bool | None = None

    @classmethod
    def from_lua_dict(cls, data: Any) -> "SignalRule | None":
        if not isinstance(data, dict):
            return None
        pos = data.get("pos")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            try:
                pos = tuple(float(v) for v in pos[:3])
            except (TypeError, ValueError):
                pos = None
        else:
            pos = None
        return cls(
            name=str(data.get("instance") or data.get("name") or ""),
            action=int(data.get("action") or 0),
            state=data.get("state"),
            rel_dist=_as_float(data.get("relDist")),
            dist=_as_float(data.get("dist")),
            dot=_as_float(data.get("dot")),
            pos=pos,
            use_lane=data.get("useLane"),
        )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _as_vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        out = tuple(float(v) for v in value[:3])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in out):
        return None
    return out


def classify_lane_direction(lanes: str | None) -> str:
    """Return ``+``, ``-`` or ``mixed`` for a BeamNG lane string."""
    if not lanes:
        return ""
    has_plus = "+" in lanes
    has_minus = "-" in lanes
    if has_plus and has_minus:
        return "mixed"
    if has_plus:
        return "+"
    if has_minus:
        return "-"
    return ""


def one_way_from_lanes(lanes: str | None) -> bool | None:
    """Game semantics: a link is one-way when either direction is missing."""
    if not lanes:
        return None
    return ("+" not in lanes) or ("-" not in lanes)


def _route_projection(px: float, py: float, route) -> tuple[float, float] | None:
    """Project a point onto a 2D polyline.

    Returns ``(arc_length_from_start, signed_lateral_offset)`` where
    lateral is positive to the left of the travel direction.  This is a
    small standalone helper so ``traffic.py`` does not depend on planner
    internals (planner already imports traffic helpers).
    """
    pts = np.asarray(route[:, :2], dtype=float)
    if len(pts) < 2:
        return None
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    best_arc: float | None = None
    best_lat = 0.0
    best_dist = math.inf
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        vx, vy = bx - ax, by - ay
        seg_len = d[i]
        if seg_len < 1e-9:
            continue
        t = ((px - ax) * vx + (py - ay) * vy) / (seg_len * seg_len)
        t = max(0.0, min(1.0, t))
        cx = ax + t * vx
        cy = ay + t * vy
        dist = math.hypot(px - cx, py - cy)
        if dist < best_dist:
            best_dist = dist
            best_arc = cum[i] + t * seg_len
            # left normal: (-vy/len, vx/len)
            best_lat = ((px - cx) * (-vy / seg_len)
                        + (py - cy) * (vx / seg_len))
    if best_arc is None:
        return None
    return float(best_arc), float(best_lat)


def find_lead_vehicle(
    obstacles,
    route,
    pos,
    heading: float | None = None,
    max_dist: float = ACC_MAX_DIST_M,
    lane_half_width: float = ACC_LANE_HALF_WIDTH_M,
):
    """Return the nearest moving/parked vehicle ahead in the ego lane.

    ``route`` may be a 2D polyline (Nx2 or Nx3); when it is None or too
    short, a straight corridor is built from ``pos`` and ``heading`` so the
    helper still works for sensor-lane-only driving.

    Returns ``(obstacle, longitudinal_distance, lateral_offset)``; when no
    lead vehicle is found the obstacle is ``None`` and distance is ``inf``.
    """
    if not obstacles:
        return None, math.inf, 0.0
    p = np.asarray(pos, dtype=float)[:2]
    if route is not None and len(route) >= 2:
        pts = np.asarray(route[:, :2], dtype=float)
    else:
        h = float(heading if heading is not None else 0.0)
        fwd = np.array([math.cos(h), math.sin(h)])
        pts = np.array([p, p + fwd * max_dist])
    best = None
    best_lon = math.inf
    best_lat = 0.0
    for ob in obstacles:
        if ob.category != "vehicle":
            continue
        proj = _route_projection(float(ob.x), float(ob.y), pts)
        if proj is None:
            continue
        lon, lat = proj
        if lon <= 0.5 or lon > max_dist:
            continue
        if abs(lat) > lane_half_width:
            continue
        if lon < best_lon:
            best = ob
            best_lon = lon
            best_lat = lat
    return best, best_lon, best_lat


def follow_speed(
    cruise: float,
    lead_dist: float,
    lead_speed: float,
    ego_speed: float,
    time_gap: float = ACC_TIME_GAP_S,
    min_gap: float = ACC_MIN_GAP_M,
) -> float:
    """ACC-style target speed for a lead vehicle ahead.

    The target keeps a ``min_gap + time_gap * ego_speed`` following gap.
    It never exceeds the cruise speed and never goes negative.  This lets
    the car follow a slow vehicle smoothly instead of waiting for the
    planner to decide whether to stop or overtake.
    """
    cruise = max(0.0, float(cruise))
    if lead_dist is None or not math.isfinite(float(lead_dist)):
        return cruise
    lead_dist = float(lead_dist)
    if lead_dist <= min_gap:
        return 0.0
    desired_gap = min_gap + time_gap * max(0.0, float(ego_speed))
    if lead_dist >= desired_gap and float(lead_speed) >= cruise - 0.5:
        return cruise
    error = lead_dist - desired_gap
    target = float(lead_speed) + error / max(time_gap, 0.1)
    return float(max(0.0, min(cruise, target)))


def should_overtake(
    lead_speed: float,
    cruise: float,
    ratio: float = ACC_OVERTAKE_SPEED_RATIO,
    min_speed: float = ACC_OVERTAKE_MIN_SPEED_MPS,
) -> bool:
    """True when a lead vehicle is slow enough to justify an overtake."""
    return (float(lead_speed) < float(cruise) * ratio
            and float(cruise) >= min_speed)


def vehicle_along_speed(
    ob,
    route,
    pos=None,
    heading: float | None = None,
) -> float:
    """Signed speed of a vehicle projected along the current route.

    This is used for ACC so an oncoming vehicle in the same lane is not
    mistaken for a slow lead moving in the ego direction.
    """
    if ob is None or ob.velocity is None:
        return 0.0
    if route is not None and len(route) >= 2:
        pts = np.asarray(route[:, :2], dtype=float)
    else:
        h = float(heading if heading is not None else 0.0)
        fwd = np.array([math.cos(h), math.sin(h)])
        p = (np.asarray(pos, dtype=float)[:2]
             if pos is not None else np.zeros(2))
        pts = np.array([p, p + fwd * 100.0])
    if len(pts) < 2:
        return 0.0
    i = int(np.argmin(np.linalg.norm(pts - np.asarray(
        [ob.x, ob.y], dtype=float), axis=1)))
    i0 = max(0, i - 1)
    i1 = min(len(pts) - 1, i + 1)
    dx = pts[i1, 0] - pts[i0, 0]
    dy = pts[i1, 1] - pts[i0, 1]
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return 0.0
    return max(0.0, float((ob.velocity[0] * dx + ob.velocity[1] * dy) / n))


def road_width_m(rule: RoadRuleView | None) -> float | None:
    """Link width used by the game's lane-offset math, or None."""
    if rule is None:
        return None
    radii = [v for v in (rule.in_radius, rule.out_radius)
             if v is not None and math.isfinite(v) and v > 0.0]
    if not radii:
        return None
    return min(radii) * 2.0


def lane_offset_m(lane_index: int, lane_count: int, width_m: float) -> float:
    """Signed offset of a lane centre from the link centre, positive right.

    Mirrors ``map.lua getLaneOffset``: lane 1 (leftmost) is negative of
    centre, the last lane is positive of centre.
    """
    return (lane_index - lane_count / 2.0 + 0.5) * width_m / lane_count


def _boundary_offset_m(boundary_index: int, lane_count: int,
                       width_m: float) -> float:
    """Offset of the boundary after ``boundary_index`` (0-based lanes)."""
    return (boundary_index + 1.0 - lane_count / 2.0) * width_m / lane_count


def _plus_runs(lanes: str) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = -1
    for i, ch in enumerate(lanes):
        if ch == "+":
            if start < 0:
                start = i
        elif start >= 0:
            runs.append((start, i - 1))
            start = -1
    if start >= 0:
        runs.append((start, len(lanes) - 1))
    return runs


def legal_lane_view(rule: RoadRuleView | None) -> LegalLaneView | None:
    """Return the legal forward lane range for a link, or None.

    ``None`` means the map snapshot has no usable lane/RHD data and the
    caller should keep its existing navigation behaviour.  A returned view
    with ``legal=False`` means the link has no forward lane at all (wrong
    way on a one-way link) and the vehicle must not drive it.
    """
    if rule is None:
        return None
    lanes = rule.lanes
    if not lanes or any(ch not in "+-" for ch in lanes):
        return None
    lane_count = len(lanes)
    width = road_width_m(rule)
    if width is None:
        width = lane_count * FALLBACK_LANE_WIDTH_M
    runs = _plus_runs(lanes)
    if not runs:
        return LegalLaneView(
            legal=False,
            lane_count=lane_count,
            lane_width_m=width,
            right_hand_drive=rule.right_hand_drive,
            lanes=lanes,
        )
    if rule.right_hand_drive is None:
        # Direction-only links are still usable, but without the road-side
        # flag we cannot choose a legal side without inventing one.
        return None
    start, end = runs[0] if rule.right_hand_drive else runs[-1]
    preferred = start if rule.right_hand_drive else end
    boundaries: list[tuple[float, float]] = []
    if start > 0:
        boundaries.append((_boundary_offset_m(start - 1, lane_count, width),
                           1.0))
    if end < lane_count - 1:
        boundaries.append((_boundary_offset_m(end, lane_count, width),
                           -1.0))
    return LegalLaneView(
        legal=True,
        lane_count=lane_count,
        lane_width_m=width,
        start=start,
        end=end,
        preferred_index=preferred,
        preferred_offset_m=lane_offset_m(preferred, lane_count, width),
        boundaries=tuple(boundaries),
        right_hand_drive=rule.right_hand_drive,
        lanes=lanes,
    )


def signal_distance(signal: SignalRule | None) -> float | None:
    """Distance to the stop line; None when the placement is unknown."""
    if signal is None or signal.rel_dist is None:
        return None
    return max(0.0, -float(signal.rel_dist))


def signal_requires_stop(signal: SignalRule | None) -> bool:
    return signal is not None and signal.action in (
        SIGNAL_ACTION_STOP,
        SIGNAL_ACTION_BRIEF_STOP,
    )


def signal_action_label(action: int | None) -> str:
    if action is None:
        return "unknown"
    return {
        SIGNAL_ACTION_GO: "go",
        SIGNAL_ACTION_ALERT: "yellow",
        SIGNAL_ACTION_STOP: "stop",
        SIGNAL_ACTION_BRIEF_STOP: "stop_sign",
        SIGNAL_ACTION_SLOW: "slow",
        SIGNAL_ACTION_YIELD: "yield",
    }.get(int(action), "unknown")


def select_signal_rule(
    signals: list[SignalRule],
    pos: Any = None,
    heading: float | None = None,
    dir_vec: Any = None,
    max_dist: float = SIGNAL_MAX_LOOKAHEAD_M,
) -> SignalRule | None:
    """Pick the nearest applicable signal still ahead of the ego."""
    if not signals:
        return None
    fwd = None
    if dir_vec is not None:
        try:
            fwd = (float(dir_vec[0]), float(dir_vec[1]))
            n = math.hypot(fwd[0], fwd[1])
            if n > 1e-9:
                fwd = (fwd[0] / n, fwd[1] / n)
            else:
                fwd = None
        except (TypeError, ValueError, IndexError):
            fwd = None
    if fwd is None and heading is not None:
        fwd = (math.cos(float(heading)), math.sin(float(heading)))
    px = py = None
    if pos is not None:
        try:
            px, py = float(pos[0]), float(pos[1])
        except (TypeError, ValueError, IndexError):
            px = py = None

    best: SignalRule | None = None
    best_dist = math.inf
    for signal in signals:
        dist = signal_distance(signal)
        if dist is None or dist > max_dist:
            continue
        if signal.dot is not None and float(signal.dot) < SIGNAL_MIN_FACING_DOT:
            continue
        if fwd is not None and px is not None and signal.pos is not None:
            tx = float(signal.pos[0]) - px
            ty = float(signal.pos[1]) - py
            if tx * fwd[0] + ty * fwd[1] < 0.0:
                continue
        if dist < best_dist:
            best = signal
            best_dist = dist
    return best


def apply_rule_speed(
    cruise: float,
    road_rule: RoadRuleView | None = None,
    signal_rule: SignalRule | None = None,
    stop_margin: float = SIGNAL_STOP_MARGIN_M,
    decel_mps2: float = SIGNAL_DECEL_MPS2,
    slow_cap_mps: float = SIGNAL_SLOW_CAP_MPS,
    alert_floor_mps: float = SIGNAL_ALERT_FLOOR_MPS,
) -> tuple[float, str | None, float | None]:
    """Return (target_speed, reason, rule_limit) after legal speed caps.

    Road speed limits cap the requested speed directly.  Stop signals use
    the same kinematic braking curve as obstacle planning so the car eases
    to a stop before the line; yellow/slow signals decelerate without
    pinning the speed at zero.  A signal already passed is ignored.
    """
    target = max(0.0, float(cruise))
    reason: str | None = None
    limit: float | None = None

    if road_rule is not None and road_rule.speed_limit_mps is not None:
        speed_limit = float(road_rule.speed_limit_mps)
        if speed_limit < target:
            target = speed_limit
            reason = "speed_limit"
            limit = speed_limit

    if signal_rule is not None and (
            signal_rule.rel_dist is None
            or signal_rule.rel_dist < 0.0):
        dist = signal_distance(signal_rule)
        if dist is not None:
            stop_dist = max(0.0, dist - stop_margin)
            v_brake = math.sqrt(2.0 * decel_mps2 * stop_dist)
            if signal_rule.action in (
                SIGNAL_ACTION_STOP,
                SIGNAL_ACTION_BRIEF_STOP,
            ):
                if v_brake < target:
                    target = v_brake
                    reason = "signal"
                    limit = v_brake
            elif signal_rule.action in (
                SIGNAL_ACTION_ALERT,
                SIGNAL_ACTION_SLOW,
                SIGNAL_ACTION_YIELD,
            ):
                # Yellow / slow zones should brake, but never pin the car
                # to a full stop just because the signal is far ahead.
                v_slow = min(slow_cap_mps,
                             max(v_brake, alert_floor_mps))
                if v_slow < target:
                    target = v_slow
                    reason = "signal"
                    limit = v_slow
    return target, reason, limit
