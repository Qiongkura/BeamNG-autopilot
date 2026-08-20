"""Routing intent: semantic travel direction from the nav route (FSD Routing).

FSD does not only follow a path - it knows *why* it is going there: the
route's upcoming geometry tells the stack "next intersection: turn
left", and the driver-facing layer announces it, while the planner uses
it for expected lateral movement.  This module gives that structure:

* ``RoutingIntent``: the semantic label (straight / left / right /
  u-turn / unknown) plus the net turn angle and the curvature of the
  upcoming segment, with a suggested slow-speed for a sharp turn.
* ``infer_route_intent``: from a nav-route polyline + the ego position,
  look ahead and classify.  Pure + game-free + testable.

The intent is a planning-layer *input*: it does not steer itself, it
informs the trajectory fan (bias the arcs) and the HUD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Intent labels.
INTENT_STRAIGHT = "straight"
INTENT_LEFT = "left"
INTENT_RIGHT = "right"
INTENT_U_TURN = "u_turn"
INTENT_UNKNOWN = "unknown"

INTENTS = (INTENT_STRAIGHT, INTENT_LEFT, INTENT_RIGHT,
           INTENT_U_TURN, INTENT_UNKNOWN)

# Segment classification thresholds (net turn over a lookahead window).
TURN_ANGLE_DEG = 25.0
U_TURN_ANGLE_DEG = 140.0

# Suggested cruise speeds (m/s) for a turn ahead.
STRAIGHT_SPEED = 12.0
TURN_SPEED = 8.0
U_TURN_SPEED = 5.0


@dataclass
class RoutingIntent:
    label: str = INTENT_UNKNOWN
    turn_deg: float = 0.0          # signed net turn of the upcoming window
    ahead_m: float = 0.0           # how far ahead the intent was judged
    suggested_speed: float = 12.0

    @property
    def is_turn(self) -> bool:
        return self.label in (INTENT_LEFT, INTENT_RIGHT, INTENT_U_TURN)


def infer_route_intent(route, pos, heading: float,
                       lookahead_m: float = 35.0,
                       min_points: int = 8
                       ) -> RoutingIntent:
    """Classify the travel intent from the nav route ahead.

    ``route`` is the world polyline (N, 2).  The ego is at ``pos`` with
    ``heading`` (radians, atan2 convention).  We find the route point
    nearest the ego, advance `lookahead_m` along it, and measure the
    signed turn of that arc relative to the ego heading:
      * |turn| < TURN_ANGLE_DEG      -> straight
      * TURN_ANGLE..U_TURN           -> left / right by sign
      * |turn| >= U_TURN_ANGLE_DEG   -> u-turn
    """
    route = np.asarray(route[:, :2], dtype=float) if route is not None \
        else np.zeros((0, 2))
    if len(route) < min_points:
        return RoutingIntent(label=INTENT_UNKNOWN, ahead_m=0.0,
                             suggested_speed=STRAIGHT_SPEED)
    pos = np.asarray(pos[:2], dtype=float)

    # nearest route point (search around the ego along the route)
    d = np.linalg.norm(route - pos, axis=1)
    i0 = int(np.argmin(d))
    # advance until lookahead arc length
    arc = 0.0
    i1 = i0
    while i1 + 1 < len(route) and arc < lookahead_m:
        arc += float(np.linalg.norm(route[i1 + 1] - route[i1]))
        i1 += 1
    if i1 - i0 < max(2, min_points // 2):
        return RoutingIntent(label=INTENT_STRAIGHT, ahead_m=arc,
                             suggested_speed=STRAIGHT_SPEED)

    # ego heading vector
    hv = np.array([math.cos(heading), math.sin(heading)])
    # route delta over the window
    dv = route[i1] - route[i0]
    if float(np.linalg.norm(dv)) < 1e-6:
        return RoutingIntent(label=INTENT_STRAIGHT, ahead_m=arc,
                             suggested_speed=STRAIGHT_SPEED)
    # signed angle from ego heading to the route direction (atan2 cross/dot)
    # positive = left turn in a right-handed xy-plane
    turn = math.degrees(math.atan2(hv[0] * dv[1] - hv[1] * dv[0],
                                   hv[0] * dv[0] + hv[1] * dv[1]))

    if abs(turn) >= U_TURN_ANGLE_DEG:
        label, speed = INTENT_U_TURN, U_TURN_SPEED
    elif turn >= TURN_ANGLE_DEG:
        label, speed = INTENT_LEFT, TURN_SPEED
    elif turn <= -TURN_ANGLE_DEG:
        label, speed = INTENT_RIGHT, TURN_SPEED
    else:
        label, speed = INTENT_STRAIGHT, STRAIGHT_SPEED

    return RoutingIntent(label=label, turn_deg=float(turn),
                         ahead_m=arc, suggested_speed=float(speed))