"""Planner constants and shared data structures.

Extracted from ``beamng_autopilot.planner`` to allow lightweight imports
of planner tuning values without pulling in the full planning module (and
its heavy transitive dependencies such as the traffic and vision packages).
"""

from __future__ import annotations

import numpy as np

from beamng_autopilot.lane import (
    LANE_MIN_CONF,
    LANE_WIDTH_MAX_M,
    LANE_WIDTH_DEFAULT_M,
)

# ── Ego vehicle geometry ─────────────────────────────────────────────
CAR_HALF_WIDTH = 1.0      # lateral half width of the ego car

# ── Obstacle / clearance ─────────────────────────────────────────────
SAFETY_MARGIN = 1.7       # extra clearance kept around obstacles (m)
# Path-planning gap for lidar clusters: car half width + 0.3 m.  Dense
# town scenes otherwise inflate every roadside bush by SAFETY_MARGIN and
# fill the A* grid; the speed planner still keeps the full clearance.
LIDAR_PATH_CLEAR_M = 1.3
MAX_LATERAL_DEV = 8.0     # how far the path may leave the nav corridor
PLAN_HORIZON_M = 48.0     # how far ahead we re-plan the path
CORRIDOR_HALF_W = 1.6     # pass-by slower than this gap from an obstacle
                          # footprint edge is treated as a tight squeeze
STOP_MARGIN_M = 2.5       # keep this much room when stopping for an obstacle
DECEL_MPS2 = 4.0          # assumed decel when computing obstacle speed limit
PASS_BY_MIN_MPS = 2.0     # slowest speed while passing a box that sits
                          # beside the path; a pass-by never stops the car
SPECK_PASS_BY_MIN_MPS = 6.0  # sparse raycast artefacts keep a gentle slow-
                             # down but must not turn a roadside grove into
                             # a 2 m/s crawl every frame

# ── Solid-line heuristics ────────────────────────────────────────────
SOLID_LINE_MARGIN = CAR_HALF_WIDTH + 0.3
SOLID_LINE_MAX_M = 8.0
SOLID_MIN_CONF = 0.55
SOLID_MIN_LEN_M = 8.0
SOLID_MAX_CORRIDOR_DEV_M = 5.0
SOLID_MIN_ALIGNMENT = 0.7
SOLID_ANCHOR_NEAR_M = 0.5   # car closer than this to a line is ambiguous:
                             # pick the side from the nearby path instead
                             # of one noisy anchor coordinate
SOLID_BLOCK_MIN_M = 3.0   # solid-line crossings closer than this are treated
                           # as noise under the car, not a legal stop point
SOLID_BLOCK_MAX_M = 30.0  # crossings farther than this are treated as a
                           # distant rule, not a reason to stand still now
SOLID_BLOCK_LANE_CONF = 0.55  # only a confident lane frame may turn a
                               # detected line into a full stop; shaky
                               # vision only nudges the path away

# ── Route shaping / sharp-corner handling ─────────────────────────────
RIGHT_OFFSET_M = 1.5
RIGHT_RAMP_M = 12.0
SHARP_ANGLE_DEG = 45.0
SHARP_CORNER_KPH = 30.0

# ── Occupancy grid A* ────────────────────────────────────────────────
GRID_RES = 0.8            # occupancy-grid cell size (m)
GRID_AHEAD = 55.0         # grid extent behind the car (m)
GRID_BEHIND = 10.0        # grid extent behind the car (m)
GRID_HALF_W = 20.0        # grid half width (m)
DEV_PENALTY = 0.15        # A* cost added per metre of lateral deviation
GRID_ANTICIPATE = 12.0    # extend blocking boxes toward the car this far (m)
                          # so the A* detour starts steering early and
                          # gently instead of swerving at the last moment
GRID_RIGHT_BIAS = 0.02    # A* cost per metre on the left side of the car:
                          # when both sides are drivable, prefer the right

# ── Roadside-wall / speck detection ──────────────────────────────────
ROADSIDE_WALL_MIN_LEN_M = 3.0
ROADSIDE_WALL_MAX_THICK_M = 3.5
ROADSIDE_WALL_MIN_EDGE_M = 0.5
SOLID_MAX_LAT_SPAN_FACTOR = 0.5
SOLID_MAX_LAT_SPAN_M = 3.5
SOLID_MAX_PERP_SPAN_M = 0.6
SOLID_MAX_PERP_SPAN_FRAC = 0.02
SPECK_RAYCAST_MAX_M = 1.2

# ── Lane-edge wall heuristics ────────────────────────────────────────
LANE_EDGE_WALL_MIN_ALIGN = 0.70
LANE_EDGE_WALL_MAX_THICK_M = 1.5
LANE_EDGE_WALL_MAX_INTRUDE_M = 0.35
LANE_EDGE_WALL_EDGE_TOL_M = 0.6

# ── Lane-correction limits ───────────────────────────────────────────
# Legacy clamp kept for import compatibility; a paired sensor lane is now
# the drive path itself and is not clamped against the nav route.
LANE_CORRECTION_MAX_M = 2.0
# Lane corrections are ramped in with frame confidence: a frame just above
# the usable threshold is too shaky to steer the car, only confident frames
# get the full capped correction.
LANE_FULL_CONF = 0.6
# A one-sided mirror fallback assumes the lane width instead of seeing it,
# so it may only nudge a nav-route path away from a boundary that is too
# close.  A two-sided lane frame is the lane centre and replaces the route.
LANE_MIRROR_CORRECTION_MAX_M = 0.4
# The LiDAR single-edge frame mirrors an assumed lane width from a wall /
# curb, which is the same one-sided assumption but with a physical edge.
# A single edge cannot prove where the other side of the lane is: a far
# guardrail says nothing about the lane centre, so the fallback is only a
# small nudge away from the wall.  The real centre comes from pairing
# that LiDAR edge with a vision lane boundary.
LANE_LIDAR_CORRECTION_MAX_M = 0.35
# A single painted line (vision mirror) is a real lane edge, but it proves
# nothing about where the opposite side is.  The nav route stays primary;
# the edge only pushes the path away when the route is already too close
# to the paint.  A single LiDAR wall / guardrail is treated the same way.
LANE_BOUNDARY_CORRECTION_MAX_M = LANE_MIRROR_CORRECTION_MAX_M
LANE_LIDAR_EDGE_CORRECTION_MAX_M = LANE_LIDAR_CORRECTION_MAX_M
# The in-game nav route follows the road/link centre, not the legal lane
# centre.  When no paired sensor lane is present, the map's legal-lane
# offset is the lateral reference; only clamp it to the widest lane the
# sensor chain may report.
MAP_LANE_OFFSET_MAX_M = LANE_WIDTH_MAX_M
# Keep this much of the car footprint clear of a single detected edge.
LANE_BOUNDARY_CLEAR_M = CAR_HALF_WIDTH + 0.4
# Boundaries farther than this are usually a different road / far lane and
# must not drag the car toward them.  ``LANE_WIDTH_MAX_M`` is the widest
# single lane the vision/LiDAR chain may report, so anything beyond it is
# not the near edge of the current lane.
LANE_BOUNDARY_MAX_M = LANE_WIDTH_MAX_M
# A paired sensor lane may replace the nav route as the driving centre
# only while it agrees with the route.  The nav route is generated by the
# game from the road graph; a detected "lane" whose centre deviates more
# than this from the route close to the car is a wrong pairing (a far
# lane's marking, a roadside line, a guardrail shadow) and must not drag
# the car sideways.  Degrades to nav-primary + single-edge protection.
LANE_NAV_MAX_DEV_M = 1.8
# A lone detected edge (painted line / wall / guardrail) may only nudge
# the nav window while it sits on its declared side of the route near
# the car.  A wrong-side or far edge is a tracker lock onto the opposite
# line or another road's boundary - on run 53 the "right" paint flipped
# to the left of the car and the boundary push dragged the car off the
# route.  An edge closer to the route than this on its declared side is
# a phantom; farther than this it is not the current lane's boundary.
LANE_EDGE_NAV_MIN_SIGN_M = 0.6
LANE_EDGE_NAV_MAX_DEV_M = 4.0


# ── Map-lane boundary helper ─────────────────────────────────────────

class _MapLaneBoundary:
    """Authoritative no-cross lane boundary derived from the map link.

    ``world`` is a short straight polyline along the current link's lane
    boundary and ``allowed_side`` is +1 when the legal lanes lie to the
    right of the boundary, -1 when they lie to the left.  The class looks
    enough like ``LaneMarking`` for ``_clamp_to_solid_lines`` to consume
    it without running CV-only heuristics.
    """

    def __init__(self, world, allowed_side: float):
        self.world = np.asarray(world, dtype=float)
        self.allowed_side = 1.0 if allowed_side >= 0.0 else -1.0
        self.kind = "solid"
        self.confidence = 1.0
        self.is_map_boundary = True
