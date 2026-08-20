"""Local obstacle-aware path and speed planning.

The global route (road-graph A* or the in-game navigation route) is only a
rough corridor / direction.  On top of it this module plans a locally
drivable path:

* no obstacle near the path      -> follow the global route as-is
* obstacle offset from the path  -> elastic-band deformation (small nudge)
* obstacle blocking the corridor -> local occupancy-grid A* detour, i.e.
  the car actually re-plans a driveable way around the thing instead of
  blindly charging into it; when no detour exists the path is truncated
  just before the obstacle so the speed controller brings the car to a
  stop in front of it.

The cruise speed is then reduced for sharp curvature and for obstacles
that sit near the planned path.  Pure pursuit follows the local path.

Sub-modules
-----------
constants  All tunable parameters in one place.
geometry   Pure math utilities (curvature, projection, distance).
obstacles  Obstacle geometry, classification, path-collision helpers.
solid      Solid lane marking detection and no-cross enforcement.
core       ``LocalPlanner`` class and ``creep_speed``.
"""

from __future__ import annotations

# Re-export everything so ``from beamng_autopilot.planner import ...``
# still works after the split.
from .constants import *  # noqa: F401,F403
from .constants import _MapLaneBoundary  # noqa: F401
from .core import LocalPlanner, creep_speed  # noqa: F401
from .geometry import (  # noqa: F401
    _clamp_path_lateral,
    _lane_correction_gain,
    _point_lat_offset,
    _point_route_pos,
    _point_route_pos_np,
    _pts_to_segments,
    _points_to_polyline_lat,
    _seg_hits_box,
    _seg_seg_dist,
    adaptive_lookahead_idx,
    corner_angle_deg,
    corner_angle_max_deg,
    corner_angle_max_deg_arc,
    corner_speed,
)
from .obstacles import (  # noqa: F401
    _find_blocker,
    _obstacle_aabb,
    _obstacle_corners,
    _obstacle_footprint_area,
    _obstacle_half_extents,
    _obstacle_oriented,
    _obstacle_seg_dist,
    _path_clear_m,
    _path_hit_index,
    _seg_hits_obstacle,
    approach_speed_limit_mps,
    emergency_speed_limit_mps,
    emergency_stop_clearance_m,
    forward_clearance_m,
    is_sparse_raycast_speck,
    is_small_lidar_clutter,
)
from .solid import (  # noqa: F401
    _clamp_to_solid_lines,
    is_lane_edge_wall,
)