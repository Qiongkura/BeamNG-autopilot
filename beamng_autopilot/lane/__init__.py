"""Sensor lane framing: markings -> lane centre, LiDAR -> free corridor.

Sub-modules
-----------
constants  All tunable parameters.
pairing    Vision lane marking pairing: markings -> lane centre.
lidar      LiDAR raycast corridor: free-space lane estimation.
fusion     Vision + LiDAR sensor fusion for lane estimation.
tracking   Lane frame tracking over time.
reference  Which lane geometry may steer the car (single policy owner).

Two layers answer "where is my lane" - do not confuse them
---------------------------------------------------------
* ``lane.reference.select_lane_reference`` is the PRODUCER: given this
  tick's sensor lane, nav route and BEV grid it decides which geometry
  may be published at all, and emits both the machine contract
  (``lane_src_sel``: sensor / map / perception-unavailable) and the
  telemetry label (``lane_src``) from that one decision.  Strict
  perception never builds map lane geometry here, so a strict tick with
  no perception lane yields no centre and the caller can only fail
  closed.
* ``planning.lateral_ref.lateral_reference`` is the CONSUMER side: given
  the Scene the producer built, it answers which Scene field may steer
  (``sensor`` -> ``envelope`` -> legacy ``route`` -> ``none``), and the
  planner, the safety monitor and the lane-alignment cost all read that
  one answer.

The producer owns *what enters* the world model; the consumer owns *what
may steer from* it.  Neither may re-derive the other's decision - the
duplicated ``lane_src`` label in ``fsd_stack`` is what the split removed.
"""

from __future__ import annotations

# Re-export everything so ``from beamng_autopilot.lane import ...``
# still works after the split.
from .constants import *  # noqa: F401,F403
from .pairing import (  # noqa: F401
    LaneFrame,
    pair_lane_markings,
)
from .envelope import SensorLaneEnvelope
from .lidar import build_lidar_corridor  # noqa: F401
from .fusion import choose_sensor_lane  # noqa: F401
from .perception_guard import (  # noqa: F401
    perception_curve_speed,
    perception_lateral_guard,
)
from .tracking import (  # noqa: F401
    LaneTracker,
    lane_frame_usable,
    _boundary_near_lat,
    _frame_near_lat,
    _fusion_center_unstable,
    _mirror_near_ok,
    _mirror_right_ok,
)
from .reference import (  # noqa: F401
    LANE_HEADING_MAX_YAW_DEG,
    LANE_ROUTE_TURN_LOOK_M,
    LANE_ROUTE_TURN_MAX_DEG,
    LaneReference,
    bev_drivable_center,
    select_lane_reference,
)
