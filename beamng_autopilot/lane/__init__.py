"""Sensor lane framing: markings -> lane centre, LiDAR -> free corridor.

Sub-modules
-----------
constants  All tunable parameters.
pairing    Vision lane marking pairing: markings -> lane centre.
lidar      LiDAR raycast corridor: free-space lane estimation.
fusion     Vision + LiDAR sensor fusion for lane estimation.
tracking   Lane frame tracking over time.
"""

from __future__ import annotations

# Re-export everything so ``from beamng_autopilot.lane import ...``
# still works after the split.
from .constants import *  # noqa: F401,F403
from .pairing import (  # noqa: F401
    LaneFrame,
    pair_lane_markings,
)
from .lidar import build_lidar_corridor  # noqa: F401
from .fusion import choose_sensor_lane  # noqa: F401
from .tracking import (  # noqa: F401
    LaneTracker,
    lane_frame_usable,
    _boundary_near_lat,
    _frame_near_lat,
    _fusion_center_unstable,
    _mirror_near_ok,
    _mirror_right_ok,
)