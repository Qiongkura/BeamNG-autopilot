"""Scene: the fused planner environment (FSD vector-space snapshot).

A planner stage should never reach for raw sensor arrays; it consumes a
``Scene`` - the occupancy grid, the route/lane reference, the ego state
and any road-rule view - so swapping a camera for a LiDAR source only
changes how the Scene is built, never how it is planned against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scene:
    """One planning-tick snapshot.

    ``pos`` / ``heading`` define the ego pose (world).  ``grid`` is an
    ``OccupancyGrid`` (or ``None`` when vector space is unavailable).
    ``route`` is the dense nav-route polyline; ``lane_ref`` an optional
    sensor lane center (world).  ``road_rule`` is the map rule view.
    """

    pos: np.ndarray
    heading: float
    grid: object = None                    # OccupancyGrid | None
    route: np.ndarray | None = None
    lane_ref: np.ndarray | None = None
    lane_left: np.ndarray | None = None
    lane_right: np.ndarray | None = None
    lane_width: float = 0.0
    road_rule: object = None
    target_speed: float = 12.0
    # obstacle boxes for legacy checks (the grid is the canonical source,
    # but existing code reads boxes; carry both).
    obstacles: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def forward(self, dist_m: float) -> np.ndarray:
        """A point `dist_m` straight ahead of the ego (world)."""
        h = float(self.heading)
        return np.array([self.pos[0] + dist_m * np.cos(h),
                         self.pos[1] + dist_m * np.sin(h)])

    def cell_occupied(self, wx: float, wy: float) -> bool:
        """True when a world point is inside an occupied grid cell."""
        if self.grid is None:
            return False
        cell = self.grid.world_to_cell(wx, wy)
        return cell is not None and bool(self.grid.obstacle[cell] > 0)