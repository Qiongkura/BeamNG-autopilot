"""Perception-only lateral road guard (no map, no nav route).

Real FSD never queries the road graph to know "how far am I from the
road centre" - it reads the semantic road/lane masks lifted into the
BEV vector space and the LiDAR-observed free space.  This guard does
exactly that: from the ego-centred occupancy grid's *drivable* cells
(the semantic road mask back-projected to the ground, plus observed
free space) it measures the perceived road centre and both edges ahead
of the car, and returns a NORMALIZED steering correction that keeps the
ego inside the perceived road.

There is deliberately no ``RoadNetwork``, no nav route and no map prior
in this module: the same function works on any map, simulator or real
vehicle, from the sensors alone.  The caller still needs *navigation*
(to know where to go - like a real stack), but staying on the road is
purely perception.
"""

from __future__ import annotations

import numpy as np

from beamng_autopilot.occupancy import OccupancyGrid


def perception_road_bands(grid, near_m: float = 2.0,
                           look_ahead_m: float = 9.0,
                           min_cells: int = 24):
    """Perceived road geometry ahead of the car (perception-only).

    Returns a dict with the road's lateral extent in a wide 2-12 m band
    plus the centre of the near (2-6 m) and far (7-12 m) sub-bands, or
    None when the road is not confidently perceived.  Used by both the
    lateral guard and the perception curve-speed governor so the two
    never disagree about where the road is.
    """
    drv = np.asarray(grid.drivable, dtype=float)
    n = grid.n_rows
    res = float(grid.res)
    extent = float(grid.extent)
    if n < 8 or res <= 0.0:
        return None

    def _rows(x0: float, x1: float) -> tuple[int, int]:
        a = int(round((extent - x1) / res))
        b = int(round((extent - x0) / res))
        return max(0, min(a, b)), min(n - 1, max(a, b))

    def _extent_of(r0: int, r1: int, min_cells_here: int):
        band = drv[r0:r1 + 1, :]
        if float(band.sum()) < min_cells_here:
            return None
        cols = np.nonzero(band.any(axis=0))[0]
        if len(cols) < 4:
            return None
        ys = extent - (cols + 0.5) * res
        return float(ys.min()), float(ys.max())

    r0w, r1w = _rows(near_m, look_ahead_m)
    wide = _extent_of(r0w, r1w, min_cells)
    if wide is None:
        return None
    r0n, r1n = _rows(near_m, near_m + 4.0)
    r0f, r1f = _rows(look_ahead_m - 4.5, look_ahead_m)
    near = _extent_of(r0n, r1n, max(6, min_cells // 2))
    far = _extent_of(r0f, r1f, max(6, min_cells // 2))
    out = {
        "y_left": wide[1],      # leftmost perceived road edge (+y)
        "y_right": wide[0],     # rightmost perceived road edge (-y side)
        "center": 0.5 * (wide[0] + wide[1]),
        "near_center": 0.5 * (near[0] + near[1]) if near is not None else None,
        "far_center": 0.5 * (far[0] + far[1]) if far is not None else None,
    }
    return out


def perception_curve_speed(grid, cruise: float,
                           corner_shift_m: float = 1.0,
                           min_speed: float = 1.5,
                           gain: float = 1.2) -> float:
    """Perception-only speed cap for a perceived bend ahead.

    Real FSD slows on what its sensors SEE (the road curving in the
    BEV), not on a map's curvature table.  When the perceived road
    centre shifts sideways by more than ``corner_shift_m`` between the
    near (2-6 m) and far (7-12 m) bands, the road turns ahead: cap the
    speed by how hard it shifts.  No road perceived / no corner -> the
    cruise speed is returned unchanged.
    """
    bands = perception_road_bands(grid)
    if bands is None or bands["near_center"] is None \
            or bands["far_center"] is None:
        return float(cruise)
    shift = abs(float(bands["far_center"]) - float(bands["near_center"]))
    if shift <= corner_shift_m:
        return float(cruise)
    cap = float(np.clip(cruise - (shift - corner_shift_m) * gain,
                        min_speed, cruise))
    return cap


def perception_lateral_guard(
        grid: OccupancyGrid,
        gate_m: float = 1.5,
        edge_margin_m: float = 1.2,
        look_ahead_m: float = 9.0,
        near_m: float = 2.0,
        min_cells: int = 24,
        gain: float = 0.7,
        max_corr: float = 0.40,
        corner_shift_m: float = 1.0,
) -> float:
    """Normalised steering correction from the PERCEIVED road only.

    Looks at the drivable (road) cells of the ego grid between
    ``near_m`` and ``look_ahead_m`` ahead of the car:

    * centre guard: when the perceived road centre is more than
      ``gate_m`` to one side of the ego, steer back toward it;
    * edge guard: when either perceived road edge is closer than
      ``edge_margin_m`` to the ego, steer away from it.

    Returns 0.0 when the road is not confidently perceived (too few
    drivable cells), so an unknown scene never fights the planner.
    Positive output = steer right, negative = steer left (BeamNG
    convention used by the rest of the stack).
    """
    bands = perception_road_bands(grid, near_m=near_m,
                                  look_ahead_m=look_ahead_m,
                                  min_cells=min_cells)
    if bands is None:
        return 0.0
    y_left = bands["y_left"]
    y_right = bands["y_right"]
    center = bands["center"]
    # Perceived corner test: when both sub-bands see road and the centre
    # shifts sideways by more than ``corner_shift_m`` the road curves
    # ahead - the perceived centre legitimately sits to one side and the
    # guard must NOT fight the turn (same reason a real stack disables
    # lane-centring on high-curvature roads).  If either sub-band is
    # empty we cannot tell, so the wide-band read is used as-is.
    if bands["near_center"] is not None and bands["far_center"] is not None:
        if abs(bands["far_center"] - bands["near_center"]) > corner_shift_m:
            return 0.0

    corr = 0.0
    # Centre guard: perceived centre left (+y) -> steer left (negative).
    if abs(center) > gate_m:
        k = min((abs(center) - gate_m) * gain, max_corr)
        corr = -float(np.sign(center)) * k
    # Edge guard: too close to the left edge -> steer right (+); too
    # close to the right edge -> steer left (-).  Only strengthen the
    # existing correction when the edge demand is larger.
    edge_corr = 0.0
    if y_left < edge_margin_m:
        edge_corr = max(edge_corr, max_corr * 0.6)
    if y_right > -edge_margin_m:
        edge_corr = min(edge_corr, -max_corr * 0.6)
    if abs(edge_corr) > abs(corr):
        corr = edge_corr
    return float(np.clip(corr, -max_corr, max_corr))
