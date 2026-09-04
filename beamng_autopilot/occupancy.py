"""BEV occupancy grid - the FSD-style vector-space core representation.

Tesla FSD plans in a bird's-eye "vector space": sensor detections, lane
geometry and road rules are fused into one ego-centred grid, and both
the (rule) planner and any future learned planner read the same tensor.

This module is the *structure* that representation needs:

* ``OccupancyGrid``: a cell `(row, col)` grid in the vehicle frame
  (longitudinal +x forward, lateral +y left, matching the axis
  convention of ``CameraModel``).  Each cell keeps fused evidence:
  ``occupancy`` (0..1 obstacle evidence), ``drivable`` (free space),
  ``obstacle`` (permanent barrier), ``height`` (mean z of hits) and a
  ``sources`` counter.  It is game-free and unit-testable.
* ``project_to_grid``: project a camera's per-pixel semantic mask onto
  the ground in front of the car (inverse pinhole), marking drivable
  space from the semantic road pixels.
* ``fuse_obstacles``: flood the grid with world obstacles / ray hits
  (LiDAR, scenario objects) - the same input the current planner already
  uses, so the grid is buildable from data we already have.
* ``query_path_cost``: a planner can sum occupancy along a candidate
  path, the primitive a trajectory scorer needs.

The grid is deliberately a plain dense ``np.ndarray``-backed structure;
it does not talk to the game (all geometry happens on the caller side).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OccupancyGrid:
    """Ego-centred BEV occupancy grid.

    ``res`` is the cell size in metres (e.g. 0.5).  Row 0 is the most
    forward (+x) row, row N-1 the most rearward; column 0 the most
    leftward (+y).  ``origin`` logs where the grid was centred (for
    debugging / world re-projection).
    """

    n_rows: int
    n_cols: int
    res: float
    origin: tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    # Persistent evidence accumulators.
    occupancy: np.ndarray = field(init=False)   # obstacle evidence 0..1
    drivable: np.ndarray = field(init=False)    # 0/1.free space flag
    obstacle: np.ndarray = field(init=False)    # 0/1 persistent barrier
    height: np.ndarray = field(init=False)      # mean z of hits (m)
    sources: np.ndarray = field(init=False)     # integration count
    observed: np.ndarray = field(init=False)    # 0/1 seen by a sensor

    def __post_init__(self) -> None:
        shape = (self.n_rows, self.n_cols)
        self.occupancy = np.zeros(shape, dtype=np.float32)
        self.drivable = np.zeros(shape, dtype=np.float32)
        self.obstacle = np.zeros(shape, dtype=np.uint8)
        self.height = np.full(shape, np.nan, dtype=np.float32)
        self.sources = np.zeros(shape, dtype=np.uint16)
        self.observed = np.zeros(shape, dtype=np.float32)
        self._recompute_extent()

    def _recompute_extent(self) -> None:
        # Symmetric BEV extent: the grid spans [+extent, -extent] in both
        # the forward (x) and left (y) ego axes, centred on the vehicle.
        # Row 0 is the most forward row, column 0 the most leftward.
        self.extent = 0.5 * self.n_rows * self.res
        self.max_x = self.max_y = self.extent
        self.center = (int(self.n_rows / 2), int(self.n_cols / 2))

    # ------------------------------------------------------------------
    def world_to_cell(self, wx: float, wy: float) -> tuple[int, int] | None:
        """Cell (row, col) for a world point (or None when out of bounds).

        The grid is centred on ``origin`` and rotated by ``heading``:
        a world point is first mapped into the ego frame with the same
        rotation the BEV renderer uses, then quantised.
        """
        dx = float(wx) - self.origin[0]
        dy = float(wy) - self.origin[1]
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        # ego: x forward (dx cos + dy sin), y left (dx -sin + dy cos)
        ex = dx * ch + dy * sh
        ey = -dx * sh + dy * ch
        return self.ego_to_cell(ex, ey)

    def ego_to_cell(self, ex: float, ey: float) -> tuple[int, int] | None:
        """Cell for an ego-frame point (x forward, y left)."""
        if abs(ex) >= self.extent or abs(ey) >= self.extent:
            return None
        r = int((self.max_x - ex) / self.res)
        c = int((self.max_y - ey) / self.res)
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            return None
        return r, c

    def add_obstacle_point(self, wx: float, wy: float,
                           z: float = 0.0, weight: float = 1.0) -> None:
        cell = self.world_to_cell(wx, wy)
        if cell is None:
            return
        r, c = cell
        if self.obstacle[r, c]:
            return
        self.occupancy[r, c] = min(
            1.0, self.occupancy[r, c] + weight * 0.4)
        if np.isfinite(self.height[r, c]):
            self.height[r, c] = (self.height[r, c] * self.sources[r, c]
                                 + float(z)) / (self.sources[r, c] + 1)
        else:
            self.height[r, c] = float(z)
        self.sources[r, c] += 1

    def add_observed_point(self, wx: float, wy: float) -> None:
        """Mark a grid cell as SEEN by a sensor (road or not).

        ``drivable`` only records road evidence; a 0 cell therefore means
        either grass/terrain OR simply "not stamped yet".  The observed
        layer records every back-projected camera pixel (road and non-
        road), so the planner can tell "seen as non-drivable" from
        "unknown" - a hard off-road gate must only punish the former.
        """
        cell = self.world_to_cell(wx, wy)
        if cell is None:
            return
        self.observed[cell] = 1.0

    def add_drivable_point(self, wx: float, wy: float) -> None:
        cell = self.world_to_cell(wx, wy)
        if cell is None:
            return
        r, c = cell
        if not self.obstacle[r, c]:
            self.drivable[r, c] = 1.0
            # evidence decay handled elsewhere; free space clears occupancy
            # evidence below the permanent barrier threshold
            self.occupancy[r, c] = max(0.0, self.occupancy[r, c] - 0.15)

    # ------------------------------------------------------------------
    def _cells_for_points(self, wxs, wys):
        """Vectorised ``world_to_cell`` for arrays of world points.

        Returns ``(rows, cols)`` int arrays containing ONLY the points
        that land inside the grid (out-of-bounds points are dropped),
        using the exact same rotation + quantisation as
        ``world_to_cell``.
        """
        dx = np.asarray(wxs, dtype=float) - self.origin[0]
        dy = np.asarray(wys, dtype=float) - self.origin[1]
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        ex = dx * ch + dy * sh
        ey = -dx * sh + dy * ch
        inside = (np.abs(ex) < self.extent) & (np.abs(ey) < self.extent)
        r = ((self.max_x - ex) / self.res).astype(np.int64)
        c = ((self.max_y - ey) / self.res).astype(np.int64)
        inside &= (r >= 0) & (r < self.n_rows) & (c >= 0) & (c < self.n_cols)
        return r[inside], c[inside]

    def add_observed_points(self, wxs, wys) -> None:
        """Batch ``add_observed_point`` (same semantics, one stamp)."""
        rs, cs = self._cells_for_points(wxs, wys)
        if len(rs):
            self.observed[rs, cs] = 1.0

    def add_drivable_points(self, wxs, wys) -> None:
        """Batch ``add_drivable_point`` (same semantics, one stamp).

        Per-hit decay semantics are preserved: a cell hit by ``k`` road
        pixels loses exactly ``0.15 * k`` occupancy (sequential
        ``max(0, x - 0.15)`` applied k times equals the batched form).
        """
        rs, cs = self._cells_for_points(wxs, wys)
        if not len(rs):
            return
        flat = rs * self.n_cols + cs
        hits = np.bincount(
            flat, minlength=self.n_rows * self.n_cols
        ).reshape(self.n_rows, self.n_cols)
        free = (hits > 0) & (self.obstacle == 0)
        self.drivable[free] = 1.0
        self.occupancy[free] = np.maximum(
            0.0, self.occupancy[free] - 0.15 * hits[free])

    def mark_obstacle_region(self, wx: float, wy: float, hw: float,
                             hh: float, z: float = 1.0,
                             axis=None, half_len: float = 0.0,
                             half_thick: float = 0.0) -> None:
        """Flood a rectangular obstacle footprint into the grid.

        Without an oriented footprint a world-axis-aligned rectangle
        ``(wx +/- hw, wy +/- hh)`` is flooded (vehicle / pillar boxes).
        When ``axis`` + ``half_len`` / ``half_thick`` are provided the
        footprint is a ROTATED rectangle along ``axis`` - a diagonal
        roadside wall must only occupy its thin strip, not the huge
        world-AABB that would otherwise cover the road (town runs
        2026-08-21: a 3.6 m roadside wall flooded a 7.2 x 3.6 m world
        rectangle and every FSD path "grazed" it).
        """
        if axis is not None and half_len is not None and half_thick is not None:
            ax = np.asarray(axis, dtype=float)[:2]
            n = float(np.linalg.norm(ax))
            if n > 1e-9:
                ax = ax / n
            else:
                ax = np.array([1.0, 0.0])
            px = np.array([-ax[1], ax[0]])
            hl = max(0.0, float(half_len))
            ht = max(0.0, float(half_thick))
            # world-axis-aligned bounds of the rotated rectangle corners
            ctr = np.array([wx, wy])
            corner_pts = []
            for sa in (-1, 1):
                for st in (-1, 1):
                    corner_pts.append(ctr + sa * hl * ax + st * ht * px)
            x0 = float(min(pt[0] for pt in corner_pts))
            x1 = float(max(pt[0] for pt in corner_pts))
            y0 = float(min(pt[1] for pt in corner_pts))
            y1 = float(max(pt[1] for pt in corner_pts))
            # Vectorised footprint sampling: same world-sample grid as the
            # scalar loop, same rotated-rect membership test, cells via
            # the shared batched world_to_cell (only the min/max of the
            # hit cells feeds the stamp, so order is irrelevant).
            gx, gy = np.meshgrid(np.arange(x0, x1 + self.res, self.res),
                                 np.arange(y0, y1 + self.res, self.res))
            dx = gx - ctr[0]
            dy = gy - ctr[1]
            sel = ((np.abs(dx * ax[0] + dy * ax[1]) <= hl)
                   & (np.abs(dx * px[0] + dy * px[1]) <= ht))
            if not sel.any():
                return
            rs, cs = self._cells_for_points(gx[sel], gy[sel])
            if not len(rs):
                return
            r0, r1 = int(rs.min()), int(rs.max())
            c0, c1 = int(cs.min()), int(cs.max())
            self.obstacle[r0:r1 + 1, c0:c1 + 1] = 1
            self.occupancy[r0:r1 + 1, c0:c1 + 1] = np.maximum(
                self.occupancy[r0:r1 + 1, c0:c1 + 1], 0.9)
            return
        x0, x1 = wx - hw, wx + hw
        y0, y1 = wy - hh, wy + hh
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        samples = []
        for wx_c in (x0, (x0 + x1) / 2.0, x1):
            for wy_c in (y0, (y0 + y1) / 2.0, y1):
                dx = wx_c - self.origin[0]
                dy = wy_c - self.origin[1]
                ex = dx * ch + dy * sh
                ey = -dx * sh + dy * ch
                # clamp ego coords into the grid, then quantise directly
                ex = min(max(ex, -self.extent + 1e-6),
                         self.extent - 1e-6)
                ey = min(max(ey, -self.extent + 1e-6),
                         self.extent - 1e-6)
                cell = self.ego_to_cell(ex, ey)
                if cell is not None:
                    samples.append(cell)
        if not samples:
            return
        rs = [c[0] for c in samples]
        cs = [c[1] for c in samples]
        r0, r1 = max(0, min(rs)), min(self.n_rows - 1, max(rs))
        c0, c1 = max(0, min(cs)), min(self.n_cols - 1, max(cs))
        self.obstacle[r0:r1 + 1, c0:c1 + 1] = 1
        self.occupancy[r0:r1 + 1, c0:c1 + 1] = np.maximum(
            self.occupancy[r0:r1 + 1, c0:c1 + 1], 0.9)
        # NOTE: do NOT erase the drivable layer here.  Drivable is the
        # sensor-observed road surface (vision), obstacle is the occupied
        # space (LiDAR/boxes).  A roadside wall must not erase the lane
        # next to it - the planner's lane centre is computed over the FREE
        # corridor (drivable AND NOT obstacle), so occupancy still blocks
        # driving while the road surface survives beside the wall (town
        # corner runs 2026-08-21: the corner wall wiped all drivable cells
        # and the planner declared "no drivable path" 8 m short of the
        # turn).
        self.height[r0:r1 + 1, c0:c1 + 1] = float(z)

    # ------------------------------------------------------------------
    def query_path_cost(self, path_xy) -> float:
        """Sum of occupancy along a world-space path (mean cell occupancy).

        A trajectory scorer uses this to reject or penalise paths that run
        through occupied cells.  Out-of-grid samples count as occupied
        (unknown space is not drivable).

        NOTE: the live trajectory scoring does NOT call this -
        ``planning.constraints`` has its own grid scans tuned for the
        planner (corridor bands, drivable/observed gating).  This is the
        standalone occupancy-cost API, kept for tools and tests.
        """
        if path_xy is None or len(path_xy) == 0:
            return 0.0
        total = 0.0
        n = 0
        for x, y in path_xy:
            cell = self.world_to_cell(x, y)
            if cell is None:
                total += 1.0
            else:
                total += float(self.occupancy[cell])
            n += 1
        return total / max(1, n)

    def as_raster(self) -> np.ndarray:
        """2D float raster (row 0 = front) of occupancy, 0..1."""
        return self.occupancy.copy()


def project_road_mask_to_grid(grid: OccupancyGrid, road_mask: np.ndarray,
                              cam, pos, heading: float,
                              max_ahead_m: float = 45.0,
                              step: int = 4) -> None:
    """Mark drivable cells from a camera road/line mask via inverse
    ground projection.

    Walks a sparse set of image pixels, back-projects each to the ground
    plane (z of the ego) through the pin-hole ``cam`` and stamps the
    corresponding grid cell drivable.  This is the camera->BEV "lift"
    primitive of a vector-space stack.  ``road_mask`` is a bool mask of
    drivable surface pixels (the semantic head's road mask).

    Fully vectorised (meshgrid over the sampled pixels): the per-pixel
    ray math is identical to the original scalar loop, only evaluated
    for every sampled pixel at once.
    """
    h, w = road_mask.shape[:2]
    C, r_vec, f_vec, u_vec = cam.camera_pose(pos, heading)
    ground_z = float(pos[2]) if len(np.asarray(pos)) > 2 else 0.0
    fx, fy = cam.fx, cam.fy
    cx, cy = cam.cx, cam.cy
    us = np.arange(0, w, step)
    vs = np.arange(0, h, step)
    uu, vv = np.meshgrid(us, vs)
    # ray from camera through pixel (u, v); camera coords: x right,
    # y down, z forward (pinhole)
    xc = (uu - cx) / fx
    yc = (vv - cy) / fy
    D = xc[..., None] * r_vec - yc[..., None] * u_vec + f_vec
    Dz = D[..., 2]
    # The ray must point downward toward the ground plane.  In the
    # (right, fwd, up) convention f_vec is up-ish; +y image (lower half)
    # points along -u_vec (down), so a ray meeting the ground has a
    # negative z component.  Rays pointing up or level never hit the
    # ground (sky / horizon pixels).
    ok = Dz < -1e-6
    t = np.zeros_like(Dz)
    np.divide(ground_z - C[2], Dz, out=t, where=ok)
    good = ok & (t > 0.0) & (t <= max_ahead_m)
    wx = C[0] + t * D[..., 0]
    wy = C[1] + t * D[..., 1]
    good &= (wx - pos[0]) ** 2 + (wy - pos[1]) ** 2 <= max_ahead_m ** 2
    if not good.any():
        return
    # Every back-projected pixel (road OR non-road) is "observed";
    # only road pixels are drivable.  The observed layer lets the
    # planner distinguish grass/terrain (seen, not drivable) from
    # unknown space beyond the sensor footprint.
    grid.add_observed_points(wx[good], wy[good])
    road = good & road_mask[vv, uu]
    if road.any():
        grid.add_drivable_points(wx[road], wy[road])


def fuse_obstacles_to_grid(grid: OccupancyGrid, obstacles,
                           ray_hits=None) -> None:
    """Flood the grid from world obstacle boxes and raw ray hits.

    ``obstacles`` are ``perception.Obstacle`` boxes; ``ray_hits`` is an
    optional iterable of ``(x, y)`` points (e.g. LiDAR footprint).
    Both come from existing channels, so building vector space does not
    require any new sensor.
    """
    for ob in obstacles or []:
        grid.mark_obstacle_region(float(ob.x), float(ob.y),
                                  float(getattr(ob, "half_w", 0.5)),
                                  float(getattr(ob, "half_h", 0.5)),
                                  z=getattr(ob, "z", 1.0),
                                  axis=getattr(ob, "axis", None),
                                  half_len=float(getattr(ob, "half_len", 0.0)),
                                  half_thick=float(getattr(ob, "half_thick", 0.0)))
    for hx, hy in (ray_hits or []):
        grid.add_obstacle_point(hx, hy, weight=0.35)