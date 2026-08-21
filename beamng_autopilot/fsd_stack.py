"""FSDStack: the integrated multi-camera perception+planning pipeline.

Wires the FSD-style modules built in previous milestones into one
callable stack that mirrors the data flow of a Tesla FSD stack:

  camera ring -> HydraNets (multi-task heads) -> BEV/vector space
  (occupancy + feature fusion) -> layered planner -> control hint.

It is intentionally *standalone*: it does not touch the existing
``AutopilotSession`` plan path, so the validated rule driving (94.6%
route result) keeps working untouched.  The stack is exercised live by
probes and shadow recording; when it is proven it can replace the old
path without disturbing it first.

Usage::

    stack = FSDStack(conn, mode="tech")
    out = stack.tick()   # one full perception+planning tick
    out.bev             # (N,N) occupancy raster
    out.best_path       # chosen trajectory or None
"""

from __future__ import annotations

import math
import time

import numpy as np

from beamng_autopilot.occupancy import (
    OccupancyGrid,
    fuse_obstacles_to_grid,
    project_road_mask_to_grid,
)
from beamng_autopilot.planning import (
    Constraints,
    Scene,
    sample_arc,
    sample_lane_shift,
    select_trajectory,
)
from beamng_autopilot.runtime import (
    build_camera_ring_provider,
    build_range_provider,
)
from beamng_autopilot.vision.hydra import FrameContext, HydraNet


class FSDTick:
    """One planning-tick result from ``FSDStack``."""

    def __init__(self):
        self.bev: np.ndarray | None = None      # (N, N) occupancy raster
        self.drivable: np.ndarray | None = None
        self.best_path: np.ndarray | None = None
        self.lane_ref: np.ndarray | None = None  # vision lane centreline
        self.best_speed: float = 0.0            # planned speed at the start
        self.min_speed: float = 0.0             # lowest speed on the path
        self.n_candidates: int = 0
        self.meta: dict = {}
        self.head_outputs: dict = {}
        self.errors: dict = {}


class FSDStack:
    """Integrated FSD-style perception + planning pipeline."""

    def __init__(self, conn, mode: str = "tech",
                 grid_n: int = 60, grid_res: float = 0.5,
                 ring_roles=None, heads=None,
                 cam_w: int = 320, cam_h: int = 240,
                 temporal: bool = True, tau_s: float = 1.5):
        self.conn = conn
        self.grid_n = int(grid_n)
        self.grid_res = float(grid_res)
        self.heads = list(heads) if heads else []

        ring, self.mode = build_camera_ring_provider(
            conn, mode, cam_w, cam_h)
        self.ring = ring
        if self.ring is None:
            self.mode = "steam-front"    # front camera only available
        self.range_prov, _ = build_range_provider(conn, mode)

        self.hydra = HydraNet()
        for head in self.heads:
            self.hydra.add(head)

        self.constraints = Constraints(
            w_collision=5.0, w_curvature=0.5, w_lane_align=1.0)
        self.target_speed = 8.0  # plan cruise speed (m/s); drive can raise

        # FSD-style temporal occupancy fusion: single-frame LiDAR glitches
        # must not create a phantom wall or erase a real one.
        self.temporal = bool(temporal)
        self.occ_filter = None
        self._tick_t0 = None
        if self.temporal:
            from beamng_autopilot.temporal import TemporalOccupancyFilter
            self.occ_filter = TemporalOccupancyFilter(
                n=int(grid_n), res=float(grid_res), tau_s=float(tau_s))

    # ------------------------------------------------------------------
    def tick(self, st=None, route_ref: np.ndarray | None = None,
             include_bev: bool = True) -> FSDTick:
        """Run one full perception + planning tick.

        ``st`` is a vehicle state with ``.pos`` / ``.heading`` / ``.speed``
        (falls back to a live ``get_state()`` when None).  ``route_ref``
        is the nav route to plan along (defaults to straight ahead).
        """
        out = FSDTick()
        if st is None:
            st = self.conn.get_state()
        pos = np.asarray(st.pos, dtype=float)
        heading = float(st.heading)

        # --- 1) camera ring -> HydraNet heads ---------------------------
        snap: dict = {}
        if self.ring is not None:
            try:
                snap = self.ring.grab_ring()
            except Exception as exc:
                out.errors["ring"] = str(exc)
        if snap:
            role = "front_main" if "front_main" in snap \
                else next(iter(snap))
            frame, cam = snap[role]
            ctx = FrameContext(
                frame_rgb=frame, cam=cam, pos=pos, heading=heading,
                ground_z=float(pos[2]) if len(pos) > 2 else 0.0,
                role=role)
            out.head_outputs = self.hydra.run(ctx)

        # --- 2) BEV vector space (occupancy grid) -----------------------
        grid = OccupancyGrid(self.grid_n, self.grid_n, self.grid_res,
                             origin=(float(pos[0]), float(pos[1])),
                             heading=heading)
        if include_bev:
            semantic = out.head_outputs.get("semantic")
            if semantic is not None and "road" in semantic.masks \
                    and "front_main" in snap:
                project_road_mask_to_grid(
                    grid, semantic.masks["road"],
                    snap["front_main"][1], pos, heading, step=4)
            try:
                rng = self.range_prov.scan(pos)
                fuse_obstacles_to_grid(grid, rng.obstacles, rng.ray_hits)
                out.meta["n_obstacles"] = len(rng.obstacles)
            except Exception as exc:
                out.errors["range"] = str(exc)
        out.bev = grid.as_raster()
        out.drivable = grid.drivable

        # Temporal fusion: smooth the single-frame occupancy before the
        # planner / safety layers read it (a one-frame glitch is neither a
        # phantom wall nor a vanished one).
        occ_filter = getattr(self, "occ_filter", None)
        if occ_filter is not None:
            now = time.time()
            t0 = getattr(self, "_tick_t0", None)
            dt = (now - t0) if t0 is not None else 0.0
            if dt < 0 or dt > 5.0:
                dt = 0.0
            self._tick_t0 = now
            occ_filter.update(out.bev, dt)
            out.bev = occ_filter.raster()
            # mark the fused occupancy back into the grid obstacle layer so
            # planner collision checks use the smoothed space too
            from beamng_autopilot.occupancy import OccupancyGrid as _OG
            if isinstance(grid, _OG):
                grid.obstacle[:] = (out.bev >= 0.6).astype(np.uint8)
                grid.occupancy[:] = np.asarray(out.bev, dtype=np.float32)

        # --- 3) layered planner -----------------------------------------
        if route_ref is None or len(route_ref) < 2:
            xs = np.linspace(0, 40, 41)
            route_ref = np.column_stack(
                [pos[0] + xs * np.cos(heading),
                 pos[1] + xs * np.sin(heading)])
        # Lane reference from the drivable-space centreline (FSD "vector
        # space" lane) when the sensor road is visible, else the nav
        # route itself (FSD uses map prior when the camera lane is not
        # available).  The safety monitor's lane-deviation check must
        # measure against a *meaningful* reference, not None.
        lane_ref = self._bev_drivable_center(grid, pos, heading)
        # The sensor (vision+LiDAR) drivable-centreline is the lane the
        # car must hold (FSD "vector space" lane): plan ALONG it when the
        # road is visible, and only fall back to the map/nav prior when
        # the sensor lane is unavailable.  A nav route that is not
        # re-anchored to the ego (or runs far from the car) must not drag
        # the planner off the real road - observed town runs where the
        # car followed the road fine but the stale nav route pinned it
        # "off-lane".
        plan_route = lane_ref if (lane_ref is not None
                                  and len(lane_ref) >= 4) else route_ref
        if lane_ref is None:
            lane_ref = route_ref
        out.lane_ref = np.asarray(plan_route, dtype=float)
        scene = Scene(pos=pos, heading=heading, grid=grid,
                      route=plan_route, lane_ref=lane_ref,
                      target_speed=getattr(self, "target_speed", 8.0))
        # Town corners need a tighter arc fan than a highway fan: a
        # 5-8 m radius bend is 0.12-0.2 rad/m, and the old 0.10 rad/m
        # cap (10 m radius) could not turn away from a corner wall
        # (town runs 2026-08-21) - it kept pressing the throttle into
        # the wall.  Sample to the physical steer limit and add wider
        # lateral shifts so the planner can actually dodge a near wall.
        fans = sample_arc(pos, heading, speed=max(2.0, float(st.speed)),
                          max_steer=0.5, n_curv=11, max_curv=0.20)
        shifts = sample_lane_shift(plan_route,
                                   offsets=(-3.0, -1.5, 1.5, 3.0))
        for c in shifts.candidates:
            fans.add(c.path, c.meta.get("kind", "shift"),
                     offset=c.meta.get("offset", 0.0))
        out.n_candidates = len(fans.candidates)
        best, meta = select_trajectory(scene, fans, self.constraints)
        out.best_path = best
        out.meta["planner"] = meta
        out.meta["total_candidates"] = out.n_candidates
        # the chosen path's speed profile (planning-side longitudinal plan)
        sp = meta.get("speed_profile")
        if best is not None and sp is not None and len(sp):
            out.best_speed = float(sp[0])
            out.min_speed = float(np.asarray(sp).min())
            out.meta["best_speed"] = out.best_speed
            out.meta["min_speed"] = out.min_speed

        # Run the topology head over the sensor lane derived from the
        # semantic markings (its graph needs a real LaneFrame, not just a
        # frame context).
        topology = self.hydra._heads.get("topology")
        if topology is not None:
            lane_frame = self._sensor_lane_from_semantic(
                out.head_outputs, pos, heading)
            ctx = FrameContext(
                frame_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                cam=None, pos=pos, heading=heading,
                ground_z=float(pos[2]) if len(pos) > 2 else 0.0,
                role="front_main")
            try:
                topo_out = topology.run(ctx, sensor_lane=lane_frame)
                out.head_outputs["topology"] = topo_out
            except Exception:
                pass
        out.meta.update(semantic_to_meta(out.head_outputs))
        return out

    def _sensor_lane_from_semantic(self, head_outputs, pos, heading):
        """Pair the semantic head's world markings into a LaneFrame."""
        try:
            from beamng_autopilot.lane import pair_lane_markings
            sem = head_outputs.get("semantic")
            if sem is None:
                return None
            markings = sem.meta.get("markings", [])
            if not markings:
                return None
            return pair_lane_markings(markings, pos, heading)
        except Exception:
            return None

    def _bev_drivable_center(self, grid, pos, heading):
        """World centreline of the drivable space in the BEV grid.

        For each longitudinal band of the grid, the lateral centre of
        the drivable cells becomes one point of a lane reference - the
        "space centreline" a real vector-space planner tracks.  Falls
        back to a straight line ahead when few drivable cells exist
        (unknown road, sensor-limited frame).
        """
        drv = getattr(grid, "drivable", None)
        if drv is None or not getattr(drv, "any", lambda: False)():
            return None
        n = int(getattr(grid, "n_rows", None) or
                getattr(grid, "n_cols", None) or 60)
        res = grid.res
        extent = grid.extent
        step = max(1, n // 24)
        pts = []
        for r in range(0, n, step):
            row = drv[r]
            cols = np.nonzero(row)[0]
            if cols.size == 0:
                continue
            c_mid = float(cols.mean())
            ex = extent - (r + 0.5) * res
            ey = extent - (c_mid + 0.5) * res
            # ego -> world
            ch = math.cos(float(heading))
            sh = math.sin(float(heading))
            wx = float(pos[0]) + ex * ch - ey * sh
            wy = float(pos[1]) + ex * sh + ey * ch
            pts.append((wx, wy))
        if len(pts) < 3:
            return None
        arr = np.asarray(pts, dtype=float)
        # Anchor the centreline at the ego and order it near -> far so the
        # planner sees a path it can actually drive from here.  The raw
        # grid rows run far -> near, so without re-anchoring the first
        # point sits metres ahead of the car and every candidate gets
        # scored as off-lane (town runs 2026-08-21: all arcs infeasible at
        # (717.8,754.6) because the reference started 8 m away).
        d = arr - np.asarray(pos[:2], dtype=float)
        fwd_m = d[:, 0] * math.cos(float(heading)) + \
                d[:, 1] * math.sin(float(heading))
        ahead = arr[fwd_m > 0.5]
        if len(ahead) < 3:
            ahead = arr
        d0 = np.linalg.norm(ahead - np.asarray(pos[:2], dtype=float), axis=1)
        ahead = ahead[np.argsort(d0)]          # near -> far
        # Anchor the reference at the ego: when the nearest drivable row
        # is already a couple of metres in front, prepend the ego so the
        # planner's shift candidates start inside the forward-progress
        # gate (a reference whose first point is beyond ~3 m got every
        # lane-shift candidate rejected - town runs 2026-08-21).
        if len(ahead) and float(np.linalg.norm(ahead[0] - np.asarray(pos[:2],
                                                                     dtype=float))) > 2.0:
            ahead = np.vstack([np.asarray(pos[:2], dtype=float), ahead])
        return ahead if len(ahead) >= 3 else None

    def reset_temporal(self) -> None:
        """Clear the temporal filter - call after a teleport so stale
        occupancy from the previous location never leaks into the new
        scene as a phantom wall."""
        if getattr(self, "occ_filter", None) is not None:
            self.occ_filter.clear()
        self._tick_t0 = None

    def close(self) -> None:
        if self.ring is not None:
            try:
                self.ring.close()
            except Exception:
                pass


def semantic_to_meta(head_outputs: dict) -> dict:
    """Flatten a few useful head outputs into tick meta (telemetry)."""
    meta: dict = {}
    sem = head_outputs.get("semantic")
    if sem is not None:
        meta["lane_markings"] = len(sem.meta.get("markings", []))
    tr = head_outputs.get("traffic")
    if tr is not None:
        meta["signal_state"] = tr.meta.get("signal_state")
        meta["signal_conf"] = tr.meta.get("signal_conf")
    topo = head_outputs.get("topology")
    if topo is not None:
        meta["change_left"] = topo.meta.get("change_left")
        meta["change_right"] = topo.meta.get("change_right")
    return meta