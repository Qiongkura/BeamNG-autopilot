"""Closed-loop offline FSD drive simulation (stubbed sensors, no game).

Runs the full FSDStack tick -> PurePursuit -> bicycle-model cycle in
two-way right-hand road worlds whose lane geometry comes ONLY from the
semantic head (yellow centre line + white right edge markings), so the
planned path, steering and progress are all driven by perception - never
by a nav-line fixed offset.

Straight world (right-hand traffic, road runs along +x):
    centre line  y =  0.0   (yellow solid, left boundary of the ego lane)
    right edge   y = -3.5   (white solid, right boundary)
    ego lane centre = -1.75

Curve world: the same two-way road bent into a constant-radius LEFT
curve (radius 30 m) - the perception lane must lead through the bend,
and the car must hold its own lane without touching the centre line or
the road edge.

The wall world adds a lane-blocking wall ahead: the FSD safety layers
(speed profile + grid/raw-sensor emergency stop) must slow the car and
stop it before contact - never drive into the wall.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.control.reverse_guard import ReverseGuard
from beamng_autopilot.occupancy import OccupancyGrid, fuse_obstacles_to_grid
from beamng_autopilot.planner.obstacles import (
    emergency_speed_limit_mps,
    emergency_stop_clearance_m,
    path_grid_clearance_m,
)


class _ClearRange:
    """Range stub: empty road ahead, no obstacles and no ray hits."""

    def scan(self, pos):
        from beamng_autopilot.runtime import RangeSample
        return RangeSample(obstacles=[], ray_hits=[])


class _StubRing:
    role = "front_main"

    def grab_ring(self):
        from beamng_autopilot.vision.projection import CameraModel
        model = CameraModel(np.array([0.0, 1.0, 1.4]),
                            np.array([0.0, 0.9999, -0.02]),
                            np.array([0.0, 0.02, 0.9999]), 65.0, 160, 120)
        frame = np.random.default_rng(0).integers(
            40, 220, (120, 160, 3), dtype=np.uint8)
        return {"front_main": (frame, model)}

    def close(self):
        return None


class _LaneWorldSemantic:
    """Semantic head stub: real painted road markings, regenerated around
    the current ego pose every frame.

    The two markings are the centre line (yellow solid at road lat 0) and
    the right edge (white solid at road lat -3.5); the paired lane centre
    therefore sits at -1.75 - the ego lane centre, from perception only.
    """

    name = "semantic"

    def run(self, ctx):
        from beamng_autopilot.vision.hydra import TaskOutput
        from beamng_autopilot.vision.lanes import LaneMarking
        h, w = ctx.frame_rgb.shape[:2]
        road = np.zeros((h, w), dtype=bool)
        road[h // 2:] = True
        hdg = float(ctx.heading)
        fwd = np.array([np.cos(hdg), np.sin(hdg)])
        left = np.array([-fwd[1], fwd[0]])
        pos = np.asarray(ctx.pos[:2], dtype=float)
        # Station origin = the point of the road centre directly below the
        # car (the road runs along +x with centre line at world y=0).
        origin = np.array([pos[0], 0.0])
        s = np.linspace(2.0, 18.0, 17)

        def line(road_lat, color, kind):
            # Real painted markings in WORLD coordinates: fixed road
            # latitude (centre line y=0, right edge y=-3.5), running
            # ahead of the car. The head projects them into the car frame.
            world = origin + s[:, None] * fwd + road_lat * left
            return LaneMarking(world=world, pixels=world,
                               color=color, kind=kind, confidence=0.9)

        return TaskOutput(
            masks={"road": road, "line": road},
            meta={"markings": [line(0.0, "yellow", "solid"),
                               line(-3.5, "white", "solid")]})


class _StubConn:
    def get_state(self):
        class S:
            pos = np.array([0.0, 0.0, 0.0])
            heading = 0.0
            speed = 5.0
        return S()


class _CurveWorldSemantic:
    """Semantic head stub over a constant-radius LEFT curve.

    The road centre line (yellow) and right edge (white) are concentric
    arcs around the same centre of curvature, so the paired lane centre
    is the parallel arc 1.75 m right-hand of the centre line - the ego
    lane centre.  Markings are regenerated every frame in WORLD
    coordinates from the car's current road station, exactly like a
    real painted-line pair seen through a bend.
    """

    name = "semantic"

    def __init__(self, radius_m: float = 30.0):
        self.radius_m = float(radius_m)

    @staticmethod
    def road_lat(pos, r):
        """Signed road latitude (m) of a world point: left = +."""
        rho = math.hypot(float(pos[0]), r - float(pos[1]))
        return r - rho

    def run(self, ctx):
        from beamng_autopilot.vision.hydra import TaskOutput
        from beamng_autopilot.vision.lanes import LaneMarking
        h, w = ctx.frame_rgb.shape[:2]
        road = np.zeros((h, w), dtype=bool)
        road[h // 2:] = True
        r = self.radius_m
        pos = np.asarray(ctx.pos[:2], dtype=float)
        rho = math.hypot(float(pos[0]), r - float(pos[1]))
        t0 = math.atan2(float(pos[0]), r - float(pos[1]))
        s = np.linspace(2.0, 20.0, 19)

        def line(road_lat):
            # centre-of-curvature distance of this marking's arc
            rho_m = r - road_lat
            tt = t0 + s / max(rho, 1e-3)
            world = np.column_stack(
                [rho_m * np.sin(tt), r - rho_m * np.cos(tt)])
            return LaneMarking(
                world=world, pixels=world,
                color="yellow" if road_lat == 0.0 else "white",
                kind="solid", confidence=0.9)

        return TaskOutput(
            masks={"road": road, "line": road},
            meta={"markings": [line(0.0), line(-3.5)]})


class _WallRange:
    """Range stub: a wall box right across the ego lane plus the lidar
    ray hits on its near face - the raw-sensor safety layer's view."""

    def __init__(self, wall_x: float = 35.0, lane_y: float = -1.75,
                 half_w: float = 1.5, half_h: float = 1.5):
        self.wall_x = float(wall_x)
        self.lane_y = float(lane_y)
        self.half_w = float(half_w)
        self.half_h = float(half_h)

    def scan(self, pos):
        from beamng_autopilot.perception import Obstacle
        from beamng_autopilot.runtime import RangeSample
        ob = Obstacle(self.wall_x, self.lane_y, half_w=self.half_w,
                      half_h=self.half_h,
                      category="wall", label="wall")
        face = self.wall_x - self.half_w
        ys = np.linspace(self.lane_y - (self.half_h - 0.15),
                         self.lane_y + (self.half_h - 0.15), 10)
        return RangeSample(obstacles=[ob],
                           ray_hits=[(face, float(y)) for y in ys])


def _stack():
    from beamng_autopilot.planning import Constraints
    from beamng_autopilot.vision.hydra import HydraNet
    from beamng_autopilot.fsd_stack import FSDStack

    st = FSDStack.__new__(FSDStack)
    st.conn = _StubConn()
    st.ring = _StubRing()
    st.mode = "tech-stub"
    st.range_prov = _ClearRange()
    st.hydra = HydraNet()
    st.hydra.add(_LaneWorldSemantic())
    st.constraints = Constraints(w_collision=5.0, w_curvature=0.5,
                                 w_lane_align=1.0)
    st.grid_n, st.grid_res = 60, 0.5
    st.target_speed = 8.0
    st.ego_half_width = 1.3
    st.lane_mode = "map"
    st.strict_sensor = False
    # deterministic throttling / temporal state (temporal off keeps the
    # occupancy smoothness layer out of this pure-simulation test)
    st.semantic_every_n = 1
    st.object_every_n = 1
    st.range_every_n = 1
    st._range_skip = 0
    st._last_range = None
    st._last_range_t = 0.0
    st._head_skip = {}
    st._last_heads = {}
    st._head_phase = {}
    st._head_retry = set()
    st._tick_num = 0
    st.temporal = False
    st.occ_filter = None
    st.tracker = None
    st.fmap = None
    return st


def test_fsd_closed_loop_drives_straight_in_own_lane() -> None:
    """Full tick->control->kinematics loop on a clear two-way road.

    The car must keep driving forward in its own lane with the perception
    lane as the lateral reference: every tick plans a drivable path that
    never crosses the centre line, steers smoothly (no oscillation),
    never reverses, and makes real forward progress.
    """
    st = _stack()
    route = np.column_stack([np.linspace(0.0, 120.0, 121),
                             np.zeros(121)])
    pos = np.array([0.0, -1.75, 0.0])     # lane centre
    heading = 0.0
    speed = 0.0
    dt = 0.25
    wheelbase = 2.9
    pp = PurePursuit(lookahead=6.0, wheelbase=wheelbase)
    guard = ReverseGuard()

    total = 0.0
    prev_steer = 0.0
    lane_srcs = set()
    planner_kinds = set()
    max_path_y = -1e9    # world y of the chosen path (left = +y)
    min_path_y = 1e9
    max_dsteer = 0.0
    for i in range(60):
        out = st.tick(
            st=SimpleNamespace(pos=pos, heading=heading, speed=speed),
            route_ref=route)
        assert out.best_path is not None, out.meta
        lane_srcs.add(out.meta.get("lane_src_sel", ""))
        planner_kinds.add(out.meta["planner"].get("kind", "?"))

        path = np.asarray(out.best_path[:, :2], dtype=float)
        # Only look past the car's own footprint (>= 2.5 m ahead).
        fwd = np.array([np.cos(heading), np.sin(heading)])
        rel = path - pos[:2]
        ahead = (rel @ fwd) >= 2.5
        if ahead.any():
            max_path_y = max(max_path_y, float(path[ahead, 1].max()))
            min_path_y = min(min_path_y, float(path[ahead, 1].min()))

        v = float(out.best_speed or 0.0)
        assert v >= 0.0, out.meta
        steer, _, _ = pp.steering(pos, heading, path, speed=v)
        assert abs(steer) <= 0.5, f"steer out of range: {steer:.3f}"
        if i > 0:
            max_dsteer = max(max_dsteer, abs(steer - prev_steer))
        prev_steer = steer

        brake, reverse = guard.decide(v, dt=dt)
        assert not reverse, "car must never drive backwards"
        assert brake == 0.0

        total += v * dt
        pos = pos + v * dt * np.array([fwd[0], fwd[1], 0.0])
        heading = heading + steer * v * dt / wheelbase
        speed = v

    # real forward progress, not a crawl or a spin
    assert total >= 15.0, f"only drove {total:.1f} m in 60 ticks"
    # the lateral reference came from the perception lane (sensor)
    assert "sensor" in lane_srcs, lane_srcs
    # the planner chose the lane-centre tracking path (perception lane)
    assert "lane_center" in planner_kinds, planner_kinds
    # the path never crossed the centre line (y=0) nor left the right edge
    assert max_path_y <= 0.05, f"path crossed the centre line: {max_path_y:.2f}"
    assert min_path_y >= -3.55, f"path left the right edge: {min_path_y:.2f}"
    # the car stayed straight (no sideways drift / oscillation)
    assert abs(heading) < 0.2, f"heading drifted: {heading:.3f} rad"
    # steering never flapped tick to tick (smooth control)
    assert max_dsteer <= 0.25, f"steering oscillation: {max_dsteer:.3f} rad"


def test_fsd_closed_loop_sensor_lane_used_every_tick() -> None:
    """The perception lane is paired and gated in on every tick - the
    planner's lateral reference is the semantic lane centre, so no
    'nav line + offset' logic is in the driving loop."""
    st = _stack()
    route = np.column_stack([np.linspace(0.0, 120.0, 121),
                             np.zeros(121)])
    pos = np.array([0.0, -1.75, 0.0])
    heading = 0.0
    speed = 0.0
    dt = 0.25
    pp = PurePursuit(lookahead=6.0, wheelbase=2.9)
    guard = ReverseGuard()
    for i in range(30):
        out = st.tick(
            st=SimpleNamespace(pos=pos, heading=heading, speed=speed),
            route_ref=route)
        assert out.meta.get("lane_paired") == 1, out.meta
        assert out.meta.get("lane_src_sel") == "sensor", out.meta
        lane_ref = out.lane_ref
        assert lane_ref is not None and len(lane_ref) >= 3
        # the perception lane centre sits at the EGO lane centre (-1.75),
        # not on the road centre line (y=0) and not at some fixed offset
        med_y = float(np.median(np.asarray(lane_ref)[:, 1]))
        assert abs(med_y - (-1.75)) < 0.4, f"lane centre at y={med_y:.2f}"
        v = float(out.best_speed or 0.0)
        steer, _, _ = pp.steering(pos, heading, out.best_path, speed=v)
        brake, reverse = guard.decide(v, dt=dt)
        assert not reverse
        fwd = np.array([np.cos(heading), np.sin(heading)])
        pos = pos + v * dt * np.array([fwd[0], fwd[1], 0.0])
        heading = heading + steer * v * dt / 2.9
        speed = v


def test_fsd_closed_loop_curve_stays_in_own_lane() -> None:
    """Full tick->control->kinematics loop through a constant-radius left
    bend whose markings are regenerated every frame from the world curve.

    The planned path must follow the curve through the bend inside the ego
    lane (road latitude [-3.55, 0]: right edge at -3.5, centre line at 0)
    every tick, never crossing the centre line or the road edge, while the
    car turns left, steers smoothly and never reverses.
    """
    st = _stack()
    st.lane_mode = "sensor"
    st.strict_sensor = False
    st.hydra._heads = {}
    st.hydra.add(_CurveWorldSemantic(radius_m=30.0))
    st.target_speed = 5.0
    r = 30.0
    tt = np.linspace(0.0, 2.2, 140)
    route = np.column_stack([r * np.sin(tt), r - r * np.cos(tt)])
    pos = np.array([0.0, -1.75, 0.0])     # ego lane centre on the curve
    heading = 0.0
    speed = 0.0
    dt = 0.25
    wheelbase = 2.9
    pp = PurePursuit(lookahead=6.0, wheelbase=wheelbase)
    guard = ReverseGuard()

    total = 0.0
    prev_steer = 0.0
    max_dsteer = 0.0
    min_path_lat = 1e9
    max_path_lat = -1e9
    min_ego_lat = 1e9
    max_ego_lat = -1e9
    for i in range(45):
        out = st.tick(
            st=SimpleNamespace(pos=pos, heading=heading, speed=speed),
            route_ref=route)
        # the lateral reference is the perception lane every single tick
        assert out.meta.get("lane_paired") == 1, out.meta
        assert out.meta.get("lane_src_sel") == "sensor", out.meta
        assert out.best_path is not None, out.meta
        path = np.asarray(out.best_path[:, :2], dtype=float)
        # road latitude from the pure curve geometry (left = +)
        path_lat = r - np.hypot(path[:, 0], r - path[:, 1])
        fwd = np.array([np.cos(heading), np.sin(heading)])
        ahead = ((path - pos[:2]) @ fwd) >= 2.5
        if ahead.any():
            min_path_lat = min(min_path_lat,
                               float(path_lat[ahead].min()))
            max_path_lat = max(max_path_lat,
                               float(path_lat[ahead].max()))
        ego_lat = r - math.hypot(float(pos[0]), r - float(pos[1]))
        min_ego_lat = min(min_ego_lat, ego_lat)
        max_ego_lat = max(max_ego_lat, ego_lat)

        v = float(out.best_speed or 0.0)
        assert v >= 0.0, out.meta
        steer, _, _ = pp.steering(pos, heading, path, speed=v)
        assert abs(steer) <= 0.5, f"steer out of range: {steer:.3f}"
        if i > 0:
            max_dsteer = max(max_dsteer, abs(steer - prev_steer))
        prev_steer = steer

        brake, reverse = guard.decide(v, dt=dt)
        assert not reverse, "car must never drive backwards"
        assert brake == 0.0

        total += v * dt
        pos = pos + v * dt * np.array([fwd[0], fwd[1], 0.0])
        heading = heading + steer * v * dt / wheelbase
        speed = v

    # real forward progress through the bend
    assert total >= 15.0, f"only drove {total:.1f} m in 45 ticks"
    # the car actually turned left with the curve
    assert heading > 0.3, f"heading did not follow the left bend: {heading:.3f}"
    # planned path never crossed the centre line (lat 0) nor left the edge
    assert max_path_lat <= -0.05, \
        f"path crossed the centre line: {max_path_lat:.2f}"
    assert min_path_lat >= -3.55, \
        f"path left the road edge: {min_path_lat:.2f}"
    # the car's own body also stayed inside the ego lane
    assert max_ego_lat <= -0.05, \
        f"ego crossed the centre line: {max_ego_lat:.2f}"
    assert min_ego_lat >= -3.55, f"ego left the road edge: {min_ego_lat:.2f}"
    # steering never flapped tick to tick (smooth curve control)
    assert max_dsteer <= 0.25, f"steering oscillation: {max_dsteer:.3f} rad"


def test_fsd_closed_loop_tight_curve_stays_in_own_lane() -> None:
    """TIGHT (r=15) left bend regression: the painted boundaries must be
    paired by world road station so the perception lane keeps leading
    through the bend.

    On a tight curve the old whole-line car-frame median swings metres
    outward as the far arc curves around the car, so the real pair was
    gated out and the mirror fallback parked the lane on the centre line
    (``lane_paired=0`` from tick 0, ``lane_src='bev/route'``).  This test
    feeds the ego-anchored forward route exactly like the product
    (``m5_fsd_drive`` calls ``local_route`` before ``stack.tick``): every
    tick must keep the PAIRED sensor lane as the lateral reference, and
    the path / ego must stay in the ego lane (road latitude
    [-3.55, 0]) through the bend.
    """
    from beamng_autopilot.planning.local_route import local_route

    st = _stack()
    st.lane_mode = "sensor"
    st.strict_sensor = False
    st.hydra._heads = {}
    st.hydra.add(_CurveWorldSemantic(radius_m=15.0))
    st.target_speed = 5.0
    r = 15.0
    # ~0.55 m vertex spacing (coarser than r=30's 140-point arc): the
    # tighter arc must NOT collapse under map_lane_local's dedup.
    tt = np.linspace(0.0, 2.2, 60)
    route = np.column_stack([r * np.sin(tt), r - r * np.cos(tt)])
    pos = np.array([0.0, -1.75, 0.0])     # ego lane centre on the curve
    heading = 0.0
    speed = 0.0
    dt = 0.25
    wheelbase = 2.9
    pp = PurePursuit(lookahead=6.0, wheelbase=wheelbase)
    guard = ReverseGuard()

    total = 0.0
    prev_steer = 0.0
    max_dsteer = 0.0
    min_path_lat = 1e9
    max_path_lat = -1e9
    min_ego_lat = 1e9
    max_ego_lat = -1e9
    # Stop well inside the 33 m route so the map-prior lane stays
    # available every tick (the end-zone handling is a separate concern).
    for i in range(22):
        route_ref = local_route(pos, heading, route)
        out = st.tick(
            st=SimpleNamespace(pos=pos, heading=heading, speed=speed),
            route_ref=route_ref)
        # the lateral reference is the paired perception lane every tick
        assert out.meta.get("lane_paired") == 1, out.meta
        assert out.meta.get("lane_src") == "sensor", out.meta
        assert out.meta.get("lane_src_sel") == "sensor", out.meta
        assert out.best_path is not None, out.meta
        path = np.asarray(out.best_path[:, :2], dtype=float)
        # road latitude from the pure curve geometry (left = +)
        path_lat = r - np.hypot(path[:, 0], r - path[:, 1])
        fwd = np.array([np.cos(heading), np.sin(heading)])
        ahead = ((path - pos[:2]) @ fwd) >= 2.5
        if ahead.any():
            min_path_lat = min(min_path_lat,
                               float(path_lat[ahead].min()))
            max_path_lat = max(max_path_lat,
                               float(path_lat[ahead].max()))
        ego_lat = r - math.hypot(float(pos[0]), r - float(pos[1]))
        min_ego_lat = min(min_ego_lat, ego_lat)
        max_ego_lat = max(max_ego_lat, ego_lat)

        v = float(out.best_speed or 0.0)
        assert v >= 0.0, out.meta
        steer, _, _ = pp.steering(pos, heading, path, speed=v)
        assert abs(steer) <= 0.5, f"steer out of range: {steer:.3f}"
        if i > 0:
            max_dsteer = max(max_dsteer, abs(steer - prev_steer))
        prev_steer = steer

        brake, reverse = guard.decide(v, dt=dt)
        assert not reverse, "car must never drive backwards"
        assert brake == 0.0

        total += v * dt
        pos = pos + v * dt * np.array([fwd[0], fwd[1], 0.0])
        heading = heading + steer * v * dt / wheelbase
        speed = v

    # real forward progress through the bend
    assert total >= 15.0, f"only drove {total:.1f} m in 22 ticks"
    # the car actually turned left with the tight curve
    assert heading > 0.3, f"heading did not follow the bend: {heading:.3f}"
    # planned path never crossed the centre line (lat 0) nor left the edge
    assert max_path_lat <= -0.05, \
        f"path crossed the centre line: {max_path_lat:.2f}"
    assert min_path_lat >= -3.55, \
        f"path left the road edge: {min_path_lat:.2f}"
    # the car's own body also stayed inside the ego lane
    assert max_ego_lat <= -0.05, \
        f"ego crossed the centre line: {max_ego_lat:.2f}"
    assert min_ego_lat >= -3.55, f"ego left the road edge: {min_ego_lat:.2f}"
    # steering never flapped tick to tick (smooth curve control)
    assert max_dsteer <= 0.25, f"steering oscillation: {max_dsteer:.3f} rad"


def test_fsd_closed_loop_wall_stop() -> None:
    """Full-width wall across the ego lane: the longitudinal safety layers
    (speed profile + grid/raw emergency stop) must halt the car before
    contact - never drive into the wall and never brake-and-reverse.

    Mirrors the product wiring (``scripts/m5_fsd_drive.py``): clearance is
    measured along the CHOSEN path in the ego-centred occupancy grid, with
    the raw-sensor heading corridor as the last-line fallback when the
    grid read is blank (a lane-blocked reference path can park its first
    samples inside the wall footprint, which the grid layer skips).
    """
    st = _stack()
    wall = _WallRange(wall_x=35.0, lane_y=-1.75, half_w=1.5, half_h=5.25)
    st.range_prov = wall
    st.target_speed = 5.0
    route = np.column_stack([np.linspace(0.0, 120.0, 121),
                             np.zeros(121)])
    pos = np.array([0.0, -1.75, 0.0])
    heading = 0.0
    speed = 0.0
    dt = 0.25
    wheelbase = 2.9
    pp = PurePursuit(lookahead=6.0, wheelbase=wheelbase)
    guard = ReverseGuard()

    face_x = wall.wall_x - wall.half_w
    ego_nose_m = 2.35      # EGO_NOSE_M: nose sits this far ahead of centre
    min_best = 1e9
    force_stop_seen = False
    speed_zeroed_after_stop = False
    max_x = -1e9
    total = 0.0
    for i in range(200):
        out = st.tick(
            st=SimpleNamespace(pos=pos, heading=heading, speed=speed),
            route_ref=route)
        rng = wall.scan(pos)
        # ego-centred grid rebuilt exactly like the stack's vector space
        grid = OccupancyGrid(st.grid_n, st.grid_n, st.grid_res,
                             origin=(float(pos[0]), float(pos[1])),
                             heading=heading)
        fuse_obstacles_to_grid(grid, rng.obstacles)
        path = np.asarray(out.best_path[:, :2], dtype=float)
        fwd_clear = min(path_grid_clearance_m(path, grid),
                        float(out.forward_clearance))
        force_stop = False
        cap = float("inf")
        if np.isfinite(fwd_clear):
            need = emergency_stop_clearance_m(speed, margin=ego_nose_m + 1.0)
            force_stop, cap = emergency_speed_limit_mps(fwd_clear, need)
        v = float(out.best_speed or 0.0)
        min_best = min(min_best, v)
        target = 0.0 if force_stop else min(v, cap)
        if force_stop:
            force_stop_seen = True
        # bounded accelerate / heavy brake toward the effective target
        if target < speed:
            speed = max(0.0, speed - 6.0 * dt)
        else:
            speed = min(max(speed, 0.0) + 2.0 * dt, target)
        if force_stop_seen and speed == 0.0:
            speed_zeroed_after_stop = True

        steer, _, _ = pp.steering(pos, heading, path, speed=speed)
        assert abs(steer) <= 0.5, f"steer out of range: {steer:.3f}"
        brake, reverse = guard.decide(speed, dt=dt)
        assert not reverse, "car must never drive backwards"
        assert brake == 0.0

        fwd = np.array([np.cos(heading), np.sin(heading)])
        total += speed * dt
        pos = pos + speed * dt * np.array([fwd[0], fwd[1], 0.0])
        heading = heading + steer * speed * dt / wheelbase
        max_x = max(max_x, float(pos[0]))

    # the speed profile saw the wall and dipped well below cruise
    assert min_best <= 1.5, f"plan never braked for the wall: {min_best:.2f}"
    # the emergency layer force-stopped at least once
    assert force_stop_seen, "emergency stop was never triggered"
    # the car made real progress toward the wall...
    assert max_x >= 20.0, f"car barely moved: max_x={max_x:.2f}"
    # ...then stopped before its nose reached the wall face
    nose_x = max_x + ego_nose_m
    assert nose_x < face_x - 0.5, \
        f"nose {nose_x:.2f} too close to wall face {face_x:.2f}"
    # it came to a full stop after the force-stop and stayed stopped
    assert speed_zeroed_after_stop, "car never reached v=0 after stop"
    assert speed == 0.0, f"car still moving at the end: {speed:.2f} m/s"
    assert total > 20.0, f"only drove {total:.1f} m in 200 ticks"
