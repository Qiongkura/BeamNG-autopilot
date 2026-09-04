"""Closed-loop offline FSD drive simulation (stubbed sensors, no game).

Runs the full FSDStack tick -> PurePursuit -> bicycle-model cycle in a
two-way right-hand road world whose lane geometry comes ONLY from the
semantic head (yellow centre line + white right edge markings), so the
planned path, steering and progress are all driven by perception - never
by a nav-line fixed offset.

World layout (right-hand traffic, road runs along +x):
    centre line  y =  0.0   (yellow solid, left boundary of the ego lane)
    right edge   y = -3.5   (white solid, right boundary)
    ego lane centre = -1.75
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.control.reverse_guard import ReverseGuard


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
