"""Offline (no-game, no-network) validation of the M5 planner changes.

Runs pure-numpy checks against the same modules the autopilot uses, so the
fixes can be verified without starting BeamNG:

* ``LocalPlanner._lateral_bypass`` - a parked car in the lane becomes a
  lane-change detour (forward / sideways parked), a wall wider than the
  allowed lateral deviation yields ``None`` (-> stop in front of it).
* ``LocalPlanner.plan`` - detour mode + not blocked for the parked car,
  blocked + truncated path for an impassable wide wall.
* ``SpeedController`` - throttle / brake change is rate-limited every step
  (linear pedals, no jumps).
* ``scan_obstacles_vehicles`` - the box follows the vehicle yaw (a car
  parked along the road stays slim; rotated 90 deg it becomes wide).
* ``merge_obstacles`` - the same spot from two sensors merges into one box.

Usage:
    .venv\\Scripts\\python.exe scripts\\m5_offline_validate.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot.control.speed import SpeedController
from beamng_autopilot.lane import (
    LaneFrame,
    LaneTracker,
    LANE_FAR_CENTER_PAIR_CONF_MAX,
    LANE_FAR_MIRROR_CONF_MAX,
    LANE_VISION_RIGHT_MIRROR_CONF_MAX,
    build_lidar_corridor,
    choose_sensor_lane,
    lane_frame_usable,
    pair_lane_markings,
)
from beamng_autopilot.perception import (
    Obstacle,
    _cluster_points,
    drop_vision_waypoint_ghosts,
    filter_self_overlap,
    merge_obstacles,
    scan_obstacles,
    scan_obstacles_raycast,
    scan_obstacles_vehicles,
)
from beamng_autopilot.planner import (
    CAR_HALF_WIDTH,
    PASS_BY_MIN_MPS,
    LocalPlanner,
    corner_angle_deg,
    corner_speed,
    _point_lat_offset,
    _seg_hits_box,
    creep_speed,
    is_lane_edge_wall,
)
from beamng_autopilot.traffic import (
    RoadRuleView,
    SignalRule,
    apply_rule_speed,
    classify_lane_direction,
    lane_offset_m,
    legal_lane_view,
    one_way_from_lanes,
    select_signal_rule,
    signal_action_label,
    signal_distance,
)
from beamng_autopilot.vision.tracking import VisionTrack, update_vision_tracks


_FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _FAILED.append(name)


def straight_route(length_m: float = 60.0, spacing: float = 1.0) -> np.ndarray:
    xs = np.arange(0.0, length_m + 1e-9, spacing)
    return np.column_stack([xs, np.zeros_like(xs)])


def curve_route() -> np.ndarray:
    """90-degree left bend with straight lead-in and tail."""
    xs1 = np.linspace(0.0, 20.0, 21)
    ys1 = np.zeros_like(xs1)
    ang = np.linspace(-np.pi / 2.0, 0.0, 41)
    cx, cy, radius = 20.0, 20.0, 20.0
    xs2 = cx + radius * np.cos(ang)
    ys2 = cy + radius * np.sin(ang)
    ys3 = np.linspace(20.0, 60.0, 41)
    xs3 = np.full_like(ys3, 40.0)
    xs = np.concatenate([xs1[:-1], xs2[:-1], xs3])
    ys = np.concatenate([ys1[:-1], ys2[:-1], ys3])
    return np.column_stack([xs, ys])


def path_hits_box(path, box: Obstacle, extra: float) -> bool:
    """True when any path segment enters the box inflated by ``extra``."""
    for i in range(len(path) - 1):
        if _seg_hits_box(
            path[i, 0], path[i, 1], path[i + 1, 0], path[i + 1, 1],
            box.x, box.y, box.half_w + extra, box.half_h + extra,
        ):
            return True
    return False


def test_bypass_forward_car(planner: LocalPlanner) -> None:
    """Car parked in the lane, nose pointing along the road."""
    route = straight_route()
    box = Obstacle(x=30.0, y=0.0, half_w=4.6, half_h=2.2,
                   category="vehicle", label="car")
    pts, i0, i1 = planner._window(route, 0)
    bypass = planner._lateral_bypass(pts, [box], i0, i1)
    check("bypass forward-car: returns a path", bypass is not None)
    if bypass is None:
        return
    # Needs at least half_lat + car half width + 0.9 ~ 4.1 m sideways.
    near = np.where((bypass[:, 0] >= box.x - box.half_w)
                    & (bypass[:, 0] <= box.x + box.half_w))[0]
    max_lat = float(np.abs(bypass[near, 1]).max()) if len(near) else 0.0
    check("bypass forward-car: lateral clearance >= need",
          max_lat >= 4.1 - 0.5, f"max_lat={max_lat:.2f}")
    check("bypass forward-car: clears obstacle footprint",
          not path_hits_box(bypass, box, CAR_HALF_WIDTH + 0.6))
    drive, blocked = planner.plan(route, [box], (0.0, 0.0), 0.0, 0)
    check("plan forward-car: mode=detour, blocked=False",
          planner.last_mode == "detour" and not blocked,
          f"mode={planner.last_mode}")
    check("plan forward-car: path clears obstacle footprint",
          not path_hits_box(np.asarray(drive), box, CAR_HALF_WIDTH + 0.6))


def test_bypass_sideways_car(planner: LocalPlanner) -> None:
    """Car parked across the road (rotated 90 deg): needs a wider shift."""
    route = straight_route()
    box = Obstacle(x=30.0, y=0.0, half_w=2.2, half_h=4.6,
                   category="vehicle", label="car")
    pts, i0, i1 = planner._window(route, 0)
    bypass = planner._lateral_bypass(pts, [box], i0, i1)
    check("bypass sideways-car: returns a path", bypass is not None)
    if bypass is None:
        return
    near = np.where((bypass[:, 0] >= box.x - box.half_w)
                    & (bypass[:, 0] <= box.x + box.half_w))[0]
    max_lat = float(np.abs(bypass[near, 1]).max()) if len(near) else 0.0
    check("bypass sideways-car: lateral clearance >= need",
          max_lat >= 6.5 - 0.5, f"max_lat={max_lat:.2f}")
    check("bypass sideways-car: clears obstacle footprint",
          not path_hits_box(bypass, box, CAR_HALF_WIDTH + 0.6))


def test_wide_wall_blocks(planner: LocalPlanner) -> None:
    """A wall wider than the max lateral deviation stops the car."""
    route = straight_route()
    # 18 m wide wall: needs ~10.9 m lateral -> more than max_dev 8 m.
    wall = Obstacle(x=30.0, y=0.0, half_w=2.0, half_h=9.0,
                    category="wall", label="wall")
    pts, i0, i1 = planner._window(route, 0)
    bypass = planner._lateral_bypass(pts, [wall], i0, i1)
    check("bypass wide-wall(18m): returns None (too wide)",
          bypass is None)
    # 60 m wide wall: neither bypass nor grid A* can find a way -> blocked,
    # and the drive path must be truncated before the wall.
    big = Obstacle(x=30.0, y=0.0, half_w=2.0, half_h=30.0,
                   category="wall", label="wall")
    drive, blocked = planner.plan(route, [big], (0.0, 0.0), 0.0, 0)
    drive = np.asarray(drive, dtype=float)
    last_x = float(drive[-1, 0]) if len(drive) else 0.0
    check("plan wide-wall(60m): blocked=True, mode=blocked",
          blocked and planner.last_mode == "blocked",
          f"mode={planner.last_mode}")
    check("plan wide-wall(60m): path truncated before wall",
          len(drive) >= 2 and last_x < 27.5, f"last_x={last_x:.1f}")
    check("plan wide-wall(60m): blocker recorded for HUD",
          planner.last_blocker is not None
          and planner.last_blocker[0] == "wall",
          f"blocker={planner.last_blocker}")


def test_roadside_wall_boundary(planner: LocalPlanner) -> None:
    """A long wall beside the road is a boundary, not a detour target.

    Regression for the reported "sudden left swerve into the map wall":
    the right-hand offset used to press the path into the right wall, then
    the planner tried to lane-change to the left.  Now the offset clamps
    away from the wall and the planner keeps following the corridor.
    """
    route = straight_route()
    right_wall = Obstacle(x=20.0, y=-3.0, half_w=18.0, half_h=0.5,
                          category="raycast", label="wall")
    drive, blocked = planner.plan(route, [right_wall], (0.0, 0.0), 0.0, 0)
    drive = np.asarray(drive, dtype=float)
    y_mid = float(drive[25, 1]) if len(drive) > 25 else 99.0
    check("roadside-wall: right wall does not trigger detour",
          not blocked and planner.last_mode in ("follow", "deform"),
          f"mode={planner.last_mode} blocked={blocked}")
    check("roadside-wall: default keep-right clears the wall",
          len(drive) > 25 and -1.5 <= y_mid <= 0.35,
          f"y25={y_mid:.2f}")
    check("roadside-wall: path clears wall footprint",
          not path_hits_box(drive, right_wall, CAR_HALF_WIDTH + 0.5))

    left_wall = Obstacle(x=20.0, y=3.0, half_w=18.0, half_h=0.5,
                         category="raycast", label="wall")
    drive2, blocked2 = planner.plan(
        route, [right_wall, left_wall], (0.0, 0.0), 0.0, 0)
    drive2 = np.asarray(drive2, dtype=float)
    max_y = float(drive2[10:, 1].max()) if len(drive2) > 10 else 99.0
    check("roadside-wall: two walls keep follow, no swerve",
          not blocked2 and planner.last_mode in ("follow", "deform"),
          f"mode={planner.last_mode} blocked={blocked2}")
    check("roadside-wall: path stays between the walls",
          len(drive2) > 10 and abs(max_y) < 1.5,
          f"max_y={max_y:.2f}")


def test_lane_edge_wall_boundary(planner: LocalPlanner) -> None:
    """A short thin wall at the detected lane edge is a boundary, not a
    blocker.

    Regression for run 45: a ~2.5 m roadside wall segment sat at the lane
    edge while its axis-aligned box poked to ~1.4 m from the centre of a
    3.5 m lane.  The fixed ``CAR_HALF_WIDTH + 0.8`` clearance then read it
    as a corridor blocker and the car parked in the middle of an open
    lane.  With a usable lane frame the wall is the lane boundary itself,
    so it must not close the whole corridor from the side.
    """
    xs = np.arange(0.0, 16.0, 1.5)
    center = np.column_stack([xs, np.zeros_like(xs)])
    frame = LaneFrame(center=center, width=3.5, confidence=0.8,
                      span_m=15.0, sources=("vision",))

    edge_wall = Obstacle(
        x=10.0, y=-1.75, half_w=0.9, half_h=1.6,
        category="raycast", label="",
        axis=np.array([1.0, 0.0]), half_len=1.23, half_thick=0.0)
    check("lane-edge-wall: classifier marks lane-edge wall",
          is_lane_edge_wall(edge_wall, center, frame.width))

    route = straight_route()
    drive, blocked = planner.plan(
        route, [edge_wall], (0.0, 0.0), 0.0, 0, sensor_lane=frame)
    drive = np.asarray(drive, dtype=float)
    check("lane-edge-wall: follows lane centre, not blocked",
          not blocked and planner.last_mode == "follow"
          and len(drive) >= 2,
          f"mode={planner.last_mode} blocked={blocked} n={len(drive)}")

    inside = Obstacle(
        x=10.0, y=-1.0, half_w=0.9, half_h=1.6,
        category="raycast", label="",
        axis=np.array([1.0, 0.0]), half_len=1.23, half_thick=0.0)
    check("lane-edge-wall: wall inside the lane is not a boundary",
          not is_lane_edge_wall(inside, center, frame.width))

    cross = Obstacle(
        x=10.0, y=-1.75, half_w=0.9, half_h=1.6,
        category="raycast", label="",
        axis=np.array([0.0, 1.0]), half_len=1.23, half_thick=0.0)
    check("lane-edge-wall: crossing wall is not a boundary",
          not is_lane_edge_wall(cross, center, frame.width))

    # A single-edge LiDAR frame knows the wall surface it was built from:
    # the right boundary polyline is the reference, so a slightly thick or
    # inflated wall cluster sitting at/outside that edge is still the
    # roadside boundary, not a corridor blocker.
    lidar_xs = np.arange(0.0, 16.0, 1.5)
    lidar_frame = LaneFrame(
        center=np.column_stack(
            [lidar_xs, np.full_like(lidar_xs, -1.69)]),
        right=np.column_stack(
            [lidar_xs, np.full_like(lidar_xs, -3.44)]),
        width=3.5, confidence=0.53, span_m=15.0,
        sources=("lidar",), paired=False)
    edge_out = Obstacle(
        x=10.0, y=-4.0, half_w=1.0, half_h=1.0,
        category="raycast", label="wall",
        axis=np.array([1.0, 0.0]), half_len=1.2, half_thick=0.8)
    check("lane-edge-wall: lidar edge marks roadside wall",
          is_lane_edge_wall(edge_out, lidar_frame.center,
                            lidar_frame.width,
                            lane_edge=lidar_frame.right))
    edge_inside = Obstacle(
        x=10.0, y=-2.0, half_w=1.0, half_h=1.0,
        category="raycast", label="wall",
        axis=np.array([1.0, 0.0]), half_len=1.2, half_thick=0.8)
    check("lane-edge-wall: lidar edge does not forgive in-lane wall",
          not is_lane_edge_wall(edge_inside, lidar_frame.center,
                                lidar_frame.width,
                                lane_edge=lidar_frame.right))
    drive_edge, blocked_edge = planner.plan(
        route, [edge_out], (0.0, 0.0), 0.0, 0, sensor_lane=lidar_frame)
    check("lane-edge-wall: lidar single-edge wall stays drivable",
          not blocked_edge and planner.last_mode in ("follow", "deform"),
          f"mode={planner.last_mode} blocked={blocked_edge}")


def test_right_offset_not_zeroed_by_corridor_hugging_wall(
        _planner: LocalPlanner) -> None:
    """A box hugging the nav corridor must not drag keep-right to zero.

    Regression for the italy finish-line behaviour: wall boxes that sit
    right on the route (or are merged into an AABB that covers it) made
    every right-hand candidate collide, so ``_safe_right_offset`` returned
    0 and the car slid back onto the centre line for the last 30 m.  When
    the zero-offset path is itself "hit", the offset cannot fix the
    collision, so the planner keeps a small right bias instead.
    """
    planner = LocalPlanner(right_offset=1.5, right_ramp_m=8.0)
    route = straight_route()
    pts, i0, i1 = planner._window(route, 0)
    hugging = Obstacle(x=20.0, y=-0.7, half_w=18.0, half_h=1.0,
                       category="raycast", label="wall")
    safe = planner._safe_right_offset(pts, i0, i1, 0.0, [hugging])
    check("hug-wall: corridor-hugging box keeps a right bias",
          safe >= 0.4, f"safe_off={safe:.2f}")

    # A genuine wall that only blocks the right side still clamps the
    # offset below the fallback, because the centre of the corridor stays
    # clear and there is no reason to bias toward a wall.
    right_wall = Obstacle(x=20.0, y=-3.0, half_w=18.0, half_h=0.5,
                          category="raycast", label="wall")
    safe_right = planner._safe_right_offset(
        pts, i0, i1, 0.0, [right_wall])
    check("hug-wall: true roadside wall still clamps, no fallback",
          0.3 <= safe_right < 0.8, f"safe_off={safe_right:.2f}")


def test_bypass_prefers_right_and_never_crosses_walls(
        planner: LocalPlanner) -> None:
    """Lateral bypass prefers the right side and refuses wall-lined sides."""
    route = straight_route()
    pts, i0, i1 = planner._window(route, 0)
    car = Obstacle(x=30.0, y=0.0, half_w=4.6, half_h=2.2,
                   category="vehicle", label="car")
    left_wall = Obstacle(x=20.0, y=5.0, half_w=18.0, half_h=0.5,
                         category="raycast", label="wall")
    bypass = planner._lateral_bypass(pts, [car, left_wall], i0, i1)
    check("bypass: left wall still allows right-side bypass",
          bypass is not None)
    if bypass is not None:
        near = np.where((bypass[:, 0] >= car.x - car.half_w)
                        & (bypass[:, 0] <= car.x + car.half_w))[0]
        max_lat = float(bypass[near, 1].max()) if len(near) else 0.0
        check("bypass: chooses the right-hand side first",
              len(near) > 0 and max_lat < -3.5,
              f"max_lat={max_lat:.2f}")

    right_wall = Obstacle(x=20.0, y=-3.0, half_w=18.0, half_h=0.5,
                          category="raycast", label="wall")
    bypass2 = planner._lateral_bypass(
        pts, [car, left_wall, right_wall], i0, i1)
    check("bypass: wall-lined sides yield no swerve", bypass2 is None)


def test_speed_ramp() -> None:
    """Pedals must ramp linearly; no step may jump."""
    ctrl = SpeedController()
    dt = 0.05
    # Accelerate from standstill toward 10 m/s.
    speed = 0.0
    prev_th, prev_br = 0.0, 0.0
    max_thr_step = 0.0
    overlap = 0.0
    for _ in range(40):
        th, br = ctrl.update(10.0, speed, dt=dt)
        max_thr_step = max(max_thr_step, abs(th - prev_th),
                           abs(br - prev_br))
        overlap = max(overlap, min(th, br))
        prev_th, prev_br = th, br
        speed = min(speed + 1.5 * dt, 10.0)
    check("speed ramp: throttle/brake step <= 0.16",
          max_thr_step <= 0.16 + 1e-9, f"max_step={max_thr_step:.3f}")
    check("speed ramp: no throttle+brake overlap",
          overlap <= 0.05 + 1e-6, f"overlap={overlap:.3f}")
    # Brake from 15 m/s toward 0.
    ctrl.reset()
    speed = 15.0
    prev_th, prev_br = 0.0, 0.0
    max_brk_step = 0.0
    for _ in range(50):
        th, br = ctrl.update(0.0, speed, dt=dt)
        max_brk_step = max(max_brk_step, abs(th - prev_th),
                           abs(br - prev_br))
        prev_th, prev_br = th, br
        speed = max(speed - 4.0 * dt, 0.0)
    check("speed ramp: brake step <= 0.16",
          max_brk_step <= 0.16 + 1e-9, f"max_step={max_brk_step:.3f}")
    check("speed ramp: brake engages", prev_br > 0.5,
          f"brake={prev_br:.2f}")


def test_speed_slip() -> None:
    """Wheel spin (wheels far faster than the body) must cut throttle."""
    ctrl = SpeedController()
    th, br = ctrl.update(10.0, 5.0, dt=0.05, wheel_speed=9.0)
    check("slip: throttle cut while spinning", th <= 0.08 + 1e-9,
          f"th={th:.2f}")
    check("slip: brake dab while rolling", br > 0.05, f"br={br:.2f}")
    check("slip: flagged for telemetry", ctrl.slip_active is True)
    for _ in range(3):
        th2, _ = ctrl.update(10.0, 5.0, dt=0.05, wheel_speed=5.2)
    check("slip: normal grip restores throttle",
          th2 > 0.05 and ctrl.slip_active is False,
          f"th={th2:.2f}")


def test_drop_waypoint_ghosts() -> None:
    """Vision boxes on the route markers (yellow start ball, red goal
    ball, moving nearest point) are ghosts; real obstacles survive."""
    start = Obstacle(x=0.0, y=0.0, half_w=1.0, half_h=1.0,
                     category="vision", label="person")
    near = Obstacle(x=3.1, y=0.0, half_w=1.0, half_h=1.0,
                    category="vision", label="person")
    goal = Obstacle(x=50.0, y=0.0, half_w=1.0, half_h=1.0,
                    category="vision", label="person")
    real = Obstacle(x=20.0, y=3.5, half_w=1.0, half_h=1.0,
                    category="vision", label="person")
    kept = drop_vision_waypoint_ghosts(
        [start, near, goal, real],
        [(0.0, 0.0), (3.0, 0.0), (50.0, 0.0)])
    check("waypoint ghosts: marker boxes dropped", len(kept) == 1,
          f"n={len(kept)}")
    if kept:
        check("waypoint ghosts: real obstacle kept", kept[0] is real,
              f"kept={kept[0].label}")
    wall = Obstacle(x=0.5, y=0.0, half_w=1.0, half_h=1.0,
                    category="raycast", label="wall")
    kept2 = drop_vision_waypoint_ghosts([wall], [(0.0, 0.0)])
    check("waypoint ghosts: non-vision boxes kept", len(kept2) == 1,
          f"n={len(kept2)}")


def test_vision_track_confirmation() -> None:
    """A camera phantom only reaches the planner after it survives two
    scans and an ego-motion world-stability check."""
    det = Obstacle(x=10.0, y=0.0, half_w=1.0, half_h=1.0,
                   category="vision", label="car")
    tracks, confirmed = update_vision_tracks(
        [], [det], (0.0, 0.0), 0.0)
    check("vision-track: single frame not confirmed",
          len(confirmed) == 0, f"n={len(confirmed)}")

    tracks, confirmed = update_vision_tracks(
        tracks, [det], (1.0, 0.0), 1.0)
    check("vision-track: static detection confirms after ego motion",
          len(confirmed) == 1, f"n={len(confirmed)}")
    if confirmed:
        check("vision-track: confirmed box keeps vision identity",
              confirmed[0].category == "vision"
              and confirmed[0].label == "car",
              f"{confirmed[0].category}/{confirmed[0].label}")

    still, still_conf = update_vision_tracks(
        [], [det], (0.0, 0.0), 0.0)
    still, still_conf = update_vision_tracks(
        still, [det], (0.0, 0.0), 1.0)
    check("vision-track: stationary ego never confirms",
          len(still_conf) == 0 and still[0].hits == 2
          and still[0].motion_seen is False,
          f"hits={still[0].hits} motion={still[0].motion_seen}")

    ph = Obstacle(x=10.0, y=0.0, half_w=1.0, half_h=1.0,
                  category="vision", label="car")
    ph_tracks, ph_conf = update_vision_tracks(
        [], [ph], (0.0, 0.0), 0.0)
    ph_tracks, ph_conf = update_vision_tracks(
        ph_tracks,
        [Obstacle(x=11.0, y=0.0, half_w=1.0, half_h=1.0,
                  category="vision", label="car")],
        (1.0, 0.0), 1.0)
    check("vision-track: camera phantom rides along and resets",
          len(ph_conf) == 0 and ph_tracks[0].hits == 1
          and ph_tracks[0].motion_seen is False,
          f"hits={ph_tracks[0].hits} motion={ph_tracks[0].motion_seen}")

    veh_tracks, veh_conf = update_vision_tracks(
        [], [det], (0.0, 0.0), 0.0)
    veh_tracks, veh_conf = update_vision_tracks(
        veh_tracks,
        [Obstacle(x=10.0, y=0.0, half_w=1.0, half_h=1.0,
                  category="vision", label="truck")],
        (1.0, 0.0), 1.0)
    check("vision-track: vehicle labels match across scans",
          len(veh_conf) == 1, f"n={len(veh_conf)}")

    person = Obstacle(x=10.0, y=0.0, half_w=1.0, half_h=1.0,
                      category="vision", label="person")
    mixed, mixed_conf = update_vision_tracks(
        [], [person], (0.0, 0.0), 0.0)
    mixed, mixed_conf = update_vision_tracks(
        mixed, [det], (1.0, 0.0), 1.0)
    check("vision-track: different labels stay separate",
          len(mixed) == 2 and len(mixed_conf) == 0,
          f"tracks={len(mixed)} confirmed={len(mixed_conf)}")


class _FakeBNG:
    """Minimal stub: returns two vehicles with different headings."""

    def queue_lua_command(self, chunk, response=True):
        return json.dumps([
            {"x": 12.0, "y": 4.0, "yaw": 0.0},        # nose along +x
            {"x": -10.0, "y": 6.0, "yaw": math.pi / 2},  # nose along +y
        ])


def test_vehicle_orientation() -> None:
    ob = scan_obstacles_vehicles(_FakeBNG(), "ego", (0.0, 0.0, 0.0))
    check("vehicle boxes: found 2", len(ob) == 2, f"n={len(ob)}")
    if len(ob) == 2:
        along, across = ob[0], ob[1]
        check("yaw=0: long side along x (hw=4.6)",
              abs(along.half_w - 4.6) < 1e-6
              and abs(along.half_h - 2.2) < 1e-6,
              f"hw={along.half_w:.2f} hh={along.half_h:.2f}")
        check("yaw=90deg: long side along y (hw/hh swapped)",
              abs(across.half_w - 2.2) < 1e-6
              and abs(across.half_h - 4.6) < 1e-6,
              f"hw={across.half_w:.2f} hh={across.half_h:.2f}")


def test_merge() -> None:
    a = Obstacle(x=10.0, y=0.0, half_w=2.3, half_h=1.1,
                 category="vehicle")
    b = Obstacle(x=11.5, y=0.0, half_w=2.3, half_h=1.1,
                 category="vision")
    merged = merge_obstacles([a, b])
    check("merge: two boxes < 2.5m apart become one",
          len(merged) == 1, f"n={len(merged)}")
    if merged:
        m = merged[0]
        covers = (m.x - m.half_w <= 10.0 - 2.3
                  and m.x + m.half_w >= 11.5 + 2.3)
        check("merge: box covers both extents", covers,
              f"x={m.x:.2f} hw={m.half_w:.2f}")
    wall = Obstacle(x=20.0, y=0.0, half_w=2.0, half_h=2.0,
                    category="wall")
    veh = Obstacle(x=21.0, y=0.0, half_w=2.3, half_h=1.1,
                   category="vehicle")
    not_merged = merge_obstacles([wall, veh])
    check("merge: wall and vehicle stay separate",
          len(not_merged) == 2, f"n={len(not_merged)}")


class _FakeRaycastBNG:
    """Stub: a wall hit (probe ray hits it too) and slope hits the probe
    ray clears (far contact / no contact)."""

    def queue_lua_command(self, chunk, response=True):
        return json.dumps([
            {"x": 10.0, "y": 0.0, "z": 1.15, "d": 10.0, "up": 10.4},
            {"x": 20.0, "y": 0.0, "z": 1.15, "d": 20.0, "up": 55.0},
            {"x": 30.0, "y": 0.0, "z": 1.15, "d": 30.0, "up": 40.0},
        ])


def test_raycast_slope_filter() -> None:
    """Ground / slope ray hits must not become obstacles."""
    obs = scan_obstacles_raycast(_FakeRaycastBNG(), (0.0, 0.0, 0.0))
    check("raycast slope-filter: slope hits dropped (1 wall kept)",
          len(obs) == 1, f"n={len(obs)}")
    if obs:
        check("raycast slope-filter: wall box near the kept hit",
              abs(obs[0].x - 10.0) < 1.5 and abs(obs[0].y) < 1.5,
              f"box=({obs[0].x:.1f}, {obs[0].y:.1f})")


class _FakeNearRaycastBNG:
    """Stub: a low rock close in (the rise probe clears it, so Lua reports
    up = radius + 1) and a far slope with the same clear probe."""

    def queue_lua_command(self, chunk, response=True):
        return json.dumps([
            {"x": 6.0, "y": 0.0, "z": 1.0, "d": 6.0, "up": 56.0,
             "fan": "mid"},
            {"x": 25.0, "y": 0.0, "z": 1.0, "d": 25.0, "up": 56.0,
             "fan": "mid"},
            {"x": 15.0, "y": 0.0, "z": 0.45, "d": 15.0, "up": 36.0,
             "fan": "low"},
            {"x": 23.0, "y": 0.0, "z": 0.45, "d": 23.0, "up": 36.0,
             "fan": "low"},
        ])


def test_raycast_near_low_obstacle() -> None:
    """A close low rock must be kept even when the rise probe passes over
    it, while the same clear probe far away is still a slope and drops.
    The low near-field fan keeps rocks / stumps out to 20 m while the mid
    fan only keeps probe-cleared hits within 10 m."""
    obs = scan_obstacles_raycast(_FakeNearRaycastBNG(), (0.0, 0.0, 0.0))
    check("raycast near: near rocks kept, far slopes dropped",
          len(obs) == 2, f"n={len(obs)}")
    if obs:
        check("raycast near: kept boxes are the near rocks",
              any(abs(o.x - 6.0) < 1.5 and abs(o.y) < 1.5 for o in obs)
              and any(abs(o.x - 15.0) < 1.5 and abs(o.y) < 1.5 for o in obs),
              f"boxes={[(round(o.x, 1), round(o.y, 1)) for o in obs]}")


class _FakeScenarioBNG:
    """Stub: an object under the ego and a real object down the road."""

    def queue_lua_command(self, chunk, response=True):
        return json.dumps([
            {"x": 0.5, "y": 0.0, "z": 0.0, "sx": 2.0, "sy": 2.0, "sz": 2.0},
            {"x": 0.8, "y": 0.3, "z": 0.0, "sx": 2.0, "sy": 2.0, "sz": 2.0},
            {"x": 10.0, "y": 0.0, "z": 0.0, "sx": 3.0, "sy": 1.5, "sz": 2.0},
        ])


def test_scenario_min_dist() -> None:
    """Objects sitting on / right under the ego must not block."""
    obs = scan_obstacles(_FakeScenarioBNG(), "ego", (0.0, 0.0, 0.0))
    check("scenario min-dist: ego-side objects dropped, road object kept",
          len(obs) == 1, f"n={len(obs)}")
    if obs:
        check("scenario min-dist: kept box is the far object",
              abs(obs[0].x - 10.0) < 0.1 and abs(obs[0].y) < 0.1,
              f"box=({obs[0].x:.1f}, {obs[0].y:.1f})")


def test_self_overlap_filter() -> None:
    """A vision box that covers the ego (own car in the chase cam) must be
    dropped, while a real box further away is kept.  A vehicle that is
    merely close (even a few metres) is a real blocker and stays."""
    ghost = Obstacle(x=2.3, y=1.1, half_w=2.4, half_h=5.5,
                     category="vision", label="bus")
    real = Obstacle(x=12.0, y=0.0, half_w=2.0, half_h=4.0,
                    category="vision", label="car")
    kept = filter_self_overlap([ghost, real], (0.0, 0.0, 0.0))
    check("self-overlap: ego-covering vision box dropped", len(kept) == 1,
          f"n={len(kept)}")
    if kept:
        check("self-overlap: far box kept", kept[0] is real,
              f"kept={kept[0].label}")
    # Non-vision sources keep their id-based ego exclusion; a real scenario
    # box that happens to sit under the car is still removed by the same
    # filter when the caller applies it across all categories.
    all_cat = filter_self_overlap(
        [ghost], (0.0, 0.0, 0.0), categories=("vision", "vehicle", "raycast"))
    check("self-overlap: works for any category", len(all_cat) == 0,
          f"n={len(all_cat)}")
    # Chase cam: the own car sits a few metres off the ego centre and the
    # footprint test misses it - but a car that close must be treated as a
    # real blocker (a vehicle 1-2 m away is exactly what the planner has
    # to react to), so it is kept and only boxes literally covering the
    # ego centre are dropped.
    chase = Obstacle(x=3.5, y=0.0, half_w=2.0, half_h=1.5,
                     category="vision", label="car")
    person = Obstacle(x=3.0, y=0.0, half_w=0.4, half_h=0.4,
                      category="vision", label="person")
    far = Obstacle(x=8.0, y=0.0, half_w=2.0, half_h=2.0,
                   category="vision", label="car")
    kept2 = filter_self_overlap([chase, person, far], (0.0, 0.0, 0.0))
    check("self-overlap: close car kept", len(kept2) == 3,
          f"n={len(kept2)}")
    if len(kept2) == 3:
        check("self-overlap: person + close car + far car all kept",
              kept2[0] is chase and kept2[1] is person and kept2[2] is far,
              f"kept={[o.label for o in kept2]}")


def test_heading_deviation() -> None:
    """The deviation guard slows the car once the heading leaves the route
    direction by more than 18 deg, down to a crawl when fully sideways."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "m5_autopilot",
        str(Path(__file__).resolve().parent / "m5_autopilot.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dev = mod.heading_deviation_deg
    cap = mod.heading_dev_speed_cap
    smooth_steer = mod.smooth_steer
    route = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])

    check("hdg: aligned heading", dev(route, 1, (1.0, 0.0)) == 0.0,
          f"dev={dev(route, 1, (1.0, 0.0))}")
    check("hdg: sideways heading",
          abs(dev(route, 1, (0.0, 1.0)) - 90.0) < 1e-9,
          f"dev={dev(route, 1, (0.0, 1.0))}")
    check("hdg: facing the wrong way",
          abs(dev(route, 1, (-1.0, 0.0)) - 180.0) < 1e-9,
          f"dev={dev(route, 1, (-1.0, 0.0))}")
    check("hdg: no cap below 18 deg", cap(15.0) is None)
    c18 = cap(18.5)
    check("hdg: cap right above 18 deg",
          c18 is not None and 2.4 < c18 <= 2.5,
          f"cap={c18}")
    c45 = cap(45.0)
    check("hdg: cap scales down",
          c45 is not None and 1.7 < c45 < 2.0,
          f"cap={c45}")
    c90 = cap(95.0)
    check("hdg: crawl when sideways",
          c90 is not None and abs(c90 - 0.6) < 1e-9,
          f"cap={c90}")
    check("hdg: no cap when not engaged below 18",
          cap(17.0) is None, f"cap={cap(17.0)}")
    c_hold = cap(17.0, engaged=True)
    check("hdg: engaged holds the cap below 18",
          c_hold is not None and abs(c_hold - 2.5) < 1e-9,
          f"cap={c_hold}")
    c_hold2 = cap(13.5, engaged=True)
    check("hdg: engaged still active above 13 deg",
          c_hold2 is not None and abs(c_hold2 - 2.5) < 1e-9,
          f"cap={c_hold2}")
    check("hdg: released below 13 deg",
          cap(12.0, engaged=True) is None,
          f"cap={cap(12.0, engaged=True)}")

    # The run-9 bend showed steering still pointed the wrong way after a
    # ~1 s sensor-blocked loop.  Smoothing must be time-based: one blocked
    # second is enough to cross from a right command to a left one, while a
    # fast 10 Hz loop only moves a small bounded step per frame.
    one_sec = smooth_steer(0.39, -0.45, 1.0)
    check("steer: one blocked second crosses the bend demand",
          one_sec <= -0.39, f"steer={one_sec:.3f}")
    tenth = smooth_steer(0.39, -0.45, 0.1)
    check("steer: fast loop is rate-limited per step",
          abs(tenth - (0.39 - 0.12)) < 1e-9, f"steer={tenth:.3f}")
    check("steer: never overshoots the demand",
          smooth_steer(-0.2, -0.45, 10.0) == -0.45)


def test_lane_reuse_staleness() -> None:
    """The vision worker only reuses a lane frame while it is fresh enough
    to transform into the current car frame."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "m5_autopilot",
        str(Path(__file__).resolve().parent / "m5_autopilot.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exp = mod.lane_reuse_expired

    check("lane-reuse: fresh frame stays reusable",
          exp(0, 0.0, 0.0) is False)
    check("lane-reuse: short miss still reusable",
          exp(3, 0.3, 1.2) is False)
    check("lane-reuse: too many misses expires",
          exp(7, 0.3, 1.2) is True)
    check("lane-reuse: age expires",
          exp(2, 1.1, 1.2) is True)
    check("lane-reuse: ego travel expires",
          exp(2, 0.3, 6.1) is True)
    check("lane-reuse: boundary stays fresh",
          exp(6, 1.0, 6.0) is False)
    check("lane-reuse: no stored frame expires",
          exp(0, 0.0, float("inf")) is True)


def test_bridge() -> None:
    import tempfile

    from beamng_autopilot.bridge import ControlBridge

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ctl.json"
        b = ControlBridge(path)
        check("bridge: empty file has seq 0", b.current_seq() == 0)
        check("bridge: send ok", b.send("autopilot"))
        cmds, seen = b.poll(0)
        check("bridge: poll delivers new command",
              cmds == [("autopilot", None)] and seen == 1,
              f"cmds={cmds} seen={seen}")
        cmds, seen = b.poll(seen)
        check("bridge: same command not replayed",
              cmds == [] and seen == 1)
        b.send("clear")
        cmds, seen = b.poll(seen)
        check("bridge: next command delivered",
              cmds == [("clear", None)] and seen == 2)
        b.send("set_speed", 12.5)
        cmds, seen = b.poll(seen)
        check("bridge: value command delivered",
              cmds == [("set_speed", 12.5)] and seen == 3)
        b2 = ControlBridge(path)
        cmds, seen2 = b2.poll(b2.current_seq())
        check("bridge: stale commands ignored at start",
              cmds == [] and seen2 == 3)


def test_roadnet_polylines() -> None:
    """nearby_polylines must return only connected road nodes within the
    radius, as (N,2) arrays, and stay empty before the network is ready."""
    from beamng_autopilot.roadnet import RoadNetwork

    rn = RoadNetwork()
    check("roadnet: empty before ready", rn.nearby_polylines((0.0, 0.0)) == [])
    rn.nodes = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
                         [30.0, 5.0], [100.0, 100.0]], dtype=float)
    rn.adj = {i: [] for i in range(5)}
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
        rn.adj[a].append((b, 1.0))
        rn.adj[b].append((a, 1.0))
    rn.ready = True
    polys = rn.nearby_polylines((15.0, 0.0), radius=20.0)
    check("roadnet: returns at least one polyline", len(polys) >= 1,
          f"n={len(polys)}")
    flat = [tuple(p) for pl in polys for p in pl]
    check("roadnet: near nodes chained", (0.0, 0.0) in flat
          and (10.0, 0.0) in flat and (20.0, 0.0) in flat,
          f"flat={flat}")
    check("roadnet: far node excluded", (100.0, 100.0) not in flat)
    check("roadnet: polylines are (N,2) arrays",
          all(pl.ndim == 2 and pl.shape[1] == 2 for pl in polys))


def test_connector_current_env() -> None:
    """current_env must prefer the live scenario level/vehicle and fall
    back to the launch args when the query fails."""
    from beamng_autopilot.connector import BeamNGConnector

    conn = BeamNGConnector("smallgrid", "etk800")

    class _Level:
        name = "hirochi_raceway"

    class _Scenario:
        level = _Level()
        name = "some_scenario"

        def get_current(self, connect=False):
            return self

    class _Bng:
        scenario = _Scenario()

    class _Veh:
        model = "etk800"

    conn.bng = _Bng()
    conn.vehicle = _Veh()
    env = conn.current_env()
    check("current_env: map from live level", env["map"] == "hirochi_raceway",
          f"map={env['map']}")
    check("current_env: vehicle from model", env["vehicle"] == "etk800",
          f"vehicle={env['vehicle']}")

    class _BrokenScenario:
        def get_current(self, connect=False):
            raise RuntimeError("no scenario")

    conn.bng = _Bng()
    conn.bng.scenario = _BrokenScenario()
    env2 = conn.current_env()
    check("current_env: falls back to launch args",
          env2["map"] == "smallgrid" and env2["vehicle"] == "etk800",
          f"env2={env2}")


def test_connector_italy_default_spawn() -> None:
    """The default validation map spawns at italy's crossroads."""
    from beamng_autopilot import config
    from beamng_autopilot import connector as conn_mod
    from beamng_autopilot.connector import BeamNGConnector

    captured: dict = {}

    class _Scenario:
        def __init__(self, map_name, name):
            captured["map"] = map_name

        def add_vehicle(self, vehicle, pos, rot_quat, cling):
            captured["pos"] = tuple(pos)
            captured["rot_quat"] = rot_quat

        def make(self, bng):
            captured["make"] = True

    class _Vehicle:
        def __init__(self, *args, **kwargs):
            pass

    class _ScenarioApi:
        def load(self, scenario):
            captured["load"] = True

        def start(self):
            captured["start"] = True

    class _Bng:
        def __init__(self):
            self.scenario = _ScenarioApi()

    old_scenario, old_vehicle = conn_mod.Scenario, conn_mod.Vehicle
    conn_mod.Scenario, conn_mod.Vehicle = _Scenario, _Vehicle
    try:
        conn = BeamNGConnector("italy", "etk800")
        conn.bng = _Bng()
        conn.load_scenario()
    finally:
        conn_mod.Scenario, conn_mod.Vehicle = old_scenario, old_vehicle

    check("italy default spawn: uses italy map",
          captured.get("map") == "italy", f"map={captured.get('map')}")
    check("italy default spawn: crossroads position",
          captured.get("pos") == config.ITALY_SPAWN_CROSSROADS_POS,
          f"pos={captured.get('pos')}")
    if captured.get("rot_quat"):
        qz, qw = captured["rot_quat"][2], captured["rot_quat"][3]
        yaw_deg = math.degrees(2.0 * math.atan2(qz, qw))
        expected = math.degrees(config.ITALY_SPAWN_CROSSROADS_HEADING) + 90.0
        check("italy default spawn: crossroads heading",
              abs(yaw_deg - expected) < 1e-6,
              f"yaw={yaw_deg:.3f} expected={expected:.3f}")


def test_launcher_imports() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "m5_launcher",
        str(Path(__file__).resolve().parent / "m5_launcher.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    check("launcher: imports + LauncherApp/BirdView defined",
          hasattr(mod, "LauncherApp") and hasattr(mod, "BirdView"))


def test_runtime_selection() -> None:
    """Core stays Steam-first: the Tech package loads only on Tech runs."""
    import sys

    from beamng_autopilot import config
    from beamng_autopilot.runtime import (
        SteamCameraProvider,
        SteamRangeProvider,
        build_camera_provider,
        build_range_provider,
        resolve_runtime,
    )
    from beamng_autopilot.config import resolve_launch_runtime

    class _FakeBng:
        def __init__(self, tech: bool) -> None:
            self._tech = tech

        def tech_enabled(self) -> bool:
            return self._tech

    class _FakeConn:
        def __init__(self, tech: bool) -> None:
            self.bng = _FakeBng(tech)

    steam = _FakeConn(False)
    tech = _FakeConn(True)
    check("runtime: auto resolves to steam without tech",
          resolve_runtime(steam, "auto") == "steam")
    check("runtime: auto resolves to tech when tech_enabled",
          resolve_runtime(tech, "auto") == "tech")
    check("runtime: explicit modes are respected",
          resolve_runtime(tech, "steam") == "steam"
          and resolve_runtime(steam, "tech") == "tech")
    check("runtime: homes select per mode",
          config.runtime_home("steam") == config.BEAMNG_HOME
          and config.runtime_home("tech") == config.BEAMNG_TECH_HOME)
    check("launch-runtime: explicit modes are respected",
          resolve_launch_runtime("steam") == "steam"
          and resolve_launch_runtime("tech") == "tech")
    orig_tech_home = config.BEAMNG_TECH_HOME
    try:
        with tempfile.TemporaryDirectory() as tmp:
            config.BEAMNG_TECH_HOME = Path(tmp)
            check("launch-runtime: auto prefers existing tech home",
                  resolve_launch_runtime("auto") == "tech"
                  and config.runtime_home("auto") == config.BEAMNG_TECH_HOME)
            config.BEAMNG_TECH_HOME = Path(tmp) / "missing"
            check("launch-runtime: auto falls back to steam",
                  resolve_launch_runtime("auto") == "steam"
                  and config.runtime_home("auto") == config.BEAMNG_HOME)
    finally:
        config.BEAMNG_TECH_HOME = orig_tech_home

    for mod in ("beamng_autopilot_tech", "beamng_autopilot_tech.providers"):
        sys.modules.pop(mod, None)
    cam, mode = build_camera_provider(steam, "auto")
    rng, rmode = build_range_provider(steam, "auto")
    check("runtime: steam providers stay in core",
          isinstance(cam, SteamCameraProvider)
          and isinstance(rng, SteamRangeProvider)
          and mode == "steam" and rmode == "steam")
    check("runtime: steam path never imports tech",
          "beamng_autopilot_tech" not in sys.modules)

    try:
        build_camera_provider(tech, "auto", 640, 480)
    except Exception:
        pass
    check("runtime: tech path lazily imports tech package",
          "beamng_autopilot_tech" in sys.modules
          and "beamng_autopilot_tech.providers" in sys.modules)


def test_creep_speed() -> None:
    # Pinned by an obstacle kinematic limit (path still exists): waits a
    # short hold, then creeps forward instead of parking forever.
    v, creep, since = creep_speed(False, 0.0, 0.0, 0.3, None, 10.0)
    check("creep: holds before delay", v == 0.0 and creep == 0 and since == 10.0)
    v, creep, since = creep_speed(False, 0.0, 0.0, 0.3, since, 12.0)
    check("creep: engages after 1.5s", v == 1.5 and creep == 1,
          f"v={v} creep={creep}")
    v, creep, since = creep_speed(False, 0.0, 0.0, 0.3, since, 13.0)
    check("creep: stays engaged", v == 1.5 and creep == 1 and since == 10.0)

    # No creep when blocked (no drivable path) - stop and hold.
    v, creep, since = creep_speed(True, 0.0, 0.0, 0.3, None, 10.0)
    check("creep: never on blocked", v == 0.0 and creep == 0 and since is None)

    # Corner-only cap (no obstacle limit) must not creep.
    v, creep, since = creep_speed(False, None, 0.0, 0.3, None, 10.0)
    check("creep: corner cap no creep", v == 0.0 and creep == 0 and since is None)

    # Obstacle limit above zero (car already has a demand) - no creep.
    v, creep, since = creep_speed(False, 0.8, 0.5, 0.3, None, 10.0)
    check("creep: positive demand no creep",
          v == 0.5 and creep == 0 and since is None)

    # Car already moving -> not pinned -> no creep.
    v, creep, since = creep_speed(False, 0.0, 0.0, 3.0, 9.0, 10.0)
    check("creep: moving no creep",
          v == 0.0 and creep == 0 and since is None)

    # State resets once the pinned condition clears.
    _, _, since = creep_speed(False, 0.0, 0.0, 0.3, 5.0, 6.0)
    v, creep, since = creep_speed(False, None, 0.0, 0.3, since, 7.0)
    check("creep: timer resets", since is None and creep == 0)


def test_speed_lon_semantics(planner: LocalPlanner) -> None:
    """speed() must only slow the car for obstacles actually ahead.

    Regression for the "twitch": a raycast box behind / beside the car
    used to project onto the route start (lon = 0), land inside the
    braking curve and pin the speed to zero every frame, so the car
    stuttered in place on an empty road.
    """

    def sp(obs):
        return planner.speed(straight_route(), obs,
                             np.array([0.0, 0.0]), 0.0, 0, 15.0)

    v, _ = sp([Obstacle(-5.0, 0.5, 1.0, 1.0)])
    check("speed: box behind the car does not limit",
          abs(v - 15.0) < 1e-6, f"v={v:.2f}")
    v, _ = sp([Obstacle(-3.0, 8.0, 1.0, 1.0)])
    check("speed: box behind and far off route does not limit",
          abs(v - 15.0) < 1e-6, f"v={v:.2f}")
    v, _ = sp([Obstacle(0.0, 3.0, 1.0, 1.0)])
    check("speed: box beside the car does not limit",
          abs(v - 15.0) < 1e-6, f"v={v:.2f}")
    # A box 2 m off the path sits just inside the pass-by corridor: the
    # car may only crawl past it, but must never be pinned to zero (that
    # zero demand every frame was the stop/creep twitch).
    v, _ = sp([Obstacle(0.0, 2.0, 1.0, 1.0)])
    check("speed: box beside the car crawls, never pins",
          abs(v - PASS_BY_MIN_MPS) < 1e-6, f"v={v:.2f}")
    v, _ = sp([Obstacle(8.0, 2.0, 1.0, 1.0)])
    check("speed: side box 8 m ahead limits but stays drivable",
          4.0 < v < 8.0, f"v={v:.2f}")
    v, _ = sp([Obstacle(45.0, 0.0, 0.9, 0.9)])
    check("speed: box beyond the 40 m window does not limit",
          abs(v - 15.0) < 1e-6, f"v={v:.2f}")
    # Real blocker 10 m ahead: kinematic braking curve (not a dead stop).
    v, _ = sp([Obstacle(10.0, 1.5, 1.0, 1.0)])
    check("speed: blocker 10 m ahead limits, not pins",
          5.0 < v < 9.0, f"v={v:.2f}")
    v, _ = sp([Obstacle(2.0, 0.0, 0.9, 0.9)])
    check("speed: blocker 2 m ahead pins to 0",
          v <= 1e-6, f"v={v:.2f}")
    # Box whose footprint reaches the car but centre is 3 m ahead: the
    # centre-arc fallback keeps a small creep demand instead of a 0.
    v, _ = sp([Obstacle(3.0, 0.0, 2.0, 2.0)])
    check("speed: box reaching the car still > 0 (arc fallback)",
          v > 1.5, f"v={v:.2f}")


def test_sharp_corner_speed(planner: LocalPlanner) -> None:
    """A corner whose total heading change is >= 45 deg is capped at 40
    km/h; a 30 deg bend keeps the curvature-based speed instead."""
    xs = np.arange(0.0, 21.0, 1.0)
    leg1 = np.column_stack([xs, np.zeros_like(xs)])
    ys = np.arange(1.0, 21.0, 1.0)
    leg2 = np.column_stack([np.full(len(ys), 20.0), ys])
    route90 = np.vstack([leg1, leg2])
    v90 = corner_speed(route90, 20, 30.0)
    check("corner: 90deg capped at 40kph",
          v90 <= 40.0 / 3.6 + 1e-9, f"v={v90:.3f}")
    v_plan, _ = planner.speed(route90, [], (0.0, 0.0), 0.0, 20, 30.0)
    check("corner: planner caps + flags sharp",
          v_plan <= 40.0 / 3.6 + 1e-9 and planner.last_sharp is True,
          f"v={v_plan:.3f} sharp={planner.last_sharp}")

    n = 61
    x = np.arange(n, dtype=float)
    y = np.where(x < 30.0, 0.0,
                 (x - 30.0) * math.tan(math.radians(30.0)))
    route30 = np.column_stack([x, y])
    a30 = corner_angle_deg(route30, 30)
    v30 = corner_speed(route30, 30, 30.0)
    check("corner: 30deg not classified sharp",
          a30 < 45.0, f"angle={a30:.1f}")
    check("corner: 30deg keeps curvature speed",
          v30 > 12.0, f"v={v30:.2f}")
    _, _ = planner.speed(route30, [], (0.0, 0.0), 0.0, 30, 30.0)
    check("corner: last_sharp false for 30deg",
          planner.last_sharp is False)


def test_right_offset_and_solid_line() -> None:
    """The planner drives on the right side and treats detected solid
    markings as no-cross boundaries."""
    from beamng_autopilot.vision.lanes import LaneMarking

    planner = LocalPlanner(right_offset=1.5, right_ramp_m=8.0)
    route = straight_route()
    drive, blocked = planner.plan(route, [], (0.0, 0.0), 0.0, 0)
    drive = np.asarray(drive, dtype=float)
    check("right: path shifted to right side",
          not blocked and len(drive) > 30
          and float(drive[30, 1]) < -1.2,
          (f"y30={float(drive[30, 1]):.2f}"
           if len(drive) > 30 else "short path"))
    check("right: no jump at start",
          len(drive) > 0 and abs(float(drive[0, 1])) < 0.05,
          f"y0={float(drive[0, 1]):.2f}")

    # A solid line on the left boundary: the right-hand path must stay on
    # its own side, clear of the paint.
    left_line = LaneMarking(
        world=np.array([[0.0, 1.5], [60.0, 1.5]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=1.0)
    drive2, blocked2 = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, solid_lines=[left_line])
    drive2 = np.asarray(drive2, dtype=float)
    max_y = float(drive2[10:, 1].max()) if len(drive2) > 10 else 99.0
    check("solid: stays right of a left-boundary line",
          not blocked2 and len(drive2) > 30
          and all(float(y) < 0.25 for y in drive2[10:, 1]),
          f"blocked={blocked2} max_y={max_y:.2f}")

    # A line inside the right lane that the offset would cross: the path
    # must be declared blocked instead of pressing the paint.
    bad_line = LaneMarking(
        world=np.array([[0.0, -0.6], [60.0, -0.6]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=1.0)
    drive3, blocked3 = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, solid_lines=[bad_line])
    check("solid: crossing a right-side line blocks",
          blocked3 and getattr(planner, "last_blocker", None) is not None
          and planner.last_blocker[0] == "solid line"
          and planner.last_blocker[1] >= 3.0,
          f"blocked={blocked3} blocker={getattr(planner, 'last_blocker', None)}")


def test_solid_line_anchor_ambiguity() -> None:
    """A pose error of a few cm on top of a line must not block the car.

    Regression for run 49: the car sat almost exactly on a detected solid
    line, so the anchor's sign chose the wrong "allowed side" and the
    legal right-hand path was reported as a crossing.  When the anchor is
    closer than ``SOLID_ANCHOR_NEAR_M`` the nearby path majority decides.
    """
    from beamng_autopilot.vision.lanes import LaneMarking

    planner = LocalPlanner(right_offset=1.5, right_ramp_m=8.0)
    route = straight_route()
    edge = LaneMarking(
        world=np.array([[0.0, -0.02], [60.0, -0.02]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=1.0)
    drive, blocked = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, solid_lines=[edge])
    drive = np.asarray(drive, dtype=float)
    check("solid-anchor: near-line path stays on its own side",
          not blocked and len(drive) > 30,
          f"blocked={blocked} n={len(drive)}")
    check("solid-anchor: right offset preserved",
          len(drive) > 30 and float(drive[30, 1]) < -1.2,
          f"y30={float(drive[30, 1]):.2f}" if len(drive) > 30 else "short")


def test_solid_line_low_confidence_gate() -> None:
    """A shaky lane frame must nudge around a line, never park the car.

    Regression for runs 49/50: low-confidence mirror-fallback lanes kept
    triggering "solid line blocked" full stops near the goal.  A crossing
    only becomes a legal stop when the lane frame itself is confident.
    """
    from beamng_autopilot.vision.lanes import LaneMarking

    planner = LocalPlanner(right_offset=1.5, right_ramp_m=8.0)
    route = straight_route()
    line = LaneMarking(
        world=np.array([[0.0, -0.6], [60.0, -0.6]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=1.0)
    xs = np.arange(0.0, 16.0, 1.5)
    center = np.column_stack([xs, np.full_like(xs, -1.5)])

    weak = LaneFrame(center=center, width=3.5, confidence=0.4,
                     span_m=15.0, sources=("vision",))
    _, blocked_weak = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, solid_lines=[line],
        sensor_lane=weak)
    check("solid-gate: weak lane frame does not block",
          not blocked_weak, f"blocked={blocked_weak}")

    from beamng_autopilot.planner import _clamp_to_solid_lines

    path = np.column_stack([np.arange(0.0, 60.0, 1.0),
                            np.linspace(0.0, -1.5, 60)])
    _, crossed_on, _ = _clamp_to_solid_lines(
        path, [line], (0.0, 0.0), corridor=None, allow_block=True)
    _, crossed_off, _ = _clamp_to_solid_lines(
        path, [line], (0.0, 0.0), corridor=None, allow_block=False)
    check("solid-gate: blocking allowed when lane frame is confident",
          crossed_on, f"crossed={crossed_on}")
    check("solid-gate: blocking suppressed when lane frame is shaky",
          not crossed_off, f"crossed={crossed_off}")

    # Regression for run 66: a single-edge vision mirror (0.56, the old
    # confidence threshold) must nudge, never stop, when the camera also
    # sees a solid line 14 m ahead.
    mirror = LaneFrame(center=center, width=3.5, confidence=0.56,
                       span_m=15.0, sources=("vision",), paired=False)
    _, blocked_mirror = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, solid_lines=[line],
        sensor_lane=mirror)
    check("solid-gate: unpaired mirror does not block",
          not blocked_mirror, f"blocked={blocked_mirror}")

    _, crossed_paired, _ = _clamp_to_solid_lines(
        path, [line], (0.0, 0.0), corridor=None, allow_block=True)
    check("solid-gate: paired lane still permits a stop",
          crossed_paired, f"crossed={crossed_paired}")


def test_solid_line_detour_gate() -> None:
    """A detour that crosses a solid line must stop, not lane-change.

    The low-confidence gate above protects ordinary follow/deform paths
    from false solid-line stops, but an A*/bypass detour deliberately
    leaves the current lane.  Crossing a validated solid boundary is a
    rule violation even when the lane frame is only a shaky mirror, so
    the planner must reject that detour and stop before the obstacle.
    """
    from beamng_autopilot.vision.lanes import LaneMarking

    planner = LocalPlanner(right_offset=1.5, right_ramp_m=8.0)
    route = straight_route()
    box = Obstacle(x=10.0, y=0.0, half_w=4.6, half_h=2.2,
                   category="vehicle", label="car")
    xs = np.arange(0.0, 16.0, 1.5)
    center = np.column_stack([xs, np.zeros_like(xs)])
    mirror = LaneFrame(center=center, width=3.5, confidence=0.56,
                       span_m=15.0, sources=("vision",), paired=False)

    _, blocked_free = planner.plan(
        route, [box], (0.0, 0.0), 0.0, 0, sensor_lane=mirror)
    mode_free = planner.last_mode
    check("solid-detour: bypass without line is a detour",
          mode_free == "detour" and not blocked_free,
          f"mode={mode_free} blocked={blocked_free}")

    line = LaneMarking(
        world=np.array([[0.0, -1.5], [60.0, -1.5]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=1.0)
    drive, blocked = planner.plan(
        route, [box], (0.0, 0.0), 0.0, 0,
        solid_lines=[line], sensor_lane=mirror)
    drive = np.asarray(drive, dtype=float)
    check("solid-detour: crossing a solid line blocks the detour",
          blocked and planner.last_mode == "blocked"
          and getattr(planner, "last_blocker", None) is not None
          and planner.last_blocker[0] == "solid line",
          f"blocked={blocked} mode={planner.last_mode} "
          f"blocker={getattr(planner, 'last_blocker', None)}")
    check("solid-detour: truncated path stays before the crossing",
          len(drive) >= 2 and float(np.max(np.abs(drive[:, 1]))) < 1.45,
          f"max_y={float(np.max(np.abs(drive[:, 1]))):.2f}")


def test_vision_marking_primary_over_nav_route() -> None:
    """A single painted edge protects, but does not replace, the lane offset.

    One line cannot prove where the opposite lane edge is, so the legal-lane
    offset from the nav route stays the driving reference; the marking only
    pushes the path away when the route is already too close to it.
    """
    planner = LocalPlanner(right_offset=1.5, right_ramp_m=8.0)
    route = straight_route()
    xs = np.arange(0.0, 16.0, 1.0)
    center = np.column_stack([xs, np.zeros_like(xs)])
    left = np.column_stack([xs, np.full_like(xs, 1.75)])
    lane = LaneFrame(center=center, left=left, right=None,
                     width=3.5, confidence=0.5, span_m=15.0,
                     sources=("vision",), paired=False)
    drive, blocked = planner.plan(route, [], (0.0, 0.0), 0.0, 0,
                                  sensor_lane=lane)
    drive = np.asarray(drive, dtype=float)
    check("marking-primary: single vision edge keeps the lane offset",
          not blocked and len(drive) > 10
          and -1.8 <= float(np.median(drive[5:, 1])) <= -1.2
          and 1.2 <= getattr(planner, "last_lane_offset", 0.0) <= 1.8
          and getattr(planner, "last_lane_mode", "") == "vision_left",
          f"blocked={blocked} n={len(drive)} "
          f"y_med={float(np.median(drive[5:, 1])):.2f} "
          f"mode={getattr(planner, 'last_lane_mode', '')} "
          f"off={getattr(planner, 'last_lane_offset', 0.0):.2f}")


def test_curve_right_offset_uses_local_tangent(planner: LocalPlanner) -> None:
    """Right offset follows the route tangent through a bend.

    Regression for the italy run-8 failure: shifting the whole planning
    window by the car's current heading pushed the path to the outside of
    a 30+ degree bend (and later onto the wrong side of the route), so
    pure pursuit drove the car into the wall.
    """
    route = curve_route()
    i0 = 20
    out = planner._right_offset_path(route, i0, 0.0, offset=1.5)
    lats = [_point_lat_offset(float(x), float(y), route)
            for x, y in out[i0 + 30:]]
    check("curve-right: offset stays right through the bend",
          len(lats) >= 20 and 1.25 <= float(np.median(lats)) <= 1.75,
          f"median={float(np.median(lats)):.2f} n={len(lats)}")
    check("curve-right: no point drifts back to the route centre",
          all(v > 0.9 for v in lats),
          f"min={min(lats):.2f}")


def test_solid_line_noise_filter() -> None:
    """Short / faint / kerb-like markings must not block the planner.

    The live camera frequently sees paint chips, kerbs and pavement edges
    as short or low-confidence "solid" segments.  Only a long, confident
    marking aligned with the driving corridor may act as a no-cross line.
    """
    from beamng_autopilot.vision.lanes import LaneMarking

    planner = LocalPlanner(right_offset=1.5, right_ramp_m=8.0)
    route = straight_route()

    def plan_with(mk):
        drive, blocked = planner.plan(
            route, [], (0.0, 0.0), 0.0, 0, solid_lines=[mk])
        return np.asarray(drive, dtype=float), blocked

    # A short 4 m solid blob in the right lane (typical kerb / paint chip).
    short = LaneMarking(
        world=np.array([[0.0, -0.6], [4.0, -0.6]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=0.95)
    _, blocked_short = plan_with(short)
    check("solid-noise: short marking does not block", not blocked_short,
          f"blocked={blocked_short}")

    # A long faint edge: confidence below the road-line threshold.
    faint = LaneMarking(
        world=np.array([[0.0, -0.6], [60.0, -0.6]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=0.2)
    _, blocked_faint = plan_with(faint)
    check("solid-noise: faint marking does not block", not blocked_faint,
          f"blocked={blocked_faint}")

    # A kerb 9 m off the corridor: geometrically it is not the road boundary
    # the car is driving next to.
    distant = LaneMarking(
        world=np.array([[0.0, -9.0], [60.0, -9.0]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=1.0)
    _, blocked_distant = plan_with(distant)
    check("solid-noise: distant line does not block", not blocked_distant,
          f"blocked={blocked_distant}")

    # A crosswalk stripe cutting across the road is not a lane boundary.
    cross = LaneMarking(
        world=np.array([[30.0, -6.0], [30.0, 6.0]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=1.0)
    _, blocked_cross = plan_with(cross)
    check("solid-noise: crossing stripe does not block", not blocked_cross,
          f"blocked={blocked_cross}")

    # A long diagonal pavement block that sweeps 8+ m laterally while
    # staying on one side of the road is not a lane boundary either.
    broad = LaneMarking(
        world=np.array([[0.0, -9.0], [4.0, -7.0], [8.0, -5.0],
                        [12.0, -2.5], [16.0, -0.5]], dtype=float),
        pixels=np.zeros((5, 2)), color="white", kind="solid",
        confidence=1.0)
    _, blocked_broad = plan_with(broad)
    check("solid-noise: broad roadside block does not block",
          not blocked_broad, f"blocked={blocked_broad}")

    # Run 80 saw a 30 m repair scar / dark patch along the road.  It is
    # long, confident and aligned with the corridor, but it is a 0.7 m+
    # wide ribbon in world space, not a painted lane line.  It must not
    # turn the car into a full stop (or later drag it toward the wall).
    scar_x = np.arange(0.0, 31.0, 3.0)
    scar = LaneMarking(
        world=np.column_stack(
            [scar_x, -1.5 + 0.7 * np.sin(0.3 * scar_x)]),
        pixels=np.zeros((len(scar_x), 2)), color="white", kind="solid",
        confidence=1.0)
    _, blocked_scar = plan_with(scar)
    check("solid-noise: run80 repair scar does not block",
          not blocked_scar, f"blocked={blocked_scar}")

    # A back-projected zig-zag with little net span must never be treated
    # as an 8 m solid line even when its summed polyline is long.
    zig = LaneMarking(
        world=np.array([[0.0, -1.0], [1.0, -3.0], [2.0, -1.0],
                        [3.0, -3.0], [4.0, -1.0], [5.0, -3.0],
                        [6.0, -1.0], [7.0, -3.0]], dtype=float),
        pixels=np.zeros((8, 2)), color="white", kind="solid",
        confidence=1.0)
    _, blocked_zig = plan_with(zig)
    check("solid-noise: zig-zag blob does not block",
          not blocked_zig, f"blocked={blocked_zig}")

    # The Italian pavement texture seen live at 80 m: an 11 m span that
    # zig-zags into a 26 m polyline.  It must never become a no-cross
    # boundary even though it is long and confident.
    lane8_x = np.linspace(0.0, 11.6, 37)
    lane8_zig = LaneMarking(
        world=np.column_stack(
            [lane8_x, -1.5 - 1.2 * np.sin(2.7 * lane8_x)]),
        pixels=np.zeros((len(lane8_x), 2)), color="white", kind="solid",
        confidence=1.0)
    _, blocked_lane8 = plan_with(lane8_zig)
    check("solid-noise: 11 m pavement zig-zag does not block",
          not blocked_lane8, f"blocked={blocked_lane8}")

    # A long, confident line that the nav route itself crosses is kerb or
    # track paint, not a boundary the route is supposed to stay beside.
    slanted = LaneMarking(
        world=np.array([[0.0, -1.0], [60.0, 2.0]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=1.0)
    _, blocked_slanted = plan_with(slanted)
    check("solid-noise: route-crossing line does not block",
          not blocked_slanted, f"blocked={blocked_slanted}")

    # The real long, confident right-side boundary still blocks a crossing.
    real = LaneMarking(
        world=np.array([[0.0, -0.6], [60.0, -0.6]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=1.0)
    _, blocked_real = plan_with(real)
    check("solid-noise: genuine boundary still blocks",
          blocked_real and planner.last_blocker[0] == "solid line",
          f"blocked={blocked_real}")

    # A line whose crossing sits under / immediately in front of the car
    # is treated as a stale or noisy marking, not a legal stop point.
    near_line = LaneMarking(
        world=np.array([[0.0, -0.15], [60.0, -0.15]], dtype=float),
        pixels=np.zeros((2, 2)), color="white", kind="solid",
        confidence=1.0)
    _, blocked_near = plan_with(near_line)
    check("solid-noise: line under the car does not block",
          not blocked_near and planner.last_blocker is None,
          f"blocked={blocked_near} blocker={planner.last_blocker}")


def test_lane_detector() -> None:
    """The classic-CV lane detector finds a white road marking and gives
    it a world-space polyline plus a solid/dashed kind."""
    import cv2

    from beamng_autopilot.vision.lanes import (
        LaneDetector,
        _kind_for,
        _row_core_points,
        _thin_line_ok,
    )
    from beamng_autopilot.vision.projection import default_camera

    check("lane: zig-zag long blob not solid",
          _kind_for(1.2, 7.0, 6.0) == "unknown",
          f"kind={_kind_for(1.2, 7.0, 6.0)}")
    check("lane: 11 m pavement jitter not solid",
          _kind_for(11.6, 26.7, 6.0) == "unknown",
          f"kind={_kind_for(11.6, 26.7, 6.0)}")
    check("lane: straight net span solid",
          _kind_for(12.0, 13.0, 6.0) == "solid",
          f"kind={_kind_for(12.0, 13.0, 6.0)}")
    core, widths = _row_core_points(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        np.array([10, 11, 12, 10, 11, 12, 10, 11, 12]))
    check("lane: row core collapses a thin stripe",
          core.shape == (3, 2)
          and core[:, 0].tolist() == [11.0, 11.0, 11.0]
          and widths.tolist() == [3.0, 3.0, 3.0],
          f"core={core.tolist()} widths={widths.tolist()}")
    check("lane: thin core accepted as thin line",
          _thin_line_ok(5.0, 5.2, 0.4, 3.0, 3.0))
    check("lane: wide patch core rejected by pixel width",
          not _thin_line_ok(5.0, 5.2, 0.4, 47.0, 90.0))
    check("lane: wide world core rejected",
          not _thin_line_ok(5.0, 5.2, 3.0, 3.0, 3.0))

    cam = default_camera(320, 240)
    frame = np.zeros((240, 320, 3), np.uint8)
    # A thin vertical line, not a filled blob: a blob back-projects as a
    # wavy 2.8 m span and is correctly noise after the jitter tightening.
    cv2.line(frame, (150, 170), (150, 230), (255, 255, 255), 2)
    det = LaneDetector(min_area=50, min_height=20)
    marks = det.detect(frame, cam, (0.0, 0.0, 0.0), 0.0, ground_z=0.0)
    check("lane: white marking detected", len(marks) >= 1,
          f"n={len(marks)}")
    if marks:
        m = marks[0]
        check("lane: classified solid/dashed",
              m.kind in ("solid", "dashed"), f"kind={m.kind}")
        check("lane: white color", m.color == "white",
              f"color={m.color}")
        check("lane: world polyline",
              m.world.ndim == 2 and len(m.world) >= 4,
              f"world={None if m.world is None else m.world.shape}")

    patch_frame = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(patch_frame, (120, 150), (180, 230),
                  (255, 255, 255), -1)
    patch_marks = det.detect(patch_frame, cam, (0.0, 0.0, 0.0), 0.0,
                             ground_z=0.0)
    check("lane: wide dark patch is never a lane line",
          not any(m.kind in ("solid", "dashed", "thin")
                  for m in patch_marks),
          f"kinds={[m.kind for m in patch_marks]}")


def test_lane_pairing_and_lidar_corridor() -> None:
    """Markings pair into a lane centre; a single side mirrors; the raw
    raycast fan builds a free-space corridor; vision is preferred."""
    from beamng_autopilot.vision.lanes import LaneMarking

    xs = np.arange(0.0, 16.0, 1.0)
    left = LaneMarking(
        world=np.column_stack([xs, np.full_like(xs, 1.75)]),
        pixels=np.zeros((len(xs), 2)), color="white", kind="solid",
        confidence=0.9)
    right = LaneMarking(
        world=np.column_stack([xs, np.full_like(xs, -1.75)]),
        pixels=np.zeros((len(xs), 2)), color="white", kind="solid",
        confidence=0.9)
    frame = pair_lane_markings([left, right], (0.0, 0.0), 0.0)
    check("lane-pair: frame produced", frame is not None)
    if frame is not None:
        y = float(np.median(frame.center[:, 1]))
        check("lane-pair: centre on the middle", abs(y) < 0.05,
              f"y={y:.2f}")
        check("lane-pair: width ~3.5 m",
              3.3 <= frame.width <= 3.7, f"width={frame.width:.2f}")
        check("lane-pair: long/confident frame",
              frame.span_m >= 12.0 and frame.confidence >= 0.45,
              f"span={frame.span_m:.1f} conf={frame.confidence:.2f}")

    only_left = pair_lane_markings([left], (0.0, 0.0), 0.0)
    check("lane-mirror: single side produces a frame",
          only_left is not None)
    if only_left is not None:
        y = float(np.median(only_left.center[:, 1]))
        check("lane-mirror: centre mirrored at 3.5 m", abs(y) < 0.05,
              f"y={y:.2f}")

    run84_left = LaneMarking(
        world=np.column_stack([np.arange(8.0, 12.2, 0.5),
                               np.full(9, 3.7)]),
        pixels=np.zeros((9, 2)), color="white", kind="dashed",
        confidence=0.9)
    run84_right = LaneMarking(
        world=np.column_stack([np.arange(6.0, 13.6, 0.5),
                               np.full(16, -2.09)]),
        pixels=np.zeros((16, 2)), color="white", kind="solid",
        confidence=0.95)
    r84dbg: dict = {}
    run84 = pair_lane_markings([run84_left, run84_right],
                               (0.0, 0.0), 0.0, debug=r84dbg)
    check("lane-pair: run84 far centre line + right solid pair",
          run84 is not None and run84.paired
          and r84dbg.get("mode") == "pair",
          f"mode={r84dbg.get('mode')}")
    if run84 is not None:
        check("lane-pair: run84 far pair is low-confidence",
              run84.confidence <= LANE_FAR_CENTER_PAIR_CONF_MAX + 1e-9,
              f"conf={run84.confidence:.2f}")
        check("lane-pair: run84 debug marks far pair",
              r84dbg.get("far_center_pair") is True,
              f"far={r84dbg.get('far_center_pair')}")
        check("lane-pair: run84 midpoint between lines",
              abs(float(np.median(run84.center[:, 1])) - 0.8) < 0.2
              and abs(run84.width - 5.79) < 0.25,
              f"center={float(np.median(run84.center[:, 1])):.2f} "
              f"width={run84.width:.2f}")

    thin_left = LaneMarking(
        world=np.column_stack([xs, np.full_like(xs, 1.75)]),
        pixels=np.zeros((len(xs), 2)), color="white", kind="thin",
        confidence=0.7)
    tdbg: dict = {}
    thin_frame = pair_lane_markings([thin_left], (0.0, 0.0), 0.0,
                                    debug=tdbg)
    check("lane-mirror: thin centre line mirrors",
          thin_frame is not None and not thin_frame.paired
          and tdbg.get("mode") == "mirror_left",
          f"mode={tdbg.get('mode')}")
    thin_hits = [(float(s) / 10.0, -3.5) for s in range(0, 130, 2)]
    thin_lidar = build_lidar_corridor(thin_hits, (0.0, 0.0), 0.0)
    thin_fused = choose_sensor_lane(thin_frame, thin_lidar,
                                    (0.0, 0.0), 0.0)
    ty = None if thin_fused is None else float(
        np.median(thin_fused.center[:, 1]))
    check("lane-fusion: thin centre line + right wall midpoint",
          thin_fused is not None and thin_fused.paired
          and thin_fused.sources == ("vision", "lidar")
          and ty is not None and abs(ty + 0.875) < 0.2,
          f"src={None if thin_fused is None else thin_fused.sources} "
          f"y={-9.0 if ty is None else ty:.2f}")

    hits = []
    for s in range(0, 200, 2):
        hits.append((float(s) / 10.0, 1.75))
        hits.append((float(s) / 10.0, -1.75))
    lidar = build_lidar_corridor(hits, (0.0, 0.0), 0.0)
    check("lidar: corridor produced", lidar is not None)
    if lidar is not None:
        y = float(np.median(lidar.center[:, 1]))
        check("lidar: corridor centre on the middle", abs(y) < 0.05,
              f"y={y:.2f}")
        check("lidar: corridor width ~3.5 m",
              3.3 <= lidar.width <= 3.7, f"width={lidar.width:.2f}")

    # A wall / guardrail seen on both sides but only intermittently on
    # one side must still become a real two-sided corridor: the missing
    # stations are interpolated and the centre stays on the midpoint.
    sparse_hits = []
    for s in range(0, 130, 2):
        sparse_hits.append((float(s) / 10.0, -3.5))
    for s in range(20, 90, 5):
        sparse_hits.append((float(s) / 10.0, 1.75))
    dbg_sparse: dict = {}
    sparse = build_lidar_corridor(
        sparse_hits, (0.0, 0.0), 0.0, debug=dbg_sparse)
    check("lidar: sparse two-sided corridor built",
          sparse is not None and sparse.paired,
          f"paired={None if sparse is None else sparse.paired}")
    if sparse is not None:
        y = float(np.median(sparse.center[:, 1]))
        check("lidar: sparse corridor centred on the midpoint",
              -1.2 <= y <= -0.55, f"y={y:.2f} width={sparse.width:.2f}")

    # Open road with only walls: the two-sided corridor fails and the
    # one-sided fallback must use the right wall first (right-hand
    # traffic), never a far left wall that is beyond the edge window.
    # A far guardrail is only a boundary hint, so the mirrored centre is
    # clipped to a small offset instead of being pulled a lane width away.
    one_sided_hits = []
    for s in range(0, 130, 2):
        one_sided_hits.append((float(s) / 10.0, -4.1))
        one_sided_hits.append((float(s) / 10.0, 6.3))
    dbg: dict = {}
    fallback = build_lidar_corridor(
        one_sided_hits, (0.0, 0.0), 0.0, debug=dbg)
    check("lidar-fallback: right edge preferred over far left wall",
          fallback is not None and dbg.get("fallback") == "right",
          f"fallback={dbg.get('fallback')}")
    if fallback is not None:
        y = float(np.median(fallback.center[:, 1]))
        check("lidar-fallback: right wall stays a small centring hint",
              -0.40 <= y <= -0.30, f"y={y:.2f}")
        check("lidar-fallback: weak single-edge frame",
              not fallback.paired and fallback.confidence <= 0.65,
              f"paired={fallback.paired} conf={fallback.confidence:.2f}")

    left_only_hits = [(float(s) / 10.0, 4.0)
                      for s in range(0, 130, 2)]
    dbg2: dict = {}
    left_fallback = build_lidar_corridor(
        left_only_hits, (0.0, 0.0), 0.0, debug=dbg2)
    check("lidar-fallback: left edge used when no right edge",
          left_fallback is not None and dbg2.get("fallback") == "left",
          f"fallback={dbg2.get('fallback')}")
    if left_fallback is not None:
        y = float(np.median(left_fallback.center[:, 1]))
        check("lidar-fallback: left wall stays a small centring hint",
              0.30 <= y <= 0.40, f"y={y:.2f}")
    chosen = choose_sensor_lane(frame, lidar)
    check("sensor-choose: vision preferred",
          chosen is not None and chosen.sources[0] == "vision",
          f"src={None if chosen is None else chosen.sources}")

    weak_xs = np.arange(0.0, 12.0, 1.5)
    weak_vis = LaneFrame(
        center=np.column_stack([weak_xs, np.zeros_like(weak_xs)]),
        width=3.5, confidence=0.32, span_m=12.0, sources=("vision",),
        paired=False)
    weak_chosen = choose_sensor_lane(weak_vis, lidar)
    check("sensor-choose: weak vision defers to lidar",
          weak_chosen is not None and weak_chosen.sources[0] == "lidar",
          f"src={None if weak_chosen is None else weak_chosen.sources}")
    weak_only = choose_sensor_lane(weak_vis, None)
    check("sensor-choose: weak mirror without lidar is refused",
          weak_only is None,
          f"src={None if weak_only is None else weak_only.sources}")

    weak_boundary_vis = LaneFrame(
        center=np.column_stack(
            [weak_xs, np.full_like(weak_xs, 0.0)]),
        left=np.column_stack(
            [weak_xs, np.full_like(weak_xs, 1.75)]),
        width=3.5, confidence=0.32, span_m=12.0, sources=("vision",),
        paired=False)
    weak_boundary_chosen = choose_sensor_lane(
        weak_boundary_vis, lidar, (0.0, 0.0), 0.0)
    check("sensor-choose: any painted boundary beats lidar",
          weak_boundary_chosen is not None
          and weak_boundary_chosen.sources[0] == "vision",
          f"src={None if weak_boundary_chosen is None else weak_boundary_chosen.sources}")

    shaky_mirror = LaneFrame(
        center=np.column_stack(
            [weak_xs, np.full_like(weak_xs, -2.3)]),
        width=3.5, confidence=0.37, span_m=3.0, sources=("vision",),
        paired=False)
    shaky_only = choose_sensor_lane(shaky_mirror, None)
    check("sensor-choose: run61 shaky mirror is refused",
          shaky_only is None,
          f"src={None if shaky_only is None else shaky_only.sources}")

    trusted_vis = LaneFrame(
        center=np.column_stack([weak_xs, np.full_like(weak_xs, -0.25)]),
        left=np.column_stack(
            [weak_xs, np.full_like(weak_xs, 1.5)]),
        width=3.5, confidence=0.5, span_m=14.0, sources=("vision",),
        paired=False)
    trusted_chosen = choose_sensor_lane(trusted_vis, fallback)
    check("sensor-choose: trusted vision mirror beats lidar fallback",
          trusted_chosen is not None
          and trusted_chosen.sources[0] == "vision",
          f"src={None if trusted_chosen is None else trusted_chosen.sources}")
    trusted_only = choose_sensor_lane(trusted_vis, None)
    check("sensor-choose: trusted mirror without lidar is kept",
          trusted_only is not None
          and trusted_only.sources[0] == "vision",
          f"src={None if trusted_only is None else trusted_only.sources}")

    far_mirror = LaneFrame(
        center=np.column_stack(
            [weak_xs, np.full_like(weak_xs, -1.5)]),
        width=3.5, confidence=0.5, span_m=15.0, sources=("vision",),
        paired=False)
    far_only = choose_sensor_lane(far_mirror, None, (0.0, 0.0), 0.0)
    check("sensor-choose: far-centre mirror without lidar refused",
          far_only is None,
          f"src={None if far_only is None else far_only.sources}")

    far_start_right = LaneFrame(
        center=np.column_stack(
            [np.arange(9.0, 17.0, 1.0),
             np.full(8, -1.25)]),
        right=np.column_stack(
            [np.arange(9.0, 17.0, 1.0),
             np.full(8, -3.0)]),
        width=3.5, confidence=0.42, span_m=7.0,
        sources=("vision",), paired=False)
    far_start_only = choose_sensor_lane(
        far_start_right, None, (0.0, 0.0), 0.0)
    check("sensor-choose: far-start right paint alone does not steer",
          far_start_only is None,
          f"src={None if far_start_only is None else far_start_only.sources}")

    disagree_vis = LaneFrame(
        center=np.column_stack(
            [weak_xs, np.full_like(weak_xs, 2.0)]),
        width=3.5, confidence=0.5, span_m=14.0, sources=("vision",),
        paired=False)
    disagree_chosen = choose_sensor_lane(
        disagree_vis, fallback, (0.0, 0.0), 0.0)
    check("sensor-choose: disagreeing vision mirror defers to lidar",
          disagree_chosen is not None
          and disagree_chosen.sources[0] == "lidar",
          f"src={None if disagree_chosen is None else disagree_chosen.sources}")

    weak_pair = LaneFrame(
        center=np.column_stack([weak_xs, np.zeros_like(weak_xs)]),
        width=3.5, confidence=0.32, span_m=12.0, sources=("vision",),
        paired=True)
    weak_pair_chosen = choose_sensor_lane(weak_pair, lidar)
    check("sensor-choose: weak paired vision still preferred",
          weak_pair_chosen is not None
          and weak_pair_chosen.sources[0] == "vision",
          f"src={None if weak_pair_chosen is None else weak_pair_chosen.sources}")

    # Markings that extend behind the car must not push the lane centre
    # behind the vehicle: the planner would otherwise steer from a point
    # that is not ahead of the car.
    behind_xs = np.arange(-4.0, 16.0, 1.0)
    behind = pair_lane_markings([
        LaneMarking(world=np.column_stack(
            [behind_xs, np.full_like(behind_xs, 1.75)]),
            pixels=np.zeros((len(behind_xs), 2)), color="white",
            kind="solid", confidence=0.9),
        LaneMarking(world=np.column_stack(
            [behind_xs, np.full_like(behind_xs, -1.75)]),
            pixels=np.zeros((len(behind_xs), 2)), color="white",
            kind="solid", confidence=0.9),
    ], (0.0, 0.0), 0.0)
    if behind is not None:
        check("lane-ahead: centre starts at the car",
              float(np.min(behind.center[:, 0])) >= -0.01,
              f"x0={float(np.min(behind.center[:, 0])):.2f}")

    wide_hits = []
    for s in range(0, 200, 2):
        wide_hits.append((float(s) / 10.0, 5.0))
        wide_hits.append((float(s) / 10.0, -5.0))
    wide_lidar = build_lidar_corridor(wide_hits, (0.0, 0.0), 0.0)
    check("sensor-choose: whole-road lidar corridor rejected",
          wide_lidar is not None
          and choose_sensor_lane(None, wide_lidar) is None,
          f"width={None if wide_lidar is None else wide_lidar.width:.1f}")

    wv_xs = np.arange(0.0, 16.0, 1.5)
    wide_vision = LaneFrame(
        center=np.column_stack([wv_xs, np.zeros_like(wv_xs)]),
        width=8.5, confidence=0.8,
                            span_m=15.0, sources=("vision",))
    check("sensor-choose: over-wide vision frame rejected",
          choose_sensor_lane(wide_vision, None) is None,
          f"width={wide_vision.width:.1f}")


def test_lane_boundary_guards() -> None:
    """Far, short or over-wide markings must not drag the lane centre away.

    Regression for run 40: a solid line 11-14 m away was picked as the
    right boundary, which mirrored the centre to -8.7 m.  Far lines are
    now rejected, short dashed centre-line segments still pair, and an
    over-wide pair falls back to a single-side mirror.
    """
    from beamng_autopilot.vision.lanes import LaneMarking

    far_xs = np.arange(0.0, 12.0, 1.0)
    far_left = LaneMarking(
        world=np.column_stack([far_xs, np.full_like(far_xs, 8.0)]),
        pixels=np.zeros((len(far_xs), 2)), color="white", kind="solid",
        confidence=0.9)
    far = pair_lane_markings([far_left], (0.0, 0.0), 0.0)
    check("lane-guard: 8 m far marking rejected", far is None)

    short_xs = np.linspace(0.0, 3.2, 5)
    left_short = LaneMarking(
        world=np.column_stack([short_xs, np.full_like(short_xs, 1.75)]),
        pixels=np.zeros((len(short_xs), 2)), color="white", kind="dashed",
        confidence=0.9)
    right_short = LaneMarking(
        world=np.column_stack([short_xs, np.full_like(short_xs, -1.75)]),
        pixels=np.zeros((len(short_xs), 2)), color="white", kind="solid",
        confidence=0.9)
    short = pair_lane_markings([left_short, right_short], (0.0, 0.0), 0.0)
    check("lane-guard: 3.2 m dashed+solid pair accepted",
          short is not None)
    if short is not None:
        y = float(np.median(short.center[:, 1]))
        check("lane-guard: short pair centre on the middle",
              abs(y) < 0.05, f"y={y:.2f}")
    check("lane-guard: short pair frame usable",
          lane_frame_usable(short),
          f"conf={short.confidence:.2f} span={short.span_m:.1f} "
          f"w={short.width:.2f}")

    # Regression for run 57: two markings that only start 9-15 m ahead
    # (no near points within 6 m) must not pair into a lane; a far pair
    # was mistaken for the current lane and pushed the car across a solid
    # line.  The mirror fallback also refuses lines with no near points.
    far_pair_xs = np.arange(9.0, 16.0, 1.0)
    far_pair = pair_lane_markings([
        LaneMarking(world=np.column_stack(
            [far_pair_xs, np.full_like(far_pair_xs, 1.75)]),
            pixels=np.zeros((len(far_pair_xs), 2)), color="white",
            kind="solid", confidence=0.9),
        LaneMarking(world=np.column_stack(
            [far_pair_xs, np.full_like(far_pair_xs, -1.75)]),
            pixels=np.zeros((len(far_pair_xs), 2)), color="white",
            kind="solid", confidence=0.9),
    ], (0.0, 0.0), 0.0)
    check("lane-guard: pair with no near points rejected",
          far_pair is not None and not far_pair.paired)
    if far_pair is not None:
        check("lane-guard: far no-near pair stays low trust",
              far_pair.confidence <= LANE_FAR_MIRROR_CONF_MAX + 1e-9,
              f"conf={far_pair.confidence:.2f}")

    unknown = LaneMarking(
        world=np.column_stack([short_xs, np.full_like(short_xs, 1.75)]),
        pixels=np.zeros((len(short_xs), 2)), color="white", kind="unknown",
        confidence=0.9)
    unk = pair_lane_markings([unknown], (0.0, 0.0), 0.0)
    check("lane-guard: short unknown blob still rejected", unk is None)

    # Regression for run 69's typical frame: the right solid line is
    # reliable but starts a few metres ahead, while a short unknown blob
    # starts behind the car.  The right line must still become the single
    # boundary instead of being shadowed by the invalid left blob.
    run69_right_xs = np.arange(3.7, 18.0, 0.5)
    run69_right = LaneMarking(
        world=np.column_stack(
            [run69_right_xs, np.full_like(run69_right_xs, -0.9)]),
        pixels=np.zeros((len(run69_right_xs), 2)), color="white",
        kind="solid", confidence=0.9)
    run69_left = LaneMarking(
        world=np.column_stack(
            [np.linspace(-1.9, 2.6, 6), np.full(6, 2.72)]),
        pixels=np.zeros((6, 2)), color="white", kind="unknown",
        confidence=0.6)
    run69_dbg: dict = {}
    run69 = pair_lane_markings(
        [run69_left, run69_right], (0.0, 0.0), 0.0, debug=run69_dbg)
    run69_ry = None if run69 is None or run69.right is None else float(
        np.median(run69.right[:, 1]))
    check("lane-guard: run69 right line survives short left blob",
          run69 is not None and not run69.paired
          and run69_dbg.get("mode") == "mirror_right"
          and run69_ry is not None and abs(run69_ry + 0.9) < 0.1,
          f"mode={run69_dbg.get('mode')} "
          f"right={-9.0 if run69_ry is None else run69_ry:.2f}")

    # A right paint that only becomes visible ahead is still the primary
    # boundary, but it must be low trust so the planner only nudges the
    # path toward the lane centre instead of hugging the roadside.
    far_right_xs = np.arange(9.0, 17.0, 0.5)
    far_right = LaneMarking(
        world=np.column_stack(
            [far_right_xs, np.full_like(far_right_xs, -3.0)]),
        pixels=np.zeros((len(far_right_xs), 2)), color="white",
        kind="solid", confidence=0.9)
    far_dbg: dict = {}
    far_mirror = pair_lane_markings([far_right], (0.0, 0.0), 0.0,
                                    debug=far_dbg)
    fy = None if far_mirror is None else float(
        np.median(far_mirror.center[:, 1]))
    check("lane-guard: far right line becomes a low-trust mirror",
          far_mirror is not None and not far_mirror.paired
          and far_dbg.get("mode") == "mirror_right"
          and fy is not None and abs(fy - (-1.25)) < 0.05
          and far_mirror.confidence <= LANE_FAR_MIRROR_CONF_MAX + 1e-9,
          f"mode={far_dbg.get('mode')} "
          f"conf={-1.0 if far_mirror is None else far_mirror.confidence:.2f} "
          f"y={-9.0 if fy is None else fy:.2f}")

    wide_xs = np.arange(0.0, 12.0, 1.0)
    wide_left = LaneMarking(
        world=np.column_stack([wide_xs, np.full_like(wide_xs, 2.5)]),
        pixels=np.zeros((len(wide_xs), 2)), color="white", kind="solid",
        confidence=0.9)
    wide_right = LaneMarking(
        world=np.column_stack([wide_xs, np.full_like(wide_xs, -2.5)]),
        pixels=np.zeros((len(wide_xs), 2)), color="white", kind="solid",
        confidence=0.9)
    wide = pair_lane_markings([wide_left, wide_right], (0.0, 0.0), 0.0)
    check("lane-guard: 5 m pair is a valid wide lane",
          wide is not None and wide.paired)
    if wide is not None:
        y = float(np.median(wide.center[:, 1]))
        check("lane-guard: 5 m pair centre on the middle",
              abs(y) < 0.05, f"y={y:.2f}")
        check("lane-guard: 5 m pair keeps its width",
              abs(wide.width - 5.0) < 1e-9, f"w={wide.width:.2f}")

    near_left = LaneMarking(
        world=np.column_stack([wide_xs, np.full_like(wide_xs, 1.75)]),
        pixels=np.zeros((len(wide_xs), 2)), color="white", kind="solid",
        confidence=0.9)
    far_right = LaneMarking(
        world=np.column_stack([wide_xs, np.full_like(wide_xs, -8.0)]),
        pixels=np.zeros((len(wide_xs), 2)), color="white", kind="solid",
        confidence=0.9)
    mixed = pair_lane_markings([near_left, far_right], (0.0, 0.0), 0.0)
    check("lane-guard: far right line rejected, near left kept",
          mixed is not None)
    if mixed is not None:
        y = float(np.median(mixed.center[:, 1]))
        check("lane-guard: centre mirrors the near left edge",
              abs(y) < 0.05, f"y={y:.2f}")

    # Regression for run 57: a near left line at +0.25 paired with a
    # roadside line 4.8 m to the right built a 4.5 m "lane" whose centre
    # sat 2 m right of the car.  The pair must be refused; the near left
    # line then mirrors the current lane to the correct centre instead.
    run57_left = LaneMarking(
        world=np.column_stack([wide_xs, np.full_like(wide_xs, 0.25)]),
        pixels=np.zeros((len(wide_xs), 2)), color="white", kind="solid",
        confidence=0.9)
    run57_far_right = LaneMarking(
        world=np.column_stack([wide_xs, np.full_like(wide_xs, -4.8)]),
        pixels=np.zeros((len(wide_xs), 2)), color="white", kind="dashed",
        confidence=0.6)
    run57 = pair_lane_markings([run57_left, run57_far_right],
                               (0.0, 0.0), 0.0)
    check("lane-guard: far roadside line never forms a paired lane",
          run57 is not None and not run57.paired)
    if run57 is not None:
        y = float(np.median(run57.center[:, 1]))
        check("lane-guard: near left edge mirrors to the lane centre",
              abs(y - (-1.5)) < 0.05, f"y={y:.2f}")
        check("lane-guard: far-centre mirror is not trusted",
              run57.confidence < 0.30,
              f"conf={run57.confidence:.2f}")

    # Regression for run 1786707239: a near right solid line paired with
    # a thin left line that only starts 19 m ahead made a phantom 6 m
    # lane whose centre the planner aimed at across the road.  The near
    # right line must stay a single-side mirror instead.
    run239_left = LaneMarking(
        world=np.column_stack([np.arange(19.4, 25.0, 0.5),
                               np.full(12, 4.33)]),
        pixels=np.zeros((12, 2)), color="white", kind="thin",
        confidence=0.7)
    run239_right = LaneMarking(
        world=np.column_stack([np.arange(-0.4, 16.0, 0.5),
                               np.full(33, -1.73)]),
        pixels=np.zeros((33, 2)), color="white", kind="solid",
        confidence=0.9)
    r239dbg: dict = {}
    run239 = pair_lane_markings([run239_left, run239_right],
                                (0.0, 0.0), 0.0, debug=r239dbg)
    check("lane-guard: near right paint never pairs with far thin left",
          run239 is not None and not run239.paired
          and r239dbg.get("mode") == "mirror_right",
          f"mode={r239dbg.get('mode')} "
          f"paired={None if run239 is None else run239.paired}")


def test_sensor_fusion_sides() -> None:
    """Vision mirrors and LiDAR edges only pair on opposite sides.

    A painted line on the right closer than half a lane is usually the
    centre line the car is riding: it is not a usable lane edge on its
    own and must not be paired into a bogus lane with a far LiDAR edge.
    """
    xs = np.arange(0.0, 14.0, 1.5)
    right_edge = LaneFrame(
        center=np.column_stack([xs, np.zeros_like(xs)]),
        right=np.column_stack([xs, np.full_like(xs, -1.75)]),
        width=3.5, confidence=0.55, span_m=13.5,
        sources=("vision",), paired=False)
    chosen = choose_sensor_lane(right_edge, None, (0.0, 0.0), 0.0)
    check("fusion: far right mirror kept",
          chosen is not None and chosen.sources[0] == "vision",
          f"src={None if chosen is None else chosen.sources}")

    # User rule: a painted right line is the strongest single boundary.
    # With one present, the wall / guardrail read must bend to it (here
    # the right paint pairs with the left wall into a real lane instead
    # of letting the wall-only corridor place the car 0.875 m further
    # left).  Without a right paint the wall corridor is the fallback.
    wall_hits = []
    for s in range(0, 130, 2):
        wall_hits.append((float(s) / 10.0, 1.75))
        wall_hits.append((float(s) / 10.0, -3.5))
    wall_pair = build_lidar_corridor(
        wall_hits, (0.0, 0.0), 0.0)
    paint_chosen = choose_sensor_lane(
        right_edge, wall_pair, (0.0, 0.0), 0.0)
    py = None if paint_chosen is None else float(
        np.median(paint_chosen.center[:, 1]))
    pr = None if paint_chosen is None or paint_chosen.right is None else float(
        np.median(paint_chosen.right[:, 1]))
    check("fusion: right paint wins over wall corridor",
          paint_chosen is not None
          and paint_chosen.sources[0] == "vision"
          and paint_chosen.paired
          and py is not None and abs(py) < 0.2
          and pr is not None and abs(pr + 1.75) < 0.2,
          f"src={None if paint_chosen is None else paint_chosen.sources} "
          f"paired={None if paint_chosen is None else paint_chosen.paired} "
          f"y={-9.0 if py is None else py:.2f} "
          f"right={-9.0 if pr is None else pr:.2f}")
    outside_right = LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, -2.5)]),
        right=np.column_stack([xs, np.full_like(xs, -4.0)]),
        width=3.5, confidence=0.7, span_m=13.5,
        sources=("vision",), paired=False)
    outside_chosen = choose_sensor_lane(
        outside_right, wall_pair, (0.0, 0.0), 0.0)
    check("fusion: vision line beyond the right wall defers to lidar",
          outside_chosen is not None
          and outside_chosen.sources[0] == "lidar"
          and outside_chosen.paired,
          f"src={None if outside_chosen is None else outside_chosen.sources} "
          f"paired={None if outside_chosen is None else outside_chosen.paired}")
    corridor_chosen = choose_sensor_lane(
        None, wall_pair, (0.0, 0.0), 0.0)
    check("fusion: no right paint falls back to wall corridor",
          corridor_chosen is not None and corridor_chosen.paired
          and corridor_chosen.sources[0] == "lidar",
          f"src={None if corridor_chosen is None else corridor_chosen.sources} "
          f"paired={None if corridor_chosen is None else corridor_chosen.paired}")

    near_right = LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, 1.35)]),
        right=np.column_stack([xs, np.full_like(xs, -0.4)]),
        width=3.5, confidence=0.45, span_m=13.5,
        sources=("vision",), paired=False)
    near_chosen = choose_sensor_lane(near_right, None, (0.0, 0.0), 0.0)
    check("fusion: ridden right paint is not a usable right boundary",
          near_chosen is None,
          f"src={None if near_chosen is None else near_chosen.sources} "
          f"conf={-1.0 if near_chosen is None else near_chosen.confidence:.2f}")

    # User rule: a painted right line is the lane's right boundary even
    # when it is closer than 1.75 m; as long as an opposite-side LiDAR
    # wall exists, the two are the real left/right boundaries and the
    # car drives their midpoint instead of the wall-only corridor.
    near_right_wall = LaneFrame(
        center=np.column_stack([xs, np.zeros_like(xs)]),
        right=np.column_stack([xs, np.full_like(xs, -0.9)]),
        width=3.5, confidence=0.55, span_m=13.5,
        sources=("vision",), paired=False)
    left_wall = LaneFrame(
        center=np.column_stack([xs, np.zeros_like(xs)]),
        left=np.column_stack([xs, np.full_like(xs, 1.75)]),
        width=3.5, confidence=0.5, span_m=13.5,
        sources=("lidar", "left"), paired=False)
    close_fused = choose_sensor_lane(
        near_right_wall, left_wall, (0.0, 0.0), 0.0)
    cy = None if close_fused is None else float(
        np.median(close_fused.center[:, 1]))
    check("fusion: close right paint + left wall midpoint",
          close_fused is not None and close_fused.paired
          and close_fused.sources == ("vision", "lidar")
          and cy is not None and abs(cy - 0.425) < 0.2,
          f"src={None if close_fused is None else close_fused.sources} "
          f"y={-9.0 if cy is None else cy:.2f}")

    lidar_right = LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, -1.75)]),
        right=np.column_stack([xs, np.full_like(xs, -3.5)]),
        width=3.5, confidence=0.5, span_m=13.5,
        sources=("lidar",), paired=False)
    same_chosen = choose_sensor_lane(
        near_right, lidar_right, (0.0, 0.0), 0.0)
    check("fusion: same-side edges never pair, lidar right wins",
          same_chosen is not None and same_chosen.sources[0] == "lidar",
          f"src={None if same_chosen is None else same_chosen.sources}")

    lidar_left = LaneFrame(
        center=np.column_stack([xs, np.zeros_like(xs)]),
        left=np.column_stack([xs, np.full_like(xs, 1.75)]),
        width=3.5, confidence=0.5, span_m=13.5,
        sources=("lidar",), paired=False)
    paired = choose_sensor_lane(
        right_edge, lidar_left, (0.0, 0.0), 0.0)
    y = None if paired is None else float(np.median(paired.center[:, 1]))
    check("fusion: opposite sides pair into one lane",
          paired is not None and paired.paired
          and paired.sources[0] == "vision" and y is not None
          and abs(y) < 0.2,
          f"src={None if paired is None else paired.sources} "
          f"y={-1.0 if y is None else y:.2f}")

    # The user case: the camera sees the centre line on the left and the
    # raycast fan sees the right wall / guardrail.  The lane centre must
    # sit on their midpoint, not on a fixed keep-right offset.
    vis_left = LaneFrame(
        center=np.column_stack([xs, np.zeros_like(xs)]),
        left=np.column_stack([xs, np.full_like(xs, 1.75)]),
        width=3.5, confidence=0.55, span_m=13.5,
        sources=("vision",), paired=False)
    lidar_right_wall = LaneFrame(
        center=np.column_stack([xs, np.zeros_like(xs)]),
        right=np.column_stack([xs, np.full_like(xs, -3.5)]),
        width=3.5, confidence=0.5, span_m=13.5,
        sources=("lidar", "right"), paired=False)
    fused_mid = choose_sensor_lane(
        vis_left, lidar_right_wall, (0.0, 0.0), 0.0)
    ym = None if fused_mid is None else float(
        np.median(fused_mid.center[:, 1]))
    check("fusion: centre line + right wall midpoint",
          fused_mid is not None and fused_mid.paired
          and ym is not None and -1.15 <= ym <= -0.6,
          f"src={None if fused_mid is None else fused_mid.sources} "
          f"y={-9.0 if ym is None else ym:.2f} "
          f"w={-1.0 if fused_mid is None else fused_mid.width:.2f}")

    # A right paint that starts ahead of the car still pairs with a left
    # wall seen from the car: sample the overlap instead of requiring a
    # near point on the painted side.
    far_vis_xs = np.arange(9.0, 16.0, 1.0)
    far_vis_right = LaneFrame(
        center=np.column_stack(
            [far_vis_xs, np.full_like(far_vis_xs, -1.25)]),
        right=np.column_stack(
            [far_vis_xs, np.full_like(far_vis_xs, -3.0)]),
        width=3.5, confidence=0.42, span_m=7.0,
        sources=("vision",), paired=False)
    far_fused = choose_sensor_lane(
        far_vis_right, left_wall, (0.0, 0.0), 0.0)
    far_fy = None if far_fused is None else float(
        np.median(far_fused.center[:, 1]))
    check("fusion: far right paint + left wall overlap midpoint",
          far_fused is not None and far_fused.paired
          and far_fused.sources == ("vision", "lidar")
          and far_fy is not None and abs(far_fy - (-0.625)) < 0.2
          and far_fused.confidence <= LANE_FAR_MIRROR_CONF_MAX + 1e-9,
          f"src={None if far_fused is None else far_fused.sources} "
          f"y={-9.0 if far_fy is None else far_fy:.2f} "
          f"conf={-1.0 if far_fused is None else far_fused.confidence:.2f}")

    lidar_left_far = LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, 0.75)]),
        left=np.column_stack([xs, np.full_like(xs, 2.5)]),
        width=3.5, confidence=0.5, span_m=13.5,
        sources=("lidar",), paired=False)
    bogus = choose_sensor_lane(
        near_right_wall, lidar_left_far, (0.0, 0.0), 0.0)
    by = None if bogus is None else float(np.median(bogus.center[:, 1]))
    check("fusion: right paint + left wall always forms the lane",
          bogus is not None and bogus.paired
          and bogus.sources == ("vision", "lidar")
          and by is not None and abs(by - 0.8) < 0.2,
          f"src={None if bogus is None else bogus.sources} "
          f"y={-9.0 if by is None else by:.2f}")

    # run_1786703265 frame 188: the only usable paint is a dashed line
    # 0.18 m to the right that starts 4.3 m ahead, while the raycast fan
    # only sees a left wall 4.63 m away.  Pairing them produced a 4.8 m
    # "lane" centred 2.2 m right of the car and drove into the right
    # wall.  The pair must be refused even though its near overlap has
    # only two stations, and the low-trust left-wall fallback must win.
    run188_vis = LaneFrame(
        center=np.column_stack(
            [np.arange(4.3, 11.0, 0.7),
             np.full(10, 1.57)]),
        right=np.column_stack(
            [np.arange(4.3, 11.0, 0.7),
             np.full(10, -0.18)]),
        width=3.5, confidence=0.42, span_m=6.7,
        sources=("vision",), paired=False)
    run188_lidar = LaneFrame(
        center=np.column_stack(
            [xs, np.full_like(xs, 0.35)]),
        left=np.column_stack(
            [xs, np.full_like(xs, 4.63)]),
        width=3.5, confidence=0.47, span_m=13.5,
        sources=("lidar", "left"), paired=False)
    run188_chosen = choose_sensor_lane(
        run188_vis, run188_lidar, (0.0, 0.0), 0.0)
    run188_y = None if run188_chosen is None else float(
        np.median(run188_chosen.center[:, 1]))
    check("fusion: near right paint + far left wall is refused",
          run188_chosen is not None
          and run188_chosen.sources == ("lidar", "left")
          and run188_y is not None and abs(run188_y - 0.35) < 0.15,
          f"src={None if run188_chosen is None else run188_chosen.sources} "
          f"y={-9.0 if run188_y is None else run188_y:.2f}")

    bad_fused = LaneFrame(
        center=np.column_stack(
            [xs, np.full_like(xs, 2.2)]),
        left=np.column_stack(
            [xs, np.full_like(xs, 4.63)]),
        right=np.column_stack(
            [xs, np.full_like(xs, -0.18)]),
        width=4.8, confidence=0.42, span_m=10.0,
        sources=("vision", "lidar"), paired=True)
    held_state: dict = {
        "src": ("vision", "lidar"),
        "frames": 8,
        "misses": 0,
        "last": bad_fused,
    }
    run188_recovered = choose_sensor_lane(
        None, run188_lidar, (0.0, 0.0), 0.0, state=held_state)
    check("fusion: bad fused frame is not held for 8 frames",
          run188_recovered is not None
          and run188_recovered.sources == ("lidar", "left"),
          f"src={None if run188_recovered is None else run188_recovered.sources}")

    # Single-edge LiDAR frames carry the side in sources so the fusion
    # hold treats a right->left flip as a source change, not a re-read of
    # the same edge.
    edge_right_src = LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, -1.75)]),
        right=np.column_stack([xs, np.full_like(xs, -3.5)]),
        width=3.5, confidence=0.5, span_m=13.5,
        sources=("lidar", "right"), paired=False)
    edge_left_src = LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, 1.75)]),
        left=np.column_stack([xs, np.full_like(xs, 3.5)]),
        width=3.5, confidence=0.5, span_m=13.5,
        sources=("lidar", "left"), paired=False)
    st: dict = {}
    f1 = choose_sensor_lane(None, edge_right_src, (0.0, 0.0), 0.0,
                            state=st)
    f2 = choose_sensor_lane(None, edge_left_src, (0.0, 0.0), 0.0,
                            state=st)
    f3 = choose_sensor_lane(None, edge_left_src, (0.0, 0.0), 0.0,
                            state=st)
    f4 = choose_sensor_lane(None, edge_left_src, (0.0, 0.0), 0.0,
                            state=st)
    check("fusion: edge side is part of the source",
          f1 is not None and f1.sources == ("lidar", "right"),
          f"src={None if f1 is None else f1.sources}")
    check("fusion: right->left edge flip is held",
          f2 is not None and f2.sources == ("lidar", "right"),
          f"src={None if f2 is None else f2.sources}")
    check("fusion: left edge wins after the hold",
          f4 is not None and f4.sources == ("lidar", "left"),
          f"src={None if f4 is None else f4.sources}")

    st2: dict = {}
    _ = choose_sensor_lane(None, edge_right_src, (0.0, 0.0), 0.0,
                           state=st2)
    held = choose_sensor_lane(None, None, (0.0, 0.0), 0.0,
                              state=st2)
    check("fusion: one missing frame keeps the last lane",
          held is not None and held.sources == ("lidar", "right"),
          f"src={None if held is None else held.sources}")


def test_lane_color_and_wide_fusion() -> None:
    """Yellow near the right is not a right lane edge, and a real
    vision-line + wall pair may be wider than a painted-lane pair."""
    from beamng_autopilot.vision.lanes import LaneMarking

    xs = np.arange(0.0, 14.0, 1.0)
    yellow_right = LaneMarking(
        world=np.column_stack([xs, np.full_like(xs, -0.4)]),
        pixels=np.zeros((len(xs), 2)), color="yellow", kind="solid",
        confidence=0.9)
    ydbg: dict = {}
    yf = pair_lane_markings([yellow_right], (0.0, 0.0), 0.0,
                            debug=ydbg)
    check("color: near yellow right is not a right mirror",
          yf is None, f"mode={ydbg.get('mode')}")

    white_left = LaneMarking(
        world=np.column_stack([xs, np.full_like(xs, 1.75)]),
        pixels=np.zeros((len(xs), 2)), color="white", kind="solid",
        confidence=0.9)
    mdbg: dict = {}
    mixed = pair_lane_markings([white_left, yellow_right],
                               (0.0, 0.0), 0.0, debug=mdbg)
    check("color: yellow right cannot shadow white left edge",
          mixed is not None and not mixed.paired
          and mdbg.get("mode") == "mirror_left",
          f"mode={mdbg.get('mode')}")

    yellow_far = LaneMarking(
        world=np.column_stack([xs, np.full_like(xs, -2.5)]),
        pixels=np.zeros((len(xs), 2)), color="yellow", kind="solid",
        confidence=0.9)
    fdbg: dict = {}
    far = pair_lane_markings([yellow_far], (0.0, 0.0), 0.0,
                             debug=fdbg)
    check("color: clear yellow right boundary still allowed",
          far is not None and fdbg.get("mode") == "mirror_right",
          f"mode={fdbg.get('mode')}")

    short_left = LaneMarking(
        world=np.column_stack([np.arange(2.5, 7.01, 0.5),
                               np.full(10, 3.3)]),
        pixels=np.zeros((10, 2)), color="white", kind="dashed",
        confidence=0.8)
    short_right = LaneMarking(
        world=np.column_stack([np.arange(4.5, 9.51, 0.5),
                               np.full(11, -2.0)]),
        pixels=np.zeros((11, 2)), color="yellow", kind="solid",
        confidence=0.7)
    sdbg: dict = {}
    short_pair = pair_lane_markings(
        [short_left, short_right], (0.0, 0.0), 0.0, debug=sdbg)
    check("pair: short overlap still forms a real pair",
          short_pair is not None and short_pair.paired
          and sdbg.get("mode") == "pair",
          f"mode={sdbg.get('mode')}")
    if short_pair is not None:
        check("pair: short overlap keeps true lane width/centre",
              abs(short_pair.width - 5.3) < 0.2
              and abs(float(np.median(short_pair.center[:, 1]))
                      - 0.65) < 0.15,
              f"w={short_pair.width:.2f}")
        check("pair: short overlap frame is usable",
              lane_frame_usable(short_pair),
              f"conf={short_pair.confidence:.2f} "
              f"span={short_pair.span_m:.2f}")

    wide_left = LaneMarking(
        world=np.column_stack([xs, np.full_like(xs, 2.5)]),
        pixels=np.zeros((len(xs), 2)), color="white", kind="solid",
        confidence=0.9)
    vis_left = pair_lane_markings([wide_left], (0.0, 0.0), 0.0)
    hits = [(float(s) / 10.0, -3.5) for s in range(0, 130, 2)]
    lidar = build_lidar_corridor(hits, (0.0, 0.0), 0.0)
    fused = choose_sensor_lane(vis_left, lidar, (0.0, 0.0), 0.0)
    fy = None if fused is None else float(np.median(fused.center[:, 1]))
    check("wide: left line + right wall fuse at midpoint",
          fused is not None and fused.paired
          and fused.sources == ("vision", "lidar")
          and fy is not None and abs(fy - (-0.5)) < 0.2,
          f"src={None if fused is None else fused.sources} "
          f"y={-9.0 if fy is None else fy:.2f}")
    check("wide: fused wide lane is usable",
          fused is not None and lane_frame_usable(fused),
          f"w={-1.0 if fused is None else fused.width:.2f}")

    sparse_hits = [(float(s) / 10.0, -3.5)
                   for s in range(0, 130, 4)]
    ndbg: dict = {}
    edge = build_lidar_corridor(sparse_hits, (0.0, 0.0), 0.0,
                                debug=ndbg)
    finite = (edge is not None and np.isfinite(edge.center).all()
              and edge.right is not None
              and np.isfinite(edge.right).all())
    check("lidar: single-edge fallback has no NaN boundary points",
          finite,
          f"fallback={ndbg.get('fallback')} "
          f"right_n={ndbg.get('edge_n')}")


def test_sensor_lane_planning() -> None:
    """A confident sensor lane centre overrides an explicit keep-right
    offset, while the default planner keeps right of the nav-route
    centre."""
    planner = LocalPlanner(right_offset=1.5, right_ramp_m=8.0)
    route = straight_route()
    xs = np.arange(0.0, 16.0, 1.5)
    center = np.column_stack([xs, np.zeros_like(xs)])
    frame = LaneFrame(center=center, width=3.5, confidence=0.8,
                      span_m=15.0, sources=("vision",))

    drive, blocked = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, sensor_lane=frame)
    drive = np.asarray(drive, dtype=float)
    check("sensor-plan: uses lane centre, not keep-right",
          not blocked and len(drive) >= 5
          and abs(float(drive[3, 1])) < 0.3,
          f"y3={float(drive[3, 1]):.2f} n={len(drive)}")
    check("sensor-plan: lane mode recorded",
          getattr(planner, "last_lane_mode", "") == "vision",
          f"mode={getattr(planner, 'last_lane_mode', '')}")

    drive_no_route, blocked_no_route = planner.plan(
        None, [], (0.0, 0.0), 0.0, 0, sensor_lane=frame)
    drive_no_route = np.asarray(drive_no_route, dtype=float)
    check("sensor-plan: paired lane drives without a nav route",
          not blocked_no_route and len(drive_no_route) >= 5
          and abs(float(drive_no_route[3, 1])) < 0.3,
          f"y3={float(drive_no_route[3, 1]):.2f} "
          f"n={len(drive_no_route)}")
    check("sensor-plan: no-route mode recorded",
          getattr(planner, "last_lane_mode", "") == "vision",
          f"mode={getattr(planner, 'last_lane_mode', '')}")

    offset_center = np.column_stack(
        [xs, np.full_like(xs, -1.75)])
    offset_frame = LaneFrame(
        center=offset_center, width=3.5, confidence=0.8,
        span_m=15.0, sources=("vision",))
    drive_off, blocked_off = planner.plan(
        None, [], (0.0, 0.0), 0.0, 0, sensor_lane=offset_frame)
    drive_off = np.asarray(drive_off, dtype=float)
    check("sensor-plan: no-route keeps the true lane midpoint",
          not blocked_off and len(drive_off) >= 5
          and abs(float(drive_off[3, 1]) + 1.75) < 0.3,
          f"y3={float(drive_off[3, 1]):.2f} n={len(drive_off)}")

    single = LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, -0.5)]),
        right=np.column_stack([xs, np.full_like(xs, -1.0)]),
        width=3.5, confidence=0.6, span_m=15.0,
        sources=("lidar",), paired=False)
    drive_single, blocked_single = planner.plan(
        None, [], (0.0, 0.0), 0.0, 0, sensor_lane=single)
    drive_single = np.asarray(drive_single, dtype=float)
    check("sensor-plan: no-route single edge still drives its centre",
          not blocked_single and len(drive_single) >= 5
          and abs(float(drive_single[3, 1]) + 0.5) < 0.3,
          f"y3={float(drive_single[3, 1]):.2f} n={len(drive_single)}")

    default_planner = LocalPlanner()
    drive_default, _ = default_planner.plan(
        route, [], (0.0, 0.0), 0.0, 0)
    drive_default = np.asarray(drive_default, dtype=float)
    check("sensor-plan: default planner keeps right of the route centre",
          len(drive_default) > 30
          and -1.8 <= float(drive_default[30, 1]) <= -1.2,
          f"y30={float(drive_default[30, 1]):.2f}")

    weak = LaneFrame(center=center, width=3.5, confidence=0.35,
                     span_m=15.0, sources=("vision",))
    drive2, _ = default_planner.plan(route, [], (0.0, 0.0), 0.0, 0,
                                     sensor_lane=weak)
    drive2 = np.asarray(drive2, dtype=float)
    check("sensor-plan: weak paired frame still follows the lane centre",
          len(drive2) >= 5
          and abs(float(drive2[len(drive2) - 1, 1])) < 0.3,
          f"ylast={float(drive2[len(drive2) - 1, 1]):.2f}")

    box = Obstacle(x=10.0, y=0.0, half_w=4.6, half_h=2.2,
                   category="vehicle", label="car")
    drive3, blocked3 = planner.plan(
        route, [box], (0.0, 0.0), 0.0, 0, sensor_lane=frame)
    drive3 = np.asarray(drive3, dtype=float)
    check("sensor-plan: obstacle handling still runs",
          not blocked3 and len(drive3) >= 2
          and not path_hits_box(drive3, box, CAR_HALF_WIDTH + 0.6),
          f"blocked={blocked3} n={len(drive3)}")

    hits = []
    for s in range(0, 200, 2):
        hits.append((float(s) / 10.0, 1.75))
        hits.append((float(s) / 10.0, -1.75))
    lidar = build_lidar_corridor(hits, (0.0, 0.0), 0.0)
    drive4, blocked4 = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, sensor_lane=lidar)
    drive4 = np.asarray(drive4, dtype=float)
    check("sensor-plan: lidar corridor drives the centre",
          not blocked4 and len(drive4) >= 5
          and abs(float(drive4[3, 1])) < 0.3
          and getattr(planner, "last_lane_mode", "") == "lidar",
          f"mode={getattr(planner, 'last_lane_mode', '')} "
          f"y3={float(drive4[3, 1]):.2f}")

    # Right-hand traffic: a single right-side laser edge is only a
    # low-trust wall hint.  The centre line stays primary and the wall
    # merely nudges the path a little, so a near wall cannot drag the
    # car onto the shoulder.
    right_only_hits = [(float(s) / 10.0, -4.1)
                       for s in range(0, 130, 2)]
    edge_lidar = build_lidar_corridor(
        right_only_hits, (0.0, 0.0), 0.0)
    drive_edge, _ = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, sensor_lane=edge_lidar)
    drive_edge = np.asarray(drive_edge, dtype=float)
    check("sensor-plan: lidar wall keeps the lane offset",
          edge_lidar is not None and len(drive_edge) > 30
          and -1.8 <= float(drive_edge[30, 1]) <= -1.2
          and getattr(planner, "last_lane_mode", "").startswith("lidar"),
          f"mode={getattr(planner, 'last_lane_mode', '')} "
          f"y30={float(drive_edge[30, 1]):.2f}")

    near_wall_hits = [(float(s) / 10.0, -2.0)
                      for s in range(0, 130, 2)]
    near_wall_lidar = build_lidar_corridor(
        near_wall_hits, (0.0, 0.0), 0.0)
    drive_near_wall, _ = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, sensor_lane=near_wall_lidar)
    drive_near_wall = np.asarray(drive_near_wall, dtype=float)
    check("sensor-plan: near right wall stays off the shoulder",
          near_wall_lidar is not None and len(drive_near_wall) > 30
          and -0.8 <= float(drive_near_wall[30, 1]) <= 0.2,
          f"y30={float(drive_near_wall[30, 1]):.2f}")

    # A single painted right line is only a boundary hint: it cannot prove
    # the lane centre, so the nav route stays primary and the marking only
    # nudges the path when the car is too close to it.
    vis_right = LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, -1.25)]),
        right=np.column_stack([xs, np.full_like(xs, -3.0)]),
        width=3.5, confidence=0.8, span_m=15.0,
        sources=("vision",), paired=False)
    drive_vis, _ = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, sensor_lane=vis_right)
    drive_vis = np.asarray(drive_vis, dtype=float)
    check("sensor-plan: single vision right keeps the lane offset",
          len(drive_vis) >= 10
          and -1.8 <= float(drive_vis[10, 1]) <= -1.2
          and 1.2 <= getattr(planner, "last_lane_offset", 0.0) <= 1.8
          and getattr(planner, "last_lane_mode", "") == "vision_right",
          f"mode={getattr(planner, 'last_lane_mode', '')} "
          f"y10={float(drive_vis[10, 1]):.2f} "
          f"off={getattr(planner, 'last_lane_offset', 0.0):.2f}")

    vis_right_near = LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, -0.5)]),
        right=np.column_stack([xs, np.full_like(xs, -1.0)]),
        width=3.5, confidence=0.8, span_m=15.0,
        sources=("vision",), paired=False)
    drive_vis_near, _ = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, sensor_lane=vis_right_near)
    drive_vis_near = np.asarray(drive_vis_near, dtype=float)
    check("sensor-plan: close right line only nudges, no half-lane pull",
          len(drive_vis_near) >= 5
          and abs(float(drive_vis_near[3, 1])) < 0.3
          and abs(getattr(planner, "last_lane_offset", 0.0)) < 0.5
          and getattr(planner, "last_lane_mode", "") == "vision_right",
          f"mode={getattr(planner, 'last_lane_mode', '')} "
          f"y3={float(drive_vis_near[3, 1]):.2f}")

    vis_right_low = LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, -1.25)]),
        right=np.column_stack([xs, np.full_like(xs, -3.0)]),
        width=3.5, confidence=0.42, span_m=15.0,
        sources=("vision",), paired=False)
    drive_vis_low, _ = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, sensor_lane=vis_right_low)
    drive_vis_low = np.asarray(drive_vis_low, dtype=float)
    check("sensor-plan: low-trust single line keeps the lane offset",
          len(drive_vis_low) >= 10
          and -1.8 <= float(drive_vis_low[10, 1]) <= -1.2
          and 1.2 <= getattr(planner, "last_lane_offset", 0.0) <= 1.8
          and getattr(planner, "last_lane_mode", "") == "vision_right",
          f"mode={getattr(planner, 'last_lane_mode', '')} "
          f"y10={float(drive_vis_low[10, 1]):.2f} "
          f"off={getattr(planner, 'last_lane_offset', 0.0):.2f}")

    # A fused vision + LiDAR lane knows both sides: the LiDAR right wall
    # is the lane boundary, so a thin wall cluster sitting there must not
    # close the corridor, and the drive path follows the real midpoint.
    fused_xs = np.arange(0.0, 16.0, 1.5)
    fused = LaneFrame(
        center=np.column_stack(
            [fused_xs, np.full_like(fused_xs, -0.875)]),
        left=np.column_stack(
            [fused_xs, np.full_like(fused_xs, 1.75)]),
        right=np.column_stack(
            [fused_xs, np.full_like(fused_xs, -3.5)]),
        width=5.25, confidence=0.72, span_m=15.0,
        sources=("vision", "lidar"), paired=True)
    fused_wall = Obstacle(
        x=10.0, y=-4.0, half_w=1.0, half_h=1.0,
        category="raycast", label="wall",
        axis=np.array([1.0, 0.0]), half_len=1.2, half_thick=0.8)
    drive_fused, blocked_fused = planner.plan(
        route, [fused_wall], (0.0, 0.0), 0.0, 0, sensor_lane=fused)
    drive_fused = np.asarray(drive_fused, dtype=float)
    fused_last = len(drive_fused) - 1
    check("sensor-plan: fused vision+lidar centres between wall and paint",
          not blocked_fused and fused_last >= 5
          and abs(float(drive_fused[fused_last, 1]) + 0.875) < 0.3
          and getattr(planner, "last_lane_mode", "") == "vision",
          f"mode={getattr(planner, 'last_lane_mode', '')} "
          f"ylast={float(drive_fused[fused_last, 1]):.2f} "
          f"n={len(drive_fused)} blocked={blocked_fused}")

    # A free corridor as wide as the whole road is not one lane: the
    # fallback stays on the nav-route centre instead of road centre.
    wide_hits = []
    for s in range(0, 200, 2):
        wide_hits.append((float(s) / 10.0, 5.0))
        wide_hits.append((float(s) / 10.0, -5.0))
    wide_lidar = build_lidar_corridor(wide_hits, (0.0, 0.0), 0.0)
    chosen_wide = choose_sensor_lane(None, wide_lidar)
    wide_planner = LocalPlanner()
    drive5, _ = wide_planner.plan(route, [], (0.0, 0.0), 0.0, 0,
                                  sensor_lane=chosen_wide)
    drive5 = np.asarray(drive5, dtype=float)
    check("sensor-plan: wide corridor keeps the lane offset",
          chosen_wide is None and len(drive5) > 30
          and -1.8 <= float(drive5[30, 1]) <= -1.2
          and getattr(wide_planner, "last_lane_mode", "") == "nav",
          f"mode={getattr(wide_planner, 'last_lane_mode', '')} "
          f"y30={float(drive5[30, 1]):.2f}")


def test_lane_frame_primary() -> None:
    """A two-sided lane frame is the lane centre, even without a nav route.

    The nav route is only a long-range direction source.  When both lane
    boundaries are real detections, the frame centre is the left/right
    midpoint and the drive path follows it directly instead of staying
    on (or right of) the nav route.
    """
    planner = LocalPlanner()
    route = straight_route()
    xs = np.arange(0.0, 16.0, 1.5)
    center = np.column_stack([xs, np.full_like(xs, -6.0)])
    frame = LaneFrame(center=center, width=3.5, confidence=0.8,
                      span_m=15.0, sources=("vision",))

    drive, blocked = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, sensor_lane=frame)
    drive = np.asarray(drive, dtype=float)
    last_i = len(drive) - 1
    y_far = float(drive[last_i, 1]) if last_i >= 5 else 99.0
    check("lane-primary: paired frame drives its midpoint",
          not blocked and last_i >= 5
          and abs(y_far + 6.0) < 0.3,
          f"ylast={y_far:.2f} blocked={blocked} n={len(drive)}")
    check("lane-primary: lane mode still recorded",
          getattr(planner, "last_lane_mode", "") == "vision",
          f"mode={getattr(planner, 'last_lane_mode', '')}")

    weak = LaneFrame(center=center, width=3.5, confidence=0.35,
                     span_m=15.0, sources=("vision",))
    drive_w, blocked_w = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, sensor_lane=weak)
    drive_w = np.asarray(drive_w, dtype=float)
    last_i = len(drive_w) - 1
    y_weak = float(drive_w[last_i, 1]) if last_i >= 5 else 99.0
    check("lane-primary: low-conf paired frame still uses the midpoint",
          not blocked_w and last_i >= 5
          and abs(y_weak + 6.0) < 0.3,
          f"ylast={y_weak:.2f} blocked={blocked_w}")


def test_lane_tracker_weighted_median() -> None:
    """A low-confidence wrong frame must not drag the smoothed lane centre."""
    tracker = LaneTracker(window=4)
    xs = np.arange(0.0, 12.0, 1.5)
    good = LaneFrame(center=np.column_stack([xs, np.zeros_like(xs)]),
                     width=3.5, confidence=0.7, span_m=12.0,
                     sources=("vision",))
    bad = LaneFrame(center=np.column_stack([xs, np.full_like(xs, 2.0)]),
                    width=3.5, confidence=0.31, span_m=12.0,
                    sources=("vision",))
    out = None
    for f in (good, bad, good, good):
        out = tracker.update(f, (0.0, 0.0), 0.0)
    y = None if out is None else float(np.median(out.center[:, 1]))
    check("tracker: weighted median ignores shaky frame",
          out is not None and y is not None and abs(y) < 0.2,
          f"y={-1.0 if y is None else y:.2f}")

    # Regression for run 67: the tracker used to drop the left/right
    # boundary polylines, so a vision centre line could never be paired
    # with the LiDAR right wall later on.
    tracker2 = LaneTracker(window=4)
    boundary = LaneFrame(
        center=np.column_stack([xs, np.zeros_like(xs)]),
        left=np.column_stack([xs, np.full_like(xs, 1.75)]),
        right=np.column_stack([xs, np.full_like(xs, -1.75)]),
        width=3.5, confidence=0.7, span_m=12.0,
        sources=("vision",), paired=False)
    out2 = None
    for _ in range(3):
        out2 = tracker2.update(boundary, (0.0, 0.0), 0.0)
    y2 = None if out2 is None else float(np.median(out2.center[:, 1]))
    ly = None if (out2 is None or out2.left is None) \
        else float(np.median(out2.left[:, 1]))
    ry = None if (out2 is None or out2.right is None) \
        else float(np.median(out2.right[:, 1]))
    check("tracker: keeps left/right boundaries for later fusion",
          out2 is not None and y2 is not None and abs(y2) < 0.2
          and ly is not None and abs(ly - 1.75) < 0.1
          and ry is not None and abs(ry + 1.75) < 0.1,
          f"y={-1.0 if y2 is None else y2:.2f} "
          f"left={-9.0 if ly is None else ly:.2f} "
          f"right={-9.0 if ry is None else ry:.2f}")

    # A tracked vision left boundary pairs with a LiDAR right wall into a
    # real two-sided lane centred between the two.
    lidar_right = LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, -1.75)]),
        right=np.column_stack([xs, np.full_like(xs, -3.5)]),
        width=3.5, confidence=0.5, span_m=12.0,
        sources=("lidar",), paired=False)
    fused = choose_sensor_lane(out2, lidar_right, (0.0, 0.0), 0.0)
    fy = None if fused is None else float(np.median(fused.center[:, 1]))
    check("tracker: vision line + lidar wall pair to the midpoint",
          fused is not None and fused.paired
          and fused.sources == ("vision", "lidar")
          and fy is not None and abs(fy - (-0.875)) < 0.2,
          f"src={None if fused is None else fused.sources} "
          f"y={-9.0 if fy is None else fy:.2f}")


def test_cluster_split() -> None:
    """Scattered hits must not merge into one giant false wall.

    Regression for the "parked on an empty road" report: a grove of trees
    beside the road chained into a single 14x16 m box that covered the
    route, so the planner reported blocked and stopped the car.
    """
    # A grove: two wavy rows, trees ~3 m apart.
    pts = []
    for i in range(10):
        pts.append((696.0 + i * 3.3, 356.0 + 1.5 * math.sin(i * 0.9)))
    for i in range(6):
        pts.append((698.0 + i * 3.1, 348.0 + 0.8 * math.cos(i * 0.7)))
    boxes = _cluster_points(pts)
    check("cluster: grove splits into several boxes", len(boxes) >= 8,
          f"n={len(boxes)}")
    oversized = [b for b in boxes
                 if b.half_w > 6.0 and b.half_h > 6.0]
    check("cluster: no oversized area box", not oversized,
          f"oversized={len(oversized)}")
    # A sparse row along the road must not become one 30 m wall either.
    row = [(20.0 + i * 3.2, 5.0) for i in range(10)]
    rboxes = _cluster_points(row)
    check("cluster: sparse row splits (no 30 m wall)", len(rboxes) >= 4,
          f"n={len(rboxes)}")
    # A real continuous wall stays one elongated obstacle.
    wall = [(20.0 + i * 0.8, 5.0) for i in range(40)]
    wboxes = _cluster_points(wall)
    check("cluster: solid wall stays one elongated box",
          len(wboxes) == 1 and wboxes[0].half_w > 10
          and wboxes[0].half_h < 1.5,
          (f"n={len(wboxes)} hw={wboxes[0].half_w:.1f} "
           f"hh={wboxes[0].half_h:.1f}") if wboxes else "no boxes")
    check("cluster: solid wall labelled wall",
          len(wboxes) == 1 and wboxes[0].label == "wall",
          (f"label={wboxes[0].label!r}"
           if wboxes else "no boxes"))
    from beamng_autopilot.planner import is_sparse_raycast_speck

    speck = _cluster_points([(20.0, 5.0)])
    check("cluster: single ray hit is a sparse speck",
          len(speck) == 1 and is_sparse_raycast_speck(speck[0]),
          (f"n={len(speck)} hw={speck[0].half_w:.2f}"
           if speck else "no boxes"))
    fat = [(20.0, 5.0), (25.0, 5.0), (20.0, 10.0), (25.0, 10.0)]
    fboxes = _cluster_points(fat, split_walls=True)
    fused = [b for b in fboxes
             if b.label != "wall"
             and 2.0 * b.half_len > 4.5
             and 2.0 * b.half_thick > 2.5]
    check("cluster: fused unlabelled fat blob split", not fused,
          f"n={len(fboxes)} fused={len(fused)}")


def test_bend_wall_split(planner: LocalPlanner) -> None:
    """A wall seen around a bend must not become one short fat box.

    Regression for the smallgrid wall report: both sides of a bend were
    fused into a 3.4 x 2.5 m (half-extent) blob whose inner edge sat only
    ~2.2 m from the route, so the planner declared blocked at d=64/65 and
    the car swerved instead of following the bend.  The split keeps each
    wall face thin and far enough from the lane that the path stays open.
    """
    pts = []
    for i in range(12):
        pts.append((2.6 + i * 0.7, -10.3 + (i / 11) * 5.6))
    for i in range(10):
        pts.append((2.8 + i * 0.9, 5.3 + (i / 9) * 3.4))
    for lon in (5.5, 6.0, 6.5):
        pts.append((lon, -4.7))
        pts.append((lon, 5.3))

    boxes = _cluster_points(pts, split_walls=True)
    check("bend-wall: fused blob split into several boxes",
          len(boxes) >= 3, f"n={len(boxes)}")
    fat = [b for b in boxes if 2.0 * b.half_thick >= 2.5]
    check("bend-wall: no short fat false blocker", not fat,
          f"fat={[(round(2*b.half_len,1), round(2*b.half_thick,1)) for b in fat]}")
    wall = [b for b in boxes if b.label == "wall"]
    check("bend-wall: wall faces stay thin wall segments",
          len(wall) >= 1 and all(2.0 * b.half_thick < 2.5 for b in wall),
          f"walls={len(wall)}")

    route = straight_route(30.0)
    drive, blocked = planner.plan(route, boxes, (0.0, 0.0), 0.0, 0)
    drive = np.asarray(drive, dtype=float)
    check("bend-wall: planner follows through the bend, not blocked",
          not blocked and planner.last_mode in ("follow", "deform"),
          f"mode={planner.last_mode} blocked={blocked}")
    check("bend-wall: drive path stays on the road",
          len(drive) >= 2 and abs(float(drive[-1, 1])) < 4.0,
          f"tail=({float(drive[-1, 0]):.1f},{float(drive[-1, 1]):.1f})"
          if len(drive) else "empty path")


def test_twitch_scene_2102(planner: LocalPlanner) -> None:
    """End-to-end regression for the 2026-08-12 21:02 twitch/park scene.

    The car was stopped on an empty road by two fused problems: a grove
    beside the road merged into one 14x16 m box covering the route (plan
    -> blocked), and speed() read the box as lon=0 (touching the car),
    pinning v to 0 every frame -> stop/creep oscillation.  This replays
    the exact recorded pose + route with a reconstructed grove point
    cloud and demands a non-blocked detour with a sane speed.
    """
    # Real route segment (smallgrid) around the recorded car pose,
    # densified to ~2 m spacing like the live connector does.
    rte = np.array([
        [687.9, 360.9], [690.7, 358.1], [693.5, 355.2], [696.2, 352.3],
        [698.8, 349.2], [701.2, 346.0], [703.4, 342.7], [705.6, 339.4],
        [707.4, 335.8], [709.1, 332.2], [710.5, 328.4], [711.6, 324.6],
        [712.5, 320.7], [713.0, 316.7], [713.6, 312.8], [713.7, 308.8],
        [713.7, 304.8], [713.7, 300.8], [713.4, 296.8], [713.0, 292.8],
        [712.4, 288.9], [711.8, 284.9],
    ], dtype=float)
    step = 2  # recorded rte keeps every 2nd point of the 2 m route
    route = np.zeros(((len(rte) - 1) * step + 1, 2), dtype=float)
    for i in range(len(rte) - 1):
        a, b = rte[i], rte[i + 1]
        for k in range(step):
            route[i * step + k] = a + (b - a) * (k / step)
    route[-1] = rte[-1]
    nearest = 4      # route index of the car (rte row 2, step=2)
    pos = np.array([693.321, 354.856])
    heading = -0.8843

    # Grove beside the road, filling the footprint of the recorded merged
    # box (x 695-710, y 350-366): three wavy rows, trees ~3 m apart.
    pts = []
    for i in range(6):
        pts.append((695.5 + 2.9 * i, 351.2 + 0.7 * math.sin(i * 1.3)))
    for i in range(6):
        pts.append((697.0 + 2.8 * i, 358.5 + 0.6 * math.cos(i * 0.8)))
    for i in range(4):
        pts.append((699.0 + 3.0 * i, 364.0 + 0.5 * math.sin(i * 1.7)))
    boxes = _cluster_points(pts)
    oversized = [b for b in boxes if b.half_w > 6.0 and b.half_h > 6.0]
    check("twitch-scene: grove no longer one giant box", not oversized,
          f"oversized={len(oversized)} boxes={len(boxes)}")
    # The two small recorded obstacles from the live frame.
    boxes += [
        Obstacle(712.0, 346.6, 0.9, 0.9, "tree"),
        Obstacle(706.9, 334.6, 1.2, 1.3, "tree"),
    ]

    drive, blocked = planner.plan(route, boxes, pos, heading, nearest)
    check("twitch-scene: plan detours, not blocked",
          not blocked and planner.last_mode == "detour",
          f"mode={planner.last_mode} blocked={blocked}")
    v, _ = planner.speed(np.asarray(drive, dtype=float), boxes, pos,
                         heading, 0, 15.0)
    check("twitch-scene: speed stays drivable (>5 m/s)",
          v > 5.0, f"v={v:.2f} m/s")

    # Replay the scene the way the live loop does (plan -> trim the path
    # to the car -> speed with nearest=0) along the route through the
    # grove.  The roadside boxes used to project onto the trimmed route
    # start (lon = 0) every frame and pin the speed to zero while the
    # detour ran metres away -> stop/creep twitch along the grove.
    pins = 0
    grove_ok = True
    for idx in range(0, 18, 2):
        fpos = route[idx]
        fwd = route[min(idx + 2, len(route) - 1)] - route[idx]
        fhead = math.atan2(float(fwd[1]), float(fwd[0]))
        fd = np.linalg.norm(route[:, :2] - fpos, axis=1)
        fnearest = int(np.argmin(fd))
        fdrive, fblocked = planner.plan(
            route, boxes, fpos, fhead, fnearest)
        fdrive = np.asarray(fdrive, dtype=float)
        if len(fdrive) >= 2:
            d0 = np.linalg.norm(fdrive[:, :2] - fpos, axis=1)
            start_i = int(np.argmin(d0))
            if start_i > 0 and len(fdrive) - start_i >= 2:
                fdrive = fdrive[start_i:]
        fv, _ = planner.speed(fdrive, boxes, fpos, fhead, 0, 15.0)
        if fblocked:
            fv = 0.0
        obs_lim = getattr(planner, "last_obs_lim", None)
        if not fblocked and obs_lim is not None and obs_lim <= 0.01:
            pins += 1
        if 2 <= idx <= 10 and fv <= 5.0:
            grove_ok = False
    check("twitch-scene: no frame pins the speed (obslim=0 gone)",
          pins == 0, f"pins={pins}")
    check("twitch-scene: grove frames stay drivable (>5 m/s)",
          grove_ok, "some grove frame fell below 5 m/s")


def test_traffic_rules() -> None:
    """Road limits and signals make pure-Python speed decisions."""
    v, reason, lim = apply_rule_speed(
        20.0, RoadRuleView(speed_limit_mps=10.0))
    check("rule: road speed limit caps cruise",
          v == 10.0 and reason == "speed_limit" and lim == 10.0,
          f"v={v} reason={reason}")

    v, reason, lim = apply_rule_speed(20.0)
    check("rule: no snapshot leaves cruise unchanged",
          v == 20.0 and reason is None and lim is None,
          f"v={v} reason={reason}")

    stop = SignalRule(action=2, rel_dist=-12.0, dist=12.0,
                      dot=1.0, state="red", name="sig1")
    v, reason, lim = apply_rule_speed(20.0, signal_rule=stop)
    expect = math.sqrt(2.0 * 3.0 * (12.0 - 4.0))
    check("rule: red signal brakes before the line",
          abs(v - expect) < 1e-9 and reason == "signal",
          f"v={v:.2f} expect={expect:.2f}")

    v, reason, lim = apply_rule_speed(
        20.0, signal_rule=SignalRule(
            action=2, rel_dist=2.0, dist=2.0, dot=1.0))
    check("rule: passed signal ignored",
          v == 20.0 and reason is None, f"v={v}")

    v, reason, lim = apply_rule_speed(
        20.0, signal_rule=SignalRule(
            action=1, rel_dist=-30.0, dist=30.0, dot=1.0))
    check("rule: yellow decelerates but keeps a floor",
          reason == "signal" and v >= 3.0 and v < 20.0,
          f"v={v:.2f}")

    v, reason, lim = apply_rule_speed(
        20.0, signal_rule=SignalRule(
            action=4, rel_dist=-30.0, dist=30.0, dot=1.0))
    check("rule: slow zone capped at 30 km/h",
          reason == "signal" and v <= 30.0 / 3.6 + 1e-9,
          f"v={v:.2f}")

    data = {
        "n1": "DR65_11", "n2": "DR65_10", "lanes": "-+",
        "speedLimit": 16.666666666667, "drivability": 1.0,
        "oneWay": False, "type": "asphalt",
        "rightHandDrive": False, "turnOnRed": False,
    }
    rule = RoadRuleView.from_lua_dict(data)
    check("rule: lua snapshot parsed",
          rule is not None
          and rule.n1 == "DR65_11"
          and rule.n2 == "DR65_10"
          and abs(float(rule.speed_limit_mps) - 16.666666666667) < 1e-6
          and rule.lane_direction == "mixed"
          and rule.one_way is False
          and rule.right_hand_drive is False,
          repr(rule))

    check("rule: lane direction classified",
          classify_lane_direction("-+") == "mixed"
          and classify_lane_direction("++") == "+"
          and classify_lane_direction("-") == "-")
    check("rule: one-way inferred from lanes",
          one_way_from_lanes("++") is True
          and one_way_from_lanes("-+") is False
          and one_way_from_lanes("") is None)
    check("rule: signal action labels",
          signal_action_label(0) == "go"
          and signal_action_label(2) == "stop"
          and signal_action_label(3) == "stop_sign"
          and signal_action_label(5) == "yield")
    check("rule: signal distance is positive-ahead",
          signal_distance(SignalRule(rel_dist=-4.0)) == 4.0
          and signal_distance(SignalRule(rel_dist=1.0)) == 0.0
          and signal_distance(None) is None)

    ahead = SignalRule(name="a", action=2, rel_dist=-8.0, dist=8.0,
                       dot=0.9, pos=(10.0, 0.0))
    behind = SignalRule(name="b", action=2, rel_dist=-5.0, dist=5.0,
                        dot=0.9, pos=(-10.0, 0.0))
    chosen = select_signal_rule(
        [ahead, behind], pos=(0.0, 0.0), dir_vec=(1.0, 0.0))
    check("rule: nearest ahead signal chosen",
          chosen is not None and chosen.name == "a",
          chosen.name if chosen is not None else "None")


def test_legal_lane_view() -> None:
    """Map lane strings choose the legal side on both LHD and RHD roads."""
    lhd = RoadRuleView(
        lanes="-+", right_hand_drive=False,
        in_radius=4.0, out_radius=4.0)
    v = legal_lane_view(lhd)
    check("legal-lane: LHD chooses the rightmost + lane",
          v is not None and v.legal
          and v.start == 1 and v.end == 1
          and v.preferred_index == 1
          and abs(v.preferred_offset_m - 2.0) < 1e-9,
          repr(v))
    check("legal-lane: LHD boundary is the centre line",
          v is not None and v.boundaries == ((0.0, 1.0),),
          repr(getattr(v, "boundaries", None)))
    check("legal-lane: geometry mirrors map.lua",
          v is not None
          and abs(v.lane_width_m - 8.0) < 1e-9
          and abs(v.lane_width_m / v.lane_count - 4.0) < 1e-9
          and abs(lane_offset_m(1, 2, 8.0) - 2.0) < 1e-9,
          f"width={getattr(v, 'lane_width_m', None)}")

    rhd = RoadRuleView(
        lanes="++--", right_hand_drive=True,
        in_radius=4.0, out_radius=4.0)
    v = legal_lane_view(rhd)
    check("legal-lane: RHD chooses the leftmost + lane",
          v is not None and v.legal
          and v.start == 0 and v.end == 1
          and v.preferred_index == 0
          and abs(v.preferred_offset_m + 3.0) < 1e-9,
          repr(v))
    check("legal-lane: RHD boundary allows the left side",
          v is not None and v.boundaries == ((0.0, -1.0),),
          repr(getattr(v, "boundaries", None)))

    inter = RoadRuleView(
        lanes="--++-++", right_hand_drive=False,
        in_radius=4.0, out_radius=4.0)
    v = legal_lane_view(inter)
    check("legal-lane: interleaved road uses the rightmost run",
          v is not None and v.legal
          and v.start == 5 and v.end == 6
          and v.preferred_index == 6,
          repr(v))

    plus = legal_lane_view(RoadRuleView(
        lanes="+", right_hand_drive=False,
        in_radius=4.0, out_radius=4.0))
    minus = legal_lane_view(RoadRuleView(
        lanes="-", right_hand_drive=False,
        in_radius=4.0, out_radius=4.0))
    check("legal-lane: one-way + is drivable, one-way - is wrong way",
          plus is not None and plus.legal
          and minus is not None and not minus.legal,
          f"plus={repr(plus)} minus={repr(minus)}")

    check("legal-lane: missing lanes or RHD flag returns None",
          legal_lane_view(RoadRuleView(lanes=None, right_hand_drive=False))
          is None
          and legal_lane_view(RoadRuleView(lanes="-+", right_hand_drive=None))
          is None)

    fallback = legal_lane_view(RoadRuleView(
        lanes="-+", right_hand_drive=False))
    check("legal-lane: fallback width works without node radii",
          fallback is not None
          and abs(fallback.lane_width_m - 7.0) < 1e-9
          and abs(fallback.lane_width_m / fallback.lane_count - 3.5) < 1e-9
          and abs(fallback.preferred_offset_m - 1.75) < 1e-9,
          repr(fallback))


def test_map_lane_planning() -> None:
    """Map legal-lane data moves the path and blocks wrong-way links."""
    from beamng_autopilot.planner import (
        _MapLaneBoundary,
        _clamp_to_solid_lines,
    )

    route = straight_route()
    rule = RoadRuleView(
        lanes="-+", right_hand_drive=False,
        in_radius=4.0, out_radius=4.0,
        in_pos=(0.0, 0.0, 0.0), out_pos=(60.0, 0.0, 0.0),
        right_vec=(0.0, -1.0, 0.0))

    planner = LocalPlanner(right_ramp_m=8.0)
    drive, blocked = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, road_rule=rule)
    drive = np.asarray(drive, dtype=float)
    check("map-lane: map boundary keeps the right lane offset",
          not blocked and len(drive) > 30
          and -1.8 <= float(drive[30, 1]) <= -1.2
          and 1.2 <= getattr(planner, "last_lane_offset", 0.0) <= 1.8,
          f"blocked={blocked} y30="
          f"{float(drive[30, 1]):.2f} off="
          f"{getattr(planner, 'last_lane_offset', 0.0):.2f}"
          if len(drive) > 30 else "short path")
    drive_old, blocked_old = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0)
    drive_old = np.asarray(drive_old, dtype=float)
    check("map-lane: no map data keeps the right lane offset",
          not blocked_old and len(drive_old) > 10
          and -1.8 <= float(np.median(drive_old[:, 1])) <= -1.2,
          f"blocked={blocked_old} "
          f"y_med={float(np.median(drive_old[:, 1])):.2f}"
          if len(drive_old) > 10 else "short path")

    wrong = RoadRuleView(
        lanes="-", right_hand_drive=False,
        in_radius=4.0, out_radius=4.0,
        in_pos=(0.0, 0.0, 0.0), out_pos=(60.0, 0.0, 0.0),
        right_vec=(0.0, -1.0, 0.0))
    drive_w, blocked_w = planner.plan(
        route, [], (0.0, 0.0), 0.0, 0, road_rule=wrong)
    check("map-lane: one-way wrong-way link is blocked",
          blocked_w and planner.last_mode == "blocked"
          and getattr(planner, "last_blocker", None) is not None
          and planner.last_blocker[0] == "wrong-way road",
          f"blocked={blocked_w} "
          f"blocker={getattr(planner, 'last_blocker', None)}")

    box = Obstacle(x=20.0, y=-4.0, half_w=2.0, half_h=2.0,
                   category="vehicle", label="car")
    off = planner._safe_lateral_offset(
        route, 0, len(route) - 1, 0.0, [box], target=2.0)
    check("map-lane: blocked target side shrinks without flipping",
          0.0 <= off < 2.0 - 0.05, f"off={off:.2f}")

    boundary = _MapLaneBoundary(
        np.array([[0.0, 0.0], [60.0, 0.0]], dtype=float),
        allowed_side=1.0)
    xs = np.arange(0.0, 61.0, 1.0)
    ys = np.where(xs < 5.0, -2.0,
                  -2.0 + np.clip((xs - 5.0) / 20.0, 0.0, 1.0) * 4.0)
    crossing = np.column_stack([xs, ys])
    out, crossed, _ = _clamp_to_solid_lines(
        crossing, [boundary], (0.0, 0.0),
        corridor=crossing, allow_block=True)
    check("map-lane: crossing the map boundary blocks",
          crossed and len(out) >= 2
          and float(np.max(out[:, 1])) <= 0.0,
          f"crossed={crossed} n={len(out)}")


def main() -> None:
    print("== M5 offline validation ==")
    planner = LocalPlanner()
    test_bypass_forward_car(planner)
    test_bypass_sideways_car(planner)
    test_wide_wall_blocks(planner)
    test_roadside_wall_boundary(planner)
    test_right_offset_not_zeroed_by_corridor_hugging_wall(planner)
    test_bypass_prefers_right_and_never_crosses_walls(planner)
    test_speed_ramp()
    test_speed_slip()
    test_drop_waypoint_ghosts()
    test_vision_track_confirmation()
    test_vehicle_orientation()
    test_merge()
    test_raycast_slope_filter()
    test_raycast_near_low_obstacle()
    test_scenario_min_dist()
    test_self_overlap_filter()
    test_heading_deviation()
    test_lane_reuse_staleness()
    test_bridge()
    test_roadnet_polylines()
    test_connector_current_env()
    test_connector_italy_default_spawn()
    test_launcher_imports()
    test_runtime_selection()
    test_creep_speed()
    test_traffic_rules()
    test_legal_lane_view()
    test_map_lane_planning()
    test_speed_lon_semantics(planner)
    test_sharp_corner_speed(planner)
    test_right_offset_and_solid_line()
    test_solid_line_anchor_ambiguity()
    test_solid_line_low_confidence_gate()
    test_solid_line_detour_gate()
    test_curve_right_offset_uses_local_tangent(planner)
    test_solid_line_noise_filter()
    test_lane_detector()
    test_lane_pairing_and_lidar_corridor()
    test_lane_boundary_guards()
    test_sensor_fusion_sides()
    test_lane_color_and_wide_fusion()
    test_lane_tracker_weighted_median()
    test_sensor_lane_planning()
    test_vision_marking_primary_over_nav_route()
    test_lane_frame_primary()
    test_lane_edge_wall_boundary(planner)
    test_cluster_split()
    test_bend_wall_split(planner)
    test_twitch_scene_2102(planner)
    print("-" * 40)
    if _FAILED:
        print(f"RESULT: {len(_FAILED)} FAILED -> {_FAILED}")
        sys.exit(1)
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()
