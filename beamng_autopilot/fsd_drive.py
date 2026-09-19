"""FSD live-driving session: FSDStack planning -> safety -> control.

Library core of the FSD drive, extracted from ``scripts/m5_fsd_drive.py``
(which stays as the thin argparse entry point): drive constants, pure
helpers and ``run(args)`` - the connection / warm-up / drive-loop /
telemetry flow driven by the parsed CLI arguments.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

import cv2

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector, angle_to_quat
from beamng_autopilot.control import gearbox
from beamng_autopilot.control.reverse_guard import ReverseGuard
from beamng_autopilot.control.reverse_maneuver import ReverseManeuver
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.control.speed import (
    SpeedController, rate_limit_pedal,
)
from beamng_autopilot.control.blend import (
    SteeringBlendWeights, blend_steering,
)
from beamng_autopilot.control.drive_mode import (
    DriveModeClassifier,
)
from beamng_autopilot.control.steering import SteeringShaper
from beamng_autopilot.control.substep import ControlSubstep
from beamng_autopilot.obstacle_risk import assess_obstacles
from beamng_autopilot.fsd_stack import FSDStack
from beamng_autopilot.fsd_realism import SRC_PAVED, SRC_SENSOR
from beamng_autopilot.lane import perception_curve_speed
from beamng_autopilot.neural.bc_runtime import (
    DEFAULT_BC_WEIGHTS, BCRuntime, steer_to_path,
)
from beamng_autopilot.rl.dqn_runtime import (
    DEFAULT_DQN_WEIGHTS, DQNRuntime, action_to_target,
)
from beamng_autopilot.neural.e2e_runtime import (
    DEFAULT_E2E_WEIGHTS, E2ERuntime,
)
from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import (
    ArbiterOutcome, Scene, anchored_rule_ref, arbitrate_fsd_tick,
    body_pose_crosses_lane, local_route, strict_lane_unavailable,
    validate_learned_path,
)
from beamng_autopilot.planning.arbiter import (
    bearing_diff_deg, polyline_bearing,
)
from beamng_autopilot.planning.constraints import _boundary_lateral
from beamng_autopilot.vehicle_body import (
    HALF_LENGTH_M, HALF_WIDTH_M, body_pose_cross_depth_m,
    footprint_corners,
)
from beamng_autopilot.planning.longitudinal import (
    LongitudinalPlanner,
)
from beamng_autopilot.planning.local_route import (
    map_lane_edges, _project_arc, _route_turn_deg,
)
from beamng_autopilot.planning.speed_profile import \
    MIN_SPEED as _PROF_MIN_SPEED, \
    speed_profile_for_path as _spf_raw
from beamng_autopilot.recording import (
    FMAP_CHANNELS,
    ShadowFrame,
    ShadowRecorder,
)
from beamng_autopilot.roadnet import RoadNetwork
from beamng_autopilot.autopilot import nearest_route_point
from beamng_autopilot.planner import (
    LocalPlanner,
    emergency_speed_limit_mps,
    emergency_stop_clearance_m,
    path_grid_clearance_m,
)
from beamng_autopilot.safety_monitor import SafetyMonitor
from beamng_autopilot.traffic import select_signal_rule
from beamng_autopilot.vision.heads import (
    LaneTopologyHead, ObjectHead, SemanticHead, TrafficSignalHead,
)
from beamng_autopilot.vision.heads.traffic import merge_signal_vision
from beamng_autopilot.watchdog import (
    arm as wd_arm,
    disarm as wd_disarm,
    heartbeat as wd_heartbeat,
)
from beamng_autopilot.vision.lanes import (
    PaintedLineLateralCorrector,
    painted_line_correction_active,
    painted_line_direction,
    painted_line_lane_center,
    painted_line_markings,
    polyline_dir_at,
)

# Reverse guard: the car must never drive backwards under the FSD mode.
# A real factory stack has lane/gear protections - m5_autopilot does too
# (REVERSE_ENGAGE_S/REVERSE_HOLD_S/gear=D).  Without this the FSD drive
# reversed into walls after an impact ("dumb reversing" seen on probes).
REVERSE_THRESHOLD_MPS = -0.35
REVERSE_CLEAR_MPS = 0.2
# Reverse-escape throttle ramp (see the rm.active branch): base is the
# gentle graded-road value; on grass the ramp lifts it until the car
# actually rolls back.  Both bounded by the -0.4 m/s target brake and
# the rear-clearance stop.
REV_THR_BASE = 0.06
REV_THR_STEP = 0.05
REV_THR_MAX = 0.45
# Slope-creep assist: seconds of full throttle before a "stuck" car is
# allowed to reverse (a car stopped at the bottom of a dip facing uphill
# is not wedged - it needs torque, not a backward roll).
CLIMB_ASSIST_S = 3.5
# Traffic-light action: the traffic head emits a colour + confidence
# (pure perception, never a map/route prior).  A confident RED light
# stops the car at the line; below this confidence the colour is
# ignored (a dusky frame must not park the car).
SIGNAL_CONF_MIN = 0.6
# Game-side road-link signal snapshot poll rate (the same authoritative
# source the M5 rule autopilot uses; fused with the vision head via
# merge_signal_vision - the game is authoritative for the light STATE,
# vision fills the gaps.  Longitudinal only: lateral stays
# perception-only.)
SIGNAL_RULE_POLL_S = 1.5
# Real-time control loop: the sim runs continuously and the loop paces
# itself to ~REALTIME_CTRL_HZ, so the car never freezes between ticks
# (the old paused-step design moved 0.33 s then froze ~1.2 s - the
# stutter).  Each tick now drives with the previous control for at most
# ~0.5 s, so the stale-control window is the tick time itself (ticks are
# ~0.4-0.6 s after warm-up, vs the 5-10 m stale drift of fix44-51).
REALTIME_CTRL_HZ = 2.0
# Control/perception decoupling (improvement plan phases A2/A4): the
# perception + planning tick keeps its natural cadence above, while the
# control command is re-issued at SUBSTEP_HZ in between from the last
# VERIFIED plan with a fresh pose/speed - a slow perception head delays
# the next plan, never the wheel.  ``BEAMNG_CONTROL_SUBSTEP=0`` disables
# it (rollback / A-B lever); the policy itself lives in
# ``control.substep.ControlSubstep``.
SUBSTEP_HZ = (float(config.FSD_CONTROL_SUBSTEP_HZ)
              if os.environ.get("BEAMNG_CONTROL_SUBSTEP", "1") != "0"
              else 0.0)
# Warm-up crawl: the first FSD ticks load YOLO and settle camera/LiDAR
# (observed 4-6 s); in real time the car would otherwise run open-loop
# during that.  Crawl until the object head is live, capped at WARMUP_S.
WARMUP_S = 8.0
# Lane-placement hold: the car stays BRAKED at spawn until the semantic
# head has placed it in its own lane (painted-line projection) or this
# deadline expires.  A start from the road-graph node sits ON the road
# centre line - driving unplaced rides the left line from metre one
# (town 2026-09-06, user-reported).  An unplaced run aborts instead.
PLACEMENT_HOLD_S = 30.0
# ``lane_src_sel`` values that mean "perception already owns the lane the
# car is on, so it may drive": a sensor lane pair, or - on a paved road
# with NO usable marking - the paved-boundary reference (AGENTS.md
# 「驾驶约束」: 标线 >> 路面边界 = 护墙 = 围栏).  Both are sensor-derived;
# a map lane is never in this set, so placement can not be unlocked by
# map geometry.
PLACEMENT_LANE_SRCS = (SRC_SENSOR, SRC_PAVED)
# Body clearance placement requires inside the detected lane boundaries
# (inflating the body rectangle by this turns the penetration helper into
# a clearance test).  2026-09-18: one spawn pose was 0.17 m inside the
# boundary under the base model and 0.31 m OUTSIDE under the fine-tuned
# one - the same pose, two models, one frozen run.  A pose that close is
# not "already centered"; it must take the alignment teleport instead.
PLACEMENT_BODY_CLEARANCE_M = 0.20
# A published boundary must sit at least this far on its OWN side of the
# car (left boundary left, right boundary right) before the pose counts as
# placed.  A boundary within +-this of the car centre is "riding the line"
# or on the wrong side entirely - both need the alignment teleport, not a
# release (see the boundary-side comment in _sensor_lane_is_centered).
PLACEMENT_BOUNDARY_SIDE_MIN_M = 0.30
# Extra seconds after heads are live before giving up on painted-line
# placement (US yellow paint / warm-up flicker).
# After the heads come live, US yellow/short-pair evidence may need several
# more camera ticks before it becomes a trusted sensor lane.  Keep the car
# braked through a 30 s grace window (total placement deadline 60 s).
PLACEMENT_GRACE_S = 30.0
# Skip the first N stack ticks after teleport so the camera settles
# before placement (east_coast first frames often lack paint).
PLACEMENT_SKIP_TICKS = 16
WARMUP_SPEED_MPS = 1.5
# If a tick ever takes longer than this, the car has been driving
# open-loop for that long - keep this frame slow instead of trusting
# stale controls.
STALE_CTRL_S = 1.2
STALE_CTRL_SPEED_MPS = 2.0
# Tick time-budget governor (smoothness): the FSD tick cost swings
# widely - a semantic + YOLO + fresh-LiDAR tick can hit 0.5-0.7 s while
# a reuse tick is ~40-100 ms, so without a bound the control loop
# visibly stutters every few frames.  The stack defers a due heavy head
# once the tick exceeds the budget and serves the cached output instead
# (the head stays due and runs on the next affordable tick).  The budget
# adapts to the measured frame time: fast runs never constrain a head,
# slow runs stay bounded.
TICK_BUDGET_MIN_S = 0.25
TICK_BUDGET_MAX_S = 0.45
TICK_BUDGET_FRAC = 0.85          # of the EMA frame time
TICK_BUDGET_EMA = 0.30           # EMA weight for the frame-time tracker
# Target-speed smoothing: the plan speed changes by whole m/s between
# ticks (corner governor, obstacle caps).  Feeding it straight into the
# SpeedController made the pedals oscillate throttle -> brake -> throttle
# every other frame (fix54/fix56).  Ramp the effective target toward the
# plan at a bounded rate instead.
SPEED_TARGET_RAMP_MPS = 1.5     # m/s per sim second
# Steering shaping (plan phase D3): the same first-order rate the FSD loop
# always used (smooth_steer rate=0.8), now with the SECOND-order (jerk)
# bound and the small-amplitude reversal guard on top - the wobble the
# plan asks to remove is a few centimetres of request flipping several
# times a second, which a rate limit alone cannot see.
FSD_STEER_RATE_PER_S = 0.8
# D2 steering blend (opt-in): Pure Pursuit stays the base; the
# lateral/heading feedback terms are added only when an A/B turns
# this on, and their speed schedule lives in control/blend.py.
STEER_BLEND_ENABLED = (
    os.environ.get("BEAMNG_STEER_BLEND", "0") == "1")
# D4 longitudinal planner (opt-in): composes curvature / lateral
# accel / obstacle TTC / perception confidence into the target, then
# shapes the reference speed with acceleration AND jerk limits.  Off
# by default: the accel/jerk shape differs from the first-order ramp
# it replaces, so it is a live A/B, not a silent default.
LONG_PLAN_ENABLED = os.environ.get("BEAMNG_LONG_PLAN", "0") == "1"
# D5 driving modes (opt-in): STARTING / LOW_SPEED_ALIGNMENT /
# CRUISING / CURVE_ENTRY / OBSTACLE_BRAKING / CONTROLLED_STOP /
# RECOVERY as an explicit classification with per-mode policy
# (launch steering authority, throttle-vs-brake suppression).
DRIVE_MODES_ENABLED = (
    os.environ.get("BEAMNG_DRIVE_MODES", "0") == "1")
# SpeedController pedal band: brake enters at err < -(deadband+hyst),
# throttle at err > +(deadband+hyst).  The old 0.35/0.4 pair made the
# car coast a +/-0.75 m/s band around cruise (5.25 <-> 7.0), which
# read as a visible accelerate -> brake -> accelerate wave; 0.2/0.25
# keeps a real hysteresis but halves the band to +/-0.45 m/s.
SPEED_DEADBAND_MPS = 0.2        # SpeedController coast deadband
SPEED_HYST_MPS = 0.25           # SpeedController brake/throttle hysteresis
# Plan-speed brake governor hysteresis: enter at +1.0 m/s overshoot,
# release once back within +0.5 m/s, and brake gently (0.25) instead of
# 1.0 - BeamNG's brake is highly nonlinear and even 0.4-0.7 stands the
# car dead in one 0.33 s burst, which then re-triggers the full-throttle
# stall loop (fix61-64).  The downhill-start guard below prevents the
# overshoot in the first place.
# Heading-error speed scrub: with a ~0.6 s control tick a 7-8 m/s
# car keeps rotating long after the pure-pursuit correction, and
# the loop oscillates instead of converging (opt12 2026-08-27: the
# heading swung -99 -> -131 -> -94 deg and the car drifted into the
# right-side wall).  Once the nose points more than START_DEG away
# from the nav route, ramp the target down to FLOOR at FULL_DEG so
# the steering loop can catch the rotation at a speed it can
# control.  Real bends (hairpins) are already slowed by the corner
# governor, so this only trims the oscillation case.
HEADING_DEV_START_DEG = 12.0
HEADING_DEV_FULL_DEG = 40.0
HEADING_DEV_CAP_MPS = 5.0
HEADING_DEV_FLOOR_MPS = 1.5
# Off-road recovery is PERCEPTION ONLY.  ``_perception_off_road_m``
# measures how far the ego BODY sticks out past a boundary the vision /
# LiDAR chain detected this tick.  The old map guard (nav centreline +
# DecalRoad edge half-width) was dead code - ``road_off`` was assigned
# 0.0 and never updated, so the crawl / recovery branches could not
# fire - and map lateral authority is banned in the FSD entry anyway.
# More than this far past a DETECTED edge means the car is definitively
# out of its lane: hard stop + hold, never creep further out, never
# reverse (``off_recover`` also suppresses the reverse escape).
ROAD_OFF_STOP_M = 2.0
# Map-prior lane centre EMA: map_lane_edges re-derives the own-lane
# centre every frame from the road edges; at junctions / edge
# sampling switches it can wiggle a few degrees between frames and
# feed the steering loop (opt12: map_lane bearing swung ~7 deg).
# Blend the new centre with the previous one (arc-aligned) so the
# reference stays smooth; boundaries are left untouched.
MAP_LANE_EMA = 0.6
# Start placement is PERCEPTION ONLY: after the semantic head warms up,
# the drive re-positions itself into its own lane by measuring the
# painted line (line right side + own-lane half width).  The pre-warm
# ground snap must not ride the route centre line, and it must not use a
# fixed "route centre + offset" constant either - that is the map-prior
# shortcut the FSD realism rules forbid.  (2026-09-03: SNAP_LANE_OFFSET_M
# removed; placement now comes from painted-line perception.)
# End-of-route handling: the nav route is finite.  When the local
# forward window reaches the route END, the car has arrived; without
# an explicit stop it creeps onto the road end / kerb and parks over
# the edge line (opt15 2026-08-28: parked at the destination with
# lat_right -0.3~-0.5 m, i.e. over the right line).  Slow from 10 m
# out and stop from 4 m out, while the planner still keeps the lane
# centre, so the car ends gently in its lane before the road end.
END_START_SLOW_M = 12.0
# Lateral re-centring starts earlier than the braking zone: the car
# enters the final stretch ~1.0-1.4 m from the centreline and the
# 12-6 m stop zone alone cannot pull it to the lane centre before the
# stop (opt35-41 parked 1.4-1.7 m off-centre).  From 20 m out the
# steering reference aims at the own-lane centre ahead; braking still
# starts at END_START_SLOW_M.
END_PULL_START_M = 20.0
END_STOP_M = 6.0
# End-zone final alignment: a mid-turn full-hold parks the car diagonal
# across the road (town 2026-09-06, user screenshot: ~35 deg yaw across
# both lanes).  Creep forward straightening until the body yaw is within
# ALIGN_YAW_DEG of the route, then hold.
ALIGN_YAW_DEG = 6.0
ALIGN_CREEP_MPS = 0.5
ALIGN_CREEP_THR = 0.25
END_SLOW_MPS = 1.0
END_BRAKE = 0.7
# End-zone last-good perception anchor: if the painted line flickers
# or fades inside the final stop zone, keep converging to the LAST
# perceived own-lane centre (still perception - never a map prior)
# for a short window, so a line dropout does not silently drop the
# lateral pull and park the car wherever it entered the zone.
END_PLC_HOLD_S = 2.0
END_PLC_MAX_LAT_M = 3.0
END_PLC_MAX_FWD_M = 15.0
# Steady painted-line lateral corrector (FSD realism, perception only):
# telemetry showed ``lane_src_sel=map`` for 100% of a run while the
# semantic LINE mask gave a confident own-lane centre - the map-prior
# own lane rides on (or too close to) the painted centre line and the
# car never uses its perception lateral reference in steady driving.  A
# real FSD stack places the car where ITS sensors say the lane is: while
# cruising with the map lane leading, nudge the near path toward the
# perceived own-lane centre (line right side + lane half width) at a
# bounded rate.  No nav-centreline + offset constant anywhere - the
# target is the same perception rule as start placement / end zone.
# The painted line IS the lateral authority (iron rule).  The town link
# runs with the map lane offset ~0.7-1.0 m RIGHT of the painted lane
# (line_lat +0.7..+1.25 vs +1.75 centred), so the corrector must be
# able to shift the FULL measured offset, and hold it through the
# line-visibility gaps (the semantic line class is sparse on this
# link: visible ~20-25% of frames).
PLC_MAX_SHIFT_M = 1.6
PLC_RATE_MPS = 1.2
PLC_HOLD_S = 4.0
PLC_HORIZON_M = 12.0
PLC_HOLD_S = 2.0
PLC_MIN_SPEED_MPS = 0.5
PLC_MIN_ENGAGE_M = 0.02
GOV_ON_MPS = 0.8
GOV_OFF_MPS = 0.4
GOV_BRAKE = 0.25
# Tight-bend governor gate: only apply the hairpin speed cap when the
# 12 m look-ahead actually TURNS this much.  A rounded junction corner /
# resample wiggle can measure R~3 m while turning <30 deg; capping there
# made the car crawl 5+ s through a widening junction (fsd opt23).
BEND_GOV_MIN_TURN_DEG = 40.0
# Plan-speed rate limit (m/s per sim second): the full-route profile is
# re-profiled every frame against the LIVE LiDAR grid, and junction /
# end-zone clutter appears and disappears between ticks.  Without a rate
# limit the plan snapped 6.0 <-> 1.0 and the controller slammed the
# brakes then relaunched (opt22 t=50.6: plan=1.00 brk=0.75, next frame
# plan=6.00).  The down rate stays above SPEED_TARGET_RAMP so real bend
# deceleration still binds; the up rate is gentler so a transient
# obstruction never ends in a relaunch kick.
PLAN_DOWN_RATE_MPS2 = 2.0
PLAN_UP_RATE_MPS2 = 1.0



def _trim_backtrack(route):
    """Drop a terminal backtracking tail from a road-graph route.

    When the goal is NOT on the road network, ``route_with_edges`` appends
    a straight segment from the last road node to the off-road goal - the
    route doubles back against the road direction and the final ~2 m of
    centreline / edge data point the wrong way (the end-zone reference
    then aimed left and the car parked 1.4-1.7 m off the lane centre,
    opt35-38).  The end-zone reference must only use the real road part,
    so cut the route at the first terminal direction reversal.
    """
    r = np.asarray(route[:, :2], dtype=float)
    n = len(r)
    if n < 6:
        return r
    seg = np.diff(r, axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    for k in range(n - 3, 2, -1):
        a = ang[max(0, k - 4):k]
        b = ang[k:k + 2]
        if len(a) and len(b):
            da = np.arctan2(np.sin(a - b[0]), np.cos(a - b[0]))
            if float(np.median(np.abs(da))) > 2.0:
                return r[:k]
    return r


def _painted_line_lat(out, pos, heading, marks=None):
    """Painted centre-line lateral relative to the ego (left = +).

    Projects the semantic LINE mask's near-field pixels to the ground
    with the tick's own camera model, so every run self-reports whether
    the car sits LEFT (oncoming) or RIGHT (own lane) of the painted
    line - the objective check that replaces eyeballing telemetry.
    Returns None when no line is detected this frame.
    """
    if out is None or out.frame is None or out.cam is None:
        return None
    sem = out.head_outputs.get("semantic")
    if sem is None or "line" not in getattr(sem, "masks", {}):
        return None
    try:
        if marks is None:
            ground_z = (float(pos[2]) - config.EGO_ORIGIN_GROUND_GAP_M
                        if len(pos) > 2 else None)
            marks = painted_line_markings(sem, out.cam, pos, heading,
                                          ground_z=ground_z,
                                          rgb=out.frame)
        if not marks:
            return None
        fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
        left = np.array([-fwd[1], fwd[0]])
        lats = []
        p2 = np.asarray(pos[:2], dtype=float)
        for m in marks:
            pts = np.asarray(m.world[:, :2], dtype=float)
            near = pts[np.linalg.norm(pts - p2, axis=1) < 25.0]
            if len(near):
                lats.append(float(((near - p2) @ left).mean()))
        return round(float(np.mean(lats)), 3) if lats else None
    except Exception:
        return None


def _path_errors_steer_frame(path, pos, heading: float):
    """(lateral_error_m, heading_error_rad) in the STEER-feedback frame.

    Both are expressed as what the wheel should do: positive = steer
    right.  Lateral is positive when the ego sits left of the path;
    heading is positive when the nose must turn right (BeamNG heading
    decreases when steering right, so ``heading - path_dir`` already has
    that sign).  Measured against the trajectory the planner published -
    never a map line or an offset constant.
    """
    if path is None:
        return 0.0, 0.0
    pth = np.asarray(path, dtype=float)[:, :2]
    if len(pth) < 2:
        return 0.0, 0.0
    p = np.asarray(pos, dtype=float)[:2]
    a, b = pth[:-1], pth[1:]
    ab = b - a
    l2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    t = np.clip(((p[None, :] - a) * ab).sum(axis=1) / l2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.linalg.norm(proj - p[None, :], axis=1)
    j = int(np.argmin(d))
    seg = ab[j]
    length = float(np.linalg.norm(seg))
    if length < 1e-9:
        return 0.0, 0.0
    u = seg / length
    off = p - proj[j]
    # cross(u, off) > 0  <=>  the ego is LEFT of the path direction
    lat = float(u[0] * off[1] - u[1] * off[0])
    head = (float(heading) - float(math.atan2(u[1], u[0])) + math.pi)         % (2.0 * math.pi) - math.pi
    return lat, head


def _ref_bearing(ref, pos, min_m: float = 1.5, max_m: float = 20.0):
    """Bearing (deg) of the near-ahead part of a reference polyline."""
    if ref is None or len(ref) < 2:
        return None
    r = np.asarray(ref[:, :2], dtype=float)
    p = np.asarray(pos[:2], dtype=float)
    d = np.linalg.norm(r - p, axis=1)
    sel = np.flatnonzero((d >= min_m) & (d <= max_m))
    if len(sel) < 2:
        sel = np.flatnonzero(d >= min_m)
    if len(sel) < 2:
        return None
    i, j = int(sel[0]), int(sel[-1])
    v = r[j] - r[i]
    L = float(np.linalg.norm(v))
    if L < 1e-9:
        return None
    return round(float(math.degrees(math.atan2(v[1], v[0]))), 1)


def _perception_off_road_m(out, pos, heading) -> float:
    """How far the ego BODY sticks out past a DETECTED boundary (m).

    Perception-only replacement for the old map ``road_off`` metric: the
    four corners of the one authoritative footprint (``vehicle_body``)
    are tested against the lane boundaries this tick's vision / LiDAR
    chain published, and the worst signed overshoot is returned (>= 0).

    No map / DecalRoad fallback: the FSD entry takes lateral authority
    from sensors only (project rule), and a missing boundary is
    "unknown" - 0.0 - never a map guess.  Callers use the value for the
    off-road telemetry, the learned decision layer's ``road_off`` input
    and the definitively-off-lane hard stop.
    """
    if out is None:
        return 0.0
    left = getattr(out, "lane_left", None)
    right = getattr(out, "lane_right", None)
    if left is None and right is None:
        return 0.0
    try:
        p = np.asarray(pos, dtype=float).ravel()[:2]
        if p.size < 2 or not np.isfinite(p).all():
            return 0.0
        corners = footprint_corners(p, float(heading))
    except Exception:
        return 0.0
    worst = 0.0
    for c in corners:
        if left is not None:
            try:
                lat, cov = _boundary_lateral(
                    float(c[0]), float(c[1]), left, None)
                if cov and float(lat) > worst:
                    worst = float(lat)
            except Exception:
                pass
        if right is not None:
            try:
                lat, cov = _boundary_lateral(
                    float(c[0]), float(c[1]), right, None)
                if cov and -float(lat) > worst:
                    worst = -float(lat)
            except Exception:
                pass
    return float(worst)


# Travel-direction sources that count as PERCEPTION in the end zone
# ("painted" = semantic line direction, "sensor_lane" = paired LiDAR /
# camera lane centreline).  "route" is only ever a plain orientation
# fallback for the stop RAY, never a steering reference.
PERCEPTION_DIR_SRCS = ("painted", "sensor_lane")


def _in_end_pull_zone(rem_end, start_m: float = END_PULL_START_M) -> bool:
    """True while the end-zone ease/hold ladder owns the longitudinal target.

    Inside this zone the learned decision policy is skipped so its
    "slow"/"ease" action cannot fight the deterministic end-stop and
    alignment creep (see the DQN layer in the drive loop).
    """
    if rem_end is None:
        return False
    try:
        return float(rem_end) < float(start_m)
    except (TypeError, ValueError):
        return False


def _needs_hard_pedal(*, has_path: bool, force_stop: bool, stuck: bool,
                      climb: bool, reverse_active: bool,
                      reversing: bool, rem_end: float | None) -> bool:
    """Whether the FINAL pedals must bypass the rate limiter this tick.

    Every branch here is a state where the previous tick's gentle pedal is
    actively dangerous:

    * ``force_stop`` / ``stuck`` / climb / reverse escape / end-zone hold -
      the original set (opt23's speed-kick guard);
    * ``not has_path`` - the planner published NO path (strict
      no-perception-lane, "no drivable path"): the stop branch commands
      brake 1.0 + handbrake, and the coast-down distance at 1.9 m/s while
      the ramp caught up measured 2.8 m - enough to leave the pavement
      and bury the car against a tree (east_coast 2026-09-19, user
      photo).  A no-path tick is an emergency brake, not a comfort
      transition.
    """
    return bool(force_stop or stuck or climb or reverse_active or reversing
                or (rem_end is not None and rem_end < END_STOP_M)
                or not has_path)


def _counts_as_stuck(*, has_path: bool, force_stop: bool, v: float,
                     thr: float, plan_speed: float, near_obs_m: float,
                     rem_end, v_eps: float = 0.35, plan_eps: float = 0.05,
                     obs_close_m: float = 2.5) -> bool:
    """True when this tick should advance the stuck timer.

    Two ways the car parks itself while the stack still reports "safe" with
    a live path:

    * holding throttle against an obstruction (``thr > 0``) - the original
      "spinning in place" case (mountain run 2026-08-27 run_fix31: wedged
      at (741.2,745.7) with thr=0.53 and v=0 for 50 s); and
    * a COMMANDED stop with an obstacle inside the brake reserve: the speed
      profile clamps to 0, the controller brakes so ``thr == 0`` and the
      first rule can never fire.  Measured 2026-09-11: closest_obs pinned
      at 1.17 m with plan_speed 0.00 for 164 of 224 town frames, so the run
      covered 18.4 m and then sat for 73% of the clock.

    The commanded-stop branch needs the obstacle to actually be CLOSE so
    waiting behind a lead vehicle (which sits further out) is not mistaken
    for a wedge, and the end-pull zone is excluded because a commanded stop
    there is the goal, not a failure.  Both branches arm the same bounded
    reverse escape (1.5 m / 2.5 s / -0.4 m/s, rear-clearance checked).
    """
    if not has_path or force_stop or float(v) >= float(v_eps):
        return False
    if float(thr) > 0.0:
        return True
    return (float(plan_speed) <= float(plan_eps)
            and float(near_obs_m) <= float(obs_close_m)
            and not _in_end_pull_zone(rem_end))


def _endzone_align_yaw_dev(heading, dir3, dir_src):
    """Yaw deviation to straighten the parking pose to, or None.

    The end zone creeps forward while straightening when a mid-turn hold
    would park the body diagonally across the lane (town 2026-09-06).
    "Which way is my lane" must come from PERCEPTION - using the nav
    route tangent as a steering reference is the map-prior shortcut the
    project rule forbids in the FSD entry (``_ref_bearing`` on the route
    used to feed the alignment creep).

    Returns the signed deviation (rad) to the perceived lane direction
    when ``dir_src`` is a perception source, else None so the caller
    holds the brake instead of steering on a map prior - the legal
    no-perception degradation (stop, never steer on map).
    """
    if dir_src not in PERCEPTION_DIR_SRCS or dir3 is None:
        return None
    try:
        d = np.asarray(dir3, dtype=float).ravel()[:2]
        if d.size < 2 or not np.isfinite(d).all():
            return None
        if float(np.hypot(float(d[0]), float(d[1]))) < 1e-6:
            return None
        bear_deg = math.degrees(math.atan2(float(d[1]), float(d[0])))
        return math.radians(
            (float(heading) * 57.29577951308232 - bear_deg + 180.0)
            % 360.0 - 180.0)
    except Exception:
        return None


def _unit_dir2(value):
    """Unit 2-D direction from a candidate, or None when unusable."""
    if value is None:
        return None
    try:
        d = np.asarray(value, dtype=float).ravel()
        if d.size < 2:
            return None
        d = d[:2]
        if not np.isfinite(d).all():
            return None
        n = float(np.hypot(float(d[0]), float(d[1])))
        if n < 1e-6:
            return None
        return d / n
    except Exception:
        return None


def _endzone_travel_direction(heading, painted=None, sensor=None,
                              route_tangent=None, strict=False):
    """Resolve the end-zone travel orientation: perception first.

    Priority: the semantic painted-line direction, then the paired
    sensor-lane centreline heading.  A nav-route tangent stays a plain
    orientation fallback ONLY in the legacy non-strict mode; strict FSD
    holds the current heading instead, because route geometry must never
    steer the car there (``docs/fsd_realism.md`` §1/§4 and the project
    lateral rule).  Returns ``(dir3, src)`` with ``src`` in
    "painted" / "sensor_lane" / "route" / "none" ("none" = heading hold,
    the legal no-perception degradation).
    """
    for cand, src in ((painted, "painted"), (sensor, "sensor_lane")):
        d = _unit_dir2(cand)
        if d is not None:
            return d, src
    if not strict:
        d = _unit_dir2(route_tangent)
        if d is not None:
            return d, "route"
    hf = _unit_dir2((math.cos(float(heading)), math.sin(float(heading))))
    if hf is None:
        hf = np.array([1.0, 0.0])
    return hf, "none"


def _path_radius_m(path, pos, heading, near_m: float = 1.5,
                   horizon_m: float = 20.0):
    """Smallest curve radius on the near-ahead path, or None.

    Same near-ahead window the curvature feed-forward uses; the radius
    comes from the standard three-point curvature
    ``2*|cross| / (n1*n2*(n1+n2))`` between consecutive path segments,
    which is independent of the sampling step.  ``None`` means "no
    measurable bend", which the longitudinal planner reads as "no
    curvature cap" - absent evidence must not invent a penalty.
    """
    if path is None:
        return None
    p = np.asarray(path, dtype=float)[:, :2]
    if len(p) < 3:
        return None
    pos2 = np.asarray(pos, dtype=float)[:2]
    d = np.linalg.norm(p - pos2, axis=1)
    i0 = int(np.argmin(d))
    fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    base = float(arc[i0])
    sel = [j for j in range(i0, len(p) - 1)
           if float(arc[j]) - base <= horizon_m
           and float((p[j] - pos2) @ fwd) >= near_m]
    if len(sel) < 3:
        return None
    seg = np.diff(p[sel[0]:sel[-1] + 1], axis=0)
    n1 = np.linalg.norm(seg, axis=1)
    n2 = n1[1:]
    cross = np.abs(seg[:-1, 0] * seg[1:, 1] - seg[:-1, 1] * seg[1:, 0])
    curv = 2.0 * cross / np.maximum(n1[:-1] * n2 * (n1[:-1] + n2), 1e-9)
    curv = curv[curv > 1e-6]
    if len(curv) == 0:
        return None
    return float(1.0 / float(np.max(curv)))


def _path_curvature_ff(path, pos, heading, near_m: float = 1.5,
                       horizon_m: float = 8.0, wheelbase: float = 2.9,
                       ratio: float = 0.6, max_ff: float = 0.40) -> float:
    """Feed-forward steering from the chosen path's near-ahead curvature.

    The ~1.4 s control loop only reacts to the PurePursuit target at the
    lookahead point, so at 2 m/s the car has already passed the entry of
    a hairpin before the pursuit asks for the turn (fix37-41 runs: the
    first -110 -> -24 deg bend was missed every time and the car ran
    straight past the apex).  A feed-forward term from the path curvature
    2-10 m ahead starts the turn as soon as the path bends.

    Returns a NORMALIZED steering input (negative = left), scaled by how
    aligned the ego heading is with the path so a sideways rejoin is not
    fought.  ``ratio`` is the rad-per-normalized-input steering ratio used
    by the pursuit conversion below.
    """
    if path is None or len(path) < 4:
        return 0.0
    p = np.asarray(path[:, :2], dtype=float)
    pos2 = np.asarray(pos[:2], dtype=float)
    n = len(p)
    d = np.linalg.norm(p - pos2, axis=1)
    i0 = int(np.argmin(d))
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    base = float(arc[i0])
    idxs = [i0]
    for tgt in (near_m, near_m + horizon_m):
        j = i0
        while j < n - 1 and float(arc[j]) - base < tgt:
            j += 1
        idxs.append(j)
    i1, i2 = idxs[1], idxs[2]
    if i2 - i1 < 2:
        return 0.0

    def _tangent(i: int) -> np.ndarray:
        a = max(0, i - 1)
        b = min(n - 1, i + 1)
        v = p[b] - p[a]
        L = float(np.linalg.norm(v))
        return (v / L) if L > 1e-9 else np.array([1.0, 0.0])

    t1 = _tangent(i1)
    t2 = _tangent(i2)
    th1 = math.atan2(float(t1[1]), float(t1[0]))
    th2 = math.atan2(float(t2[1]), float(t2[0]))
    dth = (th2 - th1 + math.pi) % (2.0 * math.pi) - math.pi
    ds = max(1e-3, float(arc[i2] - arc[i1]))
    kappa = dth / ds
    align = float(np.clip(
        math.cos(th1 - float(heading)), 0.0, 1.0))
    ff = -kappa * wheelbase / ratio   # left curve (kappa>0) -> negative input
    return float(np.clip(ff * (0.3 + 0.7 * align), -max_ff, max_ff))



def _snap_heading(nav_route, rx: float, ry: float, h_seg: float,
                  max_diff_deg: float = 45.0,
                  reference_heading: float | None = None,
                  reference_max_yaw_deg: float = 60.0) -> float:
    """Heading the spawn snap should face the car along.

    ``h_seg`` is the bearing of the route's FIRST interpolated segment
    (nearest vertex -> next vertex).  On a road-graph A* route that first
    segment can be a graph diagonal - the chain zigzags between nodes
    before settling onto the road - while the road itself runs a very
    different way.  The snap therefore faces the car along the route's
    near-ahead bearing (1.5-20 m window, which averages out the start
    zigzag) whenever the two disagree by more than ``max_diff_deg``.

    ``reference_heading`` is the current road-aligned heading.  When BOTH
    the route's first segment and its near-ahead bearing disagree with
    that road heading, the route is a backwards/U-turn route from this
    start (2026-09-18: goal (56,869) made the car spawn at 172 deg on a
    road whose local tangent was -108 deg; it immediately entered the
    obstacle/guardrail area).  In that case keep the road heading instead
    of turning the car across the lane.  The caller may then fail closed
    or choose a forward goal; it must never drive across the road just to
    satisfy a destination route.
    """
    b = polyline_bearing(np.asarray(nav_route, dtype=float)[:, :2],
                         np.array([rx, ry], dtype=float))
    if b is None:
        return (float(reference_heading) if reference_heading is not None
                else h_seg)
    if (reference_heading is not None
            and abs(bearing_diff_deg(b, reference_heading))
            > float(reference_max_yaw_deg)
            and abs(bearing_diff_deg(h_seg, reference_heading))
            > float(reference_max_yaw_deg)):
        return float(reference_heading)
    if abs(bearing_diff_deg(b, h_seg)) > max_diff_deg:
        return b
    return h_seg


def _spawn_traffic(conn, nav_route, n: int,
                   models=("pessima", "etk800", "pickuptd"),
                   offset_right_m: float = 3.4,
                   first_m: float = 80.0, gap_m: float = 60.0) -> int:
    """Park ``n`` NPC vehicles along the nav route (right roadside).

    Placed just outside the ego lane's right edge (≈ ``offset_right_m``
    right of the road centreline) so the front camera sees real vehicles
    for the YOLO head / obstacle fusion while a correct drive can still
    pass within the no-cross policy.  Uses the beamngpy vehicles API
    (``spawn`` / ``despawn``) and first clears NPCs left over from
    earlier runs.  Best-effort: failures log and the function returns
    how many vehicles actually spawned.  Pure vehicle spawning - no map
    offsets are used for the EGO's lateral reference.
    """
    if nav_route is None or len(nav_route) < 2 or int(n) <= 0:
        return 0
    bng = conn.bng
    ego_vid = getattr(conn.vehicle, "vid", None)
    placed = 0
    try:
        from beamngpy import Vehicle
        r = np.asarray(nav_route, dtype=float)[:, :2]
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))])
        total = float(arc[-1])
        # Remove vehicles left over from earlier runs so NPCs never stack
        # at the same arc positions (town run 1 left clone/clone0/...).
        for vid, veh in list(bng.get_current_vehicles().items()):
            if vid == ego_vid:
                continue
            try:
                bng.vehicles.despawn(veh)
            except Exception:
                pass
        # Spread over the route: a short route would cap every NPC onto
        # the same arc position (town: two stacked cars formed a
        # mid-lane barrier the no-cross policy correctly refuses to
        # pass), so the start shrinks to half the usable route first.
        first = min(float(first_m), max(20.0, (total - 15.0) * 0.5))
        for k in range(int(n)):
            s = min(total - 15.0, first + float(gap_m) * k)
            if s < 20.0:
                continue
            i = int(np.searchsorted(arc, s))
            i = min(max(i, 1), len(r) - 2)
            seg = r[i + 1] - r[i]
            L = float(np.linalg.norm(seg))
            if L < 1e-6:
                continue
            d = seg / L
            right = np.array([d[1], -d[0]])   # right of travel
            px, py = r[i] + float(offset_right_m) * right
            gz = conn.ground_z_at(float(px), float(py))
            z = (float(gz) + 0.5) if gz is not None else 1.0
            yaw_deg = -math.degrees(math.atan2(d[1], d[0])) - 90.0
            npc = Vehicle("npc_%d" % k, model=models[k % len(models)])
            ok = bng.vehicles.spawn(
                npc, pos=(float(px), float(py), z),
                rot_quat=angle_to_quat((0.0, 0.0, yaw_deg)),
                cling=True, connect=False)
            if ok:
                placed += 1
                print(f"[fsd-drive] traffic NPC #{k + 1}: "
                      f"{models[k % len(models)]} at ({px:.1f}, {py:.1f})")
            else:
                print(f"[fsd-drive] traffic NPC #{k + 1} spawn rejected")
    except Exception as exc:
        print(f"[fsd-drive] traffic spawn failed: {exc}")
    return placed


def _sensor_snapshot_age(out) -> float:
    """Max age of road-driving modalities used by one FSD tick.

    The canonical snapshot also reports throttled optional heads (object /
    traffic / topology) and reusable range age.  Those remain visible in
    snapshot freshness telemetry, but they must not park a fresh semantic
    lane/BEV stack; range has its own reuse horizon in SafetyMonitor.
    """
    if out is None:
        return float("inf")
    snapshot = getattr(out, "snapshot", None)
    if snapshot is not None:
        # The canonical perception snapshot is the freshness contract.
        # Legacy tick fields remain only for stubs that predate it.
        if not bool(getattr(snapshot, "valid", False)):
            return float("inf")
        ages: list[float] = []
        heads = getattr(snapshot, "head_age_s", {}) or {}
        if heads:
            semantic_age = heads.get("semantic")
            if semantic_age is None:
                return float("inf")
            ages.append(float(semantic_age))
        if getattr(snapshot, "bev_age_s", None) is not None:
            ages.append(float(snapshot.bev_age_s))
        envelope = getattr(snapshot, "lane_envelope", None)
        if envelope is not None:
            lane_age = getattr(envelope, "age_s", None)
            if lane_age is None:
                return float("inf")
            ages.append(float(lane_age))
        if not ages:
            return 0.0
        age = max(ages)
        return age if math.isfinite(age) else float("inf")
    meta = getattr(out, "meta", {}) or {}
    raw_heads = meta.get("head_age_s", {}) or {}
    ages: list[float] = []
    if raw_heads:
        semantic_age = raw_heads.get("semantic")
        if semantic_age is None:
            return float("inf")
        ages.append(float(semantic_age))
    if meta.get("bev_age_s") is not None:
        ages.append(float(meta["bev_age_s"]))
    envelope = getattr(out, "lane_envelope", None)
    if envelope is not None:
        lane_age = getattr(envelope, "age_s", None)
        if lane_age is None:
            return float("inf")
        ages.append(float(lane_age))
    # A tick with no frame and no head outputs has no fresh camera evidence.
    if getattr(out, "frame", None) is None \
            and not getattr(out, "head_outputs", None):
        return float("inf")
    return max([0.0] + ages)


def _pose_geometry_ok(out, pos, heading: float, pts: np.ndarray,
                      src: str, dbg: dict | None = None,
                      max_yaw_deg: float = 25.0) -> bool:
    """Pose-geometry half of the placement predicate.

    Checks (perception geometry only): the pose points along the observed
    lane direction, the inflated BODY fits inside the published
    boundaries, and every published boundary sits on its OWN side of the
    car.  Deliberately WITHOUT the paired-lane requirement - that
    requirement belongs to the RELEASE decision; this half also decides
    whether the pose itself still needs the alignment teleport.
    """
    if heading is None:
        return True
    p = np.asarray(pos[:2], dtype=float)
    if src == SRC_PAVED:
        dir_pts = None
        for _e in (getattr(out, "lane_right", None),
                   getattr(out, "lane_left", None)):
            if _e is None:
                continue
            _ea = np.asarray(_e, dtype=float)
            if _ea.ndim == 2 and _ea.shape[1] >= 2 and len(_ea) >= 3 \
                    and np.isfinite(_ea[:, :2]).all():
                dir_pts = _ea[:, :2]
                break
        if dir_pts is None:
            return False
    else:
        _dir_edge = None
        for _e in (getattr(out, "lane_left", None),
                   getattr(out, "lane_right", None)):
            if _e is None:
                continue
            _ea = np.asarray(_e, dtype=float)
            if _ea.ndim == 2 and _ea.shape[1] >= 2 and len(_ea) >= 3 \
                    and np.isfinite(_ea[:, :2]).all():
                _dir_edge = _ea[:, :2]
                break
        dir_pts = _dir_edge if _dir_edge is not None else pts
    local = polyline_dir_at(dir_pts, p)
    if local is None:
        if dbg is not None:
            dbg["geom"] = "no_dir"
        return False
    hf = np.array([math.cos(float(heading)), math.sin(float(heading))])
    # 25 deg, not the drive gate's 12: the near-field edge polyline is
    # short and its fitted direction carries 12-16 deg of noise on a bend
    # (live diag 2026-09-19), while a car actually lying across the lane
    # reads 60-90 deg.  The gate exists to refuse the latter.
    if not (float(local @ hf) >= math.cos(math.radians(max_yaw_deg))):
        if dbg is not None:
            _dyaw = math.degrees(math.acos(
                max(-1.0, min(1.0, float(local @ hf)))))
            dbg["geom"] = f"yaw({_dyaw:.1f}deg)"
        return False
    # The whole BODY must sit inside the lane with a margin, not just the
    # centre: a pose 0.17 m inside passes every check above and is then
    # frozen by the first tick's body-cross gate (see
    # PLACEMENT_BODY_CLEARANCE_M).  Inflating the body rectangle by the
    # margin reuses the existing penetration geometry as a clearance
    # test; a missing boundary side stays "unknown" (0.0), never a guess.
    left = getattr(out, "lane_left", None)
    right = getattr(out, "lane_right", None)
    if left is not None or right is not None:
        depth = body_pose_cross_depth_m(
            np.asarray(pos, dtype=float).ravel(), float(heading),
            left, right,
            half_len=HALF_LENGTH_M + PLACEMENT_BODY_CLEARANCE_M,
            half_width=HALF_WIDTH_M + PLACEMENT_BODY_CLEARANCE_M)
        if float(depth) > 0.0:
            if dbg is not None:
                dbg["geom"] = f"body({float(depth):.2f})"
            return False
        # Each published boundary must also be on its CORRECT side of the
        # car.  A centre-paint frame publishes the paint as the LEFT
        # boundary; a car spawned left of the paint (teleport lands on the
        # road node, the oncoming side) has that "left" boundary 1.4 m to
        # its RIGHT.  The re-anchored lane centre makes any distance test
        # vacuous (the polyline starts at the ego), and the paint fragment
        # starts ~5 m ahead, so the body-clearance check cannot see it
        # either - every forward path then crosses the physical paint
        # 5-8 m ahead and the run stalls in planned-body-cross stops
        # (base_speed3 t=2.3-7.2, 2026-09-19).  The SIDE of each observed
        # boundary is the honest placement evidence: a left boundary
        # belongs left of the car, a right boundary right of it.
        lf = np.array([-hf[1], hf[0]])
        for _poly, _side in ((left, 1.0), (right, -1.0)):
            if _poly is None:
                continue
            _pa = np.asarray(_poly, dtype=float)[:, :2]
            if _pa.ndim != 2 or len(_pa) < 2 or not np.isfinite(_pa).all():
                return False
            _rel = _pa - p[None, :]
            _lon = _rel @ hf
            _lat = _rel @ lf
            _win = (_lon >= 1.5) & (_lon <= 8.0)
            _tag = "L" if _side > 0 else "R"
            if not _win.any():
                if dbg is not None:
                    dbg[f"side_{_tag}"] = "no_evidence"
                continue          # no boundary evidence beside the car
            _med = float(np.median(_lat[_win]))
            if dbg is not None:
                dbg[f"side_{_tag}"] = round(_med, 2)
            if _side * _med < PLACEMENT_BOUNDARY_SIDE_MIN_M:
                if dbg is not None:
                    dbg["geom"] = f"side_{_tag}({_med:+.2f})"
                return False      # boundary on the wrong side: not placed
    return True


# Lateral clearance the nudge aims for on each published boundary (the
# SIDE_MIN gate plus a convergence margin so one nudge suffices).
_PLACEMENT_NUDGE_CLEAR_M = PLACEMENT_BOUNDARY_SIDE_MIN_M + 0.35


def _alignment_nudge_m(out, pos, heading: float):
    """World-space (dx, dy) that restores the flagged boundary geometry.

    Computed ONLY from the tick's published boundaries: shifting the car
    by d changes each boundary's car-frame lateral by +d (right = -lf), so
    a left boundary reading side_L below the clearance needs a rightward
    nudge of (clear - side_L), and a right boundary above -clear needs a
    leftward one.  Same perception-anchored class as the painted-line
    placement shift - no map line or fixed offset enters.  None when the
    geometry failure is not a boundary-side problem (yaw / body) or when
    both sides already read inside the target band.
    """
    if out is None:
        return None
    left = getattr(out, "lane_left", None)
    right = getattr(out, "lane_right", None)
    if left is None and right is None:
        return None
    p = np.asarray(pos[:2], dtype=float)
    hf = np.array([math.cos(float(heading)), math.sin(float(heading))])
    lf = np.array([-hf[1], hf[0]])

    def _med(poly):
        pa = np.asarray(poly, dtype=float)[:, :2]
        if pa.ndim != 2 or len(pa) < 2 or not np.isfinite(pa).all():
            return None
        rel = pa - p[None, :]
        lon = rel @ hf
        lat = rel @ lf
        win = (lon >= 1.5) & (lon <= 8.0)
        if not win.any():
            return None
        return float(np.median(lat[win]))

    side_l = _med(left) if left is not None else None
    side_r = _med(right) if right is not None else None
    shift = 0.0
    if side_l is not None and side_l < _PLACEMENT_NUDGE_CLEAR_M:
        shift = min(2.5, _PLACEMENT_NUDGE_CLEAR_M - side_l)      # rightward
    elif side_r is not None and side_r > -_PLACEMENT_NUDGE_CLEAR_M:
        shift = -min(2.5, side_r + _PLACEMENT_NUDGE_CLEAR_M)     # leftward
    else:
        return None
    return (float(p[0] - shift * lf[0]), float(p[1] - shift * lf[1]))


def _pose_needs_alignment(out, pos, heading: float,
                          dbg: dict | None = None) -> bool:
    """True when the pose itself is wrong for the observed lane.

    Used by the placement loop to decide between TELEPORT (the pose sits
    on the wrong side / the body does not fit) and HOLD (the pose is fine
    but the current frame lacks the paired confirmation).  Without the
    split, a good pose on a mirror frame re-teleported to the same spot
    forever (2026-09-19 live: alignment -> (245.5, 878.2) every loop).
    """
    if out is None:
        return False
    src = str(out.meta.get("lane_src_sel", ""))
    if src not in PLACEMENT_LANE_SRCS:
        return False
    lane = getattr(out, "lane_ref", None)
    if lane is None:
        return False
    pts = np.asarray(lane, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 3:
        return False
    pts = pts[:, :2]
    if not np.isfinite(pts).all():
        return False
    if getattr(out, "lane_left", None) is None \
            and getattr(out, "lane_right", None) is None:
        return False              # no geometry to judge the pose with
    return not _pose_geometry_ok(out, pos, heading, pts, src, dbg=dbg)


def _sensor_lane_is_centered(out, pos, heading: float | None = None,
                             max_dist_m: float = 2.0,
                             max_yaw_deg: float = 12.0) -> bool:
    """True when strict perception already owns a lane the car may drive.

    ``sensor`` source: the car must be ON the reference polyline (within
    ``max_dist_m``) AND pointing along it.  Distance alone is not enough:
    a car standing inside the lane but YAWED against the lane direction
    still has a footprint corner across the boundary - on the 2026-09-18
    live east_coast run the pit pose passed "already centered" at
    v ~= 0.2 m/s, the alignment teleport was skipped, and the body-cross
    gate then fired and froze the car across the line.

    ``paved`` source (a paved road with no usable marking): the candidate
    itself already proved that the paved right edge is observed inside the
    image, the span is road-sized and the pavement is observed along the
    car's own track, and its polyline is anchored AT the ego, so a
    distance-to-polyline test would be trivially true.  What is graded
    here instead is the ALIGNMENT with the observed pavement EDGES (the
    un-anchored geometry of the candidate): a car yawed across the road
    must not start driving from that pose.  How far it sits from the
    keep-right target is the planner's job, not a placement precondition
    - otherwise a car spawned on the centre line of an unmarked road
    could never start at all.
    """
    if out is None:
        return False
    src = str(out.meta.get("lane_src_sel", ""))
    if src not in PLACEMENT_LANE_SRCS:
        return False
    lane = getattr(out, "lane_ref", None)
    if lane is None:
        return False
    pts = np.asarray(lane, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 3:
        return False
    pts = pts[:, :2]
    if not np.isfinite(pts).all():
        return False
    p = np.asarray(pos[:2], dtype=float)
    if src == SRC_PAVED:
        # The paved candidate's polyline is anchored AT the ego, so a
        # distance test would be trivially true; its edges carry the
        # direction the pose must agree with (checked in the geometry
        # half below).
        if getattr(out, "lane_left", None) is None                 and getattr(out, "lane_right", None) is None:
            return False
    else:
        # A single painted edge can place the car, but it cannot prove the
        # whole body is inside the lane: its missing side is a mirror, not
        # a physical boundary.  The old predicate accepted that frame as
        # "already centered" and drove into a later paired boundary that
        # appeared under the car (base run: 83 planned-cross stops; split
        # run: right body cross + road_off 2.37 m).  Strict placement now
        # waits for a REAL paired sensor lane with at least one hard edge
        # before releasing the brake; an alignment teleport alone is not
        # evidence that the body fits.
        if not bool(getattr(out, "lane_paired", False)
                    or out.meta.get("lane_paired", 0)):
            return False
        if getattr(out, "lane_left", None) is None                 and getattr(out, "lane_right", None) is None:
            return False
        d = np.linalg.norm(pts - p[None, :], axis=1)
        if not (np.isfinite(d).any() and float(np.nanmin(d)) <= max_dist_m):
            return False
    if heading is None:
        return True
    return _pose_geometry_ok(out, pos, heading, pts, src,
                             max_yaw_deg=max_yaw_deg)


def resolve_provenance_env(conn, args) -> tuple[str, str]:
    """(map, vehicle) for shadow provenance — never invent italy.

    Trust live ``current_env()`` only when the level query succeeded
    (``source == "live"``). Attach without ``--map`` and a failed query
    must record ``unknown``, not the connector's italy constructor default.
    """
    env: dict = {}
    try:
        env = conn.current_env() or {}
    except Exception:
        env = {}
    launch_map = getattr(args, "map", None)
    src = str(env.get("source") or "")
    if src == "live" and env.get("map"):
        map_name = str(env["map"])
    elif launch_map:
        map_name = str(launch_map)
    else:
        map_name = "unknown"
    vehicle = str(env.get("vehicle") or getattr(args, "vehicle", None)
                  or "unknown")
    if vehicle == "None":
        vehicle = "unknown"
    return map_name, vehicle


def build_fsd_shadow_provenance(
    *,
    runtime: str,
    map_name: str | None,
    vehicle: str | None,
    speed_arg: float,
    strict: bool,
    e2e_weights: str | None = None,
    bc_weights: str | None = None,
    dqn_weights: str | None = None,
    dqn_contract: str = "disabled",
    dqn_contract_warning: str | None = None,
) -> dict:
    """Episode provenance for ShadowRecorder.

    Map/vehicle must come from the live connector — never hardcode italy
    (multi-map experiments mislabeled every east_coast episode).
    """
    return {
        "source": "fsd_drive",
        "runtime": str(runtime),
        "map": str(map_name or "unknown"),
        "vehicle": str(vehicle or "unknown"),
        "speed_arg": float(speed_arg),
        "strict": bool(strict),
        "e2e_weights": e2e_weights,
        "bc_weights": bc_weights,
        "dqn_weights": dqn_weights,
        "dqn_contract": dqn_contract,
        "dqn_contract_warning": dqn_contract_warning,
    }


class FSDriveSession:
    """Stateful live FSD drive session.

    The first extraction intentionally preserves the legacy statement order
    byte-for-byte inside ``run``.  Subsequent stages can move lifecycle
    slices into methods without changing the actuator sequence.
    """

    def __init__(self, args) -> None:
        self.args = args

    def _build_route(self, conn):
        """Build and ground-snap the navigation route for this session.

        This method is an extraction only: it preserves the original
        road-graph/in-game-route fallback and snap order.  It returns the
        navigation centreline plus optional physical edge polylines; the
        caller still owns perception placement and all actuator timing.
        """
        args = self.args
        # Navigation route: a real stack plans ALONG the destination
        # route (FSD vector-space planner), not a straight line ahead -
        # a straight reference drives the car into the building on the
        # first town bend (observed town run 2026-08-21).  A single-marker
        # ``core_groundMarkers.setPath`` route is just a straight
        # interpolation to the marker, NOT a road-following nav route, so
        # it cuts across roads/walls (town run 2026-08-21: a straight line
        # pointed at a wall at (711.8,730.8) and the car wedged there).
        # Prefer the road-graph A* route built from the actual DecalRoad
        # centre-lines; fall back to the in-game route only when the road
        # graph is unavailable.
        nav_route = None
        nav_route_ref = None
        road_left = None
        road_right = None
        st0 = conn.get_state()
        p0 = np.asarray(st0.pos[:2], dtype=float)
        if args.goal is not None:
            rn = RoadNetwork()
            t_road = time.time()
            while not rn.ready and time.time() - t_road < 90.0:
                try:
                    if rn.build(conn.bng):
                        break
                except Exception:
                    pass
                time.sleep(1.0)
            if rn.ready:
                # Start the route at the nearest ROAD node, not the raw
                # (possibly off-road) ego/teleport point: interpolating the
                # route from an off-road start crosses terrain before the
                # car reaches the road, and leaves the auto-snap disabled
                # (route[0] == start so d0 == 0).  Snap will then place the
                # car on the road and face it along the route.
                n0 = rn.nodes[rn._nearest(p0)]
                _rwe = rn.route_with_edges(
                    n0, np.asarray(args.goal, dtype=float))
                if _rwe[0] is not None and len(_rwe[0]) >= 4:
                    nav_route = np.asarray(_rwe[0][:, :2], dtype=float)
                    # Real-road reference for the end-zone steering: cut
                    # the off-road goal tail so the last metres point
                    # along the actual road (see _trim_backtrack).
                    nav_route_ref = _trim_backtrack(nav_route)
                    road_left = _rwe[1]
                    road_right = _rwe[2]
                    dseg = np.linalg.norm(np.diff(nav_route, axis=0), axis=1)
                    print(f"[fsd-drive] road-graph route: "
                          f"{len(nav_route)} pts, "
                          f"{float(np.sum(dseg)):.1f} m "
                          f"({rn.info})")
                else:
                    print("[fsd-drive] road-graph A* found no route; "
                          "falling back to in-game nav route")
            else:
                print("[fsd-drive] road graph unavailable; "
                      "falling back to in-game nav route")
        if nav_route is None and args.goal is not None:
            conn.bng.control.queue_lua_command(
                "core_groundMarkers.setPath({vec3(%.3f, %.3f, 0)})\n"
                "return 'ok'" % (float(args.goal[0]), float(args.goal[1])),
                response=True)
            time.sleep(0.8)
            nav = conn.read_navigation_route()
            if nav is not None and len(nav) >= 4:
                nav_route = np.asarray(nav[:, :2], dtype=float)
                nav_route_ref = _trim_backtrack(nav_route)
        elif nav_route is None:
            nav = conn.read_navigation_route()
            if nav is not None and len(nav) >= 4:
                nav_route = np.asarray(nav[:, :2], dtype=float)
                nav_route_ref = _trim_backtrack(nav_route)
        if nav_route is not None and len(nav_route) >= 4:
            dseg = np.linalg.norm(np.diff(nav_route, axis=0), axis=1)
            print(f"[fsd-drive] nav route: {len(nav_route)} pts, "
                  f"{float(np.sum(dseg)):.1f} m")
            # If the car was spawned far from the nav route (or the
            # supplied --teleport missed the road), driving straight at
            # the route cuts across terrain/walls and wedges the car
            # (town runs 2026-08-21: 17-24 m off-route starts all ended
            # against a wall at ~1.8 m raw clearance).  Snap determinist-
            # ically onto the route and face along it so the FSD stack
            # verifies route-following instead of a cross-country path.
            st0 = conn.get_state()
            p0 = np.asarray(st0.pos[:2], dtype=float)
            h0 = float(st0.heading)
            f0 = np.array([float(np.cos(h0)), float(np.sin(h0))])
            d0 = np.linalg.norm(nav_route - p0, axis=1)
            i = int(np.argmin(d0))
            # Route start direction: which way the map route leaves the
            # nearest route vertex (forward vs backward along the polyline).
            rdir = None
            if i + 1 < len(nav_route):
                rv = nav_route[i + 1] - nav_route[i]
                if np.linalg.norm(rv) > 1e-9:
                    rdir = rv / np.linalg.norm(rv)
            elif i > 0:
                rv = nav_route[i] - nav_route[i - 1]
                if np.linalg.norm(rv) > 1e-9:
                    rdir = rv / np.linalg.norm(rv)
            heading_ok = True
            if rdir is not None:
                cos_a = float(np.dot(f0, rdir))
                heading_ok = cos_a >= 0.0
            # Snap when the car is off the road centre OR faces more than
            # ~60 deg away from the route direction (town runs 2026-08-22:
            # a 0.4 m off-route start with the nose pointing the wrong way
            # made the local route fall back to a straight line and drove
            # across the town into a wall; mountain run 2026-08-27
            # run_fix22: a start ON the route but facing 71 deg off drove
            # straight onto the grass and spun).  Align the nose along the
            # route so the planner follows the road graph, not a
            # cross-field line - a real stack never starts a run facing
            # across its own lane.
            snap_heading = False
            if rdir is not None:
                cos_a = float(np.dot(f0, rdir))
                snap_heading = cos_a < 0.5
            if d0[i] > 1.5 or not heading_ok or snap_heading:
                rx, ry = float(nav_route[i, 0]), float(nav_route[i, 1])
                if i + 1 < len(nav_route):
                    ndx, ndy = (float(nav_route[i + 1, 0] - rx),
                                float(nav_route[i + 1, 1] - ry))
                else:
                    ndx, ndy = (float(rx - nav_route[i - 1, 0]),
                                float(ry - nav_route[i - 1, 1]))
                h = float(np.arctan2(ndy, ndx))
                h = _snap_heading(nav_route, rx, ry, h,
                                  reference_heading=h0)
                # Ground-safe snap: same connector helper as --teleport, so
                # the car is always placed on the real surface (never below
                # terrain) and facing the route direction.  No lateral
                # offset constant here: once the semantic head is warm, the
                # perception placement below moves the car into its own
                # lane from the painted line.
                conn.safe_teleport(rx, ry, heading_deg=math.degrees(h))
                st1 = conn.get_state()
                print(f"[fsd-drive] snapped onto nav route "
                      f"({float(st1.pos[0]):.1f}, {float(st1.pos[1]):.1f}, "
                      f"{float(st1.pos[2]):.1f}) "
                      f"(was {d0[i]:.1f} m off route)")
            # Town traffic: parked NPC vehicles along the route give the
            # YOLO head / obstacle fusion real vehicles to see and the
            # planner real dodges to make (verification of the
            # perception chain under live multi-object conditions).
            if getattr(args, "traffic", 0):
                _n = _spawn_traffic(conn, nav_route, int(args.traffic))
                print(f"[fsd-drive] traffic NPCs spawned: {_n}/{args.traffic}")
        else:
            print("[fsd-drive] no nav route set; falling back to "
                  "straight-ahead reference")

        return nav_route, nav_route_ref, road_left, road_right

    def _setup_runtime(self, conn, args):
        """Assemble the FSD stack and learned candidates.

        This is a resource-only extraction.  It does not start the car or
        touch the watchdog; the caller keeps the original gearbox,
        pre-warm, and actuator order.
        """
        # Proven rule planner as the arbitration fallback: it rounds
        # switchback corners into drivable arcs (the 94.6% rule-autopilot
        # path), so the FSD drive never stops dead at a hairpin apex or
        # full-locks across a kinked map-prior lane.
        rule_planner = LocalPlanner()
        _seg = None
        if getattr(args, "seg_model", None):
            try:
                from beamng_autopilot.vision.segmentation import Segmenter
                _seg = Segmenter(model_path=args.seg_model)
                print(f"[fsd-drive] segmentation model: {args.seg_model}")
            except Exception as _seg_e:
                print(f"[fsd-drive] segmentation model disabled: {_seg_e}")
        else:
            # No pin means the run uses whatever is deployed, and until it
            # said so out loud a map could be evaluated on a checkpoint
            # nobody chose - the deployed default and the v13b/v8
            # specialists disagree strongly per map (see the line-IoU
            # matrix in the README).  Build it here so the path logged is
            # the path actually used, not a second lookup.
            try:
                from beamng_autopilot.vision.segmentation import Segmenter
                _seg = Segmenter()
                print("[fsd-drive] segmentation model (UNPINNED default): "
                      f"{_seg.model_path}")
            except Exception as _seg_e:
                print(f"[fsd-drive] segmentation model disabled: {_seg_e}")
        _line_seg = None
        _line_seg_path = getattr(args, "line_seg_model", None)
        if _line_seg_path:
            try:
                from beamng_autopilot.vision.segmentation import Segmenter
                _line_seg = Segmenter(model_path=_line_seg_path)
                print(f"[fsd-drive] painted-line model: {_line_seg_path}")
            except Exception as _line_seg_e:
                print(f"[fsd-drive] painted-line model disabled: {_line_seg_e}")
        stack = FSDStack(conn, args.runtime,
                         heads=[SemanticHead(segmenter=_seg,
                                             line_segmenter=_line_seg), TrafficSignalHead(),
                                ObjectHead(), LaneTopologyHead()],
                         # The live tick consumes ONLY the front frame:
                         # semantic world markings, road-mask projection,
                         # BEV accumulation and ObjectHead are all
                         # front_main-only, so polling the other 7 ring
                         # cameras pays ~7x grab time per tick for frames
                         # that are discarded.  --ring all restores the
                         # full surround polling (side-camera work).
                         ring_roles=(None if args.ring == "all"
                                     else ("front_main",)),
                         lane_mode=args.lane_mode,
                         strict_sensor=args.strict,
                         # Pairing-free strict-mode lane fallback, off by
                         # default; --corridor-lane opts in (live A/B
                         # before it may become a default).
                         corridor_fallback=bool(getattr(
                             args, "corridor_lane", False)),
                         # Paved-boundary lane candidate for a paved road
                         # with no usable marking (AGENTS.md「驾驶约束」).
                         # OPT-IN: --paved-lane (or BEAMNG_PAVED_LANE=1,
                         # the lever the A/B harness flips) - NOT a
                         # default, see FSDStack.paved_fallback for the
                         # live failure that put it back behind the flag.
                         paved_fallback=(
                             bool(getattr(args, "paved_lane", False))
                             or os.environ.get("BEAMNG_PAVED_LANE",
                                               "0") == "1"),
                         cam_w=args.cam_w, cam_h=args.cam_h,
                         temporal=True,
                         # Strict sensor contract: FULL sensor refresh
                         # every tick.  range_every_n=3 reuse at the
                         # every-tick-semantic cadence measured 4-6 s range
                         # age (ticks 1.4-1.9 s), past STALE_RANGE_S=2.0 -
                         # every reuse cycle fail-closed into "stale
                         # sensor" stops (29 stall frames, base_slew
                         # 2026-09-19).  Non-strict keeps the n=3
                         # throughput reuse.
                         range_every_n=(1 if args.strict
                                        and args.lane_mode == "sensor"
                                        else 3),
                         # Strict lateral control cannot hold a stale lane
                         # through a curve: an A/B hold reduced unavailable
                         # ticks but caused off-road frames.  Refresh the
                         # semantic lane every strict sensor tick; the
                         # non-strict throughput path keeps n=2.
                         semantic_every_n=(1 if args.strict
                                           and args.lane_mode == "sensor"
                                           else 2),
                         # LiDAR every 3rd tick: a fresh scan costs
                         # ~280 ms while a reuse is ~40 ms; the world-frame
                         # hits stay valid for static walls and the
                         # temporal occupancy filter bridges the gap.
                         object_every_n=2)
        stack.reset_temporal()  # stale occupancy before start must not leak
        # Neural (E2E) planner candidate: a trained end-to-end network
        # over the same perception (rgb + segmentation label + BEV)
        # provides the second planning layer a real FSD stack has -
        # ranked below the layered planner but above the map/rule
        # fallback.  Loading failure (missing weights / bad checkpoint)
        # disables the candidate silently; the drive never depends on it.
        e2e_rt = None
        if not args.no_e2e:
            try:
                e2e_rt = E2ERuntime(args.e2e_model or DEFAULT_E2E_WEIGHTS,
                                    device=None)
                if e2e_rt.loaded:
                    _ck = e2e_rt.ckpt or {}
                    print(f"[fsd-drive] E2E neural planner: "
                          f"{e2e_rt.weights} "
                          f"(img={e2e_rt.img_w}x{e2e_rt.img_h}, "
                          f"history={e2e_rt.history}, "
                          f"epoch={_ck.get('epoch', '?')}, "
                          f"device={e2e_rt.device})")
                else:
                    e2e_rt = None
            except Exception as _e2e_e:
                print(f"[fsd-drive] E2E planner disabled: {_e2e_e}")
                e2e_rt = None
        if e2e_rt is not None:
            e2e_rt.reset()   # no stale frames from before the run
        # DAVE-2 imitation-learned steering (M3 BC): the third neural
        # candidate, ranked below the E2E planner and above the rule
        # backup; the safety monitor verifies its rolled-out arc every
        # tick.  Loading failure disables the candidate silently - the
        # drive never depends on it.
        bc_rt = None
        if not args.no_bc:
            try:
                bc_rt = BCRuntime(args.bc_model or DEFAULT_BC_WEIGHTS,
                                  device=None)
                if bc_rt.loaded:
                    print(f"[fsd-drive] DAVE-2 BC steering: "
                          f"{bc_rt.weights} "
                          f"(img={bc_rt.img_w}x{bc_rt.img_h}, "
                          f"val_mae={float((bc_rt.ckpt or {}).get('val_mae', -1)):.4f}, "
                          f"device={bc_rt.device})")
                else:
                    print(f"[fsd-drive] BC steering disabled: "
                          f"{bc_rt.error or 'weights not found'}")
                    bc_rt = None
            except Exception as _bc_e:
                print(f"[fsd-drive] BC steering disabled: {_bc_e}")
                bc_rt = None
        # M4 DQN decision policy: caps the plan's target speed with
        # discrete cruise/ease/slow decisions learned offline.  It can
        # only SLOW the plan - steering and every safety layer stay
        # authoritative - so a bad policy costs comfort, never safety.
        dqn_rt = None
        if not args.no_dqn:
            try:
                dqn_rt = DQNRuntime(args.dqn_model or DEFAULT_DQN_WEIGHTS)
                if dqn_rt.loaded:
                    print(f"[fsd-drive] DQN decision policy: "
                          f"{dqn_rt.weights}")
                    if dqn_rt.meta_warning:
                        print(f"[fsd-drive] DQN contract warning: "
                              f"{dqn_rt.meta_warning}")
                    elif dqn_rt.meta:
                        print(f"[fsd-drive] DQN contract: "
                              f"git={dqn_rt.meta.get('git_commit') or '?'}")
                else:
                    print(f"[fsd-drive] DQN policy disabled: "
                          f"{dqn_rt.error or 'weights not found'}")
                    dqn_rt = None
            except Exception as _dqn_e:
                print(f"[fsd-drive] DQN policy disabled: {_dqn_e}")
                dqn_rt = None
        return rule_planner, stack, e2e_rt, bc_rt, dqn_rt

    def _prewarm_and_place(self, conn, stack, nav_route, fwd_gear):
        """Warm sensors and place the ego using perception-only geometry.

        The vehicle remains braked while expensive heads warm and while
        painted-line placement retries.  The method returns ``(tick,
        placed, abort_code)``; it contains no planner/control-loop state.
        """
        _pw_out = None
        _percep_ok = False
        _strict_sensor = bool(
            getattr(self.args, "strict", False)
            and str(getattr(self.args, "lane_mode", "")) == "sensor")
        try:
            conn.control(throttle=0.0, brake=1.0, steering=0.0,
                         parkingbrake=1.0, gear=fwd_gear)
            _pw_t0 = time.time()
            _pw_ticks = 0
            _pw_teleported = False
            while True:
                _elapsed = time.time() - _pw_t0
                try:
                    wd_heartbeat(conn)
                except Exception:
                    pass
                _pw_state = conn.get_state()
                _pw_route = local_route(
                    np.asarray(_pw_state.pos[:2], dtype=float),
                    float(_pw_state.heading), nav_route)
                try:
                    _pw_out = stack.tick(st=_pw_state, route_ref=_pw_route)
                    _pw_ticks += 1
                except Exception:
                    _pw_out = None
                _head_live = bool(
                    _pw_out is not None
                    and _pw_out.meta.get("object_head"))
                if (_pw_out is not None and _pw_out.frame is not None
                        and _pw_ticks >= PLACEMENT_SKIP_TICKS):
                    if _pw_ticks % 20 == 0:
                        try:
                            _sem_pw = _pw_out.head_outputs.get("semantic")
                            _lm_pw = ((getattr(_sem_pw, "masks", {}) or {})
                                      .get("line")) if _sem_pw is not None else None
                            _mk_pw = painted_line_markings(
                                _sem_pw, _pw_out.cam, _pw_state.pos,
                                float(_pw_state.heading),
                                ground_z=(float(_pw_state.pos[2])
                                          - config.EGO_ORIGIN_GROUND_GAP_M),
                                rgb=_pw_out.frame) if _sem_pw is not None \
                                and _pw_out.cam is not None else []
                            _body_pw = body_pose_cross_depth_m(
                                np.asarray(_pw_state.pos, dtype=float),
                                float(_pw_state.heading),
                                getattr(_pw_out, "lane_left", None),
                                getattr(_pw_out, "lane_right", None))
                            _pdbg_pw = _pw_out.meta.get("lane_pair_debug") or {}
                            _fdbg_pw = _pw_out.meta.get("lane_fusion_debug") or {}
                            print("[fsd-drive] placement diag: "
                                  f"src={_pw_out.meta.get('lane_src_sel')} "
                                  f"paired={getattr(_pw_out, 'lane_paired', None)} "
                                  f"paired_meta={_pw_out.meta.get('lane_paired')} "
                                  f"line_px={int(np.count_nonzero(_lm_pw)) if _lm_pw is not None else 0} "
                                  f"marks={len(_mk_pw or [])} "
                                  f"body_cross={float(_body_pw):.2f} "
                                  f"centered={_sensor_lane_is_centered(_pw_out, _pw_state.pos, float(_pw_state.heading))} "
                                  f"lane_dev={_pw_out.meta.get('lane_dev_m')} "
                                  f"pair_mode={_pdbg_pw.get('mode')} "
                                  f"pair_rejects={_pdbg_pw.get('pair_rejects')} "
                                  f"fusion={_fdbg_pw.get('mode')}",
                                  flush=True)
                        except Exception as _pde:
                            print(f"[fsd-drive] placement diag failed: {_pde}",
                                  flush=True)
                    try:
                        if _sensor_lane_is_centered(
                                _pw_out, _pw_state.pos,
                                float(_pw_state.heading)):
                            _percep_ok = True
                            print("[fsd-drive] perception lane placement "
                                  "already centered; no teleport needed")
                            break
                        _plc_dbg: dict = {}
                        _sp_tgt = painted_line_lane_center(
                            _pw_out.head_outputs.get("semantic"),
                            _pw_out.cam, _pw_state.pos,
                            float(_pw_state.heading),
                            ground_z=(float(_pw_state.pos[2])
                                      - config.EGO_ORIGIN_GROUND_GAP_M),
                            marks=((_pw_out.head_outputs.get("semantic")
                                    .meta.get("markings", []))
                                   if _pw_out.head_outputs.get("semantic")
                                   is not None else None),
                            rgb=_pw_out.frame, debug=_plc_dbg)
                        if _pw_ticks % 20 == 0:
                            print(f"[fsd-drive] plc target={_sp_tgt} "
                                  f"dbg={_plc_dbg}", flush=True)
                        _align_dbg: dict = {}
                        _needs = _pose_needs_alignment(
                            _pw_out, _pw_state.pos,
                            float(_pw_state.heading), dbg=_align_dbg)
                        _nudge = (_alignment_nudge_m(
                            _pw_out, _pw_state.pos,
                            float(_pw_state.heading)) if _needs else None)
                        if _sp_tgt is not None and not _needs:
                            # The pose already fits the observed lane (the
                            # paint cluster reads "already on the perceived
                            # centre" and the geometry agrees): just wait
                            # for the paired confirmation frame.
                            if _pw_ticks % 20 == 0:
                                print("[fsd-drive] pose fits the lane "
                                      f"({_align_dbg}); holding for the "
                                      "paired confirmation frame",
                                      flush=True)
                        elif _needs and _nudge is not None:
                            _nd = float(np.hypot(
                                _nudge[0] - float(_pw_state.pos[0]),
                                _nudge[1] - float(_pw_state.pos[1])))
                            if _nd < 0.35:
                                if _pw_ticks % 20 == 0:
                                    print(f"[fsd-drive] nudge reached "
                                          f"({_nd:.2f} m) geom="
                                          f"{_align_dbg}; holding",
                                          flush=True)
                            else:
                                conn.safe_teleport(
                                    _nudge[0], _nudge[1],
                                    heading_deg=math.degrees(
                                        float(_pw_state.heading)))
                                try:
                                    stack.reset_temporal()
                                except Exception:
                                    pass
                                _pw_teleported = True
                                print(f"[fsd-drive] perception lane "
                                      f"alignment -> "
                                      f"({_nudge[0]:.1f}, "
                                      f"{_nudge[1]:.1f}) d={_nd:.2f} "
                                      f"geom={_align_dbg}; "
                                      "awaiting paired-body confirmation",
                                      flush=True)
                        elif _needs:
                            # No boundary-anchored nudge available (yaw /
                            # body failure, or both sides inside the band):
                            # hold and wait for a frame where the geometry
                            # and the placement target agree.  Never drive
                            # unplaced.
                            if _pw_ticks % 20 == 0:
                                print(f"[fsd-drive] alignment pending "
                                      f"geom={_align_dbg}; holding",
                                      flush=True)
                            # Do not release the brake yet.  The next
                            # fresh tick must publish a paired sensor lane
                            # and prove the full body fits inside it.
                        elif _pw_ticks % 20 == 0:
                            _canon = ((_pw_out.head_outputs.get("semantic")
                                       .meta.get("markings", []))
                                      if _pw_out.head_outputs.get("semantic")
                                      is not None else [])
                            _canon_summ = []
                            try:
                                _lf2 = np.array([-math.sin(float(_pw_state.heading)),
                                                 math.cos(float(_pw_state.heading))])
                                for _m in _canon:
                                    _w2 = np.asarray(_m.world, dtype=float)
                                    if _w2.ndim != 2 or len(_w2) < 2:
                                        continue
                                    _lat2 = float(np.median(
                                        (_w2[:, :2]
                                         - np.asarray(_pw_state.pos[:2]))
                                        @ _lf2))
                                    _canon_summ.append(
                                        f"{_m.kind}/{getattr(_m, 'color', '?')}"
                                        f"/{_lat2:+.2f}/n{len(_w2)}")
                            except Exception:
                                pass
                            print("[fsd-drive] painted-line placement "
                                  "unavailable; "
                                  f"src={_pw_out.meta.get('lane_src_sel')} "
                                  f"lane_pts={0 if getattr(_pw_out, 'lane_ref', None) is None else len(_pw_out.lane_ref)} "
                                  f"canon={len(_canon)} "
                                  f"[{'; '.join(_canon_summ[:6])}]; "
                                  "holding brake", flush=True)
                    except Exception as _spe:
                        print(f"[fsd-drive] placement attempt failed: "
                              f"{_spe}")
                if _elapsed > PLACEMENT_HOLD_S and _head_live:
                    # Yellow/US paint can appear a few ticks after the
                    # object head is live.  In strict sensor mode a failed
                    # placement is NOT a reason to exit or drive unplaced:
                    # stay braked and keep refreshing perception until the
                    # line/pose becomes usable, then the existing alignment
                    # teleport or centred-lane check releases the car.
                    # This is the "stop, recover, continue" contract; the
                    # old timeout returned 2 and made the caller either end
                    # the run or (in the old strict calibrate-after path)
                    # enter the drive loop UNPLACED.
                    if (_elapsed > PLACEMENT_HOLD_S + PLACEMENT_GRACE_S
                            and _strict_sensor):
                        if _pw_ticks % 20 == 0:
                            print("[fsd-drive] placement not ready; "
                                  "stopped, refreshing perception", flush=True)
                    elif _elapsed > PLACEMENT_HOLD_S + PLACEMENT_GRACE_S:
                        break
                conn.control(throttle=0.0, brake=1.0, steering=0.0,
                             parkingbrake=1.0, gear=fwd_gear)
            print(f"[fsd-drive] pipeline warm: "
                  f"{time.time() - _pw_t0:.1f}s, "
                  f"object_head={bool(_pw_out is not None and _pw_out.meta.get('object_head'))}, "
                  f"placed={_percep_ok}")
            if not _percep_ok:
                _strict_sensor = bool(
                    getattr(self.args, "strict", False)
                    and str(getattr(self.args, "lane_mode", "")) == "sensor")
                if getattr(self.args, "allow_unplaced", False):
                    # Explicit collection mode only: a dataset capture may
                    # start before the lane is visible, but it must never be
                    # mistaken for a valid strict driving run.
                    print("[fsd-drive] lane placement failed - continuing "
                          "UNPLACED (--allow-unplaced collection mode)")
                    return _pw_out, False, 0
                if _strict_sensor:
                    print("[fsd-drive] ABORT: strict sensor lane placement "
                          "failed - the car will NOT drive unplaced")
                    conn.control(throttle=0.0, brake=1.0, steering=0.0,
                                 parkingbrake=1.0, gear=fwd_gear)
                    conn.step(3)
                    return _pw_out, False, 2
                print("[fsd-drive] ABORT: lane placement failed after "
                      f"{PLACEMENT_HOLD_S:.0f}s - the car will NOT drive "
                      "unplaced from the road centre line")
                conn.control(throttle=0.0, brake=1.0, steering=0.0,
                             parkingbrake=1.0, gear=fwd_gear)
                conn.step(3)
                return _pw_out, False, 2
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=0.0, gear=fwd_gear)
            conn.step(3)
            try:
                wd_heartbeat(conn)
            except Exception:
                pass
        except Exception as _pw_e:
            print(f"[fsd-drive] pre-warm skipped: {_pw_e}")
        return _pw_out, _percep_ok, 0

    def run(self) -> int:
        args = self.args

        # initialised FIRST: the end-of-run finally touches these even when
        # the connection fails before the drive section (2026-09-06:
        # BNGDisconnectedError at connect -> UnboundLocalError in finally
        # masked the real error)
        hist: list[dict] = []
        rec = None

        conn = BeamNGConnector(
            getattr(args, "map", None) or "italy", "etk800",
            port=config.runtime_port(args.runtime),
            home=config.runtime_home(args.runtime))
        pp = PurePursuit(lookahead=5.0)
        speed_ctrl = SpeedController(deadband=SPEED_DEADBAND_MPS,
                                    hyst_mps=SPEED_HYST_MPS)
        monitor = SafetyMonitor(max_speed=args.speed)
        try:
            conn.open(launch=not args.attach)
            try:
                conn.attach_vehicle(already_open=True)
                if getattr(args, "map", None):
                    # attach 只挂到游戏当前关卡（昨天留在 italy 的实例被
                    # 原样驱动了一整集）——显式 --map 时强制加载该地图场景
                    conn.load_scenario()
            except Exception:
                conn.load_scenario()
            if args.teleport is not None:
                x, y, yaw = args.teleport
                # Ground-safe teleport: the connector measures the real surface
                # with a cast ray and re-checks after settling, so no map can
                # ever drop the car below terrain (hardcoded z heights did on
                # maps whose surface sits much higher - 2026-08-28).
                conn.safe_teleport(float(x), float(y), heading_deg=float(yaw))
                st1 = conn.get_state()
                print(f"[fsd-drive] teleport -> "
                      f"({float(st1.pos[0]):.1f}, {float(st1.pos[1]):.1f}, "
                      f"{float(st1.pos[2]):.1f})")

            nav_route, nav_route_ref, road_left, road_right = \
                self._build_route(conn)
            (rule_planner, stack, e2e_rt, bc_rt,
             dqn_rt) = self._setup_runtime(conn, args)
            # Realistic gearbox locked into a forward gear (D).  A real stack
            # never leaves the car in reverse; keep the D input on every
            # control frame so an impact can never leave the gearbox in R.
            fwd_gear = gearbox.forward_gear_input(conn)
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=0.0, gear=fwd_gear)
            conn.step(3)
            # Game-side input watchdog: if this Python process dies or is
            # killed, the game keeps applying the last controls - the Lua
            # watchdog stops the car once the heartbeat goes stale.
            try:
                if wd_arm(conn):
                    print("[fsd-drive] input watchdog armed")
                else:
                    print("[fsd-drive] WATCHDOG FAILED TO ARM; aborting")
                    conn.control(throttle=0.0, brake=1.0, steering=0.0,
                                 parkingbrake=1.0, gear=fwd_gear)
                    conn.step(5)
                    return 2
            except Exception as _wd_e:
                print(f"[fsd-drive] WATCHDOG ARM ERROR; aborting: {_wd_e}")
                try:
                    conn.control(throttle=0.0, brake=1.0, steering=0.0,
                                 parkingbrake=1.0, gear=fwd_gear)
                    conn.step(5)
                except Exception:
                    pass
                return 2
            # Real-time control: DO NOT pause the sim.  With ticks now
            # ~0.4-0.6 s (after warm-up) the stale-control window is a few
            # metres at cruise and a fraction of a metre at bend speeds, so
            # the sim runs continuously - the car is always moving, which
            # removes the paused-step stutter.  The warm-up crawl and
            # stale-tick scrub below bound the open-loop windows.
            # Shadow episode: the FSD drive records its own (rgb + label +
            # BEV + trajectory + executed control) frames for later
            # end-to-end training - the same ShadowFrame contract
            # m5_shadow_drive uses, so a drive IS a labelled run.
            prov_map, prov_vehicle = resolve_provenance_env(conn, args)
            rec = None if args.no_shadow else ShadowRecorder(
                config.LOGS_DIR / "m5_e2e", f"fsd_{int(time.time())}",
                provenance=build_fsd_shadow_provenance(
                    runtime=str(args.runtime),
                    map_name=prov_map,
                    vehicle=prov_vehicle,
                    speed_arg=float(args.speed),
                    strict=bool(args.strict),
                    e2e_weights=(str(e2e_rt.weights)
                                 if e2e_rt is not None else None),
                    bc_weights=(str(bc_rt.weights)
                                if bc_rt is not None else None),
                    dqn_weights=(str(dqn_rt.weights)
                                 if dqn_rt is not None else None),
                    dqn_contract=(
                        "ok" if dqn_rt is not None
                        and dqn_rt.contract is not None
                        and dqn_rt.contract.ok else "disabled"),
                    dqn_contract_warning=(
                        dqn_rt.meta_warning
                        if dqn_rt is not None else None),
                ))
            (_pw_out, _percep_ok,
             _placement_rc) = self._prewarm_and_place(
                conn, stack, nav_route, fwd_gear)
            if _placement_rc:
                return _placement_rc
            if not _percep_ok:
                print("[fsd-drive] painted line not perceived; keeping the "
                      "ground-safe route snap")
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=0.0, gear=fwd_gear)
            conn.step(3)
            print(f"[fsd-drive] gearbox realistic, forward gear input = {fwd_gear}")
            rguard = ReverseGuard(threshold_mps=REVERSE_THRESHOLD_MPS,
                                  clear_mps=REVERSE_CLEAR_MPS)
            rman = ReverseManeuver(fwd_gear=fwd_gear)
            # Strict FSD never backs up: the bounded reverse escape is a
            # rule-planner manoeuvre (back out of a dead end and re-plan),
            # while a perception-led stack that cannot find a forward path
            # fails CLOSED - stop and hold (docs/fsd_realism.md §4).
            # Gating this on "no sensor lane" was not enough: the
            # 2026-09-18 live demo reversed on 46 of 220 ticks with the
            # lane PAIRED every time, plan_blocked empty and stuck=0,
            # peaking at -2.99 m/s.
            rman.enabled = not bool(args.strict)
            print(f"[fsd-drive] runtime={stack.mode} FSD pipeline driving "
                  f"for {args.seconds}s at {args.speed} m/s "
                  f"(lane_mode={args.lane_mode})")

            prev_steer = 0.0  # rate-limited steering state (rule-autopilot convention)
            steer_shaper = SteeringShaper(
                max_rate_per_s=FSD_STEER_RATE_PER_S)
            steer_blend_w = SteeringBlendWeights(
                enabled=STEER_BLEND_ENABLED)
            steer_blend_digest = None
            long_planner = LongitudinalPlanner()
            long_digest = None
            drive_modes = DriveModeClassifier()
            mode_policy = None
            _sr_prev = 0.0    # previous applied steering rate (jerk telemetry)
            watchdog_lost = False
            map_mc_smooth = None   # EMA-smoothed map-prior lane centre
            end_plc_cache = None   # (own-lane centre xy, t_seen) last-good
                                   # perception anchor for the end zone
            plc_corr = PaintedLineLateralCorrector(
                max_shift_m=PLC_MAX_SHIFT_M, horizon_m=PLC_HORIZON_M,
                rate_m_s=PLC_RATE_MPS, hold_s=PLC_HOLD_S,
                min_speed_mps=PLC_MIN_SPEED_MPS)
            last_h = None      # previous heading for the yaw-rate steering damper
            climb_t = 0.0      # seconds spent in slope-creep assist
            stuck_t = 0.0    # seconds at near-standstill with a "safe" plan
            sig_rule_state = None   # game signal state on the current link
            sig_rule_t = 0.0        # last road-link signal poll time
            rev_thr = REV_THR_BASE  # reverse-escape throttle ramp state
            t_end = time.time() + args.seconds
            frames = 0
            stopps = 0
            hold_frames = 0
            hold_offers = 0
            substeps = 0
            _substep = ControlSubstep(
                stale_plan_s=config.FSD_SUBSTEP_STALE_PLAN_S)
            _plan_t = time.time()      # when the cached plan was built
            _sub_tracks: list = []     # its tracked obstacles
            t0 = time.time()
            # Live lane-recognition overlay (--vis): every N ticks render
            # what the perception chain saw into PNGs a human can watch.
            _vis_every = int(getattr(args, "vis", 0) or 0)
            _vis_dir = (config.LOGS_DIR / "m5_vis" / "live")
            _vis_warned = False
            warmup_until = time.time() + WARMUP_S
            target_sm = float(args.speed)
            plan_sm = float(args.speed)
            prev_thr = 0.0
            prev_brk = 0.0
            gov_brake = False
            # First-frame steering: with last_t set to NOW the first tick has
            # dt ~= 0, smooth_steer cannot move the wheel, and the car runs
            # the whole first ~7 m straight past the corner entry before any
            # steering appears (fix50: steer stayed -0.02 while the car went
            # from the spawn to (726.9,756.8)).  Pretend one control interval
            # has already elapsed so the first frame can steer immediately.
            last_t = time.time() - 1.5
            # Static full-route geometry: the nav route never changes during
            # a run, so dedup/extend/round/resample + arc lengths + per-vertex
            # radii are computed ONCE before the loop instead of every frame
            # (the old per-frame rebuild of the whole rounded route cost
            # ~10-20 ms on a long route, plus a repeated curvature scan for
            # the bend governor).
            route_round = None
            route_arc = None
            route_rad = None
            if nav_route is not None and len(nav_route) >= 4:
                try:
                    from beamng_autopilot.planning.local_route import (
                        _dedup as _rdd, _extend_back as _reb,
                        _round_corners as _rrc, _resample as _rrs,
                        CORNER_RADIUS_M as _CR, CORNER_RESAMPLE_M as _CSM)
                    route_round = _rrs(_rrc(_reb(_rdd(nav_route[:, :2])), _CR),
                                       _CSM)
                    route_arc = np.concatenate(
                        [[0.0], np.cumsum(np.linalg.norm(
                            np.diff(route_round, axis=0), axis=1))])
                    _rv1 = np.diff(route_round, axis=0)
                    _rn1 = np.linalg.norm(_rv1, axis=1)
                    _rn2 = _rn1[1:]
                    _rcr = (_rv1[:-1, 0] * _rv1[1:, 1]
                            - _rv1[:-1, 1] * _rv1[1:, 0])
                    _rcurv = 2.0 * np.abs(_rcr) / (
                        _rn1[:-1] * _rn2 * (_rn1[:-1] + _rn2))
                    route_rad = np.full(len(route_round), np.inf)
                    _rm = _rcurv > 1e-6
                    route_rad[1:-1][_rm] = 1.0 / _rcurv[_rm]
                except Exception:
                    route_round = route_arc = route_rad = None
            # Shadow episode + telemetry history are initialised HERE (before
            # the placement/pre-warm section): the end-of-run finally writes
            # telemetry, so these must exist even when the placement aborts
            # the run early (the 2026-09-06 UnboundLocalError masked the real
            # placement failure).
            _ema_tick = 0.35          # adaptive tick-budget EMA (seeded: a
                                      # typical warm tick is ~0.3-0.4 s)
            while time.time() < t_end:
                try:
                    if not wd_heartbeat(conn):
                        watchdog_lost = True
                        print("[fsd-drive] WATCHDOG HEARTBEAT LOST; "
                              "braking and aborting", flush=True)
                        conn.control(throttle=0.0, brake=1.0, steering=0.0,
                                     gear=fwd_gear, parkingbrake=1.0)
                        conn.step(5)
                        break
                except Exception as exc:
                    watchdog_lost = True
                    print(f"[fsd-drive] WATCHDOG HEARTBEAT ERROR; "
                          f"braking and aborting: {exc}", flush=True)
                    try:
                        conn.control(throttle=0.0, brake=1.0, steering=0.0,
                                     gear=fwd_gear, parkingbrake=1.0)
                        conn.step(5)
                    except Exception:
                        pass
                    break
                _f0 = time.time()
                st = conn.get_state()
                pos = np.asarray(st.pos, dtype=float)
                heading = float(st.heading)
                v = float(st.speed)
                signed = 0.0
                if st.vel is not None and st.dir is not None:
                    signed = float(np.dot(
                        np.asarray(st.vel[:2], dtype=float),
                        np.asarray(st.dir[:2], dtype=float)))
                # Control-loop dt in real time: the sim runs continuously, so
                # wall time between ticks IS the driving time.  Clamp the
                # warm-up frames (which can take seconds) so the stuck /
                # reverse / climb state machines do not count a camera stall
                # as seconds of standstill (fix53) - dt caps at 0.5 s.
                now_t = time.time()
                _wall_dt = max(0.0, now_t - last_t)
                last_t = now_t
                dt = min(0.5, max(0.05, _wall_dt))
                # Long-tick brake guard: if the PREVIOUS tick took too long
                # the car just drove open-loop for that long.  Brake now
                # (before the next, possibly long, tick) so no more distance
                # is added uncontrolled; the tick below re-plans and resumes.
                if _wall_dt > STALE_CTRL_S and v > 1.0:
                    try:
                        conn.control(throttle=0.0, brake=0.5,
                                     steering=prev_steer, gear=fwd_gear)
                    except Exception:
                        pass
                rev_brk, reversing = rguard.decide(signed, dt=dt)
                # Yaw rate for the steering damper: a low-speed car at full
                # lock keeps rotating for seconds after the wheel is centred
                # (the fix37/38 loop - the car swung -130 deg around the
                # junction because the bang-bang pursuit had no damping).
                yaw_rate = 0.0
                if last_h is not None and dt > 0.05:
                    _d = float(heading) - float(last_h)
                    _d = (_d + math.pi) % (2.0 * math.pi) - math.pi
                    yaw_rate = _d / dt
                last_h = float(heading)

                # one full FSD tick -> best trajectory (planned along the
                # LOCAL forward route anchored at the ego; the full nav
                # route tail is a map prior that can cut through a corner
                # wall when the car drifts (town runs 2026-08-21) - the
                # local forward route is what the planner may follow.
                route_local = local_route(pos, heading, nav_route)
                # Strict FSD never even builds the map-prior lane: the
                # route is navigation intent, not a lateral reference.
                _strict_lane = bool(
                    getattr(args, "strict", False)
                    and str(getattr(args, "lane_mode", "map")) == "sensor")
                map_lane = None
                if not _strict_lane and road_left is not None \
                        and road_right is not None:
                    try:
                        map_lane = map_lane_edges(
                            nav_route, road_left, road_right, pos, heading)
                    except Exception:
                        map_lane = None
                if map_lane is not None:
                    _mc = np.asarray(map_lane[0], dtype=float)[:, :2]
                    if map_mc_smooth is not None and len(_mc) >= 3 \
                            and len(map_mc_smooth) >= 3:
                        try:
                            _a_new = np.concatenate([[0.0], np.cumsum(
                                np.linalg.norm(np.diff(_mc, axis=0), axis=1))])
                            _a_old = np.concatenate([[0.0], np.cumsum(
                                np.linalg.norm(np.diff(map_mc_smooth, axis=0),
                                               axis=1))])
                            _al = np.empty_like(_mc)
                            for _j in range(len(_mc)):
                                _k = int(np.clip(np.searchsorted(
                                    _a_old, _a_new[_j]), 0,
                                    len(map_mc_smooth) - 1))
                                _al[_j] = map_mc_smooth[_k]
                            _mc = _al * (1.0 - MAP_LANE_EMA) + _mc * MAP_LANE_EMA
                        except Exception:
                            pass
                    map_mc_smooth = _mc.copy()
                    map_lane = (_mc, map_lane[1], map_lane[2])
                # End-of-route remaining distance, measured on the FULL nav
                # route (arc from the ego's projection to the route END).
                # The old end-stop checked route_local[-1] against
                # nav_route[-1]; right at the end the local window collapses
                # to a straight fallback (its last point jumps ~40 m away),
                # so rem_end became None exactly at the goal, the end-stop
                # released, and the car crept onto the centre line and
                # parked ON it (opt17: lat_left 0.00 -> +0.97 inside oncoming
                # -> parked lat_left 0.00).  Computed here so the steering
                # zone override and the target cap both use it.
                rem_end = None
                if nav_route is not None and len(nav_route) >= 2:
                    try:
                        _r2 = np.asarray(nav_route[:, :2], dtype=float)
                        _arc2 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
                            np.diff(_r2, axis=0), axis=1))])
                        _proj2 = _project_arc(_r2, pos[:2])
                        rem_end = float(max(0.0, _arc2[-1] - _proj2))
                    except Exception:
                        rem_end = None
                _ta = time.time()  # local route / map lane / FSD tick split
                # Adaptive shared time budget for the whole FSD tick (see
                # TICK_BUDGET_* above): the stack drops to cached outputs
                # once the tick exceeds the budget so one heavy tick cannot
                # freeze the control loop.
                _budget = float(np.clip(
                    _ema_tick * TICK_BUDGET_FRAC,
                    TICK_BUDGET_MIN_S, TICK_BUDGET_MAX_S))
                out = stack.tick(st=st, route_ref=route_local,
                                 map_lane_override=map_lane,
                                 time_budget_s=_budget)
                if (not _percep_ok
                        and _sensor_lane_is_centered(out, pos,
                                                     heading)):
                    _percep_ok = True
                    print("[fsd-drive] placed=True after calibrate-after "
                          "sensor lane recovery", flush=True)
                _tb = time.time()
                _ema_tick = (TICK_BUDGET_EMA * (_tb - _f0)
                             + (1.0 - TICK_BUDGET_EMA) * _ema_tick)
                best = out.best_path
                # Painted centre-line lateral (objective lane-side check).
                # The semantic LINE mask is back-projected to world markings
                # ONCE per frame and shared with the steady lateral corrector
                # below, so a frame of line detection is never done twice.
                _plmarks = None
                # funnel counters: WHERE do painted-line frames drop?
                # (mask -> world markings -> near field) 2026-09-06 diagnosis
                _pl_mask = _pl_marks = _pl_near = False
                if out is not None and out.cam is not None and                     getattr(out, "head_outputs", None):
                    _semx = out.head_outputs.get("semantic")
                    _linem = (getattr(_semx, "masks", {}) or {}).get("line")                     if _semx is not None else None
                    _pl_mask = _linem is not None and bool(
                        np.asarray(_linem).any())
                    if _pl_mask:
                        try:
                            _plmarks = painted_line_markings(
                                _semx, out.cam, pos, float(heading),
                                ground_z=(float(pos[2])
                                          - config.EGO_ORIGIN_GROUND_GAP_M
                                          if len(pos) > 2 else None))
                            _pl_marks = bool(_plmarks)
                        except Exception as _ple:
                            _plmarks = None
                            print(f"[fsd-drive] painted-line projection "
                                  f"error: {_ple}")
                        if _pl_marks:
                            _p2 = np.asarray(pos[:2], dtype=float)
                            for _mk in _plmarks:
                                _mw = np.asarray(_mk.world, dtype=float)
                                if np.linalg.norm(_mw[0, :2] - _p2) < 25.0:
                                    _pl_near = True
                                    break
                line_lat = _painted_line_lat(out, pos, heading, _plmarks)
                # Painted centre-line body gate: when the visible marking is
                # the line LEFT of the ego, project all four body corners
                # against that actual painted polyline.  This catches the
                # screenshot case where the left wheel/body is over the line
                # even though lane_left is absent or not covered.
                painted_body_cross = False
                if _plmarks:
                    try:
                        _fbody = np.array([math.cos(heading), math.sin(heading)])
                        _lbody = np.array([-_fbody[1], _fbody[0]])
                        _pbody = np.asarray(pos[:2], dtype=float)
                        _corners_body = footprint_corners(
                            _pbody, float(heading))
                        for _mk in _plmarks:
                            _mw = np.asarray(_mk.world, dtype=float)[:, :2]
                            _nearw = _mw[np.linalg.norm(_mw - _pbody, axis=1) < 25.0]
                            if len(_nearw) < 4:
                                continue
                            _mlat = float(np.mean((_nearw - _pbody) @ _lbody))
                            # A marking visibly left of the ego is the centre/
                            # left boundary under right-hand traffic.  Do not
                            # treat a right-edge marking as a centreline.
                            if _mlat <= 0.1:
                                continue
                            for _corner in _corners_body:
                                _clat, _cov = _boundary_lateral(
                                    float(_corner[0]), float(_corner[1]), _mw,
                                    _fbody)
                                if _cov and _clat > 0.05:
                                    painted_body_cross = True
                                    break
                            if painted_body_cross:
                                break
                    except Exception:
                        painted_body_cross = False

                # safety arbitration on the chosen path: evaluate against the
                # SAME world model the planner planned against - one Scene
                # per tick.  Rebuilding it here let the two layers disagree
                # about the lane reference (the rebuild fed the BEV
                # whole-road centre, i.e. the two-way centre line, into the
                # safety lateral check while the planner used the own lane).
                # The rebuild below survives only as the fallback for stubs
                # that publish no Scene (docs/fsd_realism.md §2/§4).
                _strict_perc = _strict_lane
                scene = getattr(out, "scene", None)
                if scene is not None:
                    grid = scene.grid
                    verd = monitor.evaluate(
                        scene, best, planner_age_s=0.0,
                        snapshot_age_s=_sensor_snapshot_age(out),
                        ego_speed_mps=v)
                else:
                    grid = OccupancyGrid(stack.grid_n, stack.grid_n,
                                         stack.grid_res,
                                         origin=(float(pos[0]), float(pos[1])),
                                         heading=heading)
                    # Fallback lane reference: the stack's own-lane centre
                    # when available; never the map route in strict mode.
                    lane_local = (np.asarray(out.lane_ref, dtype=float)
                                  if out.lane_ref is not None
                                  and len(out.lane_ref) >= 4
                                  else (None if _strict_perc
                                        else route_local))
                    if out.bev is not None and \
                            out.bev.shape == grid.occupancy.shape:
                        grid.occupancy[:] = np.asarray(out.bev, dtype=np.float32)
                        grid.obstacle[:] = (np.asarray(out.bev) >= 0.6
                                            ).astype(np.uint8)
                        if out.drivable is not None and \
                                out.drivable.shape == grid.drivable.shape:
                            grid.drivable[:] = np.asarray(out.drivable)
                        if out.observed is not None and \
                                out.observed.shape == grid.drivable.shape:
                            grid.observed[:] = np.asarray(out.observed)
                        scene = Scene(pos=pos, heading=heading, grid=grid,
                                      route=route_local,
                                      lane_ref=lane_local,
                                      lane_left=out.lane_left,
                                      lane_right=out.lane_right,
                                      lane_width=out.lane_width,
                                      lane_envelope=getattr(
                                          out, "lane_envelope", None),
                                      perception_snapshot=getattr(
                                          out, "snapshot", None),
                                      meta=dict(getattr(out, "meta", {}) or {}),
                                      target_speed=args.speed,
                                      strict_perception=_strict_perc)
                        verd = monitor.evaluate(
                            scene, best, planner_age_s=0.0,
                            snapshot_age_s=_sensor_snapshot_age(out),
                            ego_speed_mps=v)
                    else:
                        scene = Scene(pos=pos, heading=heading)
                        verd = monitor.evaluate(scene, best)
                # Bounded PATH_HOLD offer (plan phase B): cache the FSD
                # path whenever this tick's verdict is drivable on FRESH
                # sensors - the hold is the bounded reuse of exactly this
                # verified trajectory when a later tick loses the path.
                # Rule/map fallbacks and stale-sensor ticks are never
                # offered: the hold replays perception evidence only.
                if best is not None and len(best) >= 2 and verd.drivable \
                        and not verd.stale_sensor and not verd.stale_planner:
                    if monitor.offer_verified_path(
                            best, heading, verd.target_speed,
                            strict=_strict_perc):
                        hold_offers += 1
                _tc = time.time()

                # Neural (E2E) planner candidate: the trained end-to-end
                # network consumes the same perception the layered planner
                # saw (front RGB + segmentation label + BEV) and regresses
                # an ego-relative trajectory, inverse-transformed to world
                # here.  It is arbitrated BELOW the layered planner but
                # ABOVE the map/rule fallback - a real FSD stack ranks its
                # neural planner over the kinematic backup.  Only the
                # trajectory steers the car; the net's raw action is
                # telemetry-only, and the safety monitor re-verifies the
                # path before arbitration can select it.
                e2e_path = None
                e2e_safe = False
                e2e_ms = None
                e2e_act = None
                e2e_ext = None
                e2e_reject = ""
                e2e_val = None
                _ve = None
                if e2e_rt is not None:
                    try:
                        e2e_path, e2e_act, e2e_ms = e2e_rt.step(
                            getattr(out, "snapshot", None) or out,
                            pos, heading, float(v))
                        e2e_reject = e2e_rt.last_reject
                        e2e_val = e2e_rt.last_validation
                        if e2e_path is not None and len(e2e_path) >= 2:
                            _ve = monitor.evaluate(scene, e2e_path,
                                                   planner_age_s=0.0,
                                                   snapshot_age_s=
                                                   _sensor_snapshot_age(out),
                                                   ego_speed_mps=v)
                            e2e_safe = bool(_ve.safe)
                            e2e_ext = float(np.hypot(
                                e2e_path[-1, 0] - pos[0],
                                e2e_path[-1, 1] - pos[1]))
                    except Exception:
                        e2e_path = None
                        e2e_safe = False

                # DAVE-2 imitation-learned steering (M3 BC): the net predicts
                # a normalized steering value from the front frame; rolled
                # into a constant-curvature arc it becomes a regular candidate
                # path the safety monitor verifies, ranked below the E2E
                # planner and above the rule backup (same neural-above-rule
                # ordering).
                bc_path = None
                bc_safe = False
                bc_ms = None
                bc_steer = None
                bc_reject = ""
                bc_val = None
                if bc_rt is not None:
                    try:
                        bc_steer, bc_ms = bc_rt.predict_steer(out.frame)
                        if bc_steer is not None:
                            bc_path = steer_to_path(
                                bc_steer, pos, heading,
                                wheelbase=float(pp.wheelbase))
                            bc_val = validate_learned_path(
                                bc_path, origin=pos,
                                forward=(math.cos(heading),
                                         math.sin(heading)))
                            if not bc_val.ok:
                                bc_reject = bc_val.reason
                                bc_path = None
                            if bc_path is not None and len(bc_path) >= 2:
                                _vb = monitor.evaluate(scene, bc_path,
                                                       planner_age_s=0.0,
                                                       snapshot_age_s=
                                                       _sensor_snapshot_age(out),
                                                       ego_speed_mps=v)
                                bc_safe = bool(_vb.safe)
                            else:
                                bc_path = None
                    except Exception:
                        bc_path = None
                        bc_safe = False

                _cls = out.meta.get("cls_counts", {})
                _cls_near = out.meta.get("cls_nearest", {})
                # planner arbitration: FSD path first; when the layered
                # planner declined (even to minimal risk) fall back to the
                # rule straight-ahead reference IN WORLD COORDINATES - the
                # car must not stop dead on a transient "no drivable path"
                # unless the rule path is also unusable (then and only then
                # a minimal-risk stop).  A body-frame reference handed to
                # PurePursuit points at a wrong world target and spins the
                # car (the "dumb reversing" seen in probes).
                # The rule fallback must also be a path the car can actually
                # drive from here: anchored at the ego and heading forward.  A
                # mis-anchored map prior sitting metres away must not be an
                # excuse to push the wall - when no drivable path exists, the
                # correct FSD behaviour is a minimal-risk stop (town runs
                # 2026-08-21 pushed a wall under a far-away route reference).
                # Proven rule-autopilot fallback: the LocalPlanner rounds
                # switchback corners and keeps the car in its own lane, so the
                # FSD drive does not stop dead at a hairpin apex when the
                # layered planner declines every kinked map-prior candidate.
                # The path must still be ego-anchored and head forward; the
                # FSD safety monitor re-verifies it below (and can stop).
                rule_ref = None
                _need_rule = (best is None or len(best) < 2 or not verd.safe)
                # FSD realism (strict): with no PAIRED perception lane the
                # car must stop, not drive the map/nav route through the rule
                # fallback (docs/fsd_realism.md §4).  The rule planner below
                # is exactly that map fallback, so it is disabled here; the
                # verdict itself is owned by the planner (``plan_blocked``)
                # and consumed - not re-derived - by the runtime contract.
                _strict_no_lane = strict_lane_unavailable(
                    bool(args.strict),
                    out.meta.get("plan_blocked"),
                    out.meta.get("lane_src_sel", ""))
                # In strict mode the rule path must not drive at all: it is
                # planned from ``nav_route`` with ``sensor_lane=None``, so
                # its lateral reference is map/route geometry (AGENTS.md
                # iron rule).  Gating on "no perception lane" alone left it
                # driving whenever FSD declined WITH a lane present - that
                # was every body-crossing and off-road frame on the
                # 2026-09-11 town run.  Do not build it here; the arbiter
                # enforces the same rule independently.
                if _strict_no_lane or bool(args.strict):
                    _need_rule = False
                if _need_rule and nav_route is not None and len(nav_route) >= 2:
                    try:
                        _fwd = (np.asarray(st.dir[:2], dtype=float)
                                if st.dir is not None else np.array(
                                    [math.cos(heading), math.sin(heading)]))
                        _nidx = nearest_route_point(nav_route[:, :2], pos, _fwd)
                        _rd, _rblk = rule_planner.plan(
                            np.asarray(nav_route[:, :2], dtype=float), [],
                            pos, heading, _nidx,
                            sensor_lane=None, road_rule=None, cross_solid=False)
                        if _rd is not None and len(_rd) >= 2 and not _rblk:
                            rule_ref = anchored_rule_ref(
                                pos, heading, np.asarray(_rd, dtype=float)[:, :2])
                    except Exception:
                        rule_ref = None
                chosen = arbitrate_fsd_tick(
                    best, rule_ref,
                    # ``drivable`` (not ``safe``): a degraded verdict is
                    # drivable by definition - the monitor computed the
                    # reduced speed cap for it and the loop applies it below
                    # via ``target = min(verd.target_speed, ...)``.  Gating on
                    # ``safe`` threw that cap away and force-stopped the car
                    # (2026-09-18 live east_coast: 122 of 222 ticks stopped,
                    # 83 of them level=degraded / source=none with
                    # mon_target 3.30 m/s and an open corridor; strict mode
                    # has no rule backup to fall through to).
                    fsd_safe=(best is not None and len(best) >= 2
                              and bool(getattr(verd, "drivable", False))),
                    e2e_path=e2e_path,
                    e2e_safe=e2e_safe,
                    bc_path=bc_path,
                    bc_safe=bc_safe,
                    strict=bool(args.strict),
                    plan_blocked=out.meta.get("plan_blocked"),
                    lane_src_sel=out.meta.get("lane_src_sel", ""),
                    prefer_rule=False)
                # Re-verify the verdict against the path the car actually
                # runs: the FSD verdict above was computed on the FSD best
                # (which may be None / minimal-risk).  Arbitration then
                # handed a drivable rule path to control, but the old
                # minimal-risk stop stayed latched and braked the fallback
                # to a standstill every frame (run_fix6: src=rule with
                # brk=1.0 and v=0 for 60 s after a transient no-path frame).
                if chosen.path is not None and len(chosen.path) >= 2 and \
                        (best is None or len(best) < 2 or not verd.safe):
                    try:
                        if chosen.source == "e2e" and _ve is not None:
                            # already verified this tick against the same
                            # scene; reuse to avoid a second evaluation
                            verd = _ve
                        else:
                            verd = monitor.evaluate(scene, chosen.path,
                                                    planner_age_s=0.0,
                                                    snapshot_age_s=
                                                    _sensor_snapshot_age(out),
                                                    ego_speed_mps=v)
                    except Exception:
                        pass
                # Bounded PATH_HOLD consume (plan phase B): when
                # arbitration produced nothing this tick but the monitor
                # still holds a recently verified path (and re-checked it
                # against the CURRENT scene before serving), steer that
                # degraded path with its decaying target instead of a
                # single-frame full stop.  The hold expires into a
                # controlled stop; only a fresh verified path restarts.
                if chosen.path is None and getattr(verd, "path_hold_active",
                                                   False) \
                        and getattr(verd, "held_path", None) is not None:
                    chosen = ArbiterOutcome(
                        np.asarray(verd.held_path, dtype=float)[:, :2],
                        "hold", verd.reason or "path hold")
                    hold_frames += 1
                # End-zone steering reference: inside the final stop zone the
                # raw route tail sits ON the road centre line at the
                # destination (the road ends / edges swap there) and the
                # near-end map lane can collapse onto the centreline -
                # following it parked the car ON the line (opt17: lat_left
                # 0.00 -> +0.97 inside oncoming -> parked lat_left 0.00).
                # The stop reference is therefore a STRAIGHT ray whose
                # LATERAL anchor comes from the perceived painted line
                # (own-lane centre - perception only, no nav-route offset),
                # and which falls back to holding the current heading when
                # no line is visible, so the car brakes to a stop centred
                # in its lane instead of turning onto the line.
                steer_path = chosen.path
                _end_replaced = False   # end zone rewrote the steering path
                _end_ref = 0  # 0=straight hold 1=last-good perception hold
                              # 2=live perception (telemetry)
                _dir_src = "none"
                # Perception-first travel direction resolved inside the end
                # zone; hoisted so the final alignment creep can read it
                # (stays None / "none" while the zone is off).
                _end_dir3 = None
                _end_dir_src = "none"
                if rem_end is not None and rem_end < END_PULL_START_M:
                    _bear = None
                    _anchor = np.asarray(pos[:2], dtype=float)
                    # Perception-only end-zone lateral reference (FSD
                    # realism rule 2026-09-03: lateral placement from a
                    # nav route + offset is BANNED - a real self-driving
                    # stack puts the car where the SENSORS say its lane
                    # is).  When the semantic head sees the painted line,
                    # aim the straight stop reference at the perceived
                    # own-lane centre (line right side + lane half width,
                    # the same perception helper as the start placement),
                    # so the final stop converges into the lane instead of
                    # riding the route centreline.  If the line is
                    # invisible (faded / degenerate road end) the car keeps
                    # converging to the LAST perceived own-lane centre for
                    # a short window, and only falls back to holding the
                    # current heading straight when that expires - still no
                    # map lateral pull anywhere in the chain.
                    _plc = None
                    try:
                        _sem = (out.head_outputs.get("semantic")
                                if out is not None
                                and getattr(out, "head_outputs", None)
                                else None)
                        if _sem is not None and out.cam is not None:
                            # Own-lane centre = line right side + measured
                            # lane half width.  The half width comes from the
                            # PAIRED sensor lane when it is live (perception),
                            # else the same 1.5 m default as start placement -
                            # never a map-centre offset constant.
                            _lh = 1.5
                            try:
                                if out is not None:
                                    _lw = float(getattr(out, "lane_width",
                                                        0.0) or 0.0)
                                    _sel = str(out.meta.get("lane_src_sel", ""))
                                    if _lw >= 2.4 and _sel == "sensor":
                                        _lh = float(np.clip(_lw / 2.0,
                                                            1.2, 2.6))
                            except Exception:
                                _lh = 1.5
                            _plc = painted_line_lane_center(
                                _sem, out.cam, pos, float(heading),
                                ground_z=(float(pos[2])
                                          - config.EGO_ORIGIN_GROUND_GAP_M
                                          if len(pos) > 2 else None),
                                lane_half_m=_lh, marks=_plmarks)
                    except Exception:
                        _plc = None
                    # Travel orientation for the stop ray - PERCEPTION FIRST,
                    # never a lateral offset.  1) the painted line's own
                    # direction (the sensors' answer to "which way is my
                    # lane"); 2) the paired sensor lane centreline's local
                    # heading; 3) in the legacy non-strict mode only, the
                    # nav route tangent as a plain orientation fallback;
                    # 4) hold the current heading.  Strict FSD never even
                    # reads the route here - the route tail folds onto the
                    # centreline at road ends, which is exactly why the nose
                    # used to park angled (docs/fsd_realism.md §1/§4).
                    _painted_dir = None
                    _sensor_dir = None
                    if out is not None:
                        try:
                            if _sem is not None and out.cam is not None:
                                _pd = painted_line_direction(
                                    _sem, out.cam, pos, float(heading),
                                    ground_z=(float(pos[2])
                                              - config.EGO_ORIGIN_GROUND_GAP_M
                                              if len(pos) > 2 else None))
                                if _pd is not None:
                                    _painted_dir = np.asarray(_pd[:2],
                                                              dtype=float)
                            if _painted_dir is None and \
                                    str(out.meta.get("lane_src_sel", "")) \
                                    == "sensor":
                                # Sensor lane centreline local heading; sanity-
                                # gate it to the travel direction so a stray
                                # geometry read cannot aim the stop sideways.
                                _dl = polyline_dir_at(out.lane_ref, _anchor)
                                _hf = np.array([math.cos(float(heading)),
                                                math.sin(float(heading))])
                                if _dl is not None and \
                                        float(_dl @ _hf) >= 0.5:
                                    _sensor_dir = _dl
                        except Exception:
                            _painted_dir = None
                            _sensor_dir = None
                    _route_tan = None
                    if not _strict_lane:
                        _nav_ref = nav_route_ref \
                            if nav_route_ref is not None else nav_route
                        if _nav_ref is not None and len(_nav_ref) >= 4:
                            _r3 = np.asarray(_nav_ref[:, :2], dtype=float)
                            _d3 = np.linalg.norm(
                                _r3 - _anchor[None, :], axis=1)
                            _i3 = int(np.argmin(_d3))
                            _i3a = max(0, _i3 - 2)
                            _i3b = min(len(_r3) - 1, _i3 + 2)
                            _tv3 = _r3[_i3b] - _r3[_i3a]
                            _L3 = float(np.linalg.norm(_tv3))
                            if _L3 > 1e-9:
                                _route_tan = _tv3 / _L3
                    _dir3, _dir_src = _endzone_travel_direction(
                        float(heading), painted=_painted_dir,
                        sensor=_sensor_dir, route_tangent=_route_tan,
                        strict=_strict_lane)
                    # Publish the resolved direction (and whether it came
                    # from perception) for the alignment creep below.
                    _end_dir3 = _dir3
                    _end_dir_src = _dir_src
                    _ref_xy = None
                    if _plc is not None:
                        _ref_xy = np.asarray(_plc, dtype=float)[:2]
                        end_plc_cache = (_ref_xy, time.time())
                        _end_ref = 2
                    elif end_plc_cache is not None:
                        # Line dropout: reuse the last perceived own-lane
                        # centre while it is still fresh AND the car is
                        # still near the same lane line (pedal to the
                        # cached straight reference, projected ahead of the
                        # ego), so the stop keeps converging instead of
                        # freezing at the entry offset.
                        _cxy, _t_seen = end_plc_cache
                        _age = time.time() - float(_t_seen)
                        _s_proj = float((_anchor - _cxy) @ _dir3)
                        _perp = float((_anchor - _cxy)
                                      @ np.array([-_dir3[1], _dir3[0]]))
                        if _age <= END_PLC_HOLD_S \
                                and _s_proj >= -1.0 \
                                and _s_proj <= END_PLC_MAX_FWD_M \
                                and abs(_perp) <= END_PLC_MAX_LAT_M:
                            _ref_xy = _cxy
                            _end_ref = 1
                    if _ref_xy is not None:
                        _tgt3 = _ref_xy + _dir3 * 5.0
                        _v3 = _tgt3 - _anchor
                        if float(np.linalg.norm(_v3)) > 1e-6:
                            _bear = float(math.atan2(_v3[1], _v3[0]))
                    if _bear is None:
                        # No live line, no fresh last-good anchor: hold the
                        # current heading straight instead of following the
                        # degenerate tail or any map prior.
                        _bear = float(heading)
                    _f3 = np.array([math.cos(_bear), math.sin(_bear)])
                    steer_path = _anchor + _f3 * np.arange(
                        0.0, 8.0, 0.8)[:, None]
                    _end_replaced = True
                # End-zone post-processing is a steering-path transform too:
                # the straight stop reference REPLACED the planner's path, so
                # the verdict made on ``chosen.path`` no longer covers what is
                # being steered.  Re-run the same full-body/occupancy contract
                # and fall back to the already-approved planner path when the
                # replacement fails (fail closed on the transform, not on
                # what was already verified).
                end_rejected = False
                if _end_replaced and steer_path is not None:
                    try:
                        _end_v = monitor.evaluate(
                            scene, steer_path, planner_age_s=0.0,
                            snapshot_age_s=_sensor_snapshot_age(out),
                            ego_speed_mps=v)
                    except Exception:
                        _end_v = None
                    if _end_v is None or _end_v.level == "minimal_risk":
                        end_rejected = True
                        if chosen.path is not None:
                            steer_path = chosen.path
                    else:
                        verd = _end_v
                # Steady painted-line lateral corrector (cruising only): the
                # end zone above already owns its perception reference, so the
                # nudge engages while the car is driving normally.  When the
                # map lane keeps leading (lane_src_sel != sensor) but the
                # semantic LINE mask gives a confident own-lane centre, shift
                # the near path toward it at a bounded rate - perception pulls
                # the car into its own lane instead of hugging the centre
                # line.  The corrector holds the last perceived shift across a
                # line dropout and decays it, so the car never jerks back to
                # the map prior mid-line.  Rule-source frames (FSD declined,
                # obstacle/unsafe fallback) must not get a superimposed
                # centring pull - the fallback path already carries its own
                # avoidance shape, so the shift just holds then decays.
                _plc_shift = 0.0
                _plc_desired = None
                plc_rejected = False
                _plc_active = painted_line_correction_active(
                    str(out.meta.get("lane_src_sel", "")),
                    str(chosen.source),
                    rem_end, END_PULL_START_M)
                if _plc_active:
                    try:
                        _sem0 = (out.head_outputs.get("semantic")
                                 if out is not None
                                 and getattr(out, "head_outputs", None)
                                 else None)
                        if _sem0 is not None and out.cam is not None:
                            _olc = painted_line_lane_center(
                                _sem0, out.cam, pos, float(heading),
                                ground_z=(float(pos[2])
                                          - config.EGO_ORIGIN_GROUND_GAP_M
                                          if len(pos) > 2 else None),
                                marks=_plmarks)
                            if _olc is not None:
                                _plc_desired = plc_corr.desired_shift(
                                    _olc, pos, float(heading),
                                    max_shift_m=PLC_MAX_SHIFT_M)
                    except Exception:
                        pass
                    _plc_shift = plc_corr.update(_plc_desired, dt, v)
                    if steer_path is not None \
                            and abs(_plc_shift) >= PLC_MIN_ENGAGE_M:
                        _steer_pre_plc = steer_path
                        steer_path = plc_corr.apply(
                            steer_path, pos, float(heading))
                        # PLC is a post-processing transform.  Re-run the same
                        # full-body/occupancy safety contract after shifting;
                        # the monitor verdict made on the pre-shift path is no
                        # longer sufficient.
                        try:
                            _plc_v = monitor.evaluate(
                                scene, steer_path, planner_age_s=0.0,
                                snapshot_age_s=_sensor_snapshot_age(out),
                                ego_speed_mps=v)
                            plc_rejected = _plc_v.level == "minimal_risk"
                        except Exception:
                            plc_rejected = True
                        if plc_rejected:
                            # The lateral correction is OPTIONAL
                            # post-processing and the monitor had ALREADY
                            # approved the un-shifted path, so rejecting the
                            # shift must skip the correction - not stop the
                            # car.  Escalating it to ``force_stop`` made the
                            # state inescapable: on the 2026-09-11 town run the
                            # car sat 1.838 m off the painted line with
                            # plc_active on all 145 tail frames, the 1.0 m
                            # shift was rejected every frame, and the vehicle
                            # could never move to get back (emergency=1
                            # throughout).
                            steer_path = _steer_pre_plc
                steer = 0.0
                pp_alpha = None
                pp_tgt = None
                ff_steer = 0.0
                mode_policy = None
                if DRIVE_MODES_ENABLED:
                    # D5: classify the situation ONCE per tick, before the
                    # command is shaped, so both the steering authority and
                    # the pedal rules below answer to the same mode.  Every
                    # input is evidence this tick already produced.
                    _mode_lat_e = None
                    _mode_radius = None
                    if steer_path is not None and len(steer_path) >= 2:
                        _mode_lat_e, _ = _path_errors_steer_frame(
                            steer_path, pos, heading)
                        _mode_radius = _path_radius_m(steer_path, pos,
                                                      heading)
                    mode_policy = drive_modes.classify(
                        now=now_t, speed_mps=v,
                        target_speed=float(verd.target_speed),
                        stop_commanded=bool(
                            chosen.path is None
                            or verd.level == "minimal_risk"
                            or (rem_end is not None
                                and rem_end < END_STOP_M)),
                        risk_braking=bool(
                            getattr(verd, "risk_kind", "")
                            == "braking_obstacle"
                            and verd.target_speed > 0.0),
                        radius_m=_mode_radius,
                        lateral_error_m=_mode_lat_e,
                        perception_ok=bool(
                            str(getattr(verd, "lane_ref_src", "none"))
                            != "none"),
                        elapsed_s=max(0.0, now_t - t0))
                if steer_path is not None and len(steer_path) >= 2:
                    # Same conversion as the proven rule autopilot: PurePursuit
                    # returns the steering ANGLE (rad), BeamNG expects a
                    # normalized input with the OPPOSITE sign (left target ->
                    # positive angle -> negative input).  Feeding the raw angle
                    # (as before 2026-08-22) steered the car the WRONG WAY at
                    # the town junction - the route bent left, the raw positive
                    # steer was applied as right, yaw swung -32 -> -70 deg and
                    # every candidate crossed the lane boundaries -> wedged.
                    # Also rate-limit like m5_autopilot so the wheel does not
                    # slam from lock to lock on a flickering reference.
                    # Speed-adaptive lookahead like the proven rule autopilot:
                    # a fixed 5 m lookahead at the hairpin lands the target ON the
                    # bend instead of past it, so the wheel barely turns and the
                    # car understeers off the road before the corner (mountain run
                    # 2026-08-26 run_fix11: at (727.1,757.5) the controller only
                    # asked -0.12 and the car missed the first hairpin, then only
                    # looping arcs were left).  Computed from the BASE lookahead
                    # each frame (no compounding ratchet).
                    pp.lookahead = float(np.clip(
                        5.0 + 0.55 * max(0.0, v), 4.0, 16.0))
                    steer_rad, pp_tgt, pp_near = pp.steering(
                        pos, heading, np.asarray(steer_path))
                    steer_rad = float(steer_rad)
                    ff_steer = _path_curvature_ff(steer_path, pos, heading)
                    if STEER_BLEND_ENABLED:
                        # D2: Pure Pursuit stays the base; the blend only
                        # ADDS the lateral/heading feedback terms, with the
                        # speed schedule in control/blend.py.
                        _lat_e, _head_e = _path_errors_steer_frame(
                            steer_path, pos, heading)
                        _blend = blend_steering(
                            -steer_rad / 0.6, ff_steer, _lat_e, _head_e,
                            float(v), weights=steer_blend_w)
                        new_steer = float(np.clip(_blend.steer, -1.0, 1.0))
                        steer_blend_digest = _blend.digest()
                    else:
                        new_steer = float(np.clip(
                            -steer_rad / 0.6 + ff_steer, -1.0, 1.0))
                    if mode_policy is not None                             and mode_policy.steer_gain != 1.0:
                        # D5: a launch must not yank the wheel (STARTING),
                        # a low-speed alignment keeps its corrections gentle
                        new_steer = float(np.clip(
                            new_steer * float(mode_policy.steer_gain),
                            -1.0, 1.0))
                    # Speed-adaptive steering cap (same as the proven rule
                    # autopilot): at speed a full-lock correction swings the
                    # car far past the lane direction (the FSD runs showed
                    # 40-60 deg over-rotation at the junction exits, mountain
                    # runs 2026-08-27 fix31/32).  The rule autopilot caps the
                    # wheel by v^2 so high-speed corrections stay gentle.
                    v_sq = max(v * v, 2.0)
                    steer_cap = max(0.10, min(1.0, 5.0 * 2.9 / v_sq / 0.6))
                    # Low-speed cap: full lock at 2 m/s is a ~2.9 m radius
                    # circle and builds a yaw rate the slow control loop
                    # cannot catch - the car swings around the junction
                    # instead of converging onto the lane.  0.45 (~10.7 m
                    # radius) cannot track the 8 m hairpin fillet, so the
                    # first -110 -> -24 deg bend was run wide every time
                    # (fix44: lat_l=-7.7 at the apex, then reverse-loops).
                    # 0.55 (~8.8 m radius) matches the hairpin geometry; the
                    # old 0.55 over-rotation (fix40, 9.7 m/s downhill) is
                    # now blocked by the hard speed governor below.
                    steer_cap = min(steer_cap, 0.55)
                    new_steer = float(np.clip(new_steer, -steer_cap, steer_cap))
                    # Yaw-rate damper: oppose a fast rotation that is not
                    # being commanded (left rotation -> steer right).  It must
                    # NOT fight a hard commanded turn - in the hairpin the car
                    # needs ~0.35 rad/s of yaw and the damper cut the full-lock
                    # input from -0.55 to -0.46, so the car ran wide off the
                    # road (fix49: lat_left -7.5 m at the apex).  Only damp
                    # while the wheel is not already at a strong commanded
                    # angle.
                    if abs(yaw_rate) > 0.3 and abs(new_steer) < 0.35:
                        new_steer = float(np.clip(
                            new_steer + 0.25 * yaw_rate,
                            -steer_cap, steer_cap))
                    # D3: the shaper must start from whatever was last
                    # commanded (stop / end-zone branches assign ``steer``
                    # directly), exactly like the old smooth_steer did.
                    steer_shaper.value = float(prev_steer)
                    steer = steer_shaper.update(new_steer, dt)
                    prev_steer = steer
                    _tv = np.asarray(pp_tgt, dtype=float)[:2] - pos[:2]
                    pp_alpha = round(float(math.degrees(
                        math.atan2(_tv[1], _tv[0]) - heading)), 1)

                # Longitudinal plan from the FULL RAW nav route, never the
                # local resampled window: the local window starts ON the
                # corner and its Catmull-Rom resample rounds the hairpin into
                # a 4-5 m sweep, so the curvature profile reads 4-5 m/s into
                # the bend and the car understeers off the road (mountain run
                # 2026-08-23, run_fix6: plan_v 4.9-6.0 at the first hairpin,
                # car left the road 3-10 m west of the route and never came
                # back).  The raw road-graph polyline keeps the 90-degree
                # kink - its look-ahead profile caps the entry speed at
                # ~1.7 m/s, which is the speed the bend can actually take.
                if route_round is not None and route_arc is not None \
                        and route_rad is not None:
                    _rfull = route_round
                    _ga = route_arc
                    # Arc-length projection (not nearest-vertex): the nearest
                    # vertex can flip between two close route samples frame to
                    # frame, which snapped the sampled profile speed by whole
                    # m/s on straight segments; the projection stays stable.
                    _proj_s = float(_project_arc(_rfull, pos[:2]))
                    try:
                        # Profile the ROUNDED full route, not the raw road-graph
                        # polyline: the graph collapses the first hairpin into a
                        # sharp vertex whose curvature profile caps the bend at
                        # ~1.7 m/s - at that speed the tyres scrub and the car
                        # cannot even turn (steering probe 2026-08-27).  The
                        # rounded route (same 8 m fillet the lane centre uses)
                        # lets the bend be taken at a speed the steering can
                        # actually execute.  The rounded polyline, its arc
                        # lengths and per-vertex radii are precomputed once
                        # before the loop (route_round/route_arc/route_rad).
                        _sp_raw = _spf_raw(
                            _rfull, scene,
                            target_speed=float(args.speed),
                            # Corridor-open clutter (dense junction/end-zone
                            # LiDAR that still leaves a free band) must not
                            # pin the full-route plan to the 1 m/s MIN_SPEED -
                            # the safety monitor keeps the same cruise floor
                            # when its corridor is open.
                            obstacle_min_speed=(
                                max(_PROF_MIN_SPEED,
                                    0.4 * float(args.speed))
                                if verd.corridor_open else _PROF_MIN_SPEED))
                        if len(_sp_raw):
                            # The full-route profile is indexed along the WHOLE
                            # route; [0] is the speed at the ROUTE START, not at
                            # the car.  Sample the profile at the nearest route
                            # point so a hairpin 100 m into the route still caps
                            # the speed when the car reaches it (run_fix25:
                            # plan_speed stayed at the start speed into the bend).
                            _i = int(np.clip(
                                int(np.searchsorted(_ga, _proj_s)),
                                0, len(_rfull) - 1))
                            out.best_speed = float(_sp_raw[_i])
                            out.meta["plan_src"] = "nav_round"
                    except Exception:
                        pass
                    # Tight-bend entry governor: the ~1.4 s control tick lets the
                    # car overshoot the profiled corner speed by ~+1 m/s mid-tick
                    # (fix45: 4.4 target -> ~5.5 actual at the first hairpin, ran
                    # wide off the left edge).  For a bend tighter than 15 m in
                    # the next 12 m, cap the plan speed at sqrt(1.5*R) so the
                    # actual peak stays near 4 m/s, where the 0.55 steering cap
                    # (~8.8 m radius) can track the 8 m hairpin fillet.
                    try:
                        _gi = int(np.clip(
                            int(np.searchsorted(_ga, _proj_s)),
                            0, len(_rfull) - 1))
                        _ghi = int(np.searchsorted(_ga, _proj_s + 12.0))
                        _lo = max(1, _gi)
                        _hi = min(_ghi, len(route_rad) - 2) + 1
                        _rmin = 1e9
                        if _lo < _hi:
                            _rmin = float(np.min(route_rad[_lo:_hi]))
                        if _rmin < 15.0:
                            # Turn-angle gate: only a REAL bend deserves the
                            # hairpin speed cap.  A rounded junction corner /
                            # resample wiggle can measure R~3 m over 12 m while
                            # turning <30 deg total; capping there parked the
                            # plan at sqrt(1.3*3)=1.97 and the car crawled 5+ s
                            # through a widening junction (fsd opt23 t=36-45,
                            # lat_right 1.5->4.2).  A true hairpin turns
                            # 60-180 deg over the same window.
                            if _route_turn_deg(route_round, _lo, _hi) >= \
                                    BEND_GOV_MIN_TURN_DEG:
                                # Floor the implied radius: the 0.8 m resample
                                # can measure a hairpin fillet edge as R~0.8 m
                                # (three nearly-collinear points), which caps
                                # the plan at ~1 m/s and stands the car dead
                                # on the approach (fix65: plan=1.00 at the
                                # second bend, v=5.6 -> brake-to-0).  Real
                                # roads never bend tighter than ~3 m; anything
                                # smaller is a sampling artifact.
                                _rmin = max(_rmin, 3.0)
                                out.best_speed = float(min(
                                    out.best_speed, math.sqrt(1.3 * _rmin)))
                                out.meta["plan_src"] = "nav_round+gov"
                    except Exception:
                        pass
                    # FSD-realism speed cap: slow on what the sensors SEE.
                    # The nav-route profile above is navigation intent; when
                    # the BEV road mask curves ahead, the plan is capped from
                    # PERCEPTION only (docs/fsd_realism.md §2).
                    try:
                        if out.drivable is not None:
                            _pg = OccupancyGrid(
                                60, 60, 0.5,
                                origin=(float(pos[0]), float(pos[1])),
                                heading=float(heading))
                            _pg.drivable = np.asarray(out.drivable, dtype=float)
                            _pcap = perception_curve_speed(_pg, out.best_speed)
                            if _pcap < out.best_speed:
                                out.best_speed = _pcap
                                out.meta["plan_src"] = "perception-curve"
                    except Exception:
                        pass
                # control from the (possibly degraded) target speed, but never
                # exceed the *planned* speed along the chosen trajectory - the
                # FSD longitudinal plan (bend deceleration, obstacle brake
                # band) must govern the actual pedals.
                plan_speed = out.best_speed if out.best_speed > 0.0 \
                    else float(args.speed)
                if _strict_no_lane and not verd.path_hold_active:
                    # No current perception lane means NO MOTION in strict
                    # mode, regardless of which candidate won arbitration.
                    # E2E/BC are perception-driven models, but neither gives
                    # a reliable current lane boundary when the painted/LiDAR
                    # lane is unavailable; allowing them here let the car
                    # drift half outside the road (user screenshot,
                    # town 2026-09-07).
                    # EXCEPTION - bounded PATH_HOLD (plan phase B): the
                    # monitor is serving the last VERIFIED trajectory inside
                    # its hold window and re-checked it against this tick's
                    # scene; the hold's own (decaying) cap owns the target
                    # instead of a zero-speed latch.
                    plan_speed = 0.0
                    plan_sm = 0.0
                # Traffic-light action: vision head (colour blob) fused with
                # the game's authoritative road-link signal state via
                # merge_signal_vision.  The game state wins when it knows the
                # light; a confident vision RED stops the car at the line.
                # Green never forces motion - the planner / safety monitor
                # still own the pedals, so the car simply resumes when the
                # light turns green.
                if not args.no_signal and now_t - sig_rule_t > SIGNAL_RULE_POLL_S:
                    sig_rule_t = now_t
                    try:
                        _rr = conn.read_current_road_rule(
                            pos, (math.cos(heading), math.sin(heading), 0.0))
                        if (_rr is not None and _rr.n1 and _rr.n2
                                and conn.vehicle is not None):
                            _sigs = conn.read_signal_snapshot(
                                conn.vehicle.vid, _rr.n1, _rr.n2)
                            _sel = select_signal_rule(_sigs, pos[:2],
                                                      heading=heading)
                            sig_rule_state = (
                                _sel.state if _sel is not None else None)
                    except Exception:
                        sig_rule_state = None
                _sig_final, _sig_src = merge_signal_vision(
                    sig_rule_state,
                    str(out.meta.get("signal_state") or "none"),
                    float(out.meta.get("signal_conf", 0.0) or 0.0),
                    trust_vision_conf=SIGNAL_CONF_MIN)
                out.meta["signal_src"] = _sig_src
                if not args.no_signal and _sig_final == "red":
                    plan_speed = 0.0
                    plan_sm = 0.0
                # a rule fallback does not get the FSD plan speed; cap it to a
                # cautious creep so the L2 fallback is gentle
                if chosen.source == "rule":
                    plan_speed = min(plan_speed, 3.0)
                plan_raw_speed = float(plan_speed)
                # Perception-only off-road measure for this tick: how far
                # the body sticks out past a DETECTED lane boundary.  Used
                # by the learned decision layer below, the off-road hard
                # stop / recovery branches and the telemetry.  0.0 also
                # means "no boundary seen", never a map fallback.
                road_off = _perception_off_road_m(out, pos, heading)
                # M4 DQN decision layer: cap the plan target with the learned
                # cruise/ease/slow decision from the stack's own perception.
                # It can only SLOW the plan - steering and every safety layer
                # stay authoritative - so a bad policy costs comfort, never
                # safety.
                #
                # The end-pull zone is exempt: inside END_PULL_START_M the
                # end-zone ease/hold/alignment-creep ladder below owns the
                # longitudinal target, and a learned "slow"/"ease" action
                # there only fights the deterministic stop (the creep branch
                # already runs slow by construction).  Skipping the policy
                # keeps the final approach repeatable.
                dqn_action = None
                dqn_ms = None
                _end_zone = _in_end_pull_zone(rem_end)
                if dqn_rt is not None and not _end_zone:
                    try:
                        dqn_action, dqn_ms = dqn_rt.predict(
                            speed=v,
                            target_speed=max(0.5, float(plan_speed)),
                            fwd_clearance=out.forward_clearance,
                            closest_obs=(None if verd.closest_obs_m > 900.0
                                         else float(verd.closest_obs_m)),
                            lane_dev=float(getattr(verd, "lane_dev_m", 0.0)),
                            road_off=road_off,
                            n_tracks=len(out.tracks))
                        plan_speed = min(
                            plan_speed,
                            action_to_target(dqn_action, plan_speed))
                    except Exception:
                        dqn_action = None
                        dqn_ms = None
                # Rate-limit the plan (PLAN_* constants): transient LiDAR
                # clutter at junctions/end zones must not snap the plan
                # 6.0 <-> 1.0 between ticks and make the controller brake
                # then relaunch (opt22).  Real bends still decelerate - the
                # profile drops smoothly over many frames, well inside the
                # down rate - and the safety monitor / force-stop remain the
                # authority for genuine emergencies.
                _dplan = float(np.clip(
                    plan_raw_speed - plan_sm,
                    -PLAN_DOWN_RATE_MPS2 * dt,
                    PLAN_UP_RATE_MPS2 * dt))
                plan_sm = float(plan_sm + _dplan)
                plan_speed = plan_sm
                target = min(verd.target_speed, plan_speed, float(args.speed))
                if mode_policy is not None                         and np.isfinite(mode_policy.max_speed_mps):
                    # D5: CONTROLLED_STOP caps to zero, RECOVERY creeps
                    target = min(target, float(mode_policy.max_speed_mps))
                # Heading-error speed scrub: the nav route is the intent;
                # when the nose drifts off it (oscillation / over-rotation)
                # slow down so the steering loop converges instead of
                # feeding the swing.  Falls back to no-op when the local
                # route bearing cannot be measured.
                _rh_b = _ref_bearing(route_local, pos)
                if _rh_b is not None:
                    _hdg_dev = abs((float(heading)
                                    - math.radians(_rh_b) + math.pi)
                                   % (2.0 * math.pi) - math.pi)
                    _hdg_deg = math.degrees(_hdg_dev)
                    if _hdg_deg > HEADING_DEV_START_DEG:
                        _k = min(1.0, (_hdg_deg - HEADING_DEV_START_DEG)
                                 / (HEADING_DEV_FULL_DEG
                                    - HEADING_DEV_START_DEG))
                        _hdg_cap = (HEADING_DEV_FLOOR_MPS
                                    + (HEADING_DEV_CAP_MPS
                                       - HEADING_DEV_FLOOR_MPS)
                                    * (1.0 - _k))
                        target = min(target, _hdg_cap)
                # No-route guard: without a nav route there is no map prior to
                # keep the car on the road - a straight-line reference drives
                # straight onto grass (opt13 2026-08-28: no route after a game
                # restart, car crept on the grass at 0-4 m/s).  Never cruise
                # without a route; the caller must pass --goal.
                if nav_route is None:
                    target = min(target, 1.0)
                # Map road-edge guard: the nav centreline + real DecalRoad
                # edge rows are the map prior for "where the road is".  Once
                # the ego is beyond the local road edge (grass/verge on the
                # right, oncoming lane on the left) the car must not keep
                # driving - crawl at 0.5 m/s; the monitor still stops it if
                # the path is blocked.  The centreline is used instead of the
                # raw edge polylines because edge rows fold at junctions and
                # go stale past the last graph node (town run10: a folded
                # edge corner reported 3.2 m off-road on a straight section
                # while the car sat on the centreline).
                # Definitively off-lane (PERCEPTION only, computed above):
                # the body is more than ROAD_OFF_STOP_M past a detected
                # boundary.  The hard stop + hold is applied further down
                # together with the other override branches; ``off_recover``
                # also suppresses the reverse escape (never back further
                # out).  The old map crawl ladder that lived here was dead
                # code and map-based - see the constant comment above.
                off_recover = bool(road_off > ROAD_OFF_STOP_M)
                # End-of-route (rem_end computed above from the FULL nav route
                # arc, so it never goes None when the local window collapses):
                # ease to a stop while still in the lane instead of parking
                # over the edge line at the road end (opt15).
                if rem_end is not None:
                    if rem_end < END_STOP_M:
                        target = 0.0
                    elif rem_end < END_START_SLOW_M:
                        target = min(target, END_SLOW_MPS)
                # Warm-up crawl + stale-tick scrub (real-time mode): before
                # the object head is live, or after an unusually long tick,
                # the car has been driving open-loop - keep it slow.
                if time.time() < warmup_until and not out.meta.get("object_head"):
                    target = min(target, WARMUP_SPEED_MPS)
                if _wall_dt > STALE_CTRL_S:
                    target = min(target, STALE_CTRL_SPEED_MPS)
                # Safety clearance along the CHOSEN path (FSD vector-space
                # safety layer): the grid obstacle layer the planner itself
                # scored against is the authority, so a wall beside the nose
                # that the chosen arc turns away from does not park the car
                # (town corner run 2026-08-21: planner picked a feasible arc,
                # raw LiDAR foliage read 0.5 m and the emergency layer
                # force-stopped every frame).  A path that really is blocked
                # still forces the same stop.  ``inf`` from
                # ``path_grid_clearance_m`` MEANS the path is clear - it must
                # not be treated as "missing" (run 2026-08-22: the fallback
                # replaced a clean-path inf with the raw heading corridor
                # 0.19 m at a town corner and parked a car that was steering
                # fine).  The raw-sensor heading corridor is only the last
                # line when there is NO planned path at all.
                # ``plc_rejected`` no longer escalates to a vehicle stop: the
                # rejected shift is skipped and the approved un-shifted path
                # is driven instead (see the fallback above).  Keeping it here
                # made the state inescapable - a car parked 1.838 m off the
                # line had its correction rejected on every frame, so it was
                # force-stopped on every frame and could never drive back.
                # The telemetry field is kept for diagnosis.
                force_stop = bool(painted_body_cross)
                if force_stop:
                    target = 0.0
                fwd_clear = float("inf")
                if chosen.path is not None and len(chosen.path) >= 2:
                    fwd_clear = path_grid_clearance_m(chosen.path, grid)
                else:
                    fwd_clear = float(out.forward_clearance)
                if np.isfinite(fwd_clear):
                    need = emergency_stop_clearance_m(v)
                    force_stop, cap = emergency_speed_limit_mps(fwd_clear, need)
                    target = min(target, cap if not force_stop else 0.0)
                # Smooth the effective target: ramp toward the raw plan at a
                # bounded rate (sim time), so the corner governor stepping the
                # plan from 6 to 3.2 m/s in one tick cannot flip the pedals.
                # A safety force-stop bypasses the ramp and brakes immediately.
                if force_stop:
                    target_sm = 0.0
                elif LONG_PLAN_ENABLED:
                    # D4: compose curvature / lateral accel / obstacle TTC /
                    # perception confidence, then shape the reference with
                    # accel AND jerk limits.  The emergency branches above
                    # (force_stop / hard stop) never reach this path, so the
                    # comfort shaping cannot delay a safety action.
                    _lt = long_planner.update(
                        speed_mps=v, dt=dt, plan_speed=target,
                        radius_m=_path_radius_m(steer_path, pos, heading),
                        gap_m=getattr(verd, "risk_closest_m", None),
                        ttc_s=getattr(verd, "min_ttc_s", None),
                        road_conf=(out.meta.get("lane_envelope")
                                   or {}).get("confidence"),
                        line_conf=out.meta.get("line_conf_current"))
                    target_sm = min(float(_lt.reference), plan_speed)
                    long_digest = _lt.digest()
                else:
                    _dmax = SPEED_TARGET_RAMP_MPS * dt
                    if target > target_sm:
                        target_sm = min(target, target_sm + _dmax)
                    else:
                        target_sm = max(target, target_sm - _dmax)
                    # Never cruise above the planned corner speed.  The ramp
                    # can still be converging down from a high initial target
                    # (first frames), which let the car overshoot the bend
                    # plan and trip the hard governor every tick (fix61:
                    # v=4.45 against plan 3.23 -> brake 1.0 -> stall ->
                    # full throttle again).  Capping the smoothed target by
                    # plan_speed keeps the pedals inside the plan from the
                    # very first tick.
                    target_sm = min(target_sm, plan_speed)
                thr, brk = speed_ctrl.update(
                    target_sm, v, dt=min(0.25, max(0.01, dt)))
                # Downhill-start throttle guard: on the descent the car
                # accelerates by gravity alone, so feeding throttle near the
                # plan overshoots the bend speed and the governor then brakes
                # it to a standstill every tick (fix61-64: v 0 -> 4.4 -> 0).
                # Below a low speed, hold the pedal near idle and let gravity
                # bring the speed up to the plan; the controller resumes once
                # the speed is there.
                if v < 2.5 and signed > 0.3 and target_sm <= plan_speed:
                    thr = min(thr, 0.25)
                # Downhill acceleration cap: once the car is rolling, full
                # throttle demand plus gravity overshoots the plan in one
                # 0.66 s burst (opt8: 0.80 throttle -> 2.7 -> 7.3 m/s).
                # Cap the pedal while approaching the target so gravity does
                # most of the work; the plan governor trims the rest.
                if v > 2.5 and signed > 0.5 and v < target_sm - 0.5:
                    thr = min(thr, 0.35)
                # (The old "corner brake zone" here is gone: with the sim
                # paused and stepped in 0.33 s bursts the speed controller
                # reacts within one burst, while the zone caused an
                # accelerate -> brake-to-stop oscillation - fix54 reached
                # v=3.6 then brk=0.8 stopped it dead at every tick.  The
                # plan-speed governor below still hard-brakes overshoot.)
                # Soft overspeed governor: never let the car exceed the
                # commanded cruise speed by more than 1 m/s regardless of
                # the smoothed pedal state.  The old brk=1.0 here stopped
                # the car DEAD on the downhill, then the controller relaunched
                # with full throttle -> 0 <-> 7.8 m/s bang-bang (opt8).
                # Taper the throttle off between +0.5 and +1.3 m/s overshoot
                # instead of the old hard cut at +1.0.  A hard cut to 0 then
                # a full re-launch made a +/-0.9 m/s speed wave around cruise
                # (opt21: 49 throttle on/off flips in 169 frames); a gradual
                # taper removes the relaunch kick while the plan governor's
                # gentle GOV_BRAKE still trims the overshoot.
                _ov = v - (float(args.speed) + 0.5)
                if _ov > 0.0:
                    thr *= float(np.clip((0.8 - _ov) / 0.8, 0.0, 1.0))
                # Plan-speed governor: never let the car exceed the planned
                # corner speed by more than 0.8 m/s even within one tick -
                # the profile alone is sampled at tick boundaries and the
                # car can overshoot a 4.4 m/s hairpin plan to ~5.5 mid-tick.
                if v > plan_speed + GOV_ON_MPS:
                    gov_brake = True
                elif not (v > plan_speed + GOV_OFF_MPS):
                    gov_brake = False
                if gov_brake:
                    thr, brk = 0.0, max(brk, GOV_BRAKE)
                # Stuck detection: see ``_counts_as_stuck`` for the two
                # states it covers and why the command-stop case needed its
                # own rule.
                _near_obs = float(getattr(verd, "closest_obs_m", 999.0)
                                  or 999.0)
                if _counts_as_stuck(
                        has_path=(chosen.path is not None),
                        force_stop=bool(force_stop), v=v, thr=thr,
                        plan_speed=plan_speed, near_obs_m=_near_obs,
                        rem_end=rem_end):
                    stuck_t += max(0.0, float(dt))
                else:
                    stuck_t = 0.0
                stuck = stuck_t >= 2.5
                # hard stop when no path remains, the raw-sensor forward
                # clearance is inside the braking reserve (never grind into a
                # wall / wedge the car into a too-narrow gap), or the car is
                # stuck spinning against an obstruction
                pb = 0.0  # handbrake: hold the car on a slope while stopped
                # Slope-creep assist: the planner keeps a CLEAR forward path
                # while the car cannot move (e.g. stopped at the bottom of a
                # dip facing uphill).  Full throttle for a bounded window lets
                # it climb before the reverse escape is allowed to arm - the
                # fix41 east-side dip at (756.7,740.6) had fwd=8.6 clear but
                # the stuck detector sent the car backwards downhill instead
                # of giving it torque.
                climb = False
                if (stuck and not force_stop and not off_recover
                        and chosen.path is not None
                        and len(chosen.path) >= 2
                        and np.isfinite(fwd_clear) and fwd_clear > 3.0):
                    if climb_t < CLIMB_ASSIST_S:
                        climb = True
                        climb_t += max(0.0, float(dt))
                    else:
                        climb_t = 0.0  # give up climbing -> allow reverse
                if (chosen.path is None or force_stop or stuck) \
                        and not climb and not off_recover:
                    thr, brk = 0.0, 1.0
                    steer = 0.0
                    stopps += 1
                    pb = 1.0
                if climb:
                    thr, brk = 1.0, 0.0
                    steer = 0.0
                    pb = 0.0
                # Controlled reverse escape: when NO drivable forward path
                # remains (dead-end / wedged nose) and the space BEHIND the
                # car is clear, back up a bounded distance in R, then let the
                # planner re-attempt a forward path.  This is the "reverse to
                # find a feasible path" behaviour; it never reverses while a
                # forward path exists and never reverses blindly (no rear
                # clearance data -> stay stopped).
                has_forward_path = (chosen.path is not None
                                    and len(chosen.path) >= 2
                                    and not force_stop and not stuck) or climb \
                    or off_recover
                # Arrived at the destination: once inside the stop zone and
                # nearly stopped, treat the forward path as present so the
                # reverse escape never backs the car over the line at the end
                # (opt18: rev=1 at the goal moved the car onto the oncoming
                # lane and it parked over the line).
                if rem_end is not None and rem_end < END_STOP_M and v < 0.6:
                    has_forward_path = True
                rear_clear_m = None
                if not has_forward_path and grid is not None:
                    try:
                        _bh = float(heading) + math.pi
                        _ln = np.array([math.cos(_bh), math.sin(_bh)])
                        _lt = np.array([-_ln[1], _ln[0]])
                        _best = float("inf")
                        for _lat in (0.0, -1.5, 1.5):
                            _o = pos[:2] + _lat * _lt
                            for _ds in np.arange(1.0, 10.0, 0.4):
                                _wx = float(_o[0] + _ds * _ln[0])
                                _wy = float(_o[1] + _ds * _ln[1])
                                _cell = grid.world_to_cell(_wx, _wy)
                                if _cell is not None:
                                    _r, _c = int(_cell[0]), int(_cell[1])
                                    if (0 <= _r < grid.obstacle.shape[0]
                                            and 0 <= _c < grid.obstacle.shape[1]
                                            and grid.obstacle[_r, _c] > 0):
                                        _best = min(_best, _ds - 0.5)
                                        break
                        rear_clear_m = (float(_best) if _best < float("inf")
                                         else 40.0)
                    except Exception:
                        rear_clear_m = None
                # The reverse-escape policy lives in the state machine
                # itself (``rman.enabled``, set from --strict above) so it
                # is unit-testable instead of being an inline condition
                # here.  See ReverseManeuver.enabled for the measured
                # 2026-09-18 live numbers.
                rm = rman.decide(has_forward_path=has_forward_path,
                                 rear_clear_m=rear_clear_m,
                                 signed_speed=signed,
                                 pos2d=pos[:2], dt=dt)
                gear_use = fwd_gear
                if not rm.active:
                    rev_thr = REV_THR_BASE   # reset the escape ramp
                if rm.active:
                    # Bounded reverse: gear R, slow, straight back (steering
                    # centred) until the state machine releases the attempt.
                    gear_use = rm.gear
                    steer = 0.0
                    pb = 0.0  # the car must roll for the escape
                    if signed >= -0.05:
                        # Gentle R throttle: 0.25 ramped the car to -3 m/s in
                        # ~1.5 s (fix37); 0.10 still reached -3.0 on the
                        # east-side slope (fix41).  0.06 keeps the escape
                        # near the -0.4 m/s target even on a mild grade.
                        # No throttle once the car is ALREADY rolling back
                        # (signed < -0.05): on a downhill slope gravity does
                        # the backing, adding throttle only rolls it further
                        # before the brake catches (east-side roll-back).
                        # ON GRASS 0.06 cannot overcome rolling resistance:
                        # the escape then pulses forever without moving
                        # (fsd_benchmark mountain 2026-09-05: 28 s stuck at
                        # road_off ~0.9 with rev_state cycling).  Ramp the
                        # throttle while the car is not yet rolling back;
                        # the -0.4 m/s target brake + rear_clear stop keep
                        # the ramp bounded.
                        rev_thr = min(REV_THR_MAX, rev_thr + REV_THR_STEP)
                        thr, brk = rev_thr, 0.0
                    elif signed > rm.target_speed_mps:
                        # Approaching the reverse target: back off the
                        # throttle and brake softly instead of waiting for a
                        # full overshoot (control ticks are ~1.4 s apart).
                        thr, brk = 0.0, 0.6
                    else:
                        # Backward speed already beyond the target (downhill
                        # roll): brake hard AND handbrake - 0.8 alone let the
                        # car run away to -7 m/s on a slope (run_fix17).
                        thr, brk, pb = 0.0, 1.0, 1.0
                    # Extra stop margin when the rear space is nearly gone.
                    if rear_clear_m is not None and rear_clear_m <= 2.0:
                        thr, brk, pb = 0.0, 1.0, 1.0
                elif reversing:
                    # Passive reverse guard (unintended backward motion, e.g.
                    # a wall bounce): brake, centre the wheel, no throttle.
                    # Handbrake too while still moving backwards so a slope
                    # cannot roll the car away.
                    thr, brk = 0.0, max(brk, float(rev_brk))
                    steer = 0.0
                    if signed < -0.1:
                        pb = 1.0
                # In strict mode, hold the handbrake before a stopped car
                # rolls through zero.  The final clean run had no reverse
                # maneuver (rman.enabled=False), but braking a 0.33 m/s
                # stop on the grade let signed speed cross to -0.38 for
                # three telemetry frames.  This is not a route recovery;
                # it is a passive anti-roll hold and keeps the strict
                # contract "never reverse" true.
                if (bool(getattr(args, "strict", False))
                        and signed < REVERSE_CLEAR_MPS
                        and brk >= 0.8
                        and v < 0.6):
                    thr = 0.0
                    brk = 1.0
                    pb = 1.0
                # Off-road recovery: the hard DecalRoad edges are the ground
                # truth for "on the road".  Once the car has LEFT the road
                # (grass/verge on the right, oncoming side on the left) it
                # must not keep driving - creeping on the verge understeered
                # 2 m -> 10 m further away at the town corner (opt24).  Hard
                # stop + hold like the end zone; the driver / next teleport
                # repositions.  Reverse escape is suppressed while off-road
                # (has_forward_path above) so it never backs further off.
                if off_recover:
                    thr = 0.0
                    brk = max(brk, END_BRAKE)
                    steer = 0.0
                    if v < 0.4:
                        pb = 1.0
                # End-of-route hard stop + hold: target=0 alone only asks the
                # speed controller for a gentle ramp (brk ~0.16 at 1.3 m/s),
                # so the car rolled through the whole stop zone, the lane
                # reference collapsed onto the centre line and it parked ON
                # the line (opt18: ll -1.00 -> 0.00 at rem 4->0, final parked
                # over the line after the reverse escape).  Inside the stop
                # zone brake hard so the car stops BEFORE the degenerate road
                # end, and hold with the handbrake once stopped.
                if rem_end is not None and rem_end < END_STOP_M:
                    thr = 0.0
                    brk = max(brk, END_BRAKE)
                    # Final ALIGNMENT before the hold: brake to a crawl, then
                    # creep straight until the body yaw is within
                    # ALIGN_YAW_DEG of the PERCEIVED lane direction - a
                    # mid-turn full-hold parks the car diagonal across both
                    # lanes (town 2026-09-06, user screenshot).  The
                    # reference is perception only: the nav-route bearing
                    # used to drive this creep, which is a map-derived
                    # steering reference (banned in the FSD entry) and
                    # folded onto the centreline exactly at road ends.
                    # Without a perceived lane direction the car holds the
                    # brake instead of steering on map.  Bounded by
                    # rem_end > 2 m so the creep never pushes past the road
                    # end.
                    _yaw_dev = None
                    try:
                        _yaw_dev = _endzone_align_yaw_dev(
                            heading, _end_dir3, _end_dir_src)
                    except Exception:
                        _yaw_dev = None
                    if v < 0.5:
                        if (_yaw_dev is not None
                                and abs(_yaw_dev) > math.radians(ALIGN_YAW_DEG)
                                and rem_end > 2.0):
                            # creep forward while straightening toward the
                            # route direction (positive steer = right =
                            # heading decreases)
                            thr = ALIGN_CREEP_THR
                            brk = 0.0
                            pb = 0.0
                            steer = float(np.clip(
                                _yaw_dev * 1.2, -0.4, 0.4))
                        else:
                            steer = 0.0
                            pb = 1.0
                # Pedal rate limit: the branches above (downhill cap, taper,
                # governor, climb/reverse/hard-stop) can step thr/brk by a
                # whole pedal in one tick - a relaunch then reads as a speed
                # kick (opt23: 13 speed jumps >1.5 m/s per tick).  Ramp the
                # FINAL commanded pedals toward the previous tick's at bounded
                # rates; safety branches bypass on purpose (hard stop, climb,
                # reverse escape, end-zone hold).
                _hard_pedal = _needs_hard_pedal(
                    has_path=(chosen.path is not None),
                    force_stop=bool(force_stop),
                    stuck=bool(stuck),
                    climb=bool(climb),
                    reverse_active=bool(rm.active),
                    reversing=bool(reversing),
                    rem_end=rem_end)
                if not _hard_pedal:
                    thr, brk = rate_limit_pedal(
                        thr, brk, prev_thr, prev_brk, dt)
                if mode_policy is not None                         and not mode_policy.allow_throttle:
                    # D5: while stopping / braking for an obstacle the
                    # throttle is forced off so the pedals cannot fight.
                    # It only ever REMOVES throttle - never adds any.
                    thr = 0.0
                prev_thr, prev_brk = thr, brk
                conn.control(throttle=thr, brake=brk, steering=steer,
                             gear=gear_use, parkingbrake=pb)
                # Shadow-frame recording (same ShadowFrame contract as
                # m5_shadow_drive): a drive tick IS one labelled episode
                # sample - executed controls + the perception/planning
                # evidence (BEV, drivable, trajectory, camera + semantic
                # label) that produced them, for image / BEV end-to-end
                # training later.  A bad tick (no plan) is labelled with
                # quality=0 so it can be gated out of the dataset.
                if rec is not None:
                    try:
                        _sem_r = out.head_outputs.get("semantic")
                        _label = None
                        if _sem_r is not None and out.frame is not None \
                                and "road" in getattr(_sem_r, "masks", {}):
                            _label = np.zeros(out.frame.shape[:2], dtype=np.uint8)
                            _label[_sem_r.masks["road"]] = 1
                            if "line" in getattr(_sem_r, "masks", {}):
                                _label[_sem_r.masks["line"]] = 2
                        _traj = (chosen.path if chosen.path is not None
                                 else out.best_path)
                        _fmap = None
                        _fm = getattr(out, "feature_map", None)
                        if _fm is not None:
                            _fmap = np.stack([
                                np.asarray(_fm.get(c), dtype=np.float32)
                                for c in FMAP_CHANNELS]).astype(np.float32)
                        rec.add(ShadowFrame(
                            x=float(pos[0]), y=float(pos[1]),
                            heading=heading, speed=v,
                            throttle=thr, brake=brk, steer=steer,
                            bev_raster=(np.asarray(out.bev, dtype=np.float32)
                                        if out.bev is not None else None),
                            drivable=(np.asarray(out.drivable, dtype=np.uint8)
                                      if out.drivable is not None else None),
                            fmap=_fmap,
                            trajectory=(np.asarray(_traj, dtype=float)[:, :2]
                                        if _traj is not None
                                        and len(_traj) >= 2 else None),
                            target_speed=float(plan_speed),
                            lane_src=str(out.meta.get("lane_src", "")),
                            cost=float(out.meta.get("planner", {})
                                       .get("cost", -1.0)),
                            kind=str(out.meta.get("planner", {})
                                     .get("kind", "")),
                            rgb=(np.ascontiguousarray(out.frame, dtype=np.uint8)
                                 if out.frame is not None else None),
                            label=_label,
                            quality=1.0 if chosen.path is not None else 0.0))
                    except Exception as _rec_e:
                        print(f"[fsd-drive] shadow frame dropped: {_rec_e}")
                # Lane-position telemetry: signed lateral offset of the ego from
                # each DETECTED lane boundary (left: + = inside oncoming traffic;
                # right: - = off the road edge).  None when the boundary does
                # not extend to the ego or no lane was detected this frame.
                lat_left = lat_right = None
                fwd_lane = np.array([float(np.cos(heading)),
                                     float(np.sin(heading))])
                if out.lane_left is not None:
                    try:
                        _ll, _cl = _boundary_lateral(
                            float(pos[0]), float(pos[1]), out.lane_left, fwd_lane)
                        lat_left = round(float(_ll), 3) if _cl else None
                    except Exception:
                        pass
                if out.lane_right is not None:
                    try:
                        _lr, _cr = _boundary_lateral(
                            float(pos[0]), float(pos[1]), out.lane_right, fwd_lane)
                        _lr = float(_lr)
                        lat_right = round(float(_lr), 3) if _cr else None
                    except Exception:
                        pass
                # BODY-aware lateral position: a yawed car crosses the line
                # with its body while the centre point still reads in-lane
                # (town run 2026-09-06: crossC=0 while the user photographed
                # the left wheels ON the line).  Boundary heading ~= route
                # bearing; footprint halves are the etk800's.
                # FULL-BODY projection: the four corners of the ego footprint
                # (half 2.2 x 0.9 m) in world space, each tested against the
                # detected lane boundaries - ANY corner beyond a boundary is
                # a body crossing (the centre point + lateral-extent
                # approximation missed yawed-body crossings).
                body_cross_l = body_cross_r = 0
                try:
                    _cy, _sy = math.cos(heading), math.sin(heading)
                    _fwd = np.array([_cy, _sy])
                    _corners = footprint_corners(pos[:2], float(heading))
                    if out.lane_left is not None:
                        for _c in _corners:
                            _lc, _cov = _boundary_lateral(
                                float(_c[0]), float(_c[1]), out.lane_left,
                                _fwd)
                            if _cov and _lc > 0.05:
                                body_cross_l += 1
                    if out.lane_right is not None:
                        for _c in _corners:
                            _rc, _cov = _boundary_lateral(
                                float(_c[0]), float(_c[1]), out.lane_right,
                                _fwd)
                            if _cov and _rc < -0.05:
                                body_cross_r += 1
                except Exception:
                    pass
                # Body overshoot past each DETECTED boundary: the worst of
                # the four corners, measured straight against the boundary
                # polyline - the same perception-only footprint the gates
                # above use.  (The old version projected a route bearing
                # from the map onto the centre offset; no map in the
                # lateral metric any more.)
                body_lat_left = body_lat_right = None
                try:
                    _corners3 = footprint_corners(pos[:2], float(heading))
                    if out.lane_left is not None:
                        _l_lat = []
                        for _c in _corners3:
                            _lc, _cov = _boundary_lateral(
                                float(_c[0]), float(_c[1]), out.lane_left, None)
                            if _cov:
                                _l_lat.append(float(_lc))
                        if _l_lat:
                            body_lat_left = round(max(_l_lat), 3)
                    if out.lane_right is not None:
                        _r_lat = []
                        for _c in _corners3:
                            _rc, _cov = _boundary_lateral(
                                float(_c[0]), float(_c[1]), out.lane_right, None)
                            if _cov:
                                _r_lat.append(float(_rc))
                        if _r_lat:
                            body_lat_right = round(min(_r_lat), 3)
                except Exception:
                    pass
                # Honest off-pavement metrics (MAP used as a METRIC only,
                # never as a lateral reference): route_dist = distance
                # from the road centreline; edge_over = metres beyond the
                # road's own edge polylines (0.0 while between them).
                _route_dist = None
                _edge_over = None
                if nav_route is not None and len(nav_route) >= 2:
                    _rn = np.asarray(nav_route[:, :2], dtype=float)
                    _rp = np.asarray(pos[:2], dtype=float)
                    _d = np.linalg.norm(_rn - _rp[None, :], axis=1)
                    _k = int(np.argmin(_d))
                    _route_dist = round(float(_d[_k]), 3)
                    if (road_left is not None and road_right is not None
                            and _k < len(road_left) and _k < len(road_right)):
                        _le = np.asarray(road_left[_k], dtype=float)[:2]
                        _re = np.asarray(road_right[_k], dtype=float)[:2]
                        if np.isfinite(_le).all() and np.isfinite(_re).all():
                            _half = 0.5 * float(np.linalg.norm(_le - _re))
                            _tan = _rn[min(_k + 1, len(_rn) - 1)] - _rn[_k]
                            _L = float(np.linalg.norm(_tan))
                            if _L > 1e-9 and _half > 1e-6:
                                _lat = float(
                                    (_rp - _rn[_k])
                                    @ np.array([-_tan[1], _tan[0]]) / _L)
                                _edge_over = round(
                                    max(0.0, abs(_lat) - _half), 3)
                # snapshot for offline stability evaluation (safe / degraded
                # ratio over a long route); written once at the end.
                if _vis_every and (frames % _vis_every) == 0:
                    try:
                        from beamng_autopilot.vision.live_vis import (
                            render_lane_vis)
                        _vis_img = render_lane_vis(
                            out, pos, heading, line_lat=line_lat)
                        _vis_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(_vis_dir / f"frame_{frames:05d}.png"),
                                    _vis_img)
                        cv2.imwrite(str(_vis_dir / "last.png"), _vis_img)
                    except Exception as _vis_e:
                        if not _vis_warned:
                            _vis_warned = True
                            print(f"[fsd-drive] live-vis render failed: "
                                  f"{_vis_e}", flush=True)
                hist.append({
                    "t": round(time.time() - t0, 3),
                    "pos": [round(float(p), 3) for p in pos[:3]],
                    "heading": round(float(heading), 4),
                    "speed": round(float(v), 3),
                    "signed": round(float(signed), 3),
                    "level": str(verd.level),
                    "reason": verd.reason or "-",
                    "path_occ_frac": round(
                        float(getattr(verd, "path_occupied_frac", 0.0)), 4),
                    "corridor_open": bool(
                        getattr(verd, "corridor_open", False)),
                    "closest_obs_m": round(
                        float(getattr(verd, "closest_obs_m", 999.0)), 3),
                    "planner_kind": str(out.meta.get("planner", {})
                                         .get("kind", "")),
                    "source": str(chosen.source),
                    "e2e": int(e2e_path is not None and len(e2e_path) >= 2),
                    "e2e_safe": int(bool(e2e_safe)),
                    "e2e_ms": (round(float(e2e_ms), 1)
                               if e2e_ms is not None else None),
                    "e2e_extent": (round(float(e2e_ext), 2)
                                   if e2e_ext is not None else None),
                    "e2e_reject": e2e_reject or None,
                    "e2e_lat": (round(float(e2e_val.lateral_m), 2)
                                if e2e_val is not None else None),
                    "e2e_backstep": (round(float(e2e_val.backstep_m), 2)
                                     if e2e_val is not None else None),
                    "e2e_curv": (round(float(e2e_val.max_curvature), 4)
                                 if e2e_val is not None else None),
                    "bc": int(bc_path is not None and len(bc_path) >= 2),
                    "bc_safe": int(bool(bc_safe)),
                    "bc_steer": (round(float(bc_steer), 3)
                                 if bc_steer is not None else None),
                    "bc_ms": (round(float(bc_ms), 1)
                              if bc_ms is not None else None),
                    "bc_reject": bc_reject or None,
                    "bc_lat": (round(float(bc_val.lateral_m), 2)
                               if bc_val is not None else None),
                    "bc_backstep": (round(float(bc_val.backstep_m), 2)
                                    if bc_val is not None else None),
                    "bc_curv": (round(float(bc_val.max_curvature), 4)
                                if bc_val is not None else None),
                    "dqn_act": dqn_action,
                    "dqn_contract_ok": (
                        int(bool(dqn_rt.contract.ok))
                        if dqn_rt is not None
                        and dqn_rt.contract is not None else None),
                    "dqn_contract_reason": (
                        dqn_rt.contract.reason or None
                        if dqn_rt is not None
                        and dqn_rt.contract is not None else None),
                    "dqn_contract_warning": (
                        dqn_rt.meta_warning or None
                        if dqn_rt is not None else None),
                    "dqn_contract_git": (
                        dqn_rt.meta.get("git_commit")
                        if dqn_rt is not None and dqn_rt.meta else None),
                    "cls_tree": int(_cls.get("tree", 0)),
                    "cls_guardrail": int(_cls.get("guardrail", 0)),
                    "cls_wall": int(_cls.get("wall", 0)),
                    "cls_tree_d": _cls_near.get("tree"),
                    "body_lat_left": body_lat_left,
                    "body_lat_right": body_lat_right,
                    "body_cross_l": int(body_cross_l > 0),
                    "body_cross_r": int(body_cross_r > 0),
                    "painted_body_cross": int(painted_body_cross),
                    "plc_rejected": int(plc_rejected),
                    "end_rejected": int(end_rejected),
                    "pl_mask": int(_pl_mask),
                    "pl_marks": int(_pl_marks),
                    "pl_near": int(_pl_near),
                    "cls_guardrail_d": _cls_near.get("guardrail"),
                    "cls_wall_d": _cls_near.get("wall"),
                    "dqn_ms": (round(float(dqn_ms), 1)
                               if dqn_ms is not None else None),
                    "e2e_act": ([round(float(a), 3) for a in e2e_act]
                                if e2e_act is not None else None),
                    "plan_speed": round(float(plan_speed), 2),
                    "plan_raw": round(float(plan_raw_speed), 2),
                    "target_sm": round(float(target_sm), 2),
                    "plan_src": str(out.meta.get("plan_src", "?")),
                    # Why the planner did (or did not) publish a path.  The
                    # 2026-09-11 town run had 34 frames with a PAIRED
                    # perception lane and no path at all, and the record
                    # could not say whether the constraint layer declined
                    # every candidate, the strict gate fired, or the fan was
                    # empty - `n_eval = 0` in the planner meta is just its
                    # default when no plan exists.  Publish the decision.
                    "plan_blocked": str(out.meta.get("plan_blocked", "")),
                    "n_candidates": int(out.meta.get("total_candidates",
                                                     out.n_candidates) or 0),
                    "tick_ms": out.meta.get("tick_ms"),
                    "tick_wall_ms": round((_tb - _f0) * 1000.0, 1),
                    # cumulative control sub-commands (plan A2/A4); the
                    # per-run control rate is reported in the summary
                    "substeps": int(substeps),
                    "budget_s": round(float(_budget), 3),
                    "budget_skips": list(
                        out.meta.get("tick_budget_skips") or []),
                    'frame_ms': {
                        'local': round((_ta - _f0) * 1000.0, 1),
                        'tick': round((_tb - _ta) * 1000.0, 1),
                        'grid_mon': round((_tc - _tb) * 1000.0, 1),
                        'rest': round((time.time() - _tc) * 1000.0, 1),
                    },
                    "throttle": round(float(thr), 4),
                    "brake": round(float(brk), 4),
                    "steer": round(float(steer), 4),
                    # D3 shaping evidence: the applied steering rate, its
                    # rate of change (jerk) and the reversal guard counters
                    "steer_rate": round(float(steer_shaper.rate), 4),
                    "steer_jerk": round(float(
                        (steer_shaper.rate - _sr_prev)
                        / max(1e-3, float(dt))), 3),
                    "steer_rev": int(steer_shaper.reversals),
                    "steer_supp": int(steer_shaper.suppressed),
                    # D2 evidence: the blend's terms and scheduled weights
                    # (None while the blend is off, i.e. Pure Pursuit only)
                    "steer_blend": steer_blend_digest,
                    # D4 evidence: the composed longitudinal terms and the
                    # shaped reference (None while the planner is off)
                    "long_plan": long_digest,
                    # D5 evidence: which situation the car believed it was in
                    "drive_mode": (mode_policy.mode
                                   if mode_policy is not None
                                   else None),
                    "mode_why": (mode_policy.reason
                                 if mode_policy is not None
                                 else None),
                    "mode_switches": int(drive_modes.switches),
                    "reversing": int(bool(reversing)),
                    "rev_state": str(rman.state),
                    "rev_active": int(bool(rm.active)),
                    "rear_clear": (round(float(rear_clear_m), 2)
                                   if rear_clear_m is not None else None),
                    "fwd_clear": float(out.forward_clearance)
                        if np.isfinite(out.forward_clearance) else None,
                    "emergency": int(bool(force_stop)),
                    "stuck": int(bool(stuck)),
                    "lane_src": str(out.meta.get("lane_src", "?")),
                    "lane_mode": str(args.lane_mode),
                    "lane_reject": str(out.meta.get("lane_reject_reason", "")),
                    "lane_sel": str(out.meta.get("lane_src_sel", "")),
                    "lane_paired": int(out.meta.get("lane_paired", 0)),
                    "lane_pair_debug": out.meta.get("lane_pair_debug"),
                    "lane_fusion_debug": out.meta.get("lane_fusion_debug"),
                    # Freshness ages.  The safety monitor's "stale sensor"
                    # verdict is `max(all head ages, range, bev, lane) >
                    # STALE_SNAPSHOT_S`, and without these in the log a
                    # stale run cannot be attributed at all: on 2026-09-11
                    # 33% of town frames went stale (0% on 2026-09-07) and
                    # the telemetry gave no way to see WHICH modality aged.
                    "freshness": (out.meta.get("snapshot") or {}).get(
                        "freshness"),
                    "head_age_s": (out.meta.get("snapshot") or {}).get(
                        "head_age_s"),
                    # Line-evidence provenance (plan phase E3): the fused
                    # line mask keeps historical support, and these say
                    # how much of it is fresh observation vs. held
                    # history, how long ago the line was last seen, and
                    # whether continuous loss has expired it.
                    "line_conf_current": out.meta.get("line_conf_current"),
                    "line_conf_history": out.meta.get("line_conf_history"),
                    "line_ev_age_s": out.meta.get("line_evidence_age_s"),
                    "line_ev_expired": out.meta.get("line_evidence_expired"),
                    "n_object_obstacles": int(
                        out.meta.get("n_object_obstacles", 0)),
                    "object_head": int(out.meta.get("object_head", 0)),
                    "signal_state": str(out.meta.get("signal_state", "")),
                    "signal_conf": (round(float(out.meta["signal_conf"]), 3)
                                    if out.meta.get("signal_conf") else None),
                    "intent": str(out.meta.get("intent", "")),
                    "intent_turn_deg": out.meta.get("intent_turn_deg"),
                    "change_left": int(out.meta.get("change_left", 0)),
                    "change_right": int(out.meta.get("change_right", 0)),
                    "n_tracks": int(out.meta.get("n_tracks", 0)),
                    "fmap": int(out.feature_map is not None),
                    "lane_dev_m": round(float(getattr(verd, "lane_dev_m", 0.0)), 3),
                    "lat_left": lat_left,
                    "lat_right": lat_right,
                    "kind": str(out.meta.get("planner", {}).get("kind", "?")),
                    "cost": round(float(out.meta.get("planner", {}).get("cost", 0.0)), 3),
                    "n_eval": int(out.meta.get("planner", {}).get("n_eval", 0)),
                    # Candidate hysteresis (plan phase D1): whether this
                    # tick switched trajectory, why, and how long the
                    # current candidate has been held - the evidence the
                    # plan asks for (candidate_switch_count / reason /
                    # current_candidate_age).
                    "cand_switch": int(
                        (out.meta.get("hysteresis") or {}).get("switch", 0)),
                    "cand_reason": str(
                        (out.meta.get("hysteresis") or {}).get("reason", "")),
                    "cand_age_s": (out.meta.get("hysteresis") or {}).get(
                        "age_s"),
                    "cand_n_switch": int(
                        (out.meta.get("hysteresis") or {}).get("n_switch", 0)),
                    "pp_alpha": pp_alpha,
                    "yaw_rate": round(float(yaw_rate), 3),
                    "ff_steer": round(float(ff_steer), 3),
                    "climb": int(bool(climb)),
                    "road_off": round(float(road_off), 3),
                    # Honest off-road metric: distance from the nav-route
                    # (road centreline) polyline.  The perception-based
                    # road_off is blind in strict mode (no boundaries ->
                    # 0.0 even off the pavement); the route is the map's
                    # road geometry used as a METRIC only, never as a
                    # lateral reference.
                    "route_dist": _route_dist,
                    "edge_over": _edge_over,
                    "off_recover": int(off_recover),
                    "rem_end": (round(float(rem_end), 2)
                                if rem_end is not None else None),
                    "mon_target": round(float(verd.target_speed), 2),
                    # Bounded PATH_HOLD + boundary-cross diagnostics (plan
                    # phases B/C1): WHERE a hold served a verified path
                    # instead of a single-frame stop, and whether a
                    # boundary violation was the CURRENT body or a PLANNED
                    # sweep pose (with its distance/segment/side).
                    "path_hold_active": int(bool(
                        getattr(verd, "path_hold_active", False))),
                    "path_hold_phase": str(
                        getattr(verd, "path_hold_phase", "") or ""),
                    "path_hold_age_s": (
                        round(float(verd.path_hold_age_s), 2)
                        if getattr(verd, "path_hold_age_s", None) is not None
                        else None),
                    "body_cross_current": int(bool(
                        getattr(verd, "body_cross_current", False))),
                    "body_cross_planned": int(bool(
                        getattr(verd, "body_cross_planned", False))),
                    "first_cross_m": (
                        round(float(verd.first_crossing_distance_m), 2)
                        if getattr(verd, "first_crossing_distance_m", None)
                        else None),
                    "cross_idx": (
                        int(verd.crossing_path_index)
                        if getattr(verd, "crossing_path_index", None)
                        is not None else None),
                    "cross_side": (str(verd.crossing_boundary_side)
                                   if getattr(verd, "crossing_boundary_side",
                                              "") else None),
                    # Obstacle risk grading (plan phase C3): the worst
                    # class among the tracked obstacles, the closest
                    # time-to-collision and the nearest graded distance.
                    "risk_kind": str(getattr(verd, "risk_kind", "") or ""),
                    "min_ttc": (round(float(verd.min_ttc_s), 2)
                                if getattr(verd, "min_ttc_s", None)
                                is not None else None),
                    "risk_closest_m": (
                        round(float(verd.risk_closest_m), 2)
                        if getattr(verd, "risk_closest_m", None) is not None
                        else None),
                    "closest_obs": (round(float(verd.closest_obs_m), 2)
                                    if verd.closest_obs_m < 900.0 else None),
                    "pp_tgt": ([round(float(v), 2) for v in pp_tgt[:2]]
                               if pp_tgt is not None else None),
                    "line_lat": line_lat,
                    "plc_active": int(_plc_active),
                    "plc_shift": round(float(_plc_shift), 3),
                    "plc_desired": (round(float(_plc_desired), 3)
                                    if _plc_desired is not None else None),
                    "end_ref": _end_ref,
                    "end_dir_src": _dir_src,
                    "lane_bear": _ref_bearing(out.lane_ref, pos),
                    "route_bear": _ref_bearing(route_local, pos),
                    "best_bear": _ref_bearing(out.best_path, pos),
                })
                _sr_prev = float(steer_shaper.rate)
                # --- control sub-steps (plan phases A2/A4) -------------
                # The perception + planning tick above keeps its natural
                # cadence (~2 Hz); between ticks the sim keeps running, and
                # the car deserves fresher commands than that.  Each
                # sub-step re-reads the state (cheap), re-checks the hard
                # pose gates against the SAME cached scene/plan, and re-runs
                # the pure steering + longitudinal controllers, so a slow
                # perception head delays the next PLAN, never the wheel or
                # the pedals.  The sub-step never plans and never sees new
                # sensor evidence (ControlSubstep owns those rules); it
                # brakes only for a stale plan or a contact-risk stop, and
                # hands a body crossing back to the tick.
                _sub_driving = (
                    chosen.path is not None and len(chosen.path) >= 2
                    and not force_stop and not stuck and not climb
                    and not rm.active and not reversing and not off_recover
                    and not (rem_end is not None and rem_end < END_STOP_M)
                    and _sig_final != "red")
                # Cache the plan this sub-step run may drive: the tick
                # timestamp (age zero) and the tracked obstacles the risk
                # re-check uses.  Sub-steps never update either.
                _plan_t = time.time()
                _sub_tracks = list(getattr(out, "tracks", None) or [])
                if _sub_driving and SUBSTEP_HZ > 0.0:
                    _sub_until = min(_f0 + 1.0 / REALTIME_CTRL_HZ, t_end)
                    _sub_last = time.time()
                    while time.time() < _sub_until:
                        _sub_t0 = time.time()
                        try:
                            _ss = conn.get_state()
                        except Exception:
                            break
                        _sp = np.asarray(_ss.pos, dtype=float)
                        _sh = float(_ss.heading)
                        _sv = float(_ss.speed)
                        _sdt = min(0.5, max(0.02, _sub_t0 - _sub_last))
                        _sub_last = _sub_t0
                        _now = time.time()
                        _age = _now - _plan_t
                        _cross_now = False
                        if scene is not None:
                            try:
                                _cross_now = body_pose_crosses_lane(
                                    scene, _sp, _sh,
                                    half_len=HALF_LENGTH_M + 0.25,
                                    half_width=HALF_WIDTH_M + 0.25)
                            except Exception:
                                _cross_now = True   # unknown -> hand back
                        _risk = None
                        try:
                            _risk = assess_obstacles(
                                _sub_tracks, _sp, _sh, _sv,
                                path=np.asarray(steer_path, dtype=float)
                                if steer_path is not None
                                and len(steer_path) >= 2 else None)
                        except Exception:
                            _risk = None
                        _dec = _substep.decide(
                            plan_age_s=_age, target_speed=target_sm,
                            pose_crosses=bool(_cross_now),
                            risk_stop=bool(_risk is not None and _risk.stop),
                            risk_cap=(float(_risk.target_speed_cap)
                                      if _risk is not None else None))
                        if not _dec.ok:
                            break
                        if _dec.stop:
                            conn.control(throttle=0.0, brake=1.0,
                                         steering=prev_steer, gear=gear_use,
                                         parkingbrake=1.0)
                            break
                        _s_steer = steer
                        if steer_path is not None and len(steer_path) >= 2:
                            pp.lookahead = float(np.clip(
                                5.0 + 0.55 * max(0.0, _sv), 4.0, 16.0))
                            _sr, _stgt, _ = pp.steering(
                                _sp, _sh, np.asarray(steer_path))
                            _ff = _path_curvature_ff(steer_path, _sp, _sh)
                            _new = float(np.clip(-float(_sr) / 0.6 + _ff,
                                                 -1.0, 1.0))
                            _vsq = max(_sv * _sv, 2.0)
                            _cap = min(0.55, max(0.10, min(
                                1.0, 5.0 * 2.9 / _vsq / 0.6)))
                            _new = float(np.clip(_new, -_cap, _cap))
                            steer_shaper.value = float(prev_steer)
                            _s_steer = steer_shaper.update(_new, _sdt)
                            prev_steer = _s_steer
                        _sthr, _sbrk = speed_ctrl.update(
                            _dec.target_speed, _sv,
                            dt=min(0.25, max(0.01, _sdt)))
                        _sthr, _sbrk = rate_limit_pedal(
                            _sthr, _sbrk, prev_thr, prev_brk, _sdt)
                        prev_thr, prev_brk = _sthr, _sbrk
                        conn.control(throttle=_sthr, brake=_sbrk,
                                     steering=_s_steer, gear=gear_use,
                                     parkingbrake=pb)
                        substeps += 1
                        _sl = (1.0 / SUBSTEP_HZ) - (time.time() - _sub_t0)
                        if _sl > 0.0:
                            time.sleep(_sl)
                # Real-time cadence: the sim keeps running; the heavy tick
                # is paced to REALTIME_CTRL_HZ - the sub-steps above already
                # consumed the interval when they ran.
                _elapsed = time.time() - _f0
                _slack = (1.0 / REALTIME_CTRL_HZ) - _elapsed
                if _slack > 0.0:
                    time.sleep(_slack)
                frames += 1
                if frames % 4 == 1:
                    _e2e_s = (f"e2e={e2e_ms:.0f}ms "
                              if e2e_ms is not None else "")
                    print(f"[fsd-drive] t={time.time()-t0:5.1f} v={v:4.1f} "
                          f"level={verd.level} src={chosen.source:4s} "
                          f"reason={verd.reason or '-':22s} "
                          f"steer={steer:+.2f} thr={thr:.2f} "
                          f"plan_v={plan_speed:.1f} {_e2e_s}"
                          f"rev={int(reversing)} signed={signed:+.2f} "
                          f"lane={out.meta.get('lane_src', '?')}/"
                          f"{'P' if out.meta.get('lane_paired') else '1'} "
                          f"dev={getattr(verd, 'lane_dev_m', 0.0):.2f}")
            _ll = [f["line_lat"] for f in hist if f.get("line_lat") is not None]
            if _ll:
                _arr = np.asarray(_ll, dtype=float)
                print(f"[fsd-drive] painted line lateral (left=+): "
                      f"mean={_arr.mean():+.2f}m p50="
                      f"{np.percentile(_arr, 50):+.2f}m "
                      f"min={_arr.min():+.2f} max={_arr.max():+.2f} "
                      f"({int((_arr < -0.5).sum())} frames car left of line)")
            _ps = [float(f.get("plc_shift", 0.0)) for f in hist]
            _nplc = sum(1 for f in hist
                        if abs(f.get("plc_shift", 0.0)) > PLC_MIN_ENGAGE_M)
            if _nplc:
                print(f"[fsd-drive] painted-line steady corrector: "
                      f"active {_nplc}/{len(hist)} frames, "
                      f"mean|shift|={np.mean(np.abs(_ps)):.2f}m")
            _er = [(f.get("end_ref", 0), f.get("rem_end")) for f in hist
                   if f.get("rem_end") is not None
                   and f["rem_end"] < END_PULL_START_M]
            if _er:
                _n_live = sum(1 for _r, _ in _er if _r == 2)
                _n_hold = sum(1 for _r, _ in _er if _r == 1)
                _n_flat = sum(1 for _r, _ in _er if _r == 0)
                print(f"[fsd-drive] end-zone ref: live-perception={_n_live} "
                      f"last-good-hold={_n_hold} straight-hold={_n_flat}")
            print(f"[fsd-drive] done: {frames} frames, {stopps} stops")
            if SUBSTEP_HZ > 0.0 and substeps:
                _run_s = max(1e-3, time.time() - t0)
                print(f"[fsd-drive] control decoupling: {substeps} sub-steps "
                      f"between {frames} ticks -> "
                      f"{(frames + substeps) / _run_s:.1f} control "
                      f"commands/s (target {SUBSTEP_HZ:.0f} Hz)")
            if hist:
                _n_cand_sw = sum(1 for f in hist if f.get("cand_switch"))
                _n_cand_hold = sum(1 for f in hist
                                   if f.get("cand_reason") in ("min_dwell",
                                                               "cost_margin"))
                if _n_cand_sw or _n_cand_hold:
                    print(f"[fsd-drive] candidate hysteresis: "
                          f"{_n_cand_sw} switch frames, "
                          f"{_n_cand_hold} held frames "
                          f"(total switches {hist[-1].get('cand_n_switch')})")
            if steer_shaper.suppressed:
                print(f"[fsd-drive] steering shaper: "
                      f"{steer_shaper.suppressed} trim-wiggle reversals "
                      f"suppressed (rate limit {FSD_STEER_RATE_PER_S}/s)")
            if hold_frames or hold_offers:
                # safe_stop_count and path_hold_count stay SEPARATE (plan
                # phase B acceptance): a held tick is a degraded reuse of
                # a verified path, not a stop.
                print(f"[fsd-drive] path hold: {hold_frames}/{frames} "
                      f"frames reused a bounded verified path "
                      f"({hold_offers} verified offers) vs {stopps} "
                      f"safe-stop frames")
            _n_skip_fr = sum(1 for f in hist if f.get("budget_skips"))
            if _n_skip_fr:
                _n_skip_hd = sum(len(f.get("budget_skips") or [])
                                 for f in hist)
                _max_budget = max(float(f.get("budget_s") or 0.0)
                                  for f in hist)
                print(f"[fsd-drive] tick budget: {_n_skip_fr} frames "
                      f"deferred {_n_skip_hd} heavy head(s) "
                      f"(budget capped at {_max_budget:.2f}s)")
            if rec is not None:
                try:
                    _rec_out = rec.save()
                    if _rec_out:
                        print(f"[fsd-drive] shadow episode saved -> {_rec_out}")
                    else:
                        print("[fsd-drive] nothing recorded")
                except Exception as _rec_e:
                    print(f"[fsd-drive] shadow episode save failed: {_rec_e}")
        finally:
            # input watchdog must always be lifted, even on a crash, so a
            # later attach is not blocked by a stale Lua-frame timer
            try:
                wd_disarm(conn)
            except Exception:
                pass
            # ensure the car stops
            try:
                conn.control(throttle=0.0, brake=1.0, steering=0.0,
                             gear=locals().get("fwd_gear"))
                conn.step(3)
            except Exception:
                pass
            # Remove the Tech camera/LiDAR sensors so a next attach process
            # does not pile up sensors in the running game (leftover sensors
            # made later camera polls fail intermittently).
            try:
                stack.close()
            except Exception:
                pass
            conn.close()
            if hist and args.out:
                try:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.out).write_text(
                        json.dumps(hist, ensure_ascii=False), encoding="utf-8")
                    print(f"[fsd-drive] telemetry -> {args.out} ({len(hist)} frames)")
                except Exception as _e:
                    print(f"[fsd-drive] telemetry write failed: {_e}")
        return 2 if watchdog_lost else 0


def run(args) -> int:
    """Compatibility wrapper used by the thin script and benchmark."""
    return FSDriveSession(args).run()
