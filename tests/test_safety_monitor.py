"""Offline tests for the FSD-style safety monitor."""

from __future__ import annotations

import math
import numpy as np
import pytest

from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import Scene
from beamng_autopilot.obstacle_risk import (
    RISK_BRAKING, RISK_HARD_COLLISION)
from beamng_autopilot.safety_monitor import SafetyMonitor


def _scene(obs_at=None, n=60, obs_half=1.0):
    grid = OccupancyGrid(n, n, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    if obs_at is not None:
        x, y = obs_at
        grid.mark_obstacle_region(x, y, obs_half, obs_half)
    xs = np.linspace(0, 30, 31)
    route = np.column_stack([xs, np.zeros_like(xs)])
    return Scene(pos=np.array([0.0, 0.0]), heading=0.0, grid=grid,
                 route=route, lane_ref=route)


def _straight():
    return np.column_stack([np.linspace(0, 15, 20), np.zeros(20)])


def test_safe_open_road() -> None:
    mon = SafetyMonitor(max_speed=12.0)
    v = mon.evaluate(_scene(), _straight())
    assert v.safe
    assert v.target_speed == pytest.approx(12.0)


def test_minimal_risk_when_blocked() -> None:
    mon = SafetyMonitor()
    # a wall spanning the whole forward corridor locks every path
    scene = _scene(obs_at=(4.0, 0.0), obs_half=50.0)
    v = mon.evaluate(scene, _straight())
    assert v.level == "minimal_risk"
    assert v.target_speed == 0.0
    assert "blocked" in v.reason


def test_scattered_cluster_degrades_not_stops() -> None:
    """A cluster of roadside poles leaves the corridor open: the monitor
    must degrade (slow, find another path) instead of a full stop."""
    mon = SafetyMonitor(occ_fraction_degrade=0.05,
                        occ_fraction_stop=0.5)
    scene = _scene()
    for y in (-4.0, -2.0, 2.0, 4.0):
        scene.grid.mark_obstacle_region(6.0, y, 0.4, 0.4)
    v = mon.evaluate(scene, _straight())
    assert v.level != "minimal_risk"


def test_degraded_when_obstacle_grazed() -> None:
    mon = SafetyMonitor(occ_fraction_degrade=0.02,
                        occ_fraction_stop=0.4)
    scene = _scene(obs_at=(6.0, 0.0))
    path = np.column_stack([np.linspace(0, 15, 30),
                            np.zeros(30)])
    v = mon.evaluate(scene, path)
    # straight path runs through the box -> a high occupied fraction
    assert v.level in ("minimal_risk", "degraded")


def test_rotated_roadside_wall_does_not_false_graze_open_path() -> None:
    """A thin diagonal roadside wall must not become an AABB block."""
    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    grid.mark_obstacle_region(
        6.0, 1.0, 0.0, 0.0,
        axis=np.array([1.0, 1.0]), half_len=4.0, half_thick=0.25)
    path = _straight()
    scene = Scene(pos=np.array([0.0, 0.0]), heading=0.0,
                  grid=grid, route=path, lane_ref=path)
    verdict = SafetyMonitor().evaluate(scene, path)
    assert verdict.reason != "path grazes obstacle"
    assert verdict.level == "safe"


def test_open_corridor_treats_scattered_path_occupancy_as_soft():
    mon = SafetyMonitor(max_speed=15.0, occ_fraction_degrade=0.01,
                        occ_fraction_stop=0.9)
    scene = _scene()
    for x in (4.0, 6.0, 8.0):
        scene.grid.mark_obstacle_region(x, 0.0, 0.35, 0.35)
    verdict = mon.evaluate(scene, _straight())
    assert verdict.corridor_open is True
    assert verdict.reason == "scattered obstacle"
    assert verdict.level == "degraded"
    assert verdict.target_speed >= 15.0 * 0.55 - 1e-9


def test_stale_sensor_degrades() -> None:
    mon = SafetyMonitor(max_speed=12.0)
    v = mon.evaluate(_scene(), _straight(),
                     snapshot_age_s=2.0)
    assert v.degraded
    assert "stale" in v.reason


def test_no_path_minimal_risk() -> None:
    mon = SafetyMonitor()
    v = mon.evaluate(_scene(), None)
    assert v.level == "minimal_risk"
    assert v.target_speed == 0.0


def test_obstacle_approach_slows() -> None:
    mon = SafetyMonitor(max_speed=15.0)
    scene = _scene(obs_at=(5.0, 2.0))  # off to the side path still clears
    path = _straight()
    v = mon.evaluate(scene, path)
    # A roadside box with the forward corridor open is a lane bound, not
    # a blockage: the monitor stays safe (FSD keeps control) but eases
    # speed as the obstacle approaches (town 2026-08-21).
    assert v.safe
    assert 0.0 < v.target_speed < 15.0


def test_corridor_open_never_crawls_to_creep() -> None:
    """Dense intersection LiDAR leaves the forward corridor open: the
    target may ease, but must not drop to the 2 m/s minimal-risk creep
    (fsd opt21 t=54-60: junction clutter pulled target to ~2-3 m/s and
    the car brake->nearly-stalled).  With corridor_open_floor_frac the
    eased target stays at cruise * floor."""
    mon = SafetyMonitor(max_speed=15.0)
    scene = _scene(obs_at=(5.0, 2.0), obs_half=1.0)  # intrudes the 1.6 m ease band
    v = mon.evaluate(scene, _straight())
    assert v.safe
    assert v.target_speed >= 15.0 * 0.55 - 1e-9
    assert v.target_speed < 15.0


def test_closed_corridor_still_eases_to_creep() -> None:
    """A really blocked forward corridor (free band gone) still eases all
    the way down - the floor must not apply when the road is closed."""
    mon = SafetyMonitor(max_speed=15.0)
    scene = _scene()
    # close the corridor: occupied cells across the whole lateral width
    # at every ahead band
    for x in range(3, 28, 3):
        scene.grid.mark_obstacle_region(float(x), 0.0, 8.0, 0.5)
    v = mon.evaluate(scene, _straight())
    assert not v.safe or v.target_speed < 15.0 * 0.55


def test_roadside_clutter_beside_lane_does_not_creep() -> None:
    """Continuous roadside trees/curbs beside the lane are lane bounds:
    with the forward corridor open they must NOT pin the target to the
    2 m/s creep (run 2026-08-27: plan 6 m/s, monitor crept all run).
    Only a corridor-intruding obstacle ahead of the ego eases speed."""
    mon = SafetyMonitor(max_speed=15.0)
    scene = _scene()
    # a wall of clutter 2 m beside the straight path (outside the 1.6 m
    # ease corridor), present along the whole approach
    for x in range(3, 28, 3):
        scene.grid.mark_obstacle_region(float(x), 2.5, 0.4, 0.4)
    v = mon.evaluate(scene, _straight())
    assert v.safe
    assert v.target_speed == pytest.approx(15.0)


def test_long_route_past_grid_horizon_not_occupied() -> None:
    """A nav-route reference extends beyond the sensor grid horizon.

    The grid only holds evidence inside its extent (unknown beyond);
    the monitor must not read the far tail of a long route as a wall
    and degrade a perfectly straight path (town stall 2026-08-21).
    """
    mon = SafetyMonitor(max_speed=8.0, occ_fraction_degrade=0.05,
                        occ_fraction_stop=0.4)
    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    xs = np.linspace(0, 120, 100)
    route = np.column_stack([xs, np.zeros_like(xs)])
    scene = Scene(pos=np.array([0.0, 0.0]), heading=0.0, grid=grid,
                  route=route, lane_ref=route)
    v = mon.evaluate(scene, route)
    assert v.safe


def test_path_inside_grid_still_blocked() -> None:
    """Real in-grid walls must still trip the safety monitor."""
    mon = SafetyMonitor(occ_fraction_degrade=0.05,
                        occ_fraction_stop=0.4)
    scene = _scene(obs_at=(5.0, 0.0), obs_half=1.0)
    path = np.column_stack([np.linspace(0, 10, 30), np.zeros(30)])
    v = mon.evaluate(scene, path)
    assert v.level in ("minimal_risk", "degraded")


def test_safety_monitor_allows_a_converging_path_when_body_is_over() -> None:
    """A car already across the line may creep back along a converging path.

    The old gate refused EVERY path while a body corner was over the
    boundary - including the one that brings the car back - so the car
    froze across the line and the stuck detector armed a reverse escape
    (live east_coast 2026-09-18).  A converging path is the legal
    recovery, capped to a creep.
    """
    from beamng_autopilot.planning import Scene
    from beamng_autopilot.occupancy import OccupancyGrid
    from beamng_autopilot.safety_monitor import SafetyMonitor
    grid = OccupancyGrid(60, 60, 0.5)
    left = np.array([[0., 4.], [20., 4.]])
    right = np.array([[0., -4.], [20., -4.]])
    # the car is yawed with its front-left corner over lane_left, and the
    # path runs straight down the lane: driving it rotates the body back
    # inside, so the crossing strictly shrinks.
    scene = Scene(pos=np.array([10., 3.0]), heading=math.radians(18.),
                  grid=grid, route=np.array([[0., 0.], [20., 0.]]),
                  lane_ref=np.array([[0., 0.], [20., 0.]]),
                  lane_left=left, lane_right=right, lane_width=8.)
    scene.lane_ref_src = "sensor"
    verdict = SafetyMonitor(max_speed=6.).evaluate(
        scene, np.array([[10., 3.], [15., 3.]]))
    assert verdict.level == "degraded"
    assert verdict.reason == "lane boundary recovery"
    assert verdict.target_speed <= 1.0


def test_safety_monitor_still_stops_a_diverging_crossing() -> None:
    """A path that keeps (or deepens) the crossing must still hard-stop."""
    from beamng_autopilot.planning import Scene
    from beamng_autopilot.occupancy import OccupancyGrid
    from beamng_autopilot.safety_monitor import SafetyMonitor
    grid = OccupancyGrid(60, 60, 0.5)
    left = np.array([[0., 4.], [20., 4.]])
    right = np.array([[0., -4.], [20., -4.]])
    scene = Scene(pos=np.array([10., 3.0]), heading=math.radians(18.),
                  grid=grid, route=np.array([[0., 0.], [20., 0.]]),
                  lane_ref=np.array([[0., 0.], [20., 0.]]),
                  lane_left=left, lane_right=right, lane_width=8.)
    scene.lane_ref_src = "sensor"
    # the path keeps drifting further left, deeper across the boundary
    verdict = SafetyMonitor(max_speed=6.).evaluate(
        scene, np.array([[10., 3.], [16., 4.5]]))
    assert verdict.level == "minimal_risk"
    assert verdict.target_speed == 0.0
    assert "vehicle body" in verdict.reason



def test_strict_perception_stops_without_sensor_lane() -> None:
    """Strict FSD: a nav route is intent, never lateral geometry.

    With a perfectly good map route but no perception lane, the monitor
    must fail closed instead of measuring the path against the route
    (docs/fsd_realism.md §2 / §4).
    """
    xs = np.linspace(0, 30, 31)
    route = np.column_stack([xs, np.zeros_like(xs)])
    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    scene = Scene(pos=np.array([0.0, 0.0]), heading=0.0, grid=grid,
                  route=route, lane_ref=None, strict_perception=True)
    v = SafetyMonitor(max_speed=12.0).evaluate(scene, _straight())
    assert v.level == "minimal_risk"
    assert v.reason == "perception lane unavailable"
    assert v.target_speed == 0.0
    assert v.lane_ref_src == "none"


def test_strict_perception_measures_against_sensor_lane() -> None:
    """Strict mode aligns to the perception lane, not the road centre."""
    xs = np.linspace(0, 30, 31)
    route = np.column_stack([xs, np.zeros_like(xs)])       # road centre
    lane = np.column_stack([xs, np.full_like(xs, -1.8)])   # own lane
    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    scene = Scene(pos=np.array([0.0, 0.0]), heading=0.0, grid=grid,
                  route=route, lane_ref=lane, strict_perception=True)
    path = np.column_stack([np.linspace(0, 15, 20), np.full(20, -1.8)])
    v = SafetyMonitor(max_speed=12.0).evaluate(scene, path)
    assert v.safe
    assert v.lane_dev_m < 0.2
    assert v.lane_ref_src == "sensor"


def test_legacy_mode_keeps_route_fallback() -> None:
    """Non-strict scenes keep the old map-route fallback (M5 compat)."""
    xs = np.linspace(0, 30, 31)
    route = np.column_stack([xs, np.zeros_like(xs)])
    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    scene = Scene(pos=np.array([0.0, 0.0]), heading=0.0, grid=grid,
                  route=route, lane_ref=None)
    v = SafetyMonitor(max_speed=12.0).evaluate(scene, _straight())
    assert v.safe
    assert v.lane_ref_src == "route"


def test_degraded_verdict_is_drivable_but_minimal_risk_is_not() -> None:
    """A degraded verdict must still be driven at its reduced cap.

    Gating the arbiter on ``safe`` instead threw the degraded speed cap
    away and force-stopped the car: the 2026-09-18 live east_coast demo
    stopped on 122 of 222 ticks, 83 of them ``level=degraded`` with
    ``mon_target`` 3.30 m/s and an open corridor (strict mode has no rule
    backup to fall through to).
    """
    from beamng_autopilot.safety_monitor import SafetyVerdict

    assert SafetyVerdict(level="safe").drivable
    assert SafetyVerdict(level="degraded").drivable
    assert not SafetyVerdict(level="minimal_risk").drivable


def test_monitor_keeps_a_degraded_path_drivable_with_its_speed_cap():
    """End to end: the scattered-obstacle degrade keeps a non-zero cap."""
    from beamng_autopilot.planning import Scene
    from beamng_autopilot.occupancy import OccupancyGrid
    from beamng_autopilot.safety_monitor import SafetyMonitor
    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    for x, y in ((6.0, -0.8), (7.0, 0.8), (8.0, -0.6)):
        grid.mark_obstacle_region(x, y, 0.4, 0.4)
    route = np.column_stack([np.linspace(0, 30, 31), np.zeros(31)])
    scene = Scene(pos=np.array([0.0, 0.0]), heading=0.0, grid=grid,
                  route=route, lane_ref=route)
    verdict = SafetyMonitor(
        max_speed=6.0, occ_fraction_degrade=0.05,
        occ_fraction_stop=0.5).evaluate(scene, _straight())
    assert verdict.level == "degraded"
    assert verdict.drivable
    assert verdict.target_speed > 0.0


def _track(x, y, vx=0.0, vy=0.0, matches: int = 4, lost: int = 0,
           category: str = "object", track_id: int = 1):
    """A ``temporal.TrackedObject``-shaped detection for risk tests."""
    from types import SimpleNamespace
    return SimpleNamespace(x=float(x), y=float(y), vx=float(vx),
                           vy=float(vy), matches=int(matches),
                           lost=int(lost), category=category,
                           track_id=int(track_id))


def _monitor_scene(tracks=None):
    """Scene with an optional tracked-object snapshot (risk layer)."""
    from beamng_autopilot.occupancy import OccupancyGrid
    from beamng_autopilot.perception_snapshot import PerceptionSnapshot
    from beamng_autopilot.planning import Scene
    path = _straight()
    scene = Scene(pos=np.array([0.0, 0.0]), heading=0.0,
                  grid=OccupancyGrid(60, 60, 0.5), route=path,
                  lane_ref=path)
    if tracks is not None:
        scene.perception_snapshot = PerceptionSnapshot(
            captured_at=0.0, tick_id=1, pos=np.array([0.0, 0.0]),
            heading=0.0, tracks=list(tracks))
    return scene


def test_monitor_caps_target_for_a_closing_track() -> None:
    """The risk model must reach the verdict, not just the module."""
    from beamng_autopilot.safety_monitor import SafetyMonitor

    # no dynamic perception -> no risk fields, no risk cap
    v0 = SafetyMonitor(max_speed=8.0).evaluate(
        _monitor_scene(), _straight(), ego_speed_mps=8.0)
    assert v0.risk_kind == ""
    assert v0.min_ttc_s is None
    assert v0.target_speed == pytest.approx(8.0)

    # a vehicle closing head-on caps the target below cruise
    v1 = SafetyMonitor(max_speed=8.0).evaluate(
        _monitor_scene([_track(12.0, 0.0, vx=-6.0)]), _straight(),
        ego_speed_mps=8.0)
    assert v1.risk_kind == RISK_BRAKING
    assert v1.min_ttc_s is not None and v1.min_ttc_s < 3.0
    assert v1.risk_closest_m == pytest.approx(12.0, abs=0.5)
    assert v1.target_speed < 8.0


def test_monitor_stops_on_a_contact_band_obstacle() -> None:
    from beamng_autopilot.safety_monitor import SafetyMonitor
    v = SafetyMonitor(max_speed=8.0).evaluate(
        _monitor_scene([_track(2.0, 0.0, matches=1)]), _straight(),
        ego_speed_mps=8.0)
    assert v.level == "minimal_risk"
    assert v.reason == "obstacle contact risk"
    assert v.target_speed == 0.0
    assert v.risk_kind == RISK_HARD_COLLISION


def test_monitor_ignores_roadside_and_unconfirmed_tracks() -> None:
    from beamng_autopilot.safety_monitor import SafetyMonitor
    scene = _monitor_scene([_track(6.0, 4.0),
                            _track(20.0, 0.0, matches=1)])
    v = SafetyMonitor(max_speed=8.0).evaluate(
        scene, _straight(), ego_speed_mps=8.0)
    assert v.level != "minimal_risk"
    assert v.target_speed == pytest.approx(8.0)


def test_risk_layer_constrains_a_served_path_hold() -> None:
    """A hold may not drive toward a contact-band obstacle.

    The risk layer runs AFTER the core arbitration, so a degraded verdict
    - including a served PATH_HOLD - still gets the obstacle check.
    """
    from beamng_autopilot.safety_monitor import SafetyMonitor
    path = _monitor_scene()
    mon = SafetyMonitor(max_speed=8.0)
    assert mon.offer_verified_path(_straight(), 0.0, 5.0,
                                   now_s=100.0, strict=True)
    scene = _monitor_scene([_track(2.0, 0.0, matches=1)])
    v = mon.evaluate(scene, None, now_s=100.1, ego_speed_mps=5.0)
    assert v.path_hold_active is True          # the hold WAS served...
    assert v.level == "minimal_risk"           # ...and then stopped
    assert v.reason == "obstacle contact risk"
    assert v.target_speed == 0.0


# ---------------------------------------------------------------------
# Perceived road-surface gate (2026-09-20)
#
# ``lat_left`` / ``lat_right`` were BOTH None on 82-99% of frames in every
# 2026-09-20 town run, so an off-road metric built on detected boundaries
# read 0.0 m ("perfectly on the road") while the car finished 6.96 m past
# the pavement edge.  The road-surface gate reads the SAME 2-12 m drivable
# band the lateral guard and the corner governor use, and separates cleanly
# on the recorded runs: the 8 runs that stayed on the pavement never lost
# the band for more than 5 consecutive frames, the one that left it lost it
# for 91.
# ---------------------------------------------------------------------

def _road_scene(y_lo: float = -3.0, y_hi: float = 3.0,
                x_lo: float = 2.0, x_hi: float = 9.0):
    """Monitor scene whose BEV carries a drivable road slab."""
    scene = _scene()
    ext = scene.grid.extent
    res = scene.grid.res
    r0 = int((ext - x_hi) / res)
    r1 = int((ext - x_lo) / res)
    c0 = int((ext - y_hi) / res)
    c1 = int((ext - y_lo) / res)
    scene.grid.drivable[r0:r1 + 1, c0:c1 + 1] = 1.0
    return scene


def test_road_surface_present_keeps_the_verdict_safe() -> None:
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    for t in (0.0, 3.0, 6.0, 12.0):
        v = mon.evaluate(_road_scene(), _straight(), now_s=t)
        assert v.road_surface == "on_road"
        assert v.road_checked is True
        assert v.safe
        assert v.target_speed == pytest.approx(12.0)


def test_road_observation_is_not_skipped_when_no_path_exists() -> None:
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    v = mon.evaluate(_road_scene(), None, now_s=0.0)
    assert v.reason == "no drivable path"
    assert v.road_surface == "on_road"
    assert v.road_lost_s == 0.0
    assert v.road_checked is True
    assert v.target_speed == 0.0

    mon.evaluate(_scene(), None, now_s=1.0)
    v = mon.evaluate(_scene(), _straight(), now_s=10.0)
    assert v.reason == "perceived road surface lost"
    assert v.road_lost_s == 9.0
    assert v.target_speed == 0.0


def test_the_road_check_is_flagged_on_a_grid_with_no_evidence() -> None:
    """Consulted + no evidence and not-consulted are different verdicts."""
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    v = mon.evaluate(_scene(), _straight(), now_s=0.0)
    assert v.road_surface == "unknown"
    assert v.road_checked is True


def test_road_surface_unknown_is_not_reported_as_on_road() -> None:
    """A grid with no road evidence must read UNKNOWN, not "on the road".

    This is the contract the old off-road metric broke: ``_scene()`` has
    an empty drivable layer, so the band reader returns None - and that
    silence used to surface as 0.0 m, i.e. "perfectly on the road".
    """
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    v = mon.evaluate(_scene(), _straight(), now_s=0.0)
    assert v.road_surface == "unknown"
    assert v.road_surface != "on_road"


def test_one_blind_tick_is_not_a_fail_closed_event() -> None:
    """A single lost band must not stop the car (that is the stall trap)."""
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    v = mon.evaluate(_scene(), _straight(), now_s=0.0)
    assert v.road_surface == "unknown"
    assert v.safe
    assert v.target_speed == pytest.approx(12.0)


def test_sustained_road_loss_degrades_then_stops() -> None:
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    blind = _scene()
    mon.evaluate(blind, _straight(), now_s=0.0)          # clock starts here
    v = mon.evaluate(blind, _straight(), now_s=5.0)      # 5 s blind
    assert v.level == "degraded"
    assert v.reason == "perceived road surface lost"
    assert v.target_speed == pytest.approx(mon.min_risk_speed)
    v = mon.evaluate(blind, _straight(), now_s=9.0)      # 9 s blind
    assert v.level == "minimal_risk"
    assert v.reason == "perceived road surface lost"
    assert v.target_speed == 0.0


def test_brief_road_loss_does_not_stop() -> None:
    """~3 s of blindness is inside the recorded benign worst case."""
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    blind = _scene()
    mon.evaluate(blind, _straight(), now_s=0.0)
    v = mon.evaluate(blind, _straight(), now_s=3.0)
    assert v.safe
    v = mon.evaluate(_road_scene(), _straight(), now_s=3.1)
    assert v.road_surface == "on_road"
    assert v.safe


def test_road_loss_timer_resets_when_the_band_returns() -> None:
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    blind = _scene()
    mon.evaluate(blind, _straight(), now_s=0.0)
    mon.evaluate(_road_scene(), _straight(), now_s=3.0)   # recovered
    v = mon.evaluate(blind, _straight(), now_s=4.0)       # timer restarts
    assert v.safe
    assert v.road_surface == "unknown"


def test_band_beside_the_ego_degrades_at_once() -> None:
    """A perceived road entirely beside the car is OFF, not UNKNOWN.

    OFF degrades on the FIRST tick - positive evidence that the car is
    off the road must not read as a normal safe state - but it does not
    hard-stop immediately, because the band is read 2-12 m AHEAD and a
    bend can slide it sideways.  Sustained OFF still fails closed.
    """
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    v = mon.evaluate(_road_scene(y_lo=2.5, y_hi=8.0), _straight(),
                     now_s=0.0)
    assert v.road_surface == "off_road"
    assert v.level == "degraded"
    assert v.reason == "off perceived road surface"
    assert v.target_speed == pytest.approx(mon.min_risk_speed)


def test_sustained_off_road_stops() -> None:
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    off = _road_scene(y_lo=2.5, y_hi=8.0)
    mon.evaluate(off, _straight(), now_s=0.0)
    v = mon.evaluate(off, _straight(), now_s=9.0)
    assert v.road_surface == "off_road"
    assert v.level == "minimal_risk"
    assert v.target_speed == 0.0


def test_brief_off_road_recovers_without_a_stop() -> None:
    """A bend that slides the band off the car must not park the car."""
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    mon.evaluate(_road_scene(y_lo=2.5, y_hi=8.0), _straight(), now_s=0.0)
    v = mon.evaluate(_road_scene(), _straight(), now_s=1.0)
    assert v.road_surface == "on_road"
    assert v.safe
    assert v.target_speed == pytest.approx(12.0)


def test_gate_off_reports_the_state_but_never_acts() -> None:
    """With the switch off the state is still measured (A/B evidence)."""
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=False)
    blind = _scene()
    for t in (0.0, 5.0, 20.0):
        v = mon.evaluate(blind, _straight(), now_s=t)
        assert v.road_surface == "unknown"
        assert v.safe
        assert v.target_speed == pytest.approx(12.0)


def test_gate_off_still_reports_off_road_without_acting() -> None:
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=False)
    v = mon.evaluate(_road_scene(y_lo=2.5, y_hi=8.0), _straight(),
                     now_s=0.0)
    assert v.road_surface == "off_road"
    assert v.safe


def test_gate_defaults_to_the_module_switch() -> None:
    import beamng_autopilot.safety_monitor as sm
    mon = SafetyMonitor(max_speed=12.0)
    assert mon.road_surface_gate == bool(sm.ROAD_SURFACE_GATE_ENABLED)


def test_grid_less_scene_reports_unknown_without_starting_the_clock() -> None:
    """No BEV at all is a different failure from a silent BEV.

    A scene that never builds a grid must not accrue "road lost" time,
    or a BEV-less configuration would stop the car after the threshold
    with road evidence never having existed.
    """
    mon = SafetyMonitor(max_speed=12.0, road_surface_gate=True)
    scene = _scene()
    scene.grid = None
    for t in (0.0, 5.0, 20.0):
        v = mon.evaluate(scene, _straight(), now_s=t)
        assert v.road_surface == "unknown"
        assert v.road_lost_s == 0.0
        assert v.safe


@pytest.mark.parametrize("crossing", ["current", "planned", "off_lane"])
@pytest.mark.parametrize("soft", ["scattered", "graze", "stale", "road"])
def test_soft_verdict_cannot_mask_a_hard_lane_rule(crossing, soft):
    scene = _scene()
    path = _straight()
    if crossing == "current":
        scene.lane_right = np.array([[-10., -0.5], [30., -0.5]])
    elif crossing == "planned":
        scene.lane_left = np.array([[-10., 1.5], [30., 1.5]])
        path[:, 1] = 0.55
    else:
        scene.lane_ref[:, 1] = 7.0
    if soft == "scattered":
        scene.grid.mark_obstacle_region(12., float(path[-1, 1]), 3., 0.2)
    elif soft == "graze":
        for x in (7.75, 11.25, 14.75):
            scene.grid.mark_obstacle_region(x, 0., 0.05, 20.)
    mon = SafetyMonitor(max_speed=6., road_surface_gate=(soft == "road"),
                        occ_fraction_degrade=0.05, occ_fraction_stop=0.9)
    if soft == "road":
        mon.evaluate(scene, path, now_s=0.)
    v = mon.evaluate(scene, path, now_s=5.,
                     snapshot_age_s=2. if soft == "stale" else 0.)
    assert v.level == "minimal_risk"
    assert v.target_speed == 0.0
    rule = "path_off_lane" if crossing == "off_lane" else "body_crosses_boundary"
    assert v.effective_rule == rule
    assert rule in v.rules_evaluated
    if soft in ("scattered", "graze"):
        assert v.path_occupied_frac >= mon.occ_degrade
        assert v.corridor_open is (soft == "scattered")


@pytest.mark.parametrize("stale", [False, True])
def test_road_slowdown_cannot_mask_a_blocked_path(stale):
    scene = _scene(obs_at=(4., 0.), obs_half=50.)
    mon = SafetyMonitor(max_speed=6., road_surface_gate=True)
    mon.evaluate(scene, _straight(), now_s=0.)
    v = mon.evaluate(scene, _straight(), now_s=5.,
                     snapshot_age_s=2. if stale else 0.)
    assert v.level == "minimal_risk"
    assert v.target_speed == 0.
    assert v.reason == "path blocked by obstacle"


def test_stale_verdict_does_not_authorize_a_missing_path():
    v = SafetyMonitor(max_speed=6.).evaluate(
        _scene(), None, snapshot_age_s=2.)
    assert v.level == "minimal_risk"
    assert v.reason == "no drivable path"
    assert v.target_speed == 0.


def test_scattered_obstacles_preserve_the_convergence_recovery_cap():
    scene = _scene()
    scene.pos = np.array([10., 3.])
    scene.heading = math.radians(18.)
    scene.lane_left = np.array([[0., 4.], [30., 4.]])
    scene.lane_right = np.array([[0., -4.], [30., -4.]])
    scene.grid.mark_obstacle_region(14., 3., 0.4, 0.2)
    path = np.column_stack([np.linspace(10., 15., 21), np.full(21, 3.)])
    mon = SafetyMonitor(max_speed=6., occ_fraction_degrade=0.01,
                        occ_fraction_stop=0.9)
    v = mon.evaluate(scene, path)
    assert v.path_occupied_frac >= mon.occ_degrade
    assert v.corridor_open
    assert v.level == "degraded"
    assert v.reason == "lane boundary recovery"
    assert v.target_speed == 1.
    assert v.masked_hard_rules == []


def test_road_stop_remains_until_on_confirmation_completes():
    mon = SafetyMonitor(max_speed=6., road_surface_gate=True)
    mon.road_recover_confirm_s = 2.
    mon.evaluate(_scene(), _straight(), now_s=0.)
    assert mon.evaluate(_scene(), _straight(), now_s=9.).target_speed == 0.
    v = mon.evaluate(_road_scene(), _straight(), now_s=9.1)
    assert v.road_surface == "on_road"
    assert v.road_lost_s == pytest.approx(9.1)
    assert v.target_speed == 0.
    assert v.level == "minimal_risk"
    v = mon.evaluate(_road_scene(), _straight(), now_s=11.2)
    assert v.road_lost_s == 0.
    assert v.target_speed == 6.


def test_off_road_crawl_remains_during_unconfirmed_recovery():
    mon = SafetyMonitor(max_speed=6., road_surface_gate=True)
    mon.road_recover_confirm_s = 2.
    mon.evaluate(_road_scene(y_lo=2.5, y_hi=8.), _straight(), now_s=0.)
    v = mon.evaluate(_road_scene(), _straight(), now_s=0.1)
    assert v.road_surface == "on_road"
    assert v.level == "degraded"
    assert v.target_speed == mon.min_risk_speed


def test_stale_on_read_cannot_clear_an_existing_road_stop():
    mon = SafetyMonitor(max_speed=6., road_surface_gate=True)
    mon.evaluate(_scene(), _straight(), now_s=0.)
    mon.evaluate(_scene(), _straight(), now_s=9.)
    v = mon.evaluate(_road_scene(), _straight(), now_s=10., snapshot_age_s=2.)
    assert v.road_checked is False
    assert v.road_lost_s == 9.
    assert v.level == "minimal_risk"
    assert v.target_speed == 0.


def test_contact_risk_updates_the_final_trace_winner():
    v = SafetyMonitor(max_speed=6.).evaluate(
        _monitor_scene([_track(2., 0., matches=1)]), _straight())
    assert v.effective_rule == "obstacle_risk"
    assert "obstacle_risk" in v.rules_evaluated
    assert v.level == "minimal_risk"
    assert v.masked_hard_rules == []
