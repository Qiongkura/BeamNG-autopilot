"""Run the production drive loop with deterministic sensors and no game."""

from __future__ import annotations

import json
import math
import sys
import time as _time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from beamng_autopilot import fsd_drive as fd
from beamng_autopilot.fsd_stack import FSDTick
from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.perception_snapshot import PerceptionSnapshot
from beamng_autopilot.planning import Scene
from beamng_autopilot.safety_monitor import SafetyVerdict


class Clock:
    def __init__(self):
        self.value = 100.0

    def time(self):
        return self.value

    def monotonic(self):
        return self.value

    def perf_counter(self):
        # 预热计时的真实时钟（时长语义，与可控制的挂钟分离）
        return _time.perf_counter()

    def sleep(self, duration):
        self.value += max(0.0, duration)


class Connection:
    def __init__(self, clock, state):
        self.clock, self.state = clock, state
        self.commands = []
        self.vehicle = SimpleNamespace(vid="ego")
        self.startup_calls = []

    def open(self, **kwargs):
        self.startup_calls.append(("open", kwargs.get("launch")))

    def attach_vehicle(self, **kwargs):
        self.startup_calls.append(("attach",))

    def load_scenario(self):
        self.startup_calls.append(("load",))

    def get_state(self):
        return self.state

    def step(self, steps):
        pass

    def control(self, **controls):
        self.commands.append((self.clock.value, dict(controls)))

    def current_env(self):
        return {"map": "fixture", "vehicle": "etk800"}

    def read_damage_total(self):
        return 0.0

    def close(self):
        pass


class Stack:
    mode = "tech"
    grid_n = 60
    grid_res = 0.5

    def __init__(self, clock, state, delays, *, obstacle=None, painted=False,
                 ref_authority=None, provide_scene=True, meta_extra=None):
        self.clock, self.state = clock, state
        self.delays, self.obstacle = delays, obstacle
        self.painted = painted
        self.ref_authority = ref_authority
        self.provide_scene = provide_scene
        self.meta_extra = meta_extra
        self.created_scenes: list = []
        self.index = -1

    def tick(self, **kwargs):
        self.index += 1
        assert self.index < len(self.delays), "drive loop exceeded fixture window"
        source_t = self.clock.value
        self.clock.value += self.delays[self.index]
        pos = self.state.pos
        path = np.column_stack([np.arange(0.0, 15.0), np.zeros(15)])
        grid = OccupancyGrid(60, 60, 0.5, origin=tuple(pos[:2]),
                             heading=self.state.heading)
        grid.drivable[:] = 1
        grid.observed[:] = 1
        if self.obstacle is not None:
            grid.mark_obstacle_region(self.obstacle, 0.0, 0.25, 0.25)
        snapshot = PerceptionSnapshot(
            captured_at=source_t, tick_id=self.index, pos=pos,
            heading=self.state.heading, bev=grid.occupancy,
            drivable=grid.drivable, observed=grid.observed,
            head_age_s={"semantic": 0.0}, range_age_s=0.0, bev_age_s=0.0)
        if self.painted:
            snapshot.cam = object()
            snapshot.head_outputs = {
                "semantic": SimpleNamespace(masks={"line": np.ones((4, 4), bool)})}
        out = FSDTick()
        out.snapshot = snapshot
        out.scene = (Scene(pos=pos, heading=self.state.heading, grid=grid,
                           lane_ref=path, perception_snapshot=snapshot,
                           strict_perception=True)
                     if self.provide_scene else None)
        self.created_scenes.append(out.scene)
        out.lane_ref = path
        out.best_path = path
        out.best_speed = 6.0
        out.meta = {
            "lane_src_sel": "sensor", "lane_src": "sensor",
            "object_head": 1, "snapshot": snapshot.meta(),
            "head_sched": {"semantic": {
                "source_seq": self.index, "result_seq": self.index + 1,
                "source_t": source_t, "publish_t": self.clock.value,
                "state": "ran", "result_available": True}},
        }
        if self.meta_extra:
            out.meta.update(self.meta_extra)
        if self.ref_authority is not None:
            # Per-tick list is supported: the T04 counterexample needs the
            # authority to DROP between ticks (full -> limited) while the
            # wheel is already outside the new limit.
            ra = self.ref_authority
            if isinstance(ra, (list, tuple)):
                ra = ra[min(self.index, len(ra) - 1)]
            out.meta["ref_authority"] = ra
        return out

    def close(self):
        pass


def drive(monkeypatch, tmp_path, *, caps=(3.3,), delays=None,
          levels=None, speed=0.0, obstacle=None, dqn=False, painted=False,
          route_end=100.0, heading=0.0, ref_authority=None, lateral=0.0,
          max_wall_s=None, expected_rc=0, expected_frames=None, probe_geometry=False,
          attach=True, replay_capture=False, provide_scene=True,
          downstream_stall_s=None, e2e_rt=None, pw_out=None,
          meta_extra=None, reasons=None):
    clock = Clock()
    delays = list(delays or [0.5] * len(caps))
    state = SimpleNamespace(pos=np.array([0.0, lateral, 0.0]), heading=heading,
                            speed=speed, vel=np.array([speed, 0.0, 0.0]),
                            dir=np.array([1.0, 0.0, 0.0]))
    conn = Connection(clock, state)
    stack = Stack(clock, state, delays, obstacle=obstacle, painted=painted,
                  ref_authority=ref_authority, provide_scene=provide_scene,
                  meta_extra=meta_extra)
    disarms = []
    def disarm():
        disarms.append(clock.value)
        return {"status": "pending"}
    stack.range_prov = SimpleNamespace(disarm_replay_capture=disarm)

    class Monitor:
        def __init__(self, **kwargs):
            self.scenes: list = []
            created_monitors.append(self)

        def evaluate(self, scene, path, **kwargs):
            self.scenes.append(scene)
            if downstream_stall_s:
                # 下游停滞（规划/安全/神经头/仲裁耗时）：只推时钟，不改判定。
                clock.sleep(downstream_stall_s)
            i = stack.index
            cap = caps[i]
            level = levels[i] if levels else ("degraded" if cap < 6 else "safe")
            reason = (reasons[i] if reasons else "fixture monitor")
            v = SafetyVerdict(level=level, target_speed=cap,
                              reason=reason, closest_obs_m=999.0)
            v.corridor_open = True
            v.corridor_state = "feasible"
            v.corridor_reason = "fixture geometry"
            v.effective_rule = "fixture"
            v.rules_evaluated = ["fixture"]
            return v

        def offer_verified_path(self, *args, **kwargs):
            return False

    created_monitors: list = []
    policy = None
    if dqn:
        policy = SimpleNamespace(
            predict=lambda **kwargs: (4, 0.1), contract=None,
            meta_warning=None, meta=None)
    route = np.array([[0.0, 0.0], [route_end, 0.0]])
    monkeypatch.setattr(fd, "time", clock)
    monkeypatch.setattr(fd, "BeamNGConnector", lambda *args, **kwargs: conn)
    monkeypatch.setattr(fd, "SafetyMonitor", Monitor)
    monkeypatch.setattr(fd.gearbox, "forward_gear_input", lambda conn: 2)
    monkeypatch.setattr(fd, "wd_arm", lambda conn: True)
    monkeypatch.setattr(fd, "wd_heartbeat", lambda conn: True)
    monkeypatch.setattr(fd, "wd_disarm", lambda conn: True)
    monkeypatch.setattr(fd.FSDriveSession, "_build_route",
                        lambda self, conn: (route, route, None, None))
    monkeypatch.setattr(fd.FSDriveSession, "_setup_runtime",
                        lambda self, conn, args: (None, stack, e2e_rt, None,
                                                  policy))
    monkeypatch.setattr(fd.FSDriveSession, "_prewarm_and_place",
                        lambda *args, **kwargs: (pw_out, True, 0))
    if painted:
        marking = SimpleNamespace(world=np.column_stack([
            np.linspace(-5.0, 20.0, 26), np.full(26, 0.5)]))
        monkeypatch.setattr(fd, "painted_line_markings", lambda *a, **k: [marking])
    output = tmp_path / "drive.json"
    args = SimpleNamespace(runtime="tech", attach=attach, map=None,
                           teleport=None, speed=6.0, strict=True,
                           lane_mode="sensor", no_shadow=True, no_signal=True,
                           seconds=sum(max(0.5, x) for x in delays) - 0.01,
                           out=str(output), vis=0, max_wall_s=max_wall_s,
                           probe_geometry=probe_geometry, capture_range_replay=replay_capture)
    assert fd.FSDriveSession(args).run() == expected_rc
    frames = json.loads(output.read_text(encoding="utf-8"))
    audit = json.loads(Path(str(output) + ".session.json").read_text(encoding="utf-8"))
    if replay_capture:
        assert len(disarms) == 1
        assert audit["range_replay"]["disarmed_wall_t"] == disarms[0]
        assert disarms[0] <= audit["drive_started_wall_t"] <= frames[0]["cmd_t"]
    else:
        assert not disarms and "range_replay" not in audit
    assert len(frames) == (len(caps) if expected_frames is None else expected_frames)
    for frame in frames:
        sent = [c for t, c in conn.commands if t == frame["cmd_t"]]
        assert any(round(c["throttle"], 4) == frame["throttle"]
                   and round(c["brake"], 4) == frame["brake"]
                   and round(c["steering"], 4) == frame["steer"] for c in sent)
    conn.fixture_stack = stack
    conn.fixture_monitors = created_monitors
    return frames, conn


def test_default_loop_sends_lower_cap_and_serializes_evidence(monkeypatch, tmp_path):
    assert not fd.LONG_PLAN_ENABLED
    frames, _ = drive(monkeypatch, tmp_path, caps=(1.3,))
    frame = frames[0]
    assert frame["plan_speed"] == 6.0
    assert frame["hard_cap"] == 1.3
    assert frame["final_target_speed"] == 1.3
    assert frame["substeps_executed"] == 0
    assert frame["corridor_reason"] == "fixture geometry"
    assert frame["rules_evaluated"] == ["fixture"]
    assert frame["consumed"]["semantic"]["age_s"] == 0.5
    assert frame["consumed"]["semantic"]["publish_age_s"] == 0.0


def test_capture_disarms_before_drive_and_preserves_pending_diagnostic(monkeypatch, tmp_path):
    drive(monkeypatch, tmp_path, replay_capture=True)


def test_fresh_driver_loads_before_querying_current_vehicles(monkeypatch, tmp_path):
    frames, conn = drive(monkeypatch, tmp_path, attach=False)
    assert frames
    assert conn.startup_calls == [("open", True), ("load",)]


def test_attached_driver_reuses_existing_vehicle(monkeypatch, tmp_path):
    frames, conn = drive(monkeypatch, tmp_path, attach=True)
    assert frames
    assert conn.startup_calls == [("open", False), ("attach",)]


def test_real_loop_geometry_carries_the_final_command(monkeypatch, tmp_path):
    frames, _ = drive(monkeypatch, tmp_path, probe_geometry=True)
    frame = frames[0]
    probe = frame["geometry_probe"]
    assert "error" not in probe, probe
    assert probe["command"]["cmd_seq"] == frame["cmd_seq"]
    assert round(probe["command"]["throttle"], 4) == frame["throttle"]
    assert probe["command"]["execution_status"] == "unknown"
    assert probe["control_path"] is not None
    assert frame["geometry_probe_ms"] >= 0


def test_long_planner_cannot_raise_monitor_cap_in_real_loop(monkeypatch, tmp_path):
    monkeypatch.setattr(fd, "LONG_PLAN_ENABLED", True)
    frames, _ = drive(monkeypatch, tmp_path, caps=(1.0,), speed=5.0)
    assert frames[0]["final_target_speed"] <= 1.0


def test_real_loop_keeps_dqn_cap_after_plan_smoothing(monkeypatch, tmp_path):
    frames, _ = drive(monkeypatch, tmp_path, caps=(6.0,), dqn=True)
    frame = frames[0]
    assert frame["dqn_act"] == 4
    assert frame["dqn_cap"] == pytest.approx(0.6)
    assert frame["plan_speed"] == 6.0
    assert frame["final_target_speed"] == pytest.approx(frame["dqn_cap"])


def test_real_loop_minimal_risk_stops_after_previous_throttle(monkeypatch, tmp_path):
    frames, _ = drive(monkeypatch, tmp_path, caps=(6.0, 0.0),
                      levels=("safe", "minimal_risk"))
    assert frames[0]["throttle"] > 0.0
    assert frames[1]["final_stop"]
    assert frames[1]["throttle"] == 0.0
    assert frames[1]["brake"] == 1.0


def test_real_painted_stop_survives_clear_grid_check(monkeypatch, tmp_path):
    frames, _ = drive(monkeypatch, tmp_path, caps=(6.0,), painted=True)
    assert frames[0]["painted_body_cross"] == 1
    assert frames[0]["final_stop"]
    assert frames[0]["throttle"] == 0.0
    assert frames[0]["brake"] == 1.0


def test_real_loop_watchdog_brakes_without_substep_slack(monkeypatch, tmp_path):
    monkeypatch.setattr(fd, "CTRL_WATCHDOG_ENABLED", True)
    frames, _ = drive(monkeypatch, tmp_path, caps=(6.0,), delays=(2.0,))
    frame = frames[0]
    assert frame["substeps_executed"] == 0
    assert frame["watchdog_braked"]
    assert frame["cmd_gap_s"] == 2.0
    assert frame["throttle"] == 0.0 and frame["brake"] == 1.0


def test_drive_timeout_after_slow_sensor_never_sends_another_throttle(monkeypatch, tmp_path):
    frames, conn = drive(monkeypatch, tmp_path, caps=(6.0, 6.0),
                         delays=(0.5, 2.0), max_wall_s=1.5,
                         expected_rc=4, expected_frames=1)
    assert frames[0]["throttle"] > 0.0
    after_deadline = [cmd for t, cmd in conn.commands if t >= 101.5]
    assert after_deadline
    assert all(cmd["throttle"] == 0.0 and cmd["brake"] == 1.0
               for cmd in after_deadline)
    audit = json.loads((tmp_path / "drive.json.session.json").read_text(encoding="utf-8"))
    assert audit["exit_code"] == 4 and audit["termination"] == "drive_wall_timeout"
    assert audit["placed"] and audit["telemetry_frames"] == 1


def test_end_alignment_creep_cannot_override_monitor_stop(monkeypatch, tmp_path):
    seen = []
    apply_stop = fd._final_stop_controls

    def observe(*controls, **kwargs):
        seen.append((controls, kwargs))
        return apply_stop(*controls, **kwargs)

    monkeypatch.setattr(fd, "_final_stop_controls", observe)
    frames, _ = drive(monkeypatch, tmp_path, caps=(0.0,),
                      levels=("minimal_risk",), route_end=5.0,
                      heading=math.radians(12.0))
    assert seen[0][0][0] == fd.ALIGN_CREEP_THR
    assert seen[0][1]["stop"]
    assert frames[0]["throttle"] == 0.0
    assert frames[0]["brake"] == 1.0


def test_real_grid_clearance_stop_reaches_final_pedals(monkeypatch, tmp_path):
    frames, _ = drive(monkeypatch, tmp_path, caps=(6.0,),
                      obstacle=3.0, speed=6.0)
    frame = frames[0]
    assert frame["clear_src"] == "path_grid"
    assert frame["final_stop"]
    assert frame["throttle"] == 0.0 and frame["brake"] == 1.0


def test_protective_receipt_survives_the_next_main_send(monkeypatch, tmp_path):
    monkeypatch.setattr(fd, "CTRL_WATCHDOG_ENABLED", True)
    frames, _ = drive(monkeypatch, tmp_path, caps=(6.0, 6.0),
                      delays=(2.0, 0.5), speed=2.0)
    assert frames[1]["protective_commands"]
    receipt = frames[1]["protective_commands"][0]
    assert receipt["throttle"] == 0.0
    assert receipt["brake"] >= 0.5
    assert receipt["cmd_seq"] < frames[1]["cmd_seq"]


def test_active_climb_yields_to_next_monitor_stop(monkeypatch, tmp_path):
    frames, _ = drive(monkeypatch, tmp_path, caps=(6.0,) * 6 + (0.0,),
                      levels=("safe",) * 6 + ("minimal_risk",), obstacle=10.0)
    assert any(frame["climb"] for frame in frames[:-1])
    assert frames[-1]["final_stop"]
    assert frames[-1]["climb"] == 0
    assert frames[-1]["throttle"] == 0.0
    assert frames[-1]["brake"] == 1.0


def test_limited_ref_authority_clamps_steering_when_enabled(
        monkeypatch, tmp_path):
    """P1-2 arm C: an unpaired/unstable reference may nudge, not steer.

    The crash run held +0.55 for ~3 s while the reference flipped between
    own lane, oncoming lane and the whole-road centre; with the switch on,
    the wheel can only reach REF_STABILITY_LIMITED_STEER.
    """
    monkeypatch.setattr(fd, "REF_STABILITY_ENABLED", True)
    frames, _ = drive(monkeypatch, tmp_path, caps=(6.0,), lateral=1.2,
                      ref_authority="limited")
    frame = frames[0]
    assert frame["ref_authority"] == "limited"
    assert abs(frame["steer"]) <= fd.REF_STABILITY_LIMITED_STEER + 1e-9
    assert frame["steer_authority_clamped"] == 1


def test_full_authority_is_not_clamped(monkeypatch, tmp_path):
    monkeypatch.setattr(fd, "REF_STABILITY_ENABLED", True)
    frames, _ = drive(monkeypatch, tmp_path, caps=(6.0,), lateral=1.2,
                      ref_authority="full")
    frame = frames[0]
    assert frame["ref_authority"] == "full"
    assert frame["steer_authority_clamped"] == 0
    # The demand genuinely exceeds the limited-authority ceiling, so the
    # clamp test above cannot pass trivially.
    assert abs(frame["steer"]) > fd.REF_STABILITY_LIMITED_STEER


def test_without_the_switch_the_authority_is_only_recorded(
        monkeypatch, tmp_path):
    """Default OFF: the grading is published, the wheel is untouched."""
    assert fd.REF_STABILITY_ENABLED is False
    frames, _ = drive(monkeypatch, tmp_path, caps=(6.0,), lateral=1.2,
                      ref_authority="limited")
    frame = frames[0]
    assert frame["ref_authority"] == "limited"
    assert frame["steer_authority_clamped"] == 0


class TestFlagCapture:
    """Stage-C: hand-check frames must be capturable AND mappable."""

    def test_the_capture_decision_matrix(self):
        """Only flagged frames (plus controls) when --vis-on-flag is on."""
        v = fd.vis_should_write
        assert v(on_flag=True, every=0, control_every=0, frame=1,
                 flagged=True) == "flag"
        assert v(on_flag=True, every=0, control_every=0, frame=1,
                 flagged=False) is None
        assert v(on_flag=True, every=0, control_every=25, frame=25,
                 flagged=False) == "ctrl"
        assert v(on_flag=True, every=0, control_every=25, frame=26,
                 flagged=False) is None
        # a flagged frame is never demoted to a control
        assert v(on_flag=True, every=0, control_every=25, frame=25,
                 flagged=True) == "flag"

    def test_the_fixed_cadence_still_works(self):
        v = fd.vis_should_write
        assert v(on_flag=False, every=5, control_every=0, frame=10,
                 flagged=False) == "tick"
        assert v(on_flag=False, every=5, control_every=0, frame=11,
                 flagged=False) is None
        assert v(on_flag=False, every=0, control_every=0, frame=0,
                 flagged=False) is None

    def test_the_cli_exposes_the_flag_capture(self):
        src = (Path(__file__).resolve().parents[1] / "scripts"
               / "m5_fsd_drive.py").read_text("utf-8")
        assert "--vis-on-flag" in src and "--vis-control-every" in src

    def test_the_image_name_carries_the_join_key(self):
        """A filename index is NOT a telemetry row index (measured), so the
        timestamp is written into the name and the sidecar index."""
        src = (Path(__file__).resolve().parents[1] / "beamng_autopilot"
               / "fsd_drive.py").read_text("utf-8")
        assert "_t{" in src and "vis_index.json" in src


def test_diagnostic_success_cannot_replace_a_valid_planner_scene(
        monkeypatch, tmp_path):
    """复核 P0-A：诊断成功也不得重建安全场景。

    修复前的结构缺陷：诊断 try/else 劫持了 ``if scene is not None`` 的回退
    else——诊断成功反而用 out.bev 另建 Scene 并重新评估。回归用身份断言钉住：
    监控器收到的必须是 Stack 创建的那个 Scene 对象（grid/lane_ref 同一）。
    """
    frames, conn = drive(monkeypatch, tmp_path, caps=(6.0, 6.0),
                         delays=[0.5, 0.5])
    stack = conn.fixture_stack
    monitor = conn.fixture_monitors[0]
    assert len(monitor.scenes) == len(stack.created_scenes) == 2
    for i, created in enumerate(stack.created_scenes):
        assert created is not None
        assert monitor.scenes[i] is created
        assert monitor.scenes[i].grid is created.grid


def test_diagnostic_failure_keeps_the_planner_scene(monkeypatch, tmp_path):
    """诊断抛异常只影响诊断状态：场景不变、驾驶照常、诊断字段缺席。"""
    import beamng_autopilot.planning.geometry as geom

    def _boom(*a, **k):
        raise ValueError("fixture diagnostic failure")

    monkeypatch.setattr(geom, "polyline_point_distances", _boom)
    frames, conn = drive(monkeypatch, tmp_path, caps=(6.0, 6.0),
                         delays=[0.5, 0.5])
    stack = conn.fixture_stack
    monitor = conn.fixture_monitors[0]
    for i, created in enumerate(stack.created_scenes):
        assert monitor.scenes[i] is created
        assert monitor.scenes[i].grid is created.grid
    assert all(f.get("plan_dev_m") is None for f in frames)
    assert all(f.get("ref_gap_m") is None for f in frames)


def test_missing_scene_still_enters_the_explicit_fallback(
        monkeypatch, tmp_path):
    """缺 Scene 时才允许兼容构造：回退场景由 out.bev 重建并送同一评估。"""
    frames, conn = drive(monkeypatch, tmp_path, caps=(6.0,), delays=[0.5],
                         provide_scene=False)
    monitor = conn.fixture_monitors[0]
    assert conn.fixture_stack.created_scenes == [None]
    assert len(monitor.scenes) == 1
    scene = monitor.scenes[0]
    assert scene is not None and scene.grid is not None
    assert float(scene.pos[0]) == pytest.approx(0.0)


def test_real_monitor_verdict_is_a_pure_function_of_scene_content():
    """真实 SafetyMonitor：同一场景重复评估结果一致，内容变化才改变裁定。"""
    monitor = fd.SafetyMonitor(road_surface_gate=False)
    grid = OccupancyGrid(60, 60, 0.5, origin=(0.0, 0.0), heading=0.0)
    grid.drivable[:] = 1
    grid.observed[:] = 1
    path = np.column_stack([np.arange(0.0, 15.0), np.zeros(15)])
    snap = PerceptionSnapshot(captured_at=1000.0, tick_id=0,
                              pos=np.zeros(3), heading=0.0,
                              head_age_s={"semantic": 0.0}, bev_age_s=0.0)
    scene = Scene(pos=np.zeros(3), heading=0.0, grid=grid, lane_ref=path,
                  perception_snapshot=snap, strict_perception=True)
    v1 = monitor.evaluate(scene, path, planner_age_s=0.0, snapshot_age_s=0.0)
    v2 = monitor.evaluate(scene, path, planner_age_s=0.0, snapshot_age_s=0.0)
    assert v1.level == v2.level and v1.reason == v2.reason
    grid.mark_obstacle_region(2.0, 0.0, 0.25, 0.25)
    v3 = monitor.evaluate(scene, path, planner_age_s=0.0, snapshot_age_s=0.0)
    assert (v3.level, v3.target_speed) != (v1.level, v1.target_speed)


def test_stale_perception_at_send_cannot_send_throttle(monkeypatch, tmp_path):
    """复核 P0-B：下游停滞把年龄推进到发送边界后，过期控制不得发送。

    感知新鲜（发布年龄 0 s），但评估到发送之间停滞 2 s——发送年龄
    0+2 s 超过 STALE_PIPELINE_S。门按既有硬停契约收尾（0 油门/满刹），
    原因与下界年龄写入遥测；所有实际发出的命令都无正油门。
    """
    frames, conn = drive(monkeypatch, tmp_path, caps=(6.0,), delays=[0.5],
                         downstream_stall_s=2.0, expected_frames=1)
    frame = frames[0]
    assert frame["send_stale_stop_reason"] == "stale_snapshot"
    assert frame["stale_at_send_s"] >= 2.0
    assert frame["throttle"] == pytest.approx(0.0)
    assert frame["brake"] == pytest.approx(1.0)
    assert frame["send_chain_ms"] >= 2000.0
    assert conn.commands
    for _, c in conn.commands:
        assert c["throttle"] == pytest.approx(0.0)


def test_fresh_tick_records_no_send_hold(monkeypatch, tmp_path):
    """无下游停滞时发送门保持静默，不改变既有指令路径。"""
    frames, conn = drive(monkeypatch, tmp_path, caps=(6.0,), delays=[0.5])
    assert frames[0].get("send_stale_stop_reason") is None
    assert frames[0].get("stale_at_send_s") is None
    # 完整计时：感知返回→发送完成的后链时长随帧记录（无停滞时为时钟零点）
    assert frames[0]["send_chain_ms"] >= 0.0


def test_snapshot_age_at_send_is_a_monotonic_lower_bound(monkeypatch):
    """发布年龄 + 单调时长 = 发送边界年龄下界；不重复计、不伪装采集年龄。"""
    out = SimpleNamespace(
        snapshot=PerceptionSnapshot(captured_at=100.0, tick_id=0,
                                    pos=np.zeros(3), heading=0.0,
                                    bev=np.zeros((4, 4), dtype=np.float32),
                                    head_age_s={"semantic": 0.3},
                                    bev_age_s=0.0),
        meta={}, frame=object(), lane_envelope=None)
    # 无接收时刻（代理不可用）→ 原样返回发布年龄
    assert fd._snapshot_age_at_send(out, None) == pytest.approx(0.3)

    class _Clock:
        def __init__(self, value):
            self.value = value

        def monotonic(self):
            return self.value

    monkeypatch.setattr(fd, "time", _Clock(105.0))
    # 发布 0.3 s + 感知返回后 5 s = 5.3 s（下界，不重复加计算耗时）
    assert fd._snapshot_age_at_send(out, 100.0) == pytest.approx(5.3)

    monkeypatch.setattr(fd, "time", _Clock(99.0))
    # 时钟回拨：年龄增量按非负计，不产生负年龄
    assert fd._snapshot_age_at_send(out, 100.0) == pytest.approx(0.3)

    # 无效/缺失证据：原样返回 inf，裁定权留给 SafetyMonitor 的 stale 路径
    dead = SimpleNamespace(snapshot=None, meta={}, frame=None,
                           head_outputs=None, lane_envelope=None)
    assert fd._snapshot_age_at_send(dead, 100.0) == float("inf")


def test_warmup_runs_once_before_exposure_and_resets_state(
        monkeypatch, tmp_path):
    """复核方案阶段一第 3 项：曝光前用放置帧预热已启用运行时并清理状态。

    预热不发驾驶油门；e2e 历史在预热后 reset（reset 恰一次）；耗时/失败
    原样记入会话审计，不用预热结果伪装首帧性能。
    """
    resets: list = []
    e2e_spy = SimpleNamespace(
        weights="spy.pt",
        step=lambda *a, **k: (None, None, 0.0),
        reset=lambda: resets.append(1),
        last_reject="", last_validation=None)
    pw = FSDTick()
    pw.snapshot = PerceptionSnapshot(
        captured_at=1.0, tick_id=0, pos=np.zeros(3), heading=0.0,
        bev=np.zeros((4, 4), dtype=np.float32),
        head_age_s={"semantic": 0.0}, bev_age_s=0.0)
    pw.frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frames, conn = drive(monkeypatch, tmp_path, caps=(6.0,), delays=[0.5],
                         e2e_rt=e2e_spy, pw_out=pw)
    audit = json.loads(
        Path(str(tmp_path / "drive.json") + ".session.json")
        .read_text(encoding="utf-8"))
    assert resets == [1]
    assert audit["warmup"]["e2e"]["ms"] >= 0
    assert "error" not in audit["warmup"]["e2e"]
    # bc 运行时未启用 → 原样缺席，不伪装预热
    assert audit["warmup"].get("bc") is None


def test_warmup_skipped_is_recorded_not_faked(monkeypatch, tmp_path):
    """无放置帧输出（fixture 既有路径）时预热记为 skipped，不编造数字。"""
    frames, conn = drive(monkeypatch, tmp_path, caps=(6.0,), delays=[0.5])
    audit = json.loads(
        Path(str(tmp_path / "drive.json") + ".session.json")
        .read_text(encoding="utf-8"))
    assert audit["warmup"] == {"skipped": "no_prewarm_output"}


@pytest.mark.parametrize("meta_extra,want", [
    ({"plan_blocked": "no_perception_lane"}, "perception_lane_missing"),
    ({"plan_rejects": {"lane_cross": 11, "body_cross": 3}},
     "all_candidates_rejected"),
    (None, "no_candidates_or_unrecorded"),
])
def test_no_path_mechanism_splits_the_conflated_reason(
        monkeypatch, tmp_path, meta_extra, want):
    """停车归因（62e78bd）的实证：reason 字符串混合三种机制。

    reason 字段保持原样（冻结计数依赖），新增派生字段 no_path_mechanism
    按既有 plan_blocked / plan_rejects 拆分；其余 reason 不 annotate。
    """
    frames, conn = drive(monkeypatch, tmp_path, caps=(6.0,), delays=[0.5],
                         reasons=["no drivable path"], meta_extra=meta_extra)
    assert frames[0]["reason"] == "no drivable path"
    assert frames[0]["no_path_mechanism"] == want


def test_other_reasons_are_not_annotated(monkeypatch, tmp_path):
    frames, conn = drive(monkeypatch, tmp_path, caps=(6.0,), delays=[0.5],
                         reasons=["obstacle very close"],
                         meta_extra={"plan_blocked": "no_perception_lane"})
    assert frames[0]["reason"] == "obstacle very close"
    assert frames[0]["no_path_mechanism"] is None
