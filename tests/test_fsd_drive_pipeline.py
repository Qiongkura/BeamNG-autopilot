"""Run the production drive loop with deterministic sensors and no game."""

from __future__ import annotations

import json
import math
import sys
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

    def sleep(self, duration):
        self.value += max(0.0, duration)


class Connection:
    def __init__(self, clock, state):
        self.clock, self.state = clock, state
        self.commands = []
        self.vehicle = SimpleNamespace(vid="ego")

    def open(self, **kwargs):
        pass

    def attach_vehicle(self, **kwargs):
        pass

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
                 ref_authority=None):
        self.clock, self.state = clock, state
        self.delays, self.obstacle = delays, obstacle
        self.painted = painted
        self.ref_authority = ref_authority
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
        out.scene = Scene(pos=pos, heading=self.state.heading, grid=grid,
                          lane_ref=path, perception_snapshot=snapshot,
                          strict_perception=True)
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
          route_end=100.0, heading=0.0, ref_authority=None, lateral=0.0):
    clock = Clock()
    delays = list(delays or [0.5] * len(caps))
    state = SimpleNamespace(pos=np.array([0.0, lateral, 0.0]), heading=heading,
                            speed=speed, vel=np.array([speed, 0.0, 0.0]),
                            dir=np.array([1.0, 0.0, 0.0]))
    conn = Connection(clock, state)
    stack = Stack(clock, state, delays, obstacle=obstacle, painted=painted,
                  ref_authority=ref_authority)

    class Monitor:
        def __init__(self, **kwargs):
            pass

        def evaluate(self, scene, path, **kwargs):
            i = stack.index
            cap = caps[i]
            level = levels[i] if levels else ("degraded" if cap < 6 else "safe")
            v = SafetyVerdict(level=level, target_speed=cap,
                              reason="fixture monitor", closest_obs_m=999.0)
            v.corridor_open = True
            v.corridor_state = "feasible"
            v.corridor_reason = "fixture geometry"
            v.effective_rule = "fixture"
            v.rules_evaluated = ["fixture"]
            return v

        def offer_verified_path(self, *args, **kwargs):
            return False

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
                        lambda self, conn, args: (None, stack, None, None, policy))
    monkeypatch.setattr(fd.FSDriveSession, "_prewarm_and_place",
                        lambda *args: (None, True, 0))
    if painted:
        marking = SimpleNamespace(world=np.column_stack([
            np.linspace(-5.0, 20.0, 26), np.full(26, 0.5)]))
        monkeypatch.setattr(fd, "painted_line_markings", lambda *a, **k: [marking])
    output = tmp_path / "drive.json"
    args = SimpleNamespace(runtime="tech", attach=True, map=None,
                           teleport=None, speed=6.0, strict=True,
                           lane_mode="sensor", no_shadow=True, no_signal=True,
                           seconds=sum(max(0.5, x) for x in delays) - 0.01,
                           out=str(output), vis=0)
    assert fd.FSDriveSession(args).run() == 0
    frames = json.loads(output.read_text(encoding="utf-8"))
    assert len(frames) == len(caps)
    for frame in frames:
        sent = [c for t, c in conn.commands if t == frame["cmd_t"]]
        assert any(round(c["throttle"], 4) == frame["throttle"]
                   and round(c["brake"], 4) == frame["brake"]
                   and round(c["steering"], 4) == frame["steer"] for c in sent)
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
