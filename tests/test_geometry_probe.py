"""Exact sensor geometry can be replayed without claiming actuator execution."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from beamng_autopilot.geometry_probe import analyze_frames, capture_tick, replay_body, replay_safety
from beamng_autopilot.occupancy import OccupancyGrid


def capture(*, lateral=0.0, missing=False, path_y=0.0):
    x = np.linspace(-5, 20, 26)
    left = np.column_stack((x, np.ones_like(x) * 1.8))
    right = np.column_stack((x, -np.ones_like(x) * 1.8))
    path = np.column_stack((np.arange(15), np.full(15, path_y)))
    grid = OccupancyGrid(60, 60, 0.5)
    grid.drivable[20, 20] = 0.75
    out = SimpleNamespace(lane_ref=path, lane_left=None if missing else left,
                          lane_right=None if missing else right, best_path=path,
                          snapshot=None, meta={},
                          scene=SimpleNamespace(grid=grid, lane_ref=path))
    result = capture_tick(out, pos=[0, lateral, 0], heading=0,
                          selected_path=path, control_path=path,
                          command={"cmd_seq": 1, "throttle": 0},
                          verdict=SimpleNamespace(level="safe"))
    return result, out


def test_inside_body_and_future_crossing_are_distinct():
    inside, _ = capture()
    assert replay_body(inside)["current_crossing"] is False
    planned, _ = capture(path_y=2.0)
    result = replay_body(planned)
    assert result["current_crossing"] is False
    assert result["planned_crossing_witness_m"] > 0
    assert result["safe_to_execute"] is None


def test_body_crossing_and_missing_boundaries_never_become_safe_zero():
    crossed, _ = capture(lateral=1.0)
    assert replay_body(crossed)["current_crossing"] is True
    missing, _ = capture(missing=True)
    result = replay_body(missing)
    assert result["current_crossing"] is None
    assert result["corner_coverage"] == {"left": 0, "right": 0}


def test_capture_is_independent_copy_and_preserves_grid_values():
    probe, out = capture()
    assert probe["command"]["execution_status"] == "unknown"
    assert probe["grid"]["drivable"]["values"] == [0.75]
    out.lane_ref[:] = 99
    out.scene.grid.drivable[:] = 0
    assert probe["lane_ref"][0] == [0.0, 0.0]
    assert probe["grid"]["drivable"]["values"] == [0.75]


def test_refreshed_old_source_and_new_consumed_source_are_distinct():
    frames = [{"cmd_seq": i + 1,
               "head_sched": {"semantic": {"state": "ran", "source_seq": source,
                                           "result_seq": source}},
               "consumed": {"semantic": {"source_seq": source, "result_seq": source,
                                          "publish_age_s": 0.1}}}
              for i, source in enumerate((1, 1, 3))]
    frames[1]["head_sched"]["semantic"] = {
        "state": "keepalive_forced", "source_seq": 2, "result_seq": 1, "eligible_t": 100}
    result = analyze_frames(frames)
    assert result["geometry_unknown"] == 3
    assert len(result["forced_refresh_requests"]) == 1
    assert len(result["forced_new_source_command_joins"]) == 1
    assert result["forced_new_source_command_joins"][0]["cmd_seq"] == 3
    assert not result["forced_unconfirmed"]
    assert result["actuator_execution"] == "unknown"


def safety_capture(*, lateral=0.0, path_y=0.0, envelope=False, missing=False, held=False):
    from beamng_autopilot.planning import Scene
    from beamng_autopilot.safety_monitor import SafetyMonitor

    physical, out = capture(lateral=lateral, path_y=0)
    path = np.column_stack((np.arange(15), np.full(15, path_y)))
    left, right = out.lane_left, out.lane_right
    scene = Scene(pos=np.array([0, lateral]), heading=0, lane_ref=out.lane_ref,
                  lane_left=None if missing or envelope else left,
                  lane_right=None if missing or envelope else right,
                  lane_envelope=SimpleNamespace(left=left, right=right, age_s=0.0)
                  if envelope else None)
    out.scene = scene
    monitor = SafetyMonitor(capture_geometry=True)
    if held:
        # Isolate the recorded path selection, rather than guessing which
        # input a hold served. Hold policy itself has separate regression.
        monitor._serve_hold = lambda *a, **kw: (path, 0.1, "fresh", 1.0)
    verdict = monitor.evaluate(scene, None if held else path, now_s=100)
    probe = capture_tick(out, pos=[0, lateral, 0], heading=0,
                         selected_path=out.best_path, control_path=out.best_path,
                         command=physical["command"], verdict=verdict)
    return probe, verdict, scene, path


def test_safety_margin_and_control_path_are_replayed_with_distinct_inputs():
    from beamng_autopilot.vehicle_body import HALF_WIDTH_M

    margin, _, _, _ = safety_capture(lateral=1.8 - HALF_WIDTH_M - 0.1)
    assert replay_body(margin)["current_crossing"] is False
    safety = replay_safety(margin)
    assert safety["predicate_matches"] and safety["body_cross_current"]
    future, _, _, _ = safety_capture(path_y=2.0)
    assert replay_body(future)["planned_crossing_witness_m"] == 0
    assessed = replay_safety(future)
    assert assessed["body_cross_planned"] and assessed["predicate_matches"]
    assert assessed["safe_to_execute"] is None


def test_safety_envelope_hold_and_copy_use_actual_evaluated_inputs():
    probe, verdict, scene, path = safety_capture(envelope=True, held=True)
    result = replay_safety(probe)
    assert result["path_source"] == "held"
    assert verdict.path_hold_active
    assert result["predicate_matches"] and result["current_evidence"] is False
    path[:] = 99
    scene.lane_envelope.left[:] = 99
    verdict.assessment_geometry["pos"][0] = 99
    assert replay_safety(probe)["predicate_matches"]  # Independent post-send snapshot.


def test_missing_boundary_matches_predicate_but_evidence_stays_unknown():
    probe, _, _, _ = safety_capture(missing=True)
    result = replay_safety(probe)
    assert result["predicate_matches"]
    assert result["body_cross_current"] is False
    assert result["current_evidence"] is None
    assert result["safe_to_execute"] is None


def test_safety_replay_reports_corrupt_verdict_and_legacy_unknown():
    import pytest

    probe, _, _, _ = safety_capture()
    probe["verdict"]["body_cross_current"] = True
    result = replay_safety(probe)
    assert not result["predicate_matches"]
    assert "verdict.body_cross_current" in result["mismatched_fields"]
    legacy, _ = capture()
    with pytest.raises(ValueError, match="missing or invalid"):
        replay_safety(legacy)
    legacy["command"].update(cmd_t=100, throttle=0, brake=1, steering=0)
    report = analyze_frames([{"geometry_probe": legacy, "cmd_seq": 1, "cmd_t": 100,
                             "throttle": 0, "brake": 1, "steer": 0}])
    assert report["geometry_replayed"] == 1
    assert report["safety_assessment_unknown"] == 1
    assert report["safety_assessment_replayed"] == 0


def test_safety_capture_is_opt_in_and_does_not_change_verdict():
    from beamng_autopilot.safety_monitor import SafetyMonitor

    probe, enabled, scene, path = safety_capture(path_y=2)
    disabled = SafetyMonitor().evaluate(scene, path, now_s=100)
    assert disabled.assessment_geometry is None
    assert disabled.assessment_geometry_ms is None
    for field in ("level", "reason", "target_speed", "body_cross_current",
                  "body_cross_planned", "first_crossing_distance_m", "rules_evaluated"):
        assert getattr(enabled, field) == getattr(disabled, field)
    assert probe["safety_assessment_capture_ms"] >= 0
