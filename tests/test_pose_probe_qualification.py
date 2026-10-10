"""Counterexamples for pose evidence; these fixtures are not driving results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from beamng_autopilot.experiments import pose_sweep as ps


def write_evidence(output, request, **audit_overrides):
    cfg = {"runtime": "tech", "map": request.map_name, "strict": True,
           "lane_mode": "sensor", "teleport": list(request.teleport),
           "goal": list(request.goal), "seconds": request.seconds,
           "speed": request.speed, "max_wall_s": request.max_wall_s,
           "seg_model": request.model, "probe_geometry": request.probe_geometry,
           "capture_range_replay": request.capture_range_replay}
    frames = [{"t": i * 0.5, "pos": [1, 2, 0], "heading": 0.0,
               "reason": "no drivable path", "cmd_seq": i + 1, "cmd_t": 100 + i * 0.5,
               "throttle": 0.0, "brake": 1.0, "steer": 0.0}
              for i in range(50)]
    audit = {"schema": "fsd-session-v1", "config": cfg,
             "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
             "exit_code": 0, "placed": True, "placement_rc": 0,
             "drive_started": True, "drive_elapsed_s": 25.0,
             "termination": "completed", "telemetry_frames": len(frames),
             "source_before": {"complete": True, "sha256": "fixture-source"},
             "source_after": {"complete": True, "sha256": "fixture-source"},
             "source_unchanged": True,
             "actual_pose": {"pos": [1.0, 2.0, 0.0], "heading_rad": 0.0},
             "cleanup": {"closed": True}}
    audit.update(audit_overrides)
    output.write_text(json.dumps(frames), encoding="utf-8")
    Path(str(output) + ".session.json").write_text(json.dumps(audit), encoding="utf-8")
    return frames, audit


@pytest.fixture()
def evidence(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"fixture, not a learned checkpoint")
    request = ps.PoseRequest("italy", (1.0, 2.0, 0.0), (20.0, 2.0), str(model))
    output = tmp_path / "telemetry.json"
    write_evidence(output, request)
    return output, request


def test_complete_exposure_is_qualified_but_unknown_coverage_is_reported(evidence):
    output, request = evidence
    result = ps.qualify_probe(output, request, exit_code=0)
    assert result["qualified"] and result["status"] == "completed"
    assert result["body_coverage_unknown"] == 50
    assert result["missing"]["cmd_seq"] == 0


def test_existing_legacy_telemetry_is_not_completion(evidence):
    output, request = evidence
    Path(str(output) + ".session.json").unlink()
    assert not ps.qualify_probe(output, request, exit_code=0)["qualified"]


@pytest.mark.parametrize("field,value", [
    ("pos", [float("nan"), 2, 0]), ("heading", float("inf")),
    ("throttle", 1.2), ("brake", -0.1), ("steer", float("nan")),
    ("cmd_seq", 1.5), ("cmd_t", float("nan")),
])
def test_invalid_numeric_evidence_cannot_pass(evidence, field, value):
    output, request = evidence
    frames = json.loads(output.read_text(encoding="utf-8"))
    frames[0][field] = value
    output.write_text(json.dumps(frames), encoding="utf-8")
    assert not ps.qualify_probe(output, request, exit_code=0)["qualified"]


def test_changed_source_invalidates_completed_measurement(evidence):
    output, request = evidence
    write_evidence(output, request, source_after={"complete": True, "sha256": "other"})
    assert not ps.qualify_probe(output, request, exit_code=0)["qualified"]


@pytest.mark.parametrize("override", [
    {"drive_elapsed_s": 2.3}, {"placed": False}, {"placement_rc": 2},
    {"telemetry_frames": 1}, {"cleanup": None},
    {"cleanup": {"closed": True, "unverified": [{"pid": 7}]}},
    {"termination": "drive_wall_timeout"},
])
def test_short_unplaced_or_unverified_run_cannot_pass(evidence, override):
    output, request = evidence
    write_evidence(output, request, **override)
    assert not ps.qualify_probe(output, request, exit_code=0)["qualified"]


def test_changed_start_is_explicitly_distinguished_by_contract(evidence):
    output, request = evidence
    write_evidence(output, request,
                   actual_pose={"pos": [1.6, 2.0, 0.0], "heading_rad": 0.2})
    production = ps.qualify_probe(output, request, exit_code=0)
    assert production["qualified"] and production["position_error_m"] == pytest.approx(0.6)
    fixed = ps.qualify_probe(output, replace(request, pose_contract="fixed_actual_pose"),
                             exit_code=0)
    assert not fixed["qualified"]


def test_missing_command_or_short_telemetry_rejects_measurement(evidence):
    output, request = evidence
    frames, audit = write_evidence(output, request)
    del frames[0]["cmd_seq"]
    output.write_text(json.dumps(frames), encoding="utf-8")
    assert not ps.qualify_probe(output, request, exit_code=0)["qualified"]
    frames[0]["cmd_seq"] = 1
    frames[-1]["t"] = frames[-2]["t"]
    output.write_text(json.dumps(frames), encoding="utf-8")
    assert not ps.qualify_probe(output, request, exit_code=0)["qualified"]


def test_config_mismatch_and_exit_mismatch_are_rejected(evidence):
    output, request = evidence
    assert not ps.qualify_probe(output, replace(request, seconds=30), exit_code=0)["qualified"]
    assert not ps.qualify_probe(output, request, exit_code=3)["qualified"]


def test_placement_timeout_is_classified_without_counting_as_completed(evidence):
    output, request = evidence
    write_evidence(output, request, placed=False, placement_rc=3,
                   drive_started=False, exit_code=3)
    output.unlink()
    result = ps.qualify_probe(output, request, exit_code=3)
    assert result["status"] == "unplaceable" and not result["qualified"]


def test_runner_does_not_launch_when_an_existing_session_is_present(evidence, tmp_path, monkeypatch):
    _output, request = evidence
    monkeypatch.setattr(ps, "game_pids", lambda: {7})
    calls = []
    result = ps.run_probe(request, tmp_path / "probe", port=64257,
                          lease=None, process_factory=lambda *a, **k: calls.append(a))
    assert result["status"] == "blocked" and not calls


def test_launch_failure_preserves_request_and_failure_evidence(evidence, tmp_path, monkeypatch):
    _output, request = evidence
    monkeypatch.setattr(ps, "game_pids", lambda: set())
    monkeypatch.setattr(ps, "close_started_game", lambda *a, **k: {"closed": True})
    def fail(*args, **kwargs):
        raise OSError("interpreter launch failed")
    directory = tmp_path / "failed"
    result = ps.run_probe(request, directory, port=0, lease=None, process_factory=fail)
    assert result["status"] == "failed" and not result["qualified"]
    assert (directory / "request.json").is_file()
    assert (directory / "result.json").is_file()
    assert any("interpreter launch failed" in r for r in result["reasons"])


def test_runner_records_command_digest_and_outer_cleanup(evidence, tmp_path, monkeypatch):
    _output, request = evidence
    monkeypatch.setattr(ps, "game_pids", lambda: set())
    monkeypatch.setattr(ps, "source_snapshot",
                        lambda root: {"complete": True, "sha256": "fixture-source"})
    calls = []
    def cleanup(before, **kwargs):
        calls.append((before, kwargs))
        return {"closed": True}
    monkeypatch.setattr(ps, "close_started_game", cleanup)
    def factory(cmd, **kwargs):
        write_evidence(Path(cmd[cmd.index("--out") + 1]), request)
        assert kwargs["env"]["BEAMNG_TECH_PORT"] == "0"
        return SimpleNamespace(returncode=0, poll=lambda: 0)
    result = ps.run_probe(request, tmp_path / "completed", port=0, lease=None,
                          process_factory=factory)
    assert result["qualified"]
    assert result["request_digest"] == request.digest(result["model_sha256"])
    assert calls[0][0] == set() and "launched_after" in calls[0][1]


def test_capture_flag_is_explicit_and_missing_capture_rejects_evidence(evidence):
    output, request = evidence
    assert "--capture-range-replay" not in ps.probe_command(request, output, python="python")
    request = replace(request, capture_range_replay=True)
    assert "--capture-range-replay" in ps.probe_command(request, output, python="python")
    write_evidence(output, request)
    result = ps.qualify_probe(output, request, exit_code=0)
    assert not result["qualified"]
    assert any("range replay" in reason for reason in result["reasons"])


@pytest.mark.parametrize("mutation", [None, "late_disarm", "missing_time", "source", "corrupt"])
def test_capture_requires_valid_artifact_source_and_pre_drive_disarm(evidence, monkeypatch, mutation):
    import numpy as np
    from beamng_autopilot.experiments import range_replay as rr
    from beamng_autopilot_tech.providers import _RawRange
    output, request = evidence
    request = replace(request, capture_range_replay=True)
    monkeypatch.setattr(rr, "source_snapshot", lambda root: {"complete": True, "sha256": "fixture-source"})
    capture = rr.write_replay(output.parent / "range_replay",
                              _RawRange(cloud=np.zeros((4, 3)), heading=0., radius=55.,
                                        meta={"schema": "range-timing-v1", "source_errors": {},
                                              "finite_cloud_points": 4, "source_poll_started_t": 100.}),
                              pos=[1., 2., 0.], ego_half_len=2.2, ego_half_width=.9)
    t = capture["captured_wall_t"]
    capture["disarmed_wall_t"] = t + 1.
    _frames, audit = write_evidence(output, request, range_replay=capture, drive_started_wall_t=t + 2.)
    if mutation == "late_disarm":
        audit["range_replay"]["disarmed_wall_t"] = t + 3.
    elif mutation == "missing_time":
        del audit["drive_started_wall_t"]
    elif mutation == "source":
        audit["source_before"]["sha256"] = audit["source_after"]["sha256"] = "other"
    elif mutation == "corrupt":
        with (output.parent / "range_replay" / "cloud.npy").open("ab") as stream:
            stream.write(b"corrupt")
    Path(str(output) + ".session.json").write_text(json.dumps(audit), encoding="utf-8")
    result = ps.qualify_probe(output, request, exit_code=0)
    assert result["qualified"] is (mutation is None), result
    if mutation is None:
        assert result["range_replay"]["validated"] is True


def test_acquire_port_rotates_around_a_busy_port():
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    busy = sock.getsockname()[1]
    try:
        actual = ps.acquire_port(busy, total_wait_s=0.0, candidates=2)
        assert actual == busy + 1
    finally:
        sock.close()


def test_runner_records_actual_rotated_port(evidence, tmp_path, monkeypatch):
    _output, request = evidence
    monkeypatch.setattr(ps, "game_pids", lambda: set())
    monkeypatch.setattr(ps, "acquire_port", lambda port, **kwargs: port + 1)
    monkeypatch.setattr(ps, "source_snapshot",
                        lambda root: {"complete": True, "sha256": "fixture-source"})
    monkeypatch.setattr(ps, "close_started_game",
                        lambda *a, **k: {"closed": True})

    def factory(cmd, **kwargs):
        output = Path(cmd[cmd.index("--out") + 1])
        write_evidence(output, request)
        assert kwargs["env"]["BEAMNG_TECH_PORT"] == "64258"
        return SimpleNamespace(returncode=0, poll=lambda: 0)

    result = ps.run_probe(request, tmp_path / "rotated", port=64257,
                          lease=None, process_factory=factory)
    assert result["qualified"]
    assert result["actual_port"] == 64258
    request_record = json.loads((tmp_path / "rotated" / "request.json").read_text())
    assert request_record["requested_port"] == 64257
    assert request_record["actual_port"] == 64258

