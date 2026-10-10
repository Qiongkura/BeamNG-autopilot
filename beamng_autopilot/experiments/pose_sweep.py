"""Tech pose probes with qualified outputs and owned-process cleanup.

Offsets describe experimental spawn poses, never a lateral driving reference.
Production placement remains enabled; its actual pose is recorded separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from beamng_autopilot.config import LOGS_DIR, PROJECT_ROOT
from .collection import close_started_game, collector_python, game_pids
from .controller import MachineLease
from .reproducibility import source_snapshot


@dataclass(frozen=True)
class PoseRequest:
    map_name: str
    teleport: tuple
    goal: tuple
    model: str
    seconds: float = 25.0
    speed: float = 6.0
    max_wall_s: float = 150.0
    pose_contract: str = "production_alignment"
    position_tolerance_m: float = 0.10
    heading_tolerance_deg: float = 1.0
    probe_geometry: bool = False
    capture_range_replay: bool = False

    def __post_init__(self):
        if len(self.teleport) != 3 or len(self.goal) != 2:
            raise ValueError("teleport needs x/y/yaw and goal needs x/y")
        numbers = (*self.teleport, *self.goal, self.seconds, self.speed,
                   self.max_wall_s, self.position_tolerance_m,
                   self.heading_tolerance_deg)
        if not all(math.isfinite(float(v)) for v in numbers):
            raise ValueError("probe parameters must be finite")
        if self.seconds <= 0 or self.max_wall_s <= 0 or self.speed <= 0:
            raise ValueError("seconds, speed and wall cap must be positive")
        if self.position_tolerance_m < 0 or self.heading_tolerance_deg < 0:
            raise ValueError("pose tolerances must be non-negative")
        if self.pose_contract not in ("production_alignment", "fixed_actual_pose"):
            raise ValueError("unknown pose contract")

    def digest(self, model_sha: str) -> str:
        return hashlib.sha256(json.dumps(
            {"request": asdict(self), "model_sha256": model_sha},
            sort_keys=True).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def port_is_free(port: int) -> bool:
    """瞬时检查本地 RPC 端口是否可绑定；不触碰已有连接。"""
    if int(port) <= 0:
        return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def acquire_port(base_port: int, *, total_wait_s: float = 150.0,
                 candidates: int = 4, poll_s: float = 5.0) -> int | None:
    """总等待预算内选择一个端口（base, base+1, ...）。

    旧入口第一次 bind 失败就直接返回 blocked；E5 实测连续启动会因旧会话的
    端口/锁尚未释放而握手失败。这里在总预算内轮换端口，仍然不杀已有会话。
    """
    if int(base_port) <= 0:
        return int(base_port)
    deadline = time.monotonic() + max(0.0, float(total_wait_s))
    while True:
        for offset in range(max(1, int(candidates))):
            port = int(base_port) + offset
            if port_is_free(port):
                return port
        if time.monotonic() >= deadline:
            return None
        time.sleep(min(float(poll_s), max(0.0, deadline - time.monotonic())))


def probe_command(request: PoseRequest, output: Path, *, python: str) -> list:
    return [python, str(PROJECT_ROOT / "scripts" / "m5_fsd_drive.py"),
            "--runtime", "tech", "--map", request.map_name,
            "--teleport", *map(str, request.teleport),
            "--goal", *map(str, request.goal), "--lane-mode", "sensor", "--strict",
            "--seconds", str(request.seconds), "--speed", str(request.speed),
            "--max-wall-s", str(request.max_wall_s), "--seg-model", request.model,
            "--out", str(output), *(["--probe-geometry"] if request.probe_geometry else []),
            *(["--capture-range-replay"] if request.capture_range_replay else [])]


def qualify_probe(output: Path, request: PoseRequest, *, exit_code: int) -> dict:
    """A file's existence never proves completion; missing evidence rejects it."""
    result = {"status": "invalid", "qualified": False, "reasons": [],
              "pose_contract": request.pose_contract}
    reasons = result["reasons"]
    try:
        audit = json.loads(Path(str(output) + ".session.json").read_text(encoding="utf-8"))
        if audit.get("schema") != "fsd-session-v1":
            raise ValueError("unsupported session schema")
        cfg = audit["config"]
        expected = {"runtime": "tech", "map": request.map_name, "strict": True,
                    "lane_mode": "sensor", "teleport": list(request.teleport),
                    "goal": list(request.goal), "seconds": request.seconds,
                    "speed": request.speed, "max_wall_s": request.max_wall_s}
        for key, value in expected.items():
            if cfg.get(key) != value:
                reasons.append(f"config mismatch: {key}")
        if Path(cfg.get("seg_model") or "").resolve() != Path(request.model).resolve():
            reasons.append("config mismatch: seg_model")
        config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        if audit.get("config_hash") != config_hash:
            reasons.append("session config hash mismatch")
        if audit.get("exit_code") != exit_code:
            reasons.append("process/session exit code mismatch")
        if (audit.get("source_unchanged") is not True
                or not (audit.get("source_before") or {}).get("complete")
                or not (audit.get("source_after") or {}).get("complete")
                or not (audit.get("source_before") or {}).get("sha256")
                or (audit.get("source_before") or {}).get("sha256")
                != (audit.get("source_after") or {}).get("sha256")):
            reasons.append("source identity missing, incomplete or changed")
        if bool(cfg.get("probe_geometry")) != request.probe_geometry:
            reasons.append("config mismatch: probe_geometry")
        if bool(cfg.get("capture_range_replay")) != request.capture_range_replay:
            reasons.append("config mismatch: capture_range_replay")
        result["audit"] = audit
        if reasons:
            return result
        if exit_code == 3:
            if (audit.get("placement_rc") == 3 and audit.get("placed") is False
                    and audit.get("drive_started") is False):
                result["status"] = "unplaceable"
                reasons.append("placement deadline; no equal-exposure drive measurement")
            else:
                reasons.append("inconsistent placement timeout")
            return result
        if exit_code != 0:
            result["status"] = "incomplete" if exit_code == 4 else "failed"
            reasons.append(f"driver exit code {exit_code}")
            return result
        if (audit.get("placed") is not True or audit.get("placement_rc") != 0
                or audit.get("drive_started") is not True
                or audit.get("termination") != "completed"):
            reasons.append("successful strict placement/measurement unproven")
        cleanup = audit.get("cleanup") or {}
        if (cleanup.get("closed") is not True or cleanup.get("unverified")
                or cleanup.get("still_running")
                or audit.get("connection_close_error")):
            reasons.append("owned game cleanup unproven")
        duration = float(audit.get("drive_elapsed_s", float("nan")))
        # Frozen allowance: one 2 Hz control interval, not a post-hoc threshold.
        if not math.isfinite(duration) or duration < request.seconds - 0.5:
            reasons.append("measurement duration too short or missing")
        if request.capture_range_replay:
            from .range_replay import load_replay

            capture = audit.get("range_replay") or {}
            directory = output.parent / "range_replay"
            if (capture.get("status") != "captured"
                    or Path(capture.get("directory") or "").resolve() != directory.resolve()
                    or capture.get("manifest_sha256") != file_sha256(directory / "manifest.json")):
                raise ValueError("requested range replay capture missing/failed or digest differs")
            _payload, _sample, manifest = load_replay(directory)
            source_t = _payload.meta.get("source_poll_started_t")
            if (_payload.meta.get("schema") != "range-timing-v1"
                    or _payload.meta.get("finite_cloud_points") != manifest["points"]
                    or not isinstance(_payload.meta.get("source_errors"), dict)
                    or _payload.meta["source_errors"]
                    or not isinstance(source_t, (int, float)) or isinstance(source_t, bool)
                    or not math.isfinite(source_t) or not 0 < source_t <= manifest["captured_wall_t"]):
                raise ValueError("range replay valid source payload unproven")
            times = [float(capture.get("captured_wall_t", float("nan"))),
                     float(capture.get("disarmed_wall_t", float("nan"))),
                     float(audit.get("drive_started_wall_t", float("nan")))]
            if (not all(math.isfinite(t) for t in times)
                    or not 0 < times[0] <= times[1] <= times[2]
                    or times[0] != manifest["captured_wall_t"]
                    or capture.get("points") != manifest["points"]
                    or manifest["source"]["sha256"] != audit["source_before"]["sha256"]):
                raise ValueError("range replay source/phase/count unproven")
            result["range_replay"] = dict(capture, validated=True)
        actual = audit.get("actual_pose") or {}
        pos = np.asarray(actual.get("pos", []), dtype=float)
        heading = float(actual.get("heading_rad", float("nan")))
        if pos.shape != (3,) or not np.isfinite(pos).all() or not math.isfinite(heading):
            reasons.append("actual placement pose missing or invalid")
        else:
            shift = float(np.linalg.norm(pos[:2] - np.asarray(request.teleport[:2])))
            yaw_error = abs((math.degrees(heading) - request.teleport[2] + 180) % 360 - 180)
            result.update(position_error_m=shift, heading_error_deg=yaw_error)
            if (request.pose_contract == "fixed_actual_pose"
                    and (shift > request.position_tolerance_m
                         or yaw_error > request.heading_tolerance_deg)):
                reasons.append("production placement changed requested fixed pose")
        frames = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(frames, list) or len(frames) < 2:
            raise ValueError("fewer than two telemetry frames")
        if audit.get("telemetry_frames") != len(frames):
            reasons.append("telemetry frame count mismatch")
        required = ("t", "pos", "heading", "reason", "cmd_seq", "cmd_t",
                    "throttle", "brake", "steer")
        missing = {key: sum(key not in f or f[key] is None for f in frames)
                   for key in required}
        result["missing"] = missing
        if any(missing.values()):
            reasons.append("required control/pose columns missing")
        else:
            ts = np.asarray([f["t"] for f in frames], dtype=float)
            seq = np.asarray([f["cmd_seq"] for f in frames], dtype=float)
            cmd_ts = np.asarray([f["cmd_t"] for f in frames], dtype=float)
            if (not np.isfinite(ts).all() or not np.isfinite(seq).all()
                    or not np.isfinite(cmd_ts).all() or not (np.diff(cmd_ts) > 0).all()
                    or not (seq == np.floor(seq)).all() or not (seq > 0).all()
                    or not (np.diff(ts) > 0).all() or not (np.diff(seq) > 0).all()):
                reasons.append("non-finite or non-increasing timestamps/commands")
            for f in frames:
                pos = np.asarray(f["pos"], dtype=float)
                if pos.shape != (3,) or not np.isfinite(pos).all() or not math.isfinite(float(f["heading"])):
                    reasons.append("invalid telemetry pose")
                    break
                controls = [float(f[key]) for key in ("throttle", "brake", "steer")]
                if (not all(math.isfinite(v) for v in controls)
                        or not 0 <= controls[0] <= 1 or not 0 <= controls[1] <= 1
                        or not -1 <= controls[2] <= 1):
                    reasons.append("invalid telemetry controls")
                    break
            if request.probe_geometry:
                result["geometry_missing"] = sum(
                    not isinstance(f.get("geometry_probe"), dict)
                    or f["geometry_probe"].get("schema") != "control-geometry-v1"
                    or bool(f["geometry_probe"].get("error")) for f in frames)
                if result["geometry_missing"]:
                    reasons.append("requested geometry probe missing/failed")
            if ts[-1] - ts[0] < max(0.0, request.seconds - 1.0):
                reasons.append("telemetry exposure too short")
        result.update(frames=len(frames), reasons_histogram=dict(Counter(
            str(f.get("reason", "UNKNOWN")) for f in frames)),
            body_coverage_unknown=sum(f.get("body_cov_status") in (None, "unknown")
                                      for f in frames))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        reasons.append(f"unreadable/incomplete evidence: {exc}")
    if not reasons:
        result.update(status="completed", qualified=True)
    return result


def run_probe(request: PoseRequest, directory: Path, *, port: int,
              lease: MachineLease, process_factory=None) -> dict:
    """One fresh attempt. Never attach, kill by image name, or overwrite evidence."""
    before = game_pids()
    if before is None or before:
        return {"status": "blocked", "qualified": False,
                "reasons": ["game state unknown or another game session is active"]}
    actual_port = acquire_port(int(port), total_wait_s=min(request.max_wall_s, 150.0))
    if actual_port is None:
        return {"status": "blocked", "qualified": False,
                "reasons": [f"ports {port}..{int(port) + 3} unavailable within wait budget"]}
    model = Path(request.model)
    model_sha = file_sha256(model)
    source_before = source_snapshot(PROJECT_ROOT)
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "telemetry.json"
    python, _source = collector_python(PROJECT_ROOT)
    cmd = probe_command(request, output, python=python)
    (directory / "request.json").write_text(json.dumps(
        {"request": asdict(request), "model_sha256": model_sha,
         "request_digest": request.digest(model_sha), "command": cmd,
         "requested_port": int(port), "actual_port": int(actual_port),
         "source": source_before},
        ensure_ascii=False, indent=2), encoding="utf-8")
    env = dict(os.environ, BEAMNG_TECH_PORT=str(actual_port), PYTHONUTF8="1",
               PYTHONIOENCODING="utf-8")
    launched = time.time()
    mono = time.monotonic()
    proc = None
    timed_out = False
    cleanup = None
    failure = None
    try:
        with (directory / "driver.log").open("w", encoding="utf-8") as log:
            factory = process_factory or subprocess.Popen
            proc = factory(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log,
                           stderr=subprocess.STDOUT,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            while proc.poll() is None:
                if not lease.heartbeat():
                    raise RuntimeError("machine lease lost")
                if time.monotonic() - mono > request.max_wall_s + 120.0:
                    timed_out = True
                    break
                time.sleep(1.0)
            if timed_out:
                proc.terminate()
                proc.wait(timeout=30)
    except Exception as exc:
        failure = str(exc)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        # Reuse creation-time + launch snapshot checks; never enumerate then
        # indiscriminately terminate every BeamNG process.
        try:
            cleanup = close_started_game(before, launched_after=launched,
                                         verify_timeout_s=10.0)
        except Exception as exc:
            cleanup = {"closed": False, "reason": str(exc)}
    rc = int(proc.returncode) if proc is not None and proc.returncode is not None else 2
    result = qualify_probe(output, request, exit_code=rc)
    if failure:
        result.update(status="failed", qualified=False)
        result["reasons"].append(f"driver failure: {failure}")
    if timed_out:
        result.update(status="incomplete", qualified=False)
        result["reasons"].append("outer driver deadline; owned process terminated")
    if not model.is_file() or file_sha256(model) != model_sha:
        result.update(status="invalid", qualified=False)
        result["reasons"].append("checkpoint changed during measurement")
    source_after = source_snapshot(PROJECT_ROOT)
    audit_source = ((result.get("audit") or {}).get("source_before") or {}).get("sha256")
    if (not source_before["complete"] or not source_after["complete"]
            or source_before["sha256"] != source_after["sha256"]
            or (audit_source is not None and audit_source != source_before["sha256"])):
        result.update(status="invalid", qualified=False)
        result["reasons"].append("runner/driver source identity changed or mismatched")
    if (not cleanup.get("closed") or cleanup.get("unverified")
            or cleanup.get("still_running")):
        result.update(status="invalid", qualified=False)
        result["reasons"].append("outer cleanup unproven")
    result.update(request=asdict(request), model_sha256=model_sha,
                  request_digest=request.digest(model_sha), command=cmd,
                  requested_port=int(port), actual_port=int(actual_port),
                  exit_code=rc, outer_cleanup=cleanup,
                  source_before=source_before, source_after=source_after)
    (directory / "result.json").write_text(json.dumps(
        result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def pose_matrix(anchor: tuple, lateral: list, yaw: list) -> list:
    angle = math.radians(float(anchor[2]))
    normal = (-math.sin(angle), math.cos(angle))
    return [(float(anchor[0]) + float(d) * normal[0],
             float(anchor[1]) + float(d) * normal[1], float(anchor[2]) + float(y))
            for d in lateral for y in yaw]


def machine_lease() -> MachineLease:
    return MachineLease(LOGS_DIR / "experiments" / "machine_lease.json")
