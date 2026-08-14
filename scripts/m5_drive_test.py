"""Live autopilot drive test: run the real m5 process against a nav route.

The script attaches to the running BeamNG session, places the ego at a
probe start point, sets the same in-game navigation route the user gets
from the big map, launches ``m5_autopilot.py`` as a child process, drives
it through the same ControlBridge commands the GUI uses (F10/F9/F12), and
records high-frequency telemetry plus the autopilot's own log.

Usage:
    .venv\\Scripts\\python.exe scripts\\m5_drive_test.py --speed 6 --run 10
    .venv\\Scripts\\python.exe scripts\\m5_drive_test.py --speed 15 --run 12 --no-vision
    .venv\\Scripts\\python.exe scripts\\m5_drive_test.py --runtime tech --speed 6 --run 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control.gearbox import read_gearbox_mode
from beamng_autopilot.planner import _point_route_pos
from beamng_autopilot.runtime import resolve_runtime


START_XY = (726.6, 755.9)
GOAL_XY = (572.0, 533.5)
READY_TIMEOUT_S = 60.0
MONITOR_TIMEOUT_S = 150.0


def _ground_z(conn, x: float, y: float) -> float | None:
    chunk = (
        f"local res = Engine.castRay(vec3({x:.3f}, {y:.3f}, 10000), "
        f"vec3({x:.3f}, {y:.3f}, -1000), true, false)\n"
        "if res and res.pt then "
        "return string.format('%.3f,%.3f,%.3f', "
        "res.pt.x, res.pt.y, res.pt.z) end\n"
        "return 'nil'"
    )
    resp = conn.bng.control.queue_lua_command(chunk, response=True)
    if resp and str(resp).strip() != "nil":
        parts = str(resp).split(",")
        if len(parts) == 3:
            return float(parts[2])
    return None


def _teleport_start(conn, start_xy, goal_xy):
    """Move the ego to the probe start and face the goal."""
    from beamngpy.misc.quat import angle_to_quat

    heading = math.atan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
    yaw_deg = -math.degrees(float(heading)) - 90.0
    st0 = conn.get_state()
    z = float(st0.pos[2]) if len(st0.pos) > 2 else 0.0
    ground_z = _ground_z(conn, float(start_xy[0]), float(start_xy[1]))
    if ground_z is not None:
        z = ground_z + 0.6
    conn.vehicle.teleport(
        (float(start_xy[0]), float(start_xy[1]), z),
        rot_quat=angle_to_quat((0.0, 0.0, yaw_deg)))
    conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=0.0)
    conn.step(30)
    st = conn.get_state()
    print(f"[drive-test] teleport -> ({st.pos[0]:.1f}, {st.pos[1]:.1f}, "
          f"{st.pos[2]:.1f}) heading={math.degrees(float(st.heading)):.1f}")
    return st


def _set_nav_route(conn, goal_xy):
    """Ask the game's route planner to draw a route to ``goal_xy``."""
    conn.bng.control.queue_lua_command(
        "core_groundMarkers.setPath({vec3(%.3f, %.3f, 0)})\nreturn 'ok'"
        % (float(goal_xy[0]), float(goal_xy[1])), response=True)
    time.sleep(0.8)
    nav = conn.read_navigation_route()
    if nav is None or len(nav) < 4:
        raise RuntimeError("nav route was not created")
    dseg = np.linalg.norm(np.diff(nav[:, :2], axis=0), axis=1)
    total = float(np.sum(dseg))
    print(f"[drive-test] nav route: {len(nav)} pts, {total:.1f} m")
    return nav


def _read_live() -> dict | None:
    p = config.LOGS_DIR / "telemetry" / "live.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clear_live() -> None:
    p = config.LOGS_DIR / "telemetry" / "live.json"
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


def _ended_reason(log_text: str) -> str | None:
    for ln in reversed(log_text.splitlines()):
        if "autopilot ended:" in ln:
            return ln.split("autopilot ended:", 1)[1].strip()
    return None


def _cmd_path() -> Path:
    return config.LOGS_DIR / "autopilot_ctl.json"


def _send_cmds(cmds: list[str]) -> None:
    seq = 0
    p = _cmd_path()
    try:
        cur = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(cur, dict) and isinstance(cur.get("seq"), int):
            seq = cur["seq"]
    except Exception:
        pass
    for cmd in cmds:
        seq += 1
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "seq": seq, "cmd": cmd, "ts": time.time(),
        }), encoding="utf-8")
        tmp.replace(p)
        time.sleep(0.1)
    print(f"[drive-test] sent control cmds: {cmds}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects the connected session")
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--max-run", type=float, default=60.0)
    ap.add_argument("--no-vision", action="store_true",
                    help="disable all front-camera vision "
                         "(YOLO obstacles + lane markings)")
    ap.add_argument("--no-lanes", action="store_true",
                    help="disable lane-marking detection only "
                         "(YOLO obstacles stay on)")
    ap.add_argument("--no-nav", action="store_true",
                    help="do not set or grab a navigation route; drive "
                         "from the camera lane pair only")
    ap.add_argument("--run", type=int, default=0,
                    help="run number used in artifact names")
    ap.add_argument("--start-x", type=float, default=START_XY[0])
    ap.add_argument("--start-y", type=float, default=START_XY[1])
    ap.add_argument("--goal-x", type=float, default=GOAL_XY[0])
    ap.add_argument("--goal-y", type=float, default=GOAL_XY[1])
    args = ap.parse_args()

    run_id = args.run or int(time.time())
    out_dir = config.LOGS_DIR / "live_runs" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = out_dir / "autopilot.out.log"
    stderr_log = out_dir / "autopilot.err.log"
    telemetry_log = out_dir / "telemetry_history.json"
    report_path = out_dir / "report.json"

    start_xy = np.asarray((args.start_x, args.start_y), dtype=float)
    goal_xy = np.asarray((args.goal_x, args.goal_y), dtype=float)
    conn = BeamNGConnector(
        args.map, args.vehicle, home=config.runtime_home(args.runtime))
    route = None
    proc = None
    history: list[dict] = []
    report: dict = {
        "run": run_id,
        "runtime": args.runtime,
        "speed": args.speed,
        "max_run": args.max_run,
        "vision": not args.no_vision,
        "nav": not args.no_nav,
        "start": start_xy.tolist(),
        "goal": goal_xy.tolist(),
        "artifacts": {
            "stdout": str(stdout_log),
            "stderr": str(stderr_log),
            "telemetry": str(telemetry_log),
        },
    }

    try:
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
        session_runtime = resolve_runtime(conn, args.runtime)
        report["runtime"] = session_runtime
        print(f"[drive-test] runtime={session_runtime}")
        _teleport_start(conn, start_xy, goal_xy)
        if args.no_nav:
            print("[drive-test] no-nav mode: camera lane pair only")
        else:
            nav = _set_nav_route(conn, goal_xy)
            route = nav[:, :2]

        # Remove stale control/live files so the child starts with a clean
        # watermark and the idle-frame wait cannot match a previous run.
        _cmd_path().unlink(missing_ok=True)
        _clear_live()
        cmd = [
            sys.executable, str(Path(__file__).resolve().parent
                                / "m5_autopilot.py"),
            "--map", args.map, "--vehicle", args.vehicle,
            "--runtime", session_runtime,
            "--attach", "--speed", str(args.speed),
            "--max-run", str(args.max_run),
            "--no-hud", "--no-show", "--no-markers", "--no-overlay",
        ]
        if args.no_vision:
            cmd.append("--no-vision-obstacles")
            cmd.append("--no-lanes")
        elif args.no_lanes:
            cmd.append("--no-lanes")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        with open(stdout_log, "w", encoding="utf-8") as so, \
                open(stderr_log, "w", encoding="utf-8") as se:
            proc = subprocess.Popen(
                cmd, stdout=so, stderr=se, env=env,
                cwd=str(Path(__file__).resolve().parent.parent))
        print(f"[drive-test] autopilot pid={proc.pid}")

        # Wait until the autopilot publishes its first idle telemetry frame
        # (that is its main loop, and therefore its control poller).
        t0 = time.time()
        while time.time() - t0 < READY_TIMEOUT_S:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"autopilot exited early rc={proc.returncode}; "
                    f"see {stderr_log} / {stdout_log}")
            live = _read_live()
            if live is not None and live.get("extra", {}).get("auto") == 0:
                print("[drive-test] autopilot idle loop detected")
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("autopilot did not become ready in time")

        time.sleep(0.5)
        if args.no_nav:
            print("[drive-test] not sending navroute")
        else:
            _send_cmds(["navroute"])
            t_route = time.time()
            while time.time() - t_route < 10.0:
                live = _read_live()
                if live is not None and (
                        live.get("extra") or {}).get("route", 0):
                    print(f"[drive-test] route grabbed by autopilot "
                          f"({live['extra']['route']} pts)")
                    break
                time.sleep(0.2)
            else:
                print("[drive-test] autopilot did not grab route; retrying")
                _send_cmds(["navroute"])
                time.sleep(1.0)
        _send_cmds(["autopilot"])

        # Monitor the control loop at ~20 Hz until the autopilot ends on its
        # own (goal reached / max-run), the car stops while not near the
        # goal, the run times out, or the process exits.
        last_pos = None
        last_t = None
        stopped_t = None
        final_reason = "timeout"
        last_extra = None
        ended_frame = None
        m0 = time.time()
        while time.time() - m0 < MONITOR_TIMEOUT_S:
            if proc.poll() is not None:
                final_reason = f"autopilot exit rc={proc.returncode}"
                print(f"[drive-test] {final_reason}")
                break
            live = _read_live()
            if live is not None:
                history.append(live)
                extra = live.get("extra") or {}
                last_extra = extra
                t = float(live.get("t") or 0.0)
                speed = float(live.get("speed") or 0.0)
                pos = live.get("pos")
                if pos and len(pos) >= 2:
                    pos = np.asarray(pos[:2], dtype=float)
                    last_pos = pos
                if extra.get("ended") or extra.get("mode") == "ENDED":
                    final_reason = str(extra.get("reason") or "ended")
                    ended_frame = live
                    print(f"[drive-test] autopilot ended: {final_reason}")
                    break
                if extra.get("auto") == 1 and not args.no_nav:
                    goal_d = float(extra.get("goal_d") or 999.0)
                    if speed < 0.5 and goal_d > 12.0:
                        if stopped_t is None:
                            stopped_t = time.time()
                        elif time.time() - stopped_t > 3.0:
                            final_reason = "stopped-not-at-goal"
                            print("[drive-test] stopped while far from goal")
                            break
                    else:
                        stopped_t = None
                if t and last_t is not None and t < last_t:
                    pass
                last_t = t
            # The ENDED frame is a one-shot publish before the main loop
            # starts writing idle snapshots again, so the stdout line is
            # the reliable fallback when 20 Hz sampling misses it.
            try:
                ended_reason = _ended_reason(
                    stdout_log.read_text(encoding="utf-8",
                                         errors="replace"))
            except Exception:
                ended_reason = None
            if ended_reason:
                final_reason = ended_reason
                for _ in range(40):
                    live = _read_live()
                    if live is not None:
                        ex = live.get("extra") or {}
                        if ex.get("ended") or ex.get("mode") == "ENDED":
                            ended_frame = live
                            history.append(live)
                            break
                    time.sleep(0.05)
                print(f"[drive-test] autopilot ended: {final_reason}")
                break
            time.sleep(0.05)
        else:
            print("[drive-test] monitor timeout; stopping autopilot")

        if ended_frame is None:
            for h in reversed(history):
                ex = h.get("extra") or {}
                if (ex.get("ended") or ex.get("mode") == "ENDED"
                        or ex.get("auto") == 1):
                    ended_frame = h
                    break
        final = ended_frame or _read_live()
        if final is not None and final is not ended_frame:
            history.append(final)
            last_extra = final.get("extra") or last_extra or {}

        # Stop the autopilot cleanly through the same bridge the GUI uses.
        try:
            _send_cmds(["quit"])
        except Exception as exc:
            print(f"[drive-test] quit cmd failed: {exc}")
        if proc is not None:
            try:
                proc.wait(timeout=25.0)
                print(f"[drive-test] autopilot exited rc={proc.returncode}")
            except subprocess.TimeoutExpired:
                print("[drive-test] killing autopilot after quit timeout")
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10.0)

        # Summarize.
        autopilot_log = stdout_log.read_text(encoding="utf-8", errors="replace")
        blocked_events = [
            ln for ln in autopilot_log.splitlines() if "BLOCKED by" in ln]
        max_speed = max((float(h.get("speed") or 0.0) for h in history),
                        default=0.0)
        max_steer = max((abs(float(h.get("steer") or 0.0)) for h in history),
                        default=0.0)
        driven = 0.0
        route_cum = None
        if route is not None and len(route) >= 2:
            rseg = np.linalg.norm(np.diff(route, axis=0), axis=1)
            route_cum = np.concatenate([[0.0], np.cumsum(rseg)])
        lat_offsets = []
        max_reverse = 0.0
        reverse_episodes = 0
        rev_start = None
        rev_dist = 0.0
        route_rev_total = 0.0
        route_arc_last = None
        lane_lats: list[float] = []
        lane_widths: list[float] = []
        lane_src_counts: dict[str, int] = {}
        max_black_frames = 0
        for h in history:
            pos = h.get("pos")
            if pos and len(pos) >= 2 and route is not None:
                route_arc, lat = _point_route_pos(
                    float(pos[0]), float(pos[1]), route)
                lat_offsets.append(float(lat))
                if route_cum is not None:
                    d = np.linalg.norm(route[:, :2] - pos[:2], axis=1)
                    nearest = int(np.argmin(d))
                    driven = max(driven, float(route_cum[nearest]))
            else:
                route_arc = None
            ex = h.get("extra") or {}
            max_black_frames = max(
                max_black_frames,
                int(ex.get("black_frames") or 0))
            src = ex.get("lane_src") or ""
            if src and pos and len(pos) >= 2:
                lane_src_counts[src] = lane_src_counts.get(src, 0) + 1
                ll = ex.get("lane_lat")
                if ll is not None:
                    lane_lats.append(float(ll))
                lw = ex.get("lane_w")
                if lw is not None:
                    lane_widths.append(float(lw))
            vel = h.get("vel")
            dvec = h.get("dir_vec")
            signed = None
            if pos and len(pos) >= 2 and vel and len(vel) >= 2 \
                    and dvec and len(dvec) >= 2:
                signed = float(np.dot(np.asarray(vel[:2], dtype=float),
                                      np.asarray(dvec[:2], dtype=float)))
            route_back = 0.0
            if route_arc is not None:
                if route_arc_last is not None:
                    arc_delta = route_arc_last - route_arc
                    if 0.03 < arc_delta < 3.0:
                        route_back = arc_delta
                    elif arc_delta >= 3.0:
                        route_arc_last = route_arc
                route_arc_last = route_arc
            route_reverse = route_back > 0.12 and (
                signed is None or signed < 0.3)
            if pos and len(pos) >= 2:
                p = np.asarray(pos[:2], dtype=float)
                if (signed is not None and signed < -0.3) or route_reverse:
                    if rev_start is None:
                        rev_start = p
                        rev_dist = 0.0
                        route_rev_total = 0.0
                        reverse_episodes += 1
                    else:
                        rev_dist = float(np.linalg.norm(p - rev_start))
                        if route_reverse:
                            route_rev_total += route_back
                        max_reverse = max(
                            max_reverse, rev_dist, route_rev_total)
                else:
                    rev_start = None
                    rev_dist = 0.0
                    route_rev_total = 0.0
        report.update({
            "reason": final_reason,
            "frames": len(history),
            "max_speed": round(max_speed, 2),
            "max_abs_steer": round(max_steer, 3),
            "driven_m": round(driven, 1),
            "median_lat": round(float(np.median(lat_offsets)), 2)
            if lat_offsets else None,
            "max_abs_lat": round(float(np.max(np.abs(lat_offsets))), 2)
            if lat_offsets else None,
            "lane_frames": sum(lane_src_counts.values()),
            "max_black_frames": max_black_frames,
            "lane_src": lane_src_counts,
            "median_lane_lat": (
                round(float(np.median(lane_lats)), 2)
                if lane_lats else None),
            "max_abs_lane_lat": (
                round(float(np.max(np.abs(lane_lats))), 2)
                if lane_lats else None),
            "centered_ratio": (
                round(float(np.mean(
                    np.abs(np.asarray(lane_lats, dtype=float)) <= 0.5)), 3)
                if lane_lats else None),
            "median_lane_w": (
                round(float(np.median(lane_widths)), 2)
                if lane_widths else None),
            "max_reverse_m": round(max_reverse, 2),
            "reverse_episodes": reverse_episodes,
            "blocked_events": blocked_events[-10:],
            "last": final,
        })
        telemetry_log.write_text(
            json.dumps(history, ensure_ascii=False, indent=1),
            encoding="utf-8")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(json.dumps({k: report[k] for k in (
            "run", "reason", "frames", "max_speed", "max_abs_steer",
            "driven_m", "median_lat", "max_abs_lat",
            "lane_frames", "max_black_frames", "lane_src",
            "median_lane_lat",
            "max_abs_lane_lat", "centered_ratio", "median_lane_w",
            "max_reverse_m", "reverse_episodes",
            "blocked_events")}, ensure_ascii=False, indent=2))
    finally:
        try:
            # The autopilot hands the car back in the player's gearbox mode,
            # usually arcade.  In arcade, brake at (nearly) zero speed is a
            # reverse request, so only brake while the box is realistic or
            # the car is still rolling fast enough for brake to mean brake.
            mode = None
            try:
                mode = read_gearbox_mode(conn.vehicle)
            except Exception:
                pass
            try:
                speed = float(conn.get_state().speed)
            except Exception:
                speed = 0.0
            brake = 1.0 if (mode == "realistic" or speed > 2.0) else 0.0
            conn.control(throttle=0.0, brake=brake, steering=0.0,
                         parkingbrake=1.0)
            conn.step(10)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10.0)
            except Exception:
                pass


if __name__ == "__main__":
    main()
