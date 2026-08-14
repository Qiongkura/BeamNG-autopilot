"""M2 capture: drive along the recorded track while saving camera frames
plus ground-truth state (pos/heading/speed/nearest index) for offline
calibration of the vision pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control.pid import PID
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.runtime import build_camera_provider, resolve_runtime
from beamng_autopilot.telemetry import TelemetryBroadcaster
from beamng_autopilot.track import load_track


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--track", required=True)
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--interval", type=float, default=0.4,
                    help="save a frame every N seconds")
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    points, headings = load_track(args.track)
    n = len(points)
    start = points[0]
    heading0 = float(headings[0])

    pp = PurePursuit(lookahead=6.0)
    pid_throttle = PID(kp=0.55, ki=0.15, kd=0.06, output_limits=(0.0, 1.0))

    out_dir = config.LOGS_DIR / "m2_capture"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    live_telemetry = TelemetryBroadcaster()

    with BeamNGConnector(
            args.map, args.vehicle, port=args.port,
            home=config.runtime_home(args.runtime)) as conn:
        conn.load_scenario(spawn_pos=(float(start[0]), float(start[1]), 0.0),
                           spawn_heading=heading0)
        runtime_mode = resolve_runtime(conn, args.runtime)
        if runtime_mode == "steam":
            conn.set_front_camera()
        camera_provider, _ = build_camera_provider(conn, runtime_mode)
        print(f"[capture] scenario started; saving to {run_dir}")

        nearest, lap, t0 = 0, 0, time.time()
        last_save = 0.0
        stalled_since = None

        while time.time() - t0 < args.duration and lap < args.laps:
            st = conn.get_state()
            speed = st.speed

            pp.lookahead = pp.adaptive_lookahead(speed)
            steer_rad, _, nearest = pp.steering(st.pos, st.heading, points, nearest)
            steer = float(np.clip(-steer_rad / 0.6, -1.0, 1.0))
            throttle = pid_throttle.update(args.speed - speed)
            brake = 0.0
            if speed > args.speed + 1.5:
                brake = min(0.7, (speed - args.speed) / 8.0)
                throttle = 0.0
            if speed < 0.5:
                throttle = max(throttle, 0.6)

            live_telemetry.publish(
                t=time.time() - t0,
                speed=float(speed),
                throttle=throttle,
                brake=brake,
                steer=steer,
                vel=st.vel,
                dir_vec=st.dir,
                up_vec=st.up,
                pos=st.pos,
                heading=float(st.heading),
                lap=lap,
                nearest=int(nearest),
            )

            conn.control(throttle=throttle, steering=steer, brake=brake)
            conn.step(1)

            if len(meta) > 0 and meta[-1]["nearest"] > 0.7 * n and nearest < 0.3 * n:
                lap += 1
                print(f"[capture] lap {lap} done at t={time.time() - t0:.1f}s")

            now = time.time()
            if now - last_save >= args.interval:
                last_save = now
                try:
                    img = camera_provider.grab()
                except Exception as exc:
                    print(f"[capture] frame error: {exc}")
                    continue
                idx = len(meta)
                cv2.imwrite(str(run_dir / f"frame_{idx:05d}.jpg"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                meta.append({
                    "idx": idx,
                    "t": now - t0,
                    "pos": [float(st.pos[0]), float(st.pos[1]), float(st.pos[2])],
                    "heading": float(st.heading),
                    "speed": float(speed),
                    "steer": float(steer),
                    "nearest": int(nearest),
                    "lap": int(lap),
                })

            if speed < 0.5:
                stalled_since = stalled_since if stalled_since is not None else now
                if now - stalled_since > 5.0:
                    print("[capture] stalled, nudging")
                    conn.control(throttle=0.9, steering=steer, brake=0.0)
                    conn.step(30)
                    stalled_since = None
            else:
                stalled_since = None

        conn.control(throttle=0.0, brake=1.0)
        conn.step(30)
        camera_provider.close()

    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    live_telemetry.close()
    print(f"[capture] done: {len(meta)} frames, {lap} laps -> {run_dir}")


if __name__ == "__main__":
    main()
