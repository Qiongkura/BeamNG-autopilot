"""M3 data collection for behavioural cloning.

Drives the car with the Pure Pursuit expert while capturing camera frames at
maximum rate, paired with the exact steering/throttle/brake commands.  The
frame is grabbed BEFORE the command is computed so that image and label refer
to the same vehicle state (DAVE-2 style alignment).

Output layout (one run dir under logs/m3_bc):
    frames/frame_00000.jpg ...     downscaled camera frames (RGB jpg)
    meta.jsonl                     one JSON object per line
"""

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
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--port", type=int, default=None,
                    help="game port (default: resolved from --runtime)")
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--resize", default="320x180",
                    help="downscale frames to WxH before saving")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after this many frames (0 = unlimited)")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    ap.add_argument("--tech-annot", action="store_true",
                    help="Tech only: enable annotation rendering and drop "
                         "frames whose road share is below --min-road-share")
    ap.add_argument("--min-road-share", type=float, default=0.15,
                    help="min road pixel share in the lower frame for a "
                         "frame to be kept (with --tech-annot)")
    args = ap.parse_args()

    w, h = (int(x) for x in args.resize.lower().split("x"))
    points, headings = load_track(args.track)
    n = len(points)
    start = points[0]
    heading0 = float(headings[0])

    pp = PurePursuit(lookahead=6.0)
    pid_throttle = PID(kp=0.55, ki=0.15, kd=0.06, output_limits=(0.0, 1.0))

    out_root = config.LOGS_DIR / "m3_bc"
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / time.strftime("%Y%m%d_%H%M%S")
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "meta.jsonl"
    meta_file = open(meta_path, "w", encoding="utf-8")
    telemetry = TelemetryBroadcaster()

    with BeamNGConnector(
            args.map, args.vehicle,
            port=(args.port or config.runtime_port(args.runtime)),
            home=config.runtime_home(args.runtime)) as conn:
        conn.load_scenario(spawn_pos=(float(start[0]), float(start[1]), 0.0),
                           spawn_heading=heading0)
        runtime_mode = resolve_runtime(conn, args.runtime)
        if runtime_mode == "steam":
            conn.set_front_camera()
        use_annot = bool(args.tech_annot and runtime_mode == "tech")
        if args.tech_annot and runtime_mode != "tech":
            print("[bc-collect] WARNING: --tech-annot requires the tech "
                  "runtime; falling back to plain frames")
        camera_provider, _ = build_camera_provider(
            conn, runtime_mode, annotations=use_annot)
        if use_annot:
            from beamng_autopilot_tech.annotations import road_share
        print(f"[bc-collect] scenario started; saving to {run_dir} "
              f"(annot={use_annot})")

        nearest, prev_nearest, lap, t0 = 0, 0, 0, time.time()
        last_status = 0.0
        stalled_since = None
        idx = 0
        dropped = 0

        while time.time() - t0 < args.duration and lap < args.laps:
            st = conn.get_state()
            speed = st.speed

            # grab the view BEFORE computing the command -> aligned (image, label)
            try:
                if use_annot:
                    img, ann = camera_provider.grab_annotated()
                else:
                    img = camera_provider.grab()
                    ann = None
            except Exception as exc:
                print(f"[bc-collect] frame error: {exc}")
                conn.step(1)
                continue

            pp.lookahead = pp.adaptive_lookahead(speed)
            steer_rad, _, nearest = pp.steering(st.pos, st.heading, points, nearest)
            if prev_nearest > 0.7 * n and nearest < 0.3 * n:
                lap += 1
                print(f"[bc-collect] lap {lap} done at t={time.time() - t0:.1f}s")
            prev_nearest = nearest
            steer = float(np.clip(-steer_rad / 0.6, -1.0, 1.0))
            throttle = pid_throttle.update(args.speed - speed)
            brake = 0.0
            if speed > args.speed + 1.5:
                brake = min(0.7, (speed - args.speed) / 8.0)
                throttle = 0.0
            if speed < 0.5:
                throttle = max(throttle, 0.6)

            telemetry.publish(
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

            small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            road = None
            if ann is not None:
                # Annotation-based quality gate: an off-road / black frame
                # (low road share) teaches the network the wrong label, so
                # it is dropped at the source instead of poisoning the run.
                road = float(road_share(ann))
                if road < args.min_road_share:
                    dropped += 1
                    conn.control(throttle=throttle, steering=steer, brake=brake)
                    conn.step(1)
                    continue
            cv2.imwrite(str(run_dir / "frames" / f"frame_{idx:05d}.jpg"),
                        cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            meta_file.write(json.dumps({
                "idx": idx,
                "t": round(time.time() - t0, 4),
                "steer": round(steer, 4),
                "throttle": round(throttle, 4),
                "brake": round(brake, 4),
                "speed": round(float(speed), 4),
                "pos": [round(float(v), 3) for v in st.pos],
                "heading": round(float(st.heading), 4),
                "nearest": int(nearest),
                "lap": int(lap),
                "road_share": None if road is None else round(road, 4),
            }) + "\n")
            idx += 1
            if args.max_frames and idx >= args.max_frames:
                break

            conn.control(throttle=throttle, steering=steer, brake=brake)
            conn.step(1)

            if idx % 10 == 0 and time.time() - last_status > 2.0:
                last_status = time.time()
                fps = idx / (time.time() - t0)
                print(f"[bc-collect] frame {idx}  {fps:.1f} fps  "
                      f"steer={steer:+.2f}  v={speed:.1f} m/s")

            if speed < 0.5:
                stalled_since = stalled_since if stalled_since is not None else time.time()
                if time.time() - stalled_since > 5.0:
                    print("[bc-collect] stalled, nudging")
                    conn.control(throttle=0.9, steering=steer, brake=0.0)
                    conn.step(30)
                    stalled_since = None
            else:
                stalled_since = None

        conn.control(throttle=0.0, brake=1.0)
        conn.step(30)
        camera_provider.close()

    meta_file.close()
    telemetry.close()
    dt = time.time() - t0
    print(f"[bc-collect] done: {idx} frames ({dropped} dropped by quality "
          f"gate), {lap} laps, {idx / dt:.1f} fps -> {run_dir}")


if __name__ == "__main__":
    main()
