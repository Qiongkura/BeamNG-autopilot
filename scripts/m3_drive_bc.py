"""M3 drive with a trained behavioural-cloning steering model.

Loads a checkpoint saved by m3_train_bc.py, grabs the front camera view each
loop, predicts the steering command, and keeps the target speed with a PID.
Telemetry is streamed to a CSV under logs/m3_drive/<timestamp>.csv.

Usage:
    python scripts/m3_drive_bc.py --model logs/m3_bc/bc_steer.pt \
        --track data/track_smallgrid.npz --speed 8.0 --duration 180
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.bc import Dave2, conv_feature_size, preprocess_frame
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control.pid import PID
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.hud import LiveHUD
from beamng_autopilot.runtime import build_camera_provider, resolve_runtime
from beamng_autopilot.telemetry import TelemetryBroadcaster
from beamng_autopilot.track import load_track


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--track", default=None,
                    help="track npz for lap/ref logging (optional in --attach mode)")
    ap.add_argument("--attach", action="store_true",
                    help="drive the vehicle already present in a running "
                         "BeamNG session (no scenario load)")
    ap.add_argument("--attach-vid", default=None,
                    help="vehicle id to attach to (default: first active vehicle)")
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--no-hud", action="store_true",
                    help="disable the live telemetry HUD window")
    ap.add_argument("--no-camera", action="store_true",
                    help="hide the front-camera preview in the HUD")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    w, h = ckpt.get("resize", (200, 66))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dave2(feat_in=conv_feature_size(h, w))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"[drive] model {args.model}  input={w}x{h}  device={device}")

    points, headings = (None, None)
    if args.track:
        points, headings = load_track(args.track)
    n = len(points) if points is not None else 0
    start = points[0] if points is not None else None
    heading0 = float(headings[0]) if headings is not None else 0.0
    pid_throttle = PID(kp=0.55, ki=0.15, kd=0.06, output_limits=(0.0, 1.0))
    pp_ref = PurePursuit(lookahead=6.0)

    out_dir = config.LOGS_DIR / "m3_drive"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / time.strftime("%Y%m%d_%H%M%S.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    telemetry = TelemetryBroadcaster()
    hud = None if args.no_hud else LiveHUD(show_camera=not args.no_camera)
    writer = csv.writer(csv_file)
    writer.writerow(["t", "steer_pred", "steer_ref", "throttle", "brake",
                     "speed", "pos_x", "pos_y", "heading", "nearest", "lap"])

    with BeamNGConnector(
            args.map, args.vehicle, port=args.port,
            home=config.runtime_home(args.runtime)) as conn:
        if args.attach:
            conn.attach_vehicle(vid=args.attach_vid)
            print(f"[drive] attached mode; target {args.speed} m/s, "
                  f"max {args.duration}s")
        else:
            conn.load_scenario(spawn_pos=(float(start[0]), float(start[1]), 0.0),
                               spawn_heading=heading0)
            print(f"[drive] scenario started; target {args.speed} m/s, "
                  f"max {args.duration}s / {args.laps} laps")
        runtime_mode = resolve_runtime(conn, args.runtime)
        if runtime_mode == "steam":
            conn.set_front_camera()
        camera_provider, _ = build_camera_provider(conn, runtime_mode)

        nearest, prev_nearest, lap, t0 = 0, 0, 0, time.time()
        stalled_since = None

        with torch.no_grad():
            while time.time() - t0 < args.duration and lap < args.laps:
                st = conn.get_state()
                speed = st.speed
                if points is not None:
                    nearest = int(np.argmin(
                        np.linalg.norm(points[:, :2] - np.asarray(st.pos)[:2],
                                       axis=1)))
                    if prev_nearest > 0.7 * n and nearest < 0.3 * n:
                        lap += 1
                        print(f"[drive] lap {lap} done at t={time.time() - t0:.1f}s")
                    prev_nearest = nearest
                img = camera_provider.grab()

                tensor = preprocess_frame(img, w, h).to(device)
                steer = float(model(tensor).cpu().numpy().ravel()[0])
                steer = float(np.clip(steer, -1.0, 1.0))

                steer_ref = 0.0
                if points is not None:
                    pp_ref.lookahead = pp_ref.adaptive_lookahead(speed)
                    steer_ref_rad, _, _ = pp_ref.steering(
                        st.pos, st.heading, points, nearest)
                    steer_ref = float(np.clip(-steer_ref_rad / 0.6, -1.0, 1.0))

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
                    nearest=nearest,
                    extra={"ref": f"{steer_ref:+.2f}"},
                )

                if hud is not None:
                    if not hud.update(telemetry.latest, cam=img):
                        print("[drive] HUD closed, stopping")
                        break

                conn.control(throttle=throttle, steering=steer, brake=brake)
                conn.step(1)

                writer.writerow([round(time.time() - t0, 3), f"{steer:.4f}",
                                 f"{steer_ref:.4f}", f"{throttle:.4f}", f"{brake:.4f}",
                                 f"{speed:.3f}", f"{st.pos[0]:.3f}",
                                 f"{st.pos[1]:.3f}", f"{st.heading:.4f}",
                                 nearest, lap])

                if speed < 0.5:
                    stalled_since = stalled_since if stalled_since is not None else time.time()
                    if time.time() - stalled_since > 5.0:
                        print("[drive] stalled, nudging")
                        conn.control(throttle=0.9, steering=steer, brake=0.0)
                        conn.step(30)
                        stalled_since = None
                else:
                    stalled_since = None

        conn.control(throttle=0.0, brake=1.0)
        conn.step(30)
        camera_provider.close()

    csv_file.close()
    telemetry.close()
    if hud is not None:
        hud.close()
    print(f"[drive] done -> {csv_path}")


if __name__ == "__main__":
    main()
