"""M2 camera calibration: query the in-game camera pose/FOV via Lua while the
vehicle sits at known positions, and save frames so we can build a projection
model that maps world track points into the grabbed image."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.runtime import build_camera_provider, resolve_runtime
from beamng_autopilot.track import load_track


LUA_QUERY = (
    "local p=getCameraPosition(); local f=getCameraForward(); local u=getCameraUp(); "
    "return string.format('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f', "
    "p.x, p.y, p.z, f.x, f.y, f.z, u.x, u.y, u.z, getCameraFovDeg())"
)


def query_camera(conn: BeamNGConnector, camera_provider, runtime_mode: str,
                 st, width: int, height: int):
    if runtime_mode == "tech":
        model = camera_provider.camera_model(st.pos, st.heading, width, height)
        cam_pos, _, cam_fwd, cam_up = model.camera_pose(st.pos, st.heading)
        return {
            "cam_pos": [float(v) for v in cam_pos],
            "cam_fwd": [float(v) for v in cam_fwd],
            "cam_up": [float(v) for v in cam_up],
            "fov_deg": float(model.fov_deg),
        }
    resp = conn.bng.queue_lua_command(LUA_QUERY, response=True)
    if not resp:
        raise RuntimeError(f"empty Lua response: {resp!r}")
    vals = [float(v) for v in str(resp).split(",")]
    if len(vals) != 10:
        raise RuntimeError(f"unexpected Lua response: {resp!r}")
    return {
        "cam_pos": vals[0:3],
        "cam_fwd": vals[3:6],
        "cam_up": vals[6:9],
        "fov_deg": vals[9],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--track", required=True)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--drive", type=float, default=12.0,
                    help="drive straight this many metres before the 2nd sample")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    points, headings = load_track(args.track)
    start = points[0]
    heading0 = float(headings[0])

    out_dir = config.LOGS_DIR / "m2_calib"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with BeamNGConnector(
            args.map, args.vehicle, port=args.port,
            home=config.runtime_home(args.runtime)) as conn:
        conn.load_scenario(spawn_pos=(float(start[0]), float(start[1]), 0.0),
                           spawn_heading=heading0)
        runtime_mode = resolve_runtime(conn, args.runtime)
        if runtime_mode == "steam":
            conn.set_front_camera()
        camera_provider, _ = build_camera_provider(conn, runtime_mode)
        print("[calib] scenario started; settling camera")
        for _ in range(30):
            conn.step(1)
        time.sleep(0.5)

        def record(tag: str) -> None:
            img = camera_provider.grab()
            h, w = img.shape[:2]
            st = conn.get_state()
            cam = query_camera(conn, camera_provider, runtime_mode, st, w, h)
            sample = {
                "tag": tag,
                "veh_pos": [float(st.pos[0]), float(st.pos[1]), float(st.pos[2])],
                "veh_heading": float(st.heading),
                "cam": cam,
                "win": [w, h],
            }
            samples.append(sample)
            path = out_dir / f"calib_{tag}_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"[calib] {tag}: cam_pos={cam['cam_pos']} fov={cam['fov_deg']:.1f} "
                  f"veh=({st.pos[0]:.1f},{st.pos[1]:.1f}) h={st.heading:.3f} win={w}x{h} -> {path}")

        record("a_start")

        # drive straight ahead a little to get a second, different pose
        t0 = time.time()
        while time.time() - t0 < 15.0:
            st = conn.get_state()
            if st.speed < 0.5:
                throttle = 0.35
            else:
                throttle = 0.18
            conn.control(throttle=throttle, steering=0.0, brake=0.0)
            conn.step(1)
            if st.pos[1] > float(start[1]) + args.drive or st.pos[1] < float(start[1]) - args.drive:
                break
        conn.control(throttle=0.0, brake=1.0)
        for _ in range(10):
            conn.step(1)
        record("b_after_drive")
        camera_provider.close()

    json_path = out_dir / "camera_calib.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=1)
    print(f"[calib] done -> {json_path}")


if __name__ == "__main__":
    main()
