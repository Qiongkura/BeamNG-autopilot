"""M5 vision obstacle probe: show what the YOLO channel currently sees.

Attaches to a running BeamNG session (like ``m5_autopilot --attach``),
grabs the game window every ~0.3 s, runs YOLOv8n and prints every detected
car / truck / bus / motorcycle / person with its world position, distance
and a 2D box.  With ``--save`` annotated frames are written to
``logs/m5_vision/`` so you can check the detector without staring at the
terminal.  Useful to validate the camera back-projection before engaging
autopilot.

Exit: Ctrl+C, or hold ``q`` in the probe window when ``--show`` is used.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.runtime import build_camera_provider, resolve_runtime


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 vision obstacle probe")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--rate", type=float, default=3.0,
                    help="probe rate in Hz (default 3)")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="YOLO confidence threshold")
    ap.add_argument("--max-dist", type=float, default=55.0,
                    help="ignore detections farther than this (m)")
    ap.add_argument("--once", action="store_true",
                    help="run a single frame then exit")
    ap.add_argument("--save", action="store_true",
                    help="save annotated frames to logs/m5_vision/")
    ap.add_argument("--show", action="store_true",
                    help="show the annotated frame in a window (q to quit)")
    ap.add_argument("--no-lanes", action="store_true",
                    help="skip lane-marking detection")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    from beamng_autopilot.vision.detection import (
        VisionDetector,
    )
    from beamng_autopilot.vision.lanes import LaneDetector

    conn = BeamNGConnector(
        args.map, args.vehicle, port=args.port,
        home=config.runtime_home(args.runtime))
    camera_provider = None
    try:
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
    except Exception as exc:
        print(f"[probe] cannot attach to a running session: {exc}")
        print("[probe] start the game and enter a map first, then rerun.")
        return
    runtime_mode = resolve_runtime(conn, args.runtime)
    if runtime_mode == "steam":
        conn.set_front_camera()
    camera_provider, _ = build_camera_provider(conn, runtime_mode)
    print(f"[probe] runtime={runtime_mode}")

    detector = VisionDetector(conf=args.conf, max_dist=args.max_dist)
    lane_det = None if args.no_lanes else LaneDetector()
    print("[probe] vision detector ready; Ctrl+C to stop")
    out_dir = config.LOGS_DIR / "m5_vision"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_next = 0.0
    frame_i = 0
    try:
        while True:
            now = time.time()
            if now < t_next:
                time.sleep(min(0.05, t_next - now))
                continue
            t_next = now + 1.0 / max(1.0, args.rate)
            frame_i += 1

            st = conn.get_state()
            img = camera_provider.grab()
            h, w = img.shape[:2]
            vmodel = camera_provider.camera_model(
                st.pos, st.heading, w, h)
            t0 = time.time()
            obstacles, boxes = detector.detect(
                img, vmodel, st.pos, st.heading)
            ms = (time.time() - t0) * 1000.0

            print(f"[probe] frame {frame_i} {w}x{h} in {ms:.0f} ms: "
                  f"{len(obstacles)} obstacle(s)")
            for ob in obstacles:
                d = float(((ob.x - st.pos[0]) ** 2
                           + (ob.y - st.pos[1]) ** 2) ** 0.5)
                print(f"  - {ob.label or ob.category:9s} at "
                      f"({ob.x:7.1f}, {ob.y:7.1f})  dist {d:5.1f} m  "
                      f"box {ob.half_w*2:.1f} x {ob.half_h*2:.1f} m")

            lanes = []
            if lane_det is not None:
                lanes = lane_det.detect(
                    img, vmodel, st.pos, st.heading,
                    ground_z=(float(st.pos[2]) - config.EGO_ORIGIN_GROUND_GAP_M
                              if len(st.pos) > 2 else 0.0))
                print(f"[probe] {len(lanes)} lane marking(s)")
                for mk in lanes:
                    pts = np.asarray(mk.world, dtype=float)
                    world_len = float(np.sum(np.linalg.norm(
                        np.diff(pts, axis=0), axis=1))) if len(pts) > 1 \
                        else 0.0
                    print(f"  - {mk.color:6s} {mk.kind:7s} conf {mk.confidence:.2f} "
                          f"len {world_len:.1f} m")

            if args.save or args.show:
                for x1, y1, x2, y2, label, conf in boxes:
                    cv2.rectangle(img, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 200, 255), 2)
                    cv2.putText(img, f"{label} {conf:.2f}",
                                (int(x1), max(16, int(y1) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 200, 255), 2)
                for mk in lanes:
                    pix = np.asarray(mk.pixels, dtype=np.int32)
                    color = (0, 255, 255) if mk.color == "yellow" \
                        else (255, 255, 255)
                    cv2.polylines(
                        img, [pix], False, color,
                        3 if mk.kind == "solid" else 2)
                    if len(pix):
                        cv2.putText(img, f"{mk.kind} {mk.confidence:.2f}",
                                    (int(pix[0, 0]), max(16, int(pix[0, 1]) - 3)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    color, 2)
                if args.save:
                    p = out_dir / f"vision_{time.strftime('%H%M%S')}_{frame_i}.jpg"
                    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if args.show:
                    cv2.imshow("m5 vision probe", cv2.cvtColor(
                        img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if args.once:
                break
    except KeyboardInterrupt:
        print("[probe] stopped")
    finally:
        if args.show:
            cv2.destroyAllWindows()
        if camera_provider is not None:
            try:
                camera_provider.close()
            except Exception:
                pass
        conn.close()


if __name__ == "__main__":
    main()
