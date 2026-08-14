"""Annotate one Tech frame with road-network lanes, body width and markings.

Connects to a running BeamNG.tech session, grabs one front-camera frame,
queries the scenario road-network lane geometry and the ego bounding box,
runs the vision lane-marking detector, then saves an overlay image that
shows why road-network "distance to lane edge" can differ from the painted
lines the driver sees.  The overlay rendering lives in
``beamng_autopilot.vision.lane_overlay``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.lane import _boundary_near_lat, pair_lane_markings
from beamng_autopilot.runtime import build_camera_provider, resolve_runtime
from beamng_autopilot.vision.lane_overlay import (
    ego_extents,
    estimate_pavement_edges,
    merge_boundary_geometry,
    render_lane_overlay,
    road_lane_geometry,
    unit_fwd,
)
from beamng_autopilot.vision.lanes import LaneDetector


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate road-lane vs body geometry")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE)
    args = ap.parse_args()

    conn = BeamNGConnector(
        args.map, args.vehicle, port=args.port,
        home=config.runtime_home(args.runtime))
    camera_provider = None
    try:
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
    except Exception as exc:
        print(f"[annotate] cannot attach to a running session: {exc}")
        return

    runtime_mode = resolve_runtime(conn, args.runtime)
    if runtime_mode == "steam":
        conn.set_front_camera()
    camera_provider, _ = build_camera_provider(conn, runtime_mode)
    detector = LaneDetector()
    print(f"[annotate] runtime={runtime_mode}")

    try:
        st = conn.get_state()
        img = camera_provider.grab()
        h, w = img.shape[:2]
        fwd = unit_fwd(st)
        heading = float(st.heading)
        pos = st.pos[:2]

        roadnet_geometry = road_lane_geometry(conn, st.pos, fwd)
        half_w = ego_extents(conn)[1]
        cam = camera_provider.camera_model(st.pos, heading, w, h)
        vision_geometry = estimate_pavement_edges(
            img, cam, st.pos, heading,
            ground_z=(float(st.pos[2]) if len(st.pos) > 2 else 0.0))
        geometry = merge_boundary_geometry(
            roadnet_geometry, vision_geometry)
        markings = detector.detect(
            img, cam, st.pos, heading,
            ground_z=(float(st.pos[2]) if len(st.pos) > 2 else 0.0))
        debug: dict = {}
        frame = pair_lane_markings(
            markings, st.pos, heading, fwd=st.dir, debug=debug)

        vision_text = (
            f"vision: {len(markings)} markings mode={debug.get('mode')} "
            f"paired={frame.paired if frame is not None else None}")
        overlay = render_lane_overlay(
            img, st, geometry, markings, cam, half_w, vision_text=vision_text)

        out_dir = config.LOGS_DIR / "m5_lane_state"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = (Path(args.out) if args.out
               else out_dir / f"lane_annotate_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"[annotate] saved -> {out}")

        if frame is not None:
            if frame.left is not None:
                print(f"  vision left = "
                      f"{_boundary_near_lat(frame.left, pos, heading, fwd=fwd)}")
            if frame.right is not None:
                rl = _boundary_near_lat(frame.right, pos, heading, fwd=fwd)
                print(f"  vision right = {None if rl is None else -float(rl)}")
        if vision_geometry is not None:
            print(f"  pavement vision: conf={vision_geometry['confidence']:.2f} "
                  f"left={vision_geometry['left_lat']} "
                  f"right={vision_geometry['right_lat']} "
                  f"Lconf={vision_geometry['left_confidence']} "
                  f"Rconf={vision_geometry['right_confidence']}")
        if geometry is not None:
            print(f"  boundary: left {geometry['left_dist']:.2f} m "
                  f"{geometry['source_left']} | "
                  f"right {geometry['right_dist']:.2f} m "
                  f"{geometry['source_right']} | "
                  f"width {geometry['lane_width']:.2f} m")
    finally:
        if camera_provider is not None:
            try:
                camera_provider.close()
            except Exception:
                pass
        conn.close()


if __name__ == "__main__":
    main()
