"""Read-only probe: current vehicle lateral position inside the lane.

Attaches to a running BeamNG session, grabs one front-camera frame, runs
the lane-marking detector and prints how far the ego is from the left /
right boundary and from the lane centre.  Nothing is written unless
``--save`` is used (annotated frame under logs/m5_lane_state/).
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
from beamng_autopilot.lane import (
    _boundary_near_lat,
    pair_lane_markings,
)
from beamng_autopilot.runtime import build_camera_provider, resolve_runtime
from beamng_autopilot.vision.lanes import LaneDetector


def _lat(world, pos, fwd):
    """Signed lateral offset: positive when the point is to the left."""
    left = np.array([-fwd[1], fwd[0]])
    return float((np.asarray(world, dtype=float)[:2] - pos) @ left)


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 lane-state probe")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--frames", type=int, default=3,
                    help="probe this many frames (default 3)")
    ap.add_argument("--save", action="store_true",
                    help="save the last annotated frame")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    conn = BeamNGConnector(
        args.map, args.vehicle, port=args.port,
        home=config.runtime_home(args.runtime))
    camera_provider = None
    try:
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
    except Exception as exc:
        print(f"[probe] cannot attach to a running session: {exc}")
        return

    runtime_mode = resolve_runtime(conn, args.runtime)
    if runtime_mode == "steam":
        conn.set_front_camera()
    camera_provider, _ = build_camera_provider(conn, runtime_mode)
    detector = LaneDetector()
    print(f"[probe] runtime={runtime_mode}")

    results: list[dict] = []
    last_img = None
    try:
        for i in range(args.frames):
            st = conn.get_state()
            img = camera_provider.grab()
            h, w = img.shape[:2]
            vmodel = camera_provider.camera_model(
                st.pos, st.heading, w, h)
            markings = detector.detect(
                img, vmodel, st.pos, st.heading,
                ground_z=(float(st.pos[2]) if len(st.pos) > 2 else 0.0))
            debug: dict = {}
            frame = pair_lane_markings(
                markings, st.pos, st.heading, fwd=st.dir, debug=debug)

            fwd = np.asarray(st.dir[:2], dtype=float)
            fn = float(np.linalg.norm(fwd))
            if fn > 1e-9:
                fwd = fwd / fn
            else:
                fwd = np.array(
                    [np.cos(st.heading), np.sin(st.heading)])
            pos = np.asarray(st.pos[:2], dtype=float)

            center_offset = None
            left_dist = None
            right_dist = None
            width = None
            conf = None
            span = None
            paired = None
            sources = None
            if frame is not None:
                center_offset = _lat(frame.center[0], pos, fwd)
                width = float(frame.width)
                conf = float(frame.confidence)
                span = float(frame.span_m)
                paired = bool(frame.paired)
                sources = tuple(frame.sources)
                if frame.left is not None:
                    left_dist = _boundary_near_lat(
                        frame.left, pos, st.heading, fwd=fwd)
                if frame.right is not None:
                    rl = _boundary_near_lat(
                        frame.right, pos, st.heading, fwd=fwd)
                    if rl is not None:
                        right_dist = -rl

            res = {
                "i": i,
                "speed": float(st.speed),
                "pos": [round(float(v), 3) for v in st.pos],
                "heading": round(float(st.heading), 4),
                "n_markings": len(markings),
                "mode": debug.get("mode"),
                "center_offset": center_offset,
                "left_dist": left_dist,
                "right_dist": right_dist,
                "width": width,
                "conf": conf,
                "span": span,
                "paired": paired,
                "sources": sources,
                "markings": [
                    {
                        "color": m.color,
                        "kind": m.kind,
                        "conf": round(float(m.confidence), 2),
                    }
                    for m in markings
                ],
            }
            results.append(res)
            last_img = img

            print(f"[probe] frame {i}: v={st.speed:.2f} m/s "
                  f"pos=({st.pos[0]:.1f},{st.pos[1]:.1f},{st.pos[2]:.1f}) "
                  f"heading={st.heading:.3f}")
            print(f"  markings={len(markings)}  "
                  f"lane={debug.get('mode')}  "
                  f"conf={conf if conf is None else round(conf, 2)}  "
                  f"width={width if width is None else round(width, 2)} m  "
                  f"span={span if span is None else round(span, 1)} m  "
                  f"paired={paired}  sources={sources}")
            if center_offset is not None:
                print(f"  center_offset={center_offset:+.2f} m (left +)  "
                      f"left={left_dist if left_dist is None else round(left_dist, 2)} m  "
                      f"right={right_dist if right_dist is None else round(right_dist, 2)} m")
            elif debug:
                print(f"  debug mode={debug.get('mode')}  "
                      f"cands={len(debug.get('cands', []))}  "
                      f"axes={len(debug.get('axes', []))}")
                for c in debug.get("cands", []):
                    print(f"    cand {c['kind']:6s} {c['color']:6s} "
                          f"lat={c['med_lat']:+.2f} span={c['span']:.1f} "
                          f"conf={c['conf']:.2f} score={c['score']:.3f} "
                          f"start={c['start_s']:.1f} "
                          f"near_n={c['near_n']} "
                          f"near_med={c['near_med']}")
            for m in markings:
                pts = np.asarray(m.world, dtype=float)
                world_len = float(np.sum(np.linalg.norm(
                    np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0
                print(f"  - {m.color:6s} {m.kind:7s} conf {m.confidence:.2f} "
                      f"len {world_len:.1f} m")
            if i + 1 < args.frames:
                time.sleep(0.15)
    finally:
        if args.save and last_img is not None and results:
            out_dir = config.LOGS_DIR / "m5_lane_state"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Draw the detected markings over the raw frame for a quick
            # visual check; this does not touch the M3 capture pipeline.
            overlay = last_img.copy()
            st = conn.get_state()
            h, w = overlay.shape[:2]
            vmodel = camera_provider.camera_model(
                st.pos, st.heading, w, h)
            markings = detector.detect(
                overlay, vmodel, st.pos, st.heading,
                ground_z=(float(st.pos[2]) if len(st.pos) > 2 else 0.0))
            for mk in markings:
                pix = np.asarray(mk.pixels, dtype=np.int32)
                color = (0, 255, 255) if mk.color == "yellow" \
                    else (255, 255, 255)
                cv2.polylines(overlay, [pix], False, color,
                              3 if mk.kind == "solid" else 2)
                if len(pix):
                    cv2.putText(overlay, f"{mk.kind} {mk.confidence:.2f}",
                                (int(pix[0, 0]), max(16, int(pix[0, 1]) - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            p = out_dir / f"lane_state_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"[probe] annotated frame -> {p}")
        if camera_provider is not None:
            try:
                camera_provider.close()
            except Exception:
                pass
        conn.close()


if __name__ == "__main__":
    main()
