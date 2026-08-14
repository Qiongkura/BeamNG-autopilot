"""Live lane-state viewer: roadnet / body / vision overlay in a window.

Attaches to a running BeamNG session (Steam or Tech), grabs the front
camera continuously and shows the same diagnostic overlay as
``m5_lane_state_annotate.py`` in a standalone cv2 window.  The window is
persistent: if the game connection drops it keeps running and reconnects
automatically, so it can stay open across game restarts.

Keys: q/ESC quit, space pause, s save the current annotated frame,
r force a road-network refresh.

``--once`` renders a single frame and saves it under
``logs/m5_lane_state/`` without opening a window (used for smoke tests).
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
from beamng_autopilot.lane import pair_lane_markings
from beamng_autopilot.runtime import build_camera_provider, resolve_runtime
from beamng_autopilot.vision.lane_overlay import (
    ego_extents,
    estimate_pavement_edges,
    merge_boundary_geometry,
    render_lane_overlay,
    road_lane_geometry,
    unit_fwd,
)
from beamng_autopilot.vision.lanes import LaneDetector, MarkingSmoother

WINDOW_NAME = "BeamNG Lane State Live"
RECONNECT_DELAY_S = 2.0


def _build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Live lane-state viewer")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE)
    ap.add_argument("--rate", type=float, default=6.0,
                    help="maximum refresh rate in Hz (default 6)")
    ap.add_argument("--road-refresh", type=float, default=10.0,
                    help="road-network geometry refresh interval in s")
    ap.add_argument("--road-move", type=float, default=3.0,
                    help="refresh road geometry after this movement (m)")
    ap.add_argument("--once", action="store_true",
                    help="render one frame to logs and exit (no window)")
    ap.add_argument("--out", default=None,
                    help="output path for --once / snapshot")
    return ap.parse_args()


def _open_session(args) -> tuple:
    """Connect to the running game and build the sensor providers."""
    conn = BeamNGConnector(
        args.map, args.vehicle, port=args.port,
        home=config.runtime_home(args.runtime))
    try:
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        raise
    runtime_mode = resolve_runtime(conn, args.runtime)
    if runtime_mode == "steam":
        conn.set_front_camera()
    camera_provider, _ = build_camera_provider(conn, runtime_mode)
    detector = LaneDetector()
    smoother = MarkingSmoother()
    print(f"[view] runtime={runtime_mode}")
    return conn, camera_provider, detector, smoother


def _close_session(session) -> None:
    if session is None:
        return
    conn, camera_provider, _detector, _smoother = session
    if camera_provider is not None:
        try:
            camera_provider.close()
        except Exception:
            pass
    try:
        conn.close()
    except Exception:
        pass


def _render_frame(conn, camera_provider, detector, smoother, args, *,
                  geometry, geo_cache: dict, force_refresh: bool = False,
                  warmup: bool = False) -> dict:
    """Grab state, camera and sensors, then render one annotated frame."""
    st = conn.get_state()
    img = camera_provider.grab()
    h, w = img.shape[:2]
    fwd = unit_fwd(st)
    heading = float(st.heading)

    now = time.time()
    if geometry is None or force_refresh:
        geometry = road_lane_geometry(conn, st.pos, fwd)
        geo_cache["t"] = now
        geo_cache["pos"] = np.asarray(st.pos[:2], dtype=float).copy()
    elif (now - geo_cache["t"] >= args.road_refresh
          or float(np.hypot(*(np.asarray(st.pos[:2]) - geo_cache["pos"])))
          >= args.road_move):
        geometry = road_lane_geometry(conn, st.pos, fwd)
        geo_cache["t"] = now
        geo_cache["pos"] = np.asarray(st.pos[:2], dtype=float).copy()

    half_w = ego_extents(conn)[1]
    cam = camera_provider.camera_model(st.pos, heading, w, h)
    vision_geometry = estimate_pavement_edges(
        img, cam, st.pos, heading,
        ground_z=(float(st.pos[2]) if len(st.pos) > 2 else 0.0))
    geometry = merge_boundary_geometry(geometry, vision_geometry)
    raw_markings = detector.detect(
        img, cam, st.pos, heading,
        ground_z=(float(st.pos[2]) if len(st.pos) > 2 else 0.0))
    markings = smoother.update(
        raw_markings, cam, st.pos, heading,
        ground_z=(float(st.pos[2]) if len(st.pos) > 2 else 0.0),
        warmup=warmup, speed=float(st.speed))
    debug: dict = {}
    frame = pair_lane_markings(
        markings, st.pos, heading, fwd=st.dir, debug=debug)

    vision_text = (
        f"vision: {len(raw_markings)} raw / {len(markings)} stable "
        f"mode={debug.get('mode')} "
        f"paired={frame.paired if frame is not None else None}")
    overlay = render_lane_overlay(
        img, st, geometry, markings, cam, half_w, vision_text=vision_text)
    return {
        "state": st,
        "geometry": geometry,
        "vision": vision_geometry,
        "half_w": half_w,
        "markings": markings,
        "frame": frame,
        "debug": debug,
        "overlay": overlay,
    }


def _save_frame(overlay, args, tag: str) -> Path:
    out_dir = config.LOGS_DIR / "m5_lane_state"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (Path(args.out) if args.out
           else out_dir / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return out


def _print_summary(res: dict, saved: Path | None = None) -> None:
    st = res["state"]
    geometry = res["geometry"]
    half_w = res["half_w"]
    print(f"[view] pos=({st.pos[0]:.2f}, {st.pos[1]:.2f}, {st.pos[2]:.2f}) "
          f"heading={st.heading:.3f} speed={st.speed:.2f} m/s")
    if geometry is not None:
        print(f"  boundary: left {geometry['left_dist']:.2f} m "
              f"{geometry['source_left']} | "
              f"right {geometry['right_dist']:.2f} m "
              f"{geometry['source_right']} | "
              f"width {geometry['lane_width']:.2f} m")
        print(f"  center offset {geometry['center_offset']:+.2f} m | "
              f"body half-width {half_w:.2f} m")
        print(f"  body-edge clearances: left "
              f"{geometry['lane_left_lat'] - half_w:.2f} m | right "
              f"{-geometry['lane_right_lat'] - half_w:.2f} m")
    if res.get("vision") is not None:
        vis = res["vision"]
        print(f"  pavement vision: conf={vis['confidence']:.2f} "
              f"left={vis['left_lat']} right={vis['right_lat']} "
              f"Lconf={vis['left_confidence']} Rconf={vis['right_confidence']}")
    if saved is not None:
        print(f"[view] saved -> {saved}")


def _placeholder(text: str) -> np.ndarray:
    img = np.full((806, 1076, 3), (18, 20, 26), np.uint8)
    cv2.putText(img, "BeamNG Lane State Live", (60, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 200, 120), 2,
                cv2.LINE_AA)
    cv2.putText(img, text, (60, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(img, "waiting for the game to become reachable...",
                (60, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (150, 150, 160), 1, cv2.LINE_AA)
    return img


def _run_once(args) -> None:
    session = _open_session(args)
    try:
        geo_cache = {"t": 0.0, "pos": np.zeros(2, dtype=float)}
        res = _render_frame(
            session[0], session[1], session[2], session[3], args,
            geometry=None, geo_cache=geo_cache, warmup=True)
        saved = _save_frame(res["overlay"], args, "live_smoke")
        _print_summary(res, saved)
    finally:
        _close_session(session)


def main() -> None:
    args = _build_args()
    if args.once:
        _run_once(args)
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("[view] persistent window open - q/ESC quit, space pause, "
          "s save, r refresh")
    session = None
    geo_cache = {"t": 0.0, "pos": np.zeros(2, dtype=float)}
    geometry = None
    last_overlay = None
    paused = False
    status = "connecting"
    errors = 0
    t0 = time.time()
    fps_frames = 0
    fps = 0.0
    try:
        while True:
            frame_start = time.time()
            if session is None:
                try:
                    session = _open_session(args)
                    geo_cache = {"t": 0.0, "pos": np.zeros(2, dtype=float)}
                    geometry = None
                    status = "live"
                    errors = 0
                    print("[view] connected")
                except Exception as exc:
                    status = f"waiting for BeamNG ({type(exc).__name__})"
                    if errors % 10 == 0:
                        print(f"[view] {status}: {exc}")
                    errors += 1
                    time.sleep(RECONNECT_DELAY_S)
            elif not paused:
                try:
                    res = _render_frame(
                        session[0], session[1], session[2], session[3],
                        args,
                        geometry=geometry, geo_cache=geo_cache)
                    geometry = res["geometry"]
                    last_overlay = res["overlay"]
                    errors = 0
                    fps_frames += 1
                    now = time.time()
                    if now - t0 >= 1.0:
                        fps = fps_frames / (now - t0)
                        fps_frames = 0
                        t0 = now
                except Exception as exc:
                    print(f"[view] connection lost: {exc}")
                    _close_session(session)
                    session = None
                    status = "reconnecting"
                    continue

            if last_overlay is not None:
                display_rgb = last_overlay.copy()
                if status != "live":
                    hh, ww = display_rgb.shape[:2]
                    cv2.rectangle(display_rgb, (0, 0), (ww, 54), (0, 0, 0), -1)
                    cv2.putText(display_rgb, "DISCONNECTED - RECONNECTING",
                                (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (60, 60, 235), 2, cv2.LINE_AA)
                display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
            else:
                display_bgr = _placeholder(status)

            hh, ww = display_bgr.shape[:2]
            cv2.putText(display_bgr,
                        f"{fps:4.1f} fps" + (" | PAUSED" if paused else ""),
                        (ww - 180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, display_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                print(f"[view] {'paused' if paused else 'resumed'}")
            if key == ord("s") and last_overlay is not None:
                saved = _save_frame(last_overlay, args, "live_snapshot")
                print(f"[view] snapshot -> {saved}")
            if key == ord("r"):
                geo_cache["t"] = 0.0
                print("[view] forcing road-network refresh")

            try:
                if cv2.getWindowProperty(WINDOW_NAME,
                                         cv2.WND_PROP_VISIBLE) < 1:
                    print("[view] window closed")
                    break
            except cv2.error:
                break

            if session is not None and not paused and args.rate > 0:
                rem = 1.0 / args.rate - (time.time() - frame_start)
                if rem > 0:
                    time.sleep(rem)
            else:
                time.sleep(0.05)
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        _close_session(session)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[view] stopped")
        raise SystemExit(0)
    except Exception as exc:
        print(f"[view] fatal: {exc!r}")
        raise SystemExit(1)
