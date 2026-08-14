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

Pipeline (since the async rework): a background perception thread does all
the heavy work (grab, road-network geometry on a low cadence, pavement
edges, marking detection, smoothing, pairing) and publishes the latest
snapshot; the main loop only renders the newest snapshot at a fixed,
smooth frame rate.  Connection-level failures are detected through a stale
snapshot and trigger a reconnect; a failed single grab just skips a frame.
"""

from __future__ import annotations

import argparse
import sys
import threading
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
    draw_lane_frame,
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
SNAPSHOT_STALE_S = 3.0  # 快照超过此时长未更新 -> 视为断线


def _build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Live lane-state viewer")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--port", type=int, default=None,
                    help="comms port (default: per-runtime - Steam 64256 / "
                         "Tech 64257)")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE)
    ap.add_argument("--rate", type=float, default=6.0,
                    help="perception scan rate in Hz (default 6)")
    ap.add_argument("--display-rate", type=float, default=20.0,
                    help="render frame rate in Hz (default 20, smooth)")
    ap.add_argument("--road-refresh", type=float, default=30.0,
                    help="road-network geometry refresh interval in s "
                         "(each refresh is ~1.5s, runs in the worker)")
    ap.add_argument("--road-move", type=float, default=10.0,
                    help="refresh road geometry after this movement (m)")
    ap.add_argument("--once", action="store_true",
                    help="render one frame to logs and exit (no window)")
    ap.add_argument("--out", default=None,
                    help="output path for --once / snapshot")
    ap.add_argument("--seg-model", default=None,
                    help="learned segmentation model path (default: "
                         "logs/m5_seg/seg_model/best.pt when present; "
                         "without it classic CV is used)")
    ap.add_argument("--no-seg", action="store_true",
                    help="force classic CV detection (skip the learned "
                         "segmentation model)")
    return ap.parse_args()


def _build_segmenter(args):
    """Load the learned segmenter, or None to keep classic CV."""
    if args.no_seg:
        return None
    if args.seg_model is None:
        try:
            from beamng_autopilot.vision.segmentation \
                import default_model_path
            args.seg_model = default_model_path()
        except Exception:
            args.seg_model = None
    if args.seg_model is None:
        return None
    try:
        from beamng_autopilot.vision.segmentation import Segmenter

        seg = Segmenter(model_path=args.seg_model)
        print(f"[view] 学习式分割: {args.seg_model}")
        return seg
    except Exception as exc:
        print(f"[view] WARNING: 分割模型加载失败（回退 CV）: {exc}")
        return None


def _open_session(args) -> tuple:
    """Connect to the running game and build the sensor providers."""
    conn = BeamNGConnector(
        args.map, args.vehicle,
        port=(args.port or config.runtime_port(args.runtime)),
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
    # 显示用途用宽松平滑参数（2 帧确认、匹配距离放宽），车道框架更连续；
    # autopilot 的实例保持保守参数不受影响。
    smoother = MarkingSmoother(min_frames=2, match_max_m=1.5,
                               stale_s=3.0, stale_stop_s=8.0,
                               stale_speed_mps=4.0)
    segmenter = _build_segmenter(args)
    print(f"[view] runtime={runtime_mode}")
    return conn, camera_provider, detector, smoother, segmenter


def _close_session(session) -> None:
    if session is None:
        return
    conn, camera_provider, _detector, _smoother, _segmenter = session
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
                  warmup: bool = False, segmenter=None) -> dict | None:
    """Grab state, camera and sensors, then render one annotated frame.

    Returns None when only the frame grab failed (black/stale frame): the
    caller keeps the last overlay and must NOT treat this as a connection
    loss (real-vehicle sensor health logic: degrade a frame, never rebuild
    the session).  Connection-level failures still raise.
    """
    st = conn.get_state()
    try:
        img = camera_provider.grab()
    except Exception as exc:
        print(f"[view] grab failed (skipping frame): {exc}")
        return None
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

    half_w = geo_cache.get("half_w")
    if half_w is None:
        # get_bbox 是 Lua round-trip（~18ms）；车身尺寸不变，缓存一次即可
        half_w = ego_extents(conn)[1]
        geo_cache["half_w"] = half_w
    cam = camera_provider.camera_model(st.pos, heading, w, h,
                                       rotation=st.rotation)
    ground_z = (float(st.pos[2]) if len(st.pos) > 2 else 0.0)
    if segmenter is not None:
        # 学习式分割：路面边界与标线都来自 UNet 掩码
        vision_geometry = estimate_pavement_edges(
            img, cam, st.pos, heading, ground_z=ground_z,
            offroad_mask=segmenter.offroad_mask(img))
        raw_markings = segmenter.detect_lines(
            img, cam, st.pos, heading, ground_z=ground_z)
    else:
        vision_geometry = estimate_pavement_edges(
            img, cam, st.pos, heading, ground_z=ground_z)
        raw_markings = detector.detect(
            img, cam, st.pos, heading, ground_z=ground_z)
    geometry = merge_boundary_geometry(geometry, vision_geometry)
    markings = smoother.update(
        raw_markings, cam, st.pos, heading,
        ground_z=ground_z,
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
    # 感知车道框架：pair_lane_markings 的配对结果（真实标线识别输出）
    draw_lane_frame(overlay, frame, cam, st, heading)
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


def _perception_worker(session, args, shared: dict, stop: threading.Event):
    """Background perception: grab + detect, publish the latest snapshot.

    The worker owns all the heavy work (road geometry, pavement edges,
    marking detection, smoothing, pairing) so the render loop stays
    smooth.  A failed single grab skips the frame; any connection-level
    exception marks ``shared["error"]`` and the main loop reconnects.
    """
    conn, camera_provider, detector, smoother, segmenter = session
    geo_cache = {"t": 0.0, "pos": np.zeros(2, dtype=float)}
    geometry = None
    warmup = True
    while not stop.is_set():
        with shared["lock"]:
            # 心跳：主循环用它判断 worker 是否活着，避免把"首帧还没
            # 算完（路网刷新 ~1.5s）"误判成断线后关连接掐死 worker。
            shared["ts"] = time.time()
        t0 = time.time()
        try:
            # 推进模拟：beamngpy 连接后游戏暂停，不 step 则 Camera 传感器
            # 不再产生新帧（黑帧循环），快照永不产出。
            conn.step(10)
            st = conn.get_state()
            img = camera_provider.grab()
            h, w = img.shape[:2]
            fwd = unit_fwd(st)
            heading = float(st.heading)
            now = time.time()

            with shared["lock"]:
                force_geo = bool(shared.get("force_geometry"))
                shared["force_geometry"] = False

            if geometry is None or force_geo:
                geometry = road_lane_geometry(conn, st.pos, fwd)
                geo_cache["t"] = now
                geo_cache["pos"] = np.asarray(st.pos[:2], dtype=float).copy()
            elif (now - geo_cache["t"] >= args.road_refresh
                  or float(np.hypot(*(np.asarray(st.pos[:2])
                                      - geo_cache["pos"])))
                  >= args.road_move):
                geometry = road_lane_geometry(conn, st.pos, fwd)
                geo_cache["t"] = now
                geo_cache["pos"] = np.asarray(st.pos[:2], dtype=float).copy()

            half_w = geo_cache.get("half_w")
            if half_w is None:
                half_w = ego_extents(conn)[1]
                geo_cache["half_w"] = half_w
            cam = camera_provider.camera_model(st.pos, heading, w, h,
                                               rotation=st.rotation)
            ground_z = (float(st.pos[2]) if len(st.pos) > 2 else 0.0)
            if segmenter is not None:
                vision_geometry = estimate_pavement_edges(
                    img, cam, st.pos, heading, ground_z=ground_z,
                    offroad_mask=segmenter.offroad_mask(img))
                raw_markings = segmenter.detect_lines(
                    img, cam, st.pos, heading, ground_z=ground_z)
            else:
                vision_geometry = estimate_pavement_edges(
                    img, cam, st.pos, heading, ground_z=ground_z)
                raw_markings = detector.detect(
                    img, cam, st.pos, heading, ground_z=ground_z)
            merged = merge_boundary_geometry(geometry, vision_geometry)
            markings = smoother.update(
                raw_markings, cam, st.pos, heading, ground_z=ground_z,
                warmup=warmup, speed=float(st.speed))
            warmup = False
            debug: dict = {}
            frame = pair_lane_markings(
                markings, st.pos, heading, fwd=st.dir, debug=debug)
            vision_text = (
                f"vision: {len(raw_markings)} raw / {len(markings)} stable "
                f"mode={debug.get('mode')} "
                f"paired={frame.paired if frame is not None else None}")

            with shared["lock"]:
                shared["res"] = {
                    "state": st, "img": img, "cam": cam,
                    "geometry": merged, "vision": vision_geometry,
                    "half_w": half_w, "markings": markings,
                    "frame": frame, "debug": debug,
                    "vision_text": vision_text, "ts": time.time(),
                }
                shared["error"] = None
        except Exception as exc:
            # 单帧抓帧失败：降级跳过（不重连）；连接级异常：标记断线
            msg = str(exc)
            if "black frame" in msg or "no colour frame" in msg \
                    or "grab" in msg.lower():
                print(f"[view] grab failed (skipping frame): {msg}")
                time.sleep(0.2)
                continue
            print(f"[view] worker error: {msg}")
            with shared["lock"]:
                shared["error"] = msg
            return

        rem = 1.0 / max(1.0, args.rate) - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)


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
            geometry=None, geo_cache=geo_cache, warmup=True,
            segmenter=session[4])
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
    worker = None
    worker_stop = threading.Event()
    shared: dict = {"lock": threading.Lock(), "res": None, "error": None,
                    "force_geometry": False}
    last_overlay = None
    paused = False
    status = "connecting"
    errors = 0
    t0 = time.time()
    fps_frames = 0
    fps = 0.0

    def start_worker():
        nonlocal worker, worker_stop
        worker_stop = threading.Event()
        worker = threading.Thread(
            target=_perception_worker,
            args=(session, args, shared, worker_stop), daemon=True)
        worker.start()

    try:
        while True:
            frame_start = time.time()
            if session is None:
                try:
                    session = _open_session(args)
                    with shared["lock"]:
                        shared["res"] = None
                        shared["error"] = None
                    start_worker()
                    status = "live"
                    errors = 0
                    print("[view] connected")
                except Exception as exc:
                    status = f"waiting for BeamNG ({type(exc).__name__})"
                    if errors % 10 == 0:
                        print(f"[view] {status}: {exc}")
                    errors += 1
                    time.sleep(RECONNECT_DELAY_S)
                    continue
            elif not paused:
                with shared["lock"]:
                    res = shared["res"]
                    werr = shared["error"]
                    hb = float(shared.get("ts") or 0.0)
                # 断线 = worker 心跳超时（含首帧未产出的情况，worker 每帧
                # 都在跳心跳）或 worker 明确报连接级错误。
                stale = time.time() - hb > SNAPSHOT_STALE_S
                if werr is not None or stale:
                    print(f"[view] connection lost: "
                          f"{werr or 'worker heartbeat stale'}")
                    _close_session(session)
                    session = None
                    worker_stop.set()
                    if worker is not None:
                        worker.join(timeout=2.0)
                    worker = None
                    status = "reconnecting"
                    continue
                if res is None:
                    # worker 活着但在算第一帧：保持上一帧画面，不重连
                    time.sleep(0.05)
                    continue
                try:
                    overlay = render_lane_overlay(
                        res["img"], res["state"], res["geometry"],
                        res["markings"], res["cam"], res["half_w"],
                        vision_text=res["vision_text"])
                    draw_lane_frame(overlay, res["frame"], res["cam"],
                                    res["state"], float(res["state"].heading))
                    last_overlay = overlay
                    fps_frames += 1
                    now = time.time()
                    if now - t0 >= 1.0:
                        fps = fps_frames / (now - t0)
                        fps_frames = 0
                        t0 = now
                except Exception as exc:
                    print(f"[view] render error: {exc}")
                    time.sleep(0.1)

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
                with shared["lock"]:
                    shared["force_geometry"] = True
                print("[view] forcing road-network refresh")

            try:
                if cv2.getWindowProperty(WINDOW_NAME,
                                         cv2.WND_PROP_VISIBLE) < 1:
                    print("[view] window closed")
                    break
            except cv2.error:
                break

            if session is not None and not paused and args.display_rate > 0:
                rem = 1.0 / args.display_rate - (time.time() - frame_start)
                if rem > 0:
                    time.sleep(rem)
            else:
                time.sleep(0.05)
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        worker_stop.set()
        if worker is not None:
            worker.join(timeout=2.0)
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