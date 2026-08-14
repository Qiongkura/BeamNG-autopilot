"""Capture the current perfectly-centred driving state as BC samples.

Attaches to a running BeamNG session (BeamNG.tech by default), reads the
road-network lane geometry around the ego and saves M3-compatible
behavioural-cloning samples (frames + meta.jsonl) whose steering label is
0.0, i.e. "keep the car exactly where it is now".  The vision lane
detector is also run on every frame and its result is stored as metadata,
so the samples can be audited later without reconnecting to the game.

Output layout (one run dir under logs/m3_bc):
    frames/frame_00000.jpg ...     downscaled camera frames (RGB jpg)
    meta.jsonl                     one JSON object per line
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.lane import _boundary_near_lat, pair_lane_markings
from beamng_autopilot.runtime import build_camera_provider, resolve_runtime
from beamng_autopilot.vision.lanes import LaneDetector


def _unit_fwd(state) -> np.ndarray:
    """2D unit forward vector from a VehicleState."""
    fwd = np.asarray(state.dir[:2], dtype=float)
    n = float(np.linalg.norm(fwd))
    if n > 1e-9:
        return fwd / n
    return np.array([math.cos(state.heading), math.sin(state.heading)])


def _lat_of(world_xy, pos, left) -> float:
    """Signed lateral offset: positive when the point is to the left."""
    p = np.asarray(world_xy, dtype=float)[:2]
    return float((p - pos) @ left)


def _interp_edge(a, b, t):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a + t * (b - a)


def _road_lane_geometry(conn, pos, fwd):
    """Return current-lane geometry from the scenario road network.

    The lane the car drives in is derived from the nearest road edge row:
    the road's lanesLeft/lanesRight counts split the road into equal
    lanes, and the car is placed into the lane matching its lateral
    position.  All lateral values follow the car frame (left is +).
    """
    pos = np.asarray(pos, dtype=float)[:2]
    with conn.io_lock:
        roads = conn.bng.scenario.get_road_network(
            include_edges=True, drivable_only=True)

    candidates: list[dict] = []
    for rid, meta in roads.items():
        if not isinstance(meta, dict):
            continue
        edges = meta.get("edges")
        if not isinstance(edges, list) or len(edges) < 2:
            continue
        mids = []
        for row in edges:
            m = row.get("middle")
            mids.append(None if m is None else np.asarray(m, dtype=float))
        candidates.append((rid, meta, mids, edges))

    best = None
    for rid, meta, mids, edges in candidates:
        for i in range(len(mids) - 1):
            a, b = mids[i], mids[i + 1]
            if a is None or b is None:
                continue
            ab = b[:2] - a[:2]
            d2 = float(ab @ ab)
            if d2 < 1e-9:
                continue
            t = float((pos - a[:2]) @ ab / d2)
            t = min(1.0, max(0.0, t))
            p = a[:2] + t * ab
            dist = float(np.hypot(*(p - pos)))
            if best is None or dist < best[0]:
                best = (dist, rid, meta, i, t, edges, mids)
    if best is None:
        return None

    _, rid, meta, i, t, edges, mids = best
    row_a, row_b = edges[i], edges[i + 1]
    left_pt = _interp_edge(row_a["left"], row_b["left"], t)
    mid_pt = _interp_edge(row_a["middle"], row_b["middle"], t)
    right_pt = _interp_edge(row_a["right"], row_b["right"], t)

    left = np.array([-fwd[1], fwd[0]])
    left_lat = _lat_of(left_pt, pos, left)
    mid_lat = _lat_of(mid_pt, pos, left)
    right_lat = _lat_of(right_pt, pos, left)
    if left_lat < right_lat:
        # The road spline is stored against the traffic direction, so the
        # "left" edge point sits on the car's right; swap for car frame.
        left_lat, right_lat = right_lat, left_lat

    width = float(left_lat - right_lat)
    half = width / 2.0
    n_left = int(meta.get("lanesLeft") or 0)
    n_right = int(meta.get("lanesRight") or 0)
    car_rel = float(-mid_lat)  # car lateral relative to the road centre

    if n_left <= 0 and n_right <= 0:
        total = 1
        lane_w = width
    elif n_left <= 0 or n_right <= 0:
        total = max(1, n_left + n_right)
        lane_w = width / total
    else:
        total = 1
        lane_w = width

    if n_left <= 0 or n_right <= 0:
        d_from_left = float(left_lat)
        k = int(math.floor(max(0.0, d_from_left) / lane_w))
        k = min(max(0, k), total - 1)
        lane_left_lat = left_lat - k * lane_w
        lane_right_lat = left_lat - (k + 1) * lane_w
    elif car_rel < 0.0:
        lane_w = half / n_right
        k = int(math.floor(-car_rel / lane_w)) if lane_w > 0 else 0
        k = min(max(0, k), n_right - 1)
        lane_left_lat = mid_lat - half + (k + 1) * lane_w
        lane_right_lat = mid_lat - half + k * lane_w
    else:
        lane_w = half / n_left
        k = int(math.floor(car_rel / lane_w)) if lane_w > 0 else 0
        k = min(max(0, k), n_left - 1)
        lane_left_lat = mid_lat + half - k * lane_w
        lane_right_lat = mid_lat + half - (k + 1) * lane_w

    lane_center_lat = 0.5 * (lane_left_lat + lane_right_lat)
    return {
        "road_id": rid,
        "lanes_left": n_left,
        "lanes_right": n_right,
        "road_width": round(width, 3),
        "road_center_lat": round(mid_lat, 3),
        "left_edge_lat": round(left_lat, 3),
        "right_edge_lat": round(right_lat, 3),
        "lane_left_lat": round(lane_left_lat, 3),
        "lane_right_lat": round(lane_right_lat, 3),
        "lane_center_lat": round(lane_center_lat, 3),
        "center_offset": round(-lane_center_lat, 3),
        "left_dist": round(lane_left_lat, 3),
        "right_dist": round(-lane_right_lat, 3),
        "lane_width": round(abs(lane_right_lat - lane_left_lat), 3),
    }


def _visual_state(detector, camera_provider, img, state):
    """Run the lane-marking detector and return lane-frame metadata."""
    h, w = img.shape[:2]
    vmodel = camera_provider.camera_model(
        state.pos, state.heading, w, h)
    markings = detector.detect(
        img, vmodel, state.pos, state.heading,
        ground_z=(float(state.pos[2]) if len(state.pos) > 2 else 0.0))
    debug: dict = {}
    frame = pair_lane_markings(
        markings, state.pos, state.heading, fwd=state.dir, debug=debug)
    fwd = _unit_fwd(state)
    pos = np.asarray(state.pos[:2], dtype=float)
    left = np.array([-fwd[1], fwd[0]])

    out = {
        "n_markings": len(markings),
        "lane_mode": debug.get("mode"),
        "center_offset": None,
        "left_dist": None,
        "right_dist": None,
        "width": None,
        "conf": None,
        "span": None,
        "paired": None,
    }
    if frame is not None:
        out["center_offset"] = round(_lat_of(frame.center[0], pos, left), 3)
        out["width"] = round(float(frame.width), 3)
        out["conf"] = round(float(frame.confidence), 3)
        out["span"] = round(float(frame.span_m), 3)
        out["paired"] = bool(frame.paired)
        if frame.left is not None:
            out["left_dist"] = round(
                _boundary_near_lat(frame.left, pos, state.heading,
                                   fwd=fwd) or 0.0, 3)
        if frame.right is not None:
            rl = _boundary_near_lat(
                frame.right, pos, state.heading, fwd=fwd)
            if rl is not None:
                out["right_dist"] = round(-float(rl), 3)
    return out


def _verify_run(run_dir: Path, w: int, h: int) -> None:
    """Reuse the M3 loader to prove the run is consumable by training."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from m3_train_bc import load_run

        samples = load_run(run_dir, w, h)
        print(f"[verify] m3_train_bc.load_run -> {len(samples)} samples")
        if samples:
            first = samples[0]
            print(f"[verify] first sample idx/steer/t = "
                  f"{Path(first[0]).stem}  {first[1]:+.4f}  {first[2]:.3f}")
    except Exception as exc:
        print(f"[verify] loader check failed: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Capture the current centred lane state as BC samples")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--frames", type=int, default=30,
                    help="number of frames to capture (default 30)")
    ap.add_argument("--out", default=None,
                    help="output run dir (default logs/m3_bc/lane_center_<ts>)")
    ap.add_argument("--resize", default="320x180",
                    help="downscale frames to WxH before saving")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    w, h = (int(x) for x in args.resize.lower().split("x"))
    if args.frames < 20:
        ap.error("--frames must be >= 20 (m3_train_bc needs that many)")

    if args.out:
        run_dir = Path(args.out)
    else:
        run_dir = (config.LOGS_DIR / "m3_bc"
                   / f"lane_center_{time.strftime('%Y%m%d_%H%M%S')}")
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)

    conn = BeamNGConnector(
        args.map, args.vehicle, port=args.port,
        home=config.runtime_home(args.runtime))
    camera_provider = None
    try:
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
    except Exception as exc:
        print(f"[capture] cannot attach to a running session: {exc}")
        return

    runtime_mode = resolve_runtime(conn, args.runtime)
    if runtime_mode == "steam":
        conn.set_front_camera()
    camera_provider, _ = build_camera_provider(conn, runtime_mode)
    detector = LaneDetector()
    print(f"[capture] runtime={runtime_mode}  -> {run_dir}")

    st0 = conn.get_state()
    fwd0 = _unit_fwd(st0)
    geometry = _road_lane_geometry(conn, st0.pos, fwd0)
    if geometry is None:
        print("[capture] WARNING: no road-network geometry near the ego; "
              "samples will still be saved without lane metadata")

    meta_path = run_dir / "meta.jsonl"
    t0 = time.time()
    idx = 0
    last_status = 0.0
    try:
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            while idx < args.frames:
                st = conn.get_state()
                try:
                    img = camera_provider.grab()
                except Exception as exc:
                    print(f"[capture] frame error: {exc}")
                    conn.step(1)
                    continue

                visual = _visual_state(detector, camera_provider, img, st)
                small = cv2.resize(
                    img, (w, h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(
                    str(run_dir / "frames" / f"frame_{idx:05d}.jpg"),
                    cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])

                meta = {
                    "idx": idx,
                    "t": round(time.time() - t0, 4),
                    "steer": 0.0,
                    "throttle": 0.0,
                    "brake": 0.0,
                    "speed": round(float(st.speed), 4),
                    "pos": [round(float(v), 3) for v in st.pos],
                    "heading": round(float(st.heading), 4),
                    "nearest": 0,
                    "lap": 0,
                    "source": "roadnet",
                }
                if geometry is not None:
                    meta.update(geometry)
                meta["vision"] = visual
                meta_file.write(json.dumps(meta) + "\n")

                if idx == 0 or time.time() - last_status > 1.0:
                    last_status = time.time()
                    if geometry is not None:
                        print(
                            f"[capture] frame {idx}: "
                            f"center_offset={geometry['center_offset']:+.3f} m "
                            f"left={geometry['left_dist']:.2f} m "
                            f"right={geometry['right_dist']:.2f} m "
                            f"lane_width={geometry['lane_width']:.2f} m")
                    print(
                        f"  vision markings={visual['n_markings']} "
                        f"mode={visual['lane_mode']} "
                        f"paired={visual['paired']} "
                        f"conf={visual['conf']}")
                idx += 1
    finally:
        if camera_provider is not None:
            try:
                camera_provider.close()
            except Exception:
                pass
        conn.close()

    print(f"[capture] done: {idx} frames -> {run_dir}")
    _verify_run(run_dir, w, h)


if __name__ == "__main__":
    main()
