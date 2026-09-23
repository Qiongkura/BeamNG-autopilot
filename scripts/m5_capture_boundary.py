"""Capture boundary evidence inputs for the T07 offline arms.

Records what the arm comparison needs and nothing else: the LiDAR cloud,
the ego pose, the ground plane, and the semantic road mask with the camera
model it came from.  One tick, one file - the point is to be able to argue
about the boundary detectors offline, not to drive.

``--frames K --step-m S`` records a SEQUENCE: the car is placed at the
given pose, captured, then re-placed ``S`` metres further along its own
heading and captured again, K times.  The sensor data and the poses are
real; the motion is placement, not driving, so a sequence is labelled
``placed_step_replay`` and may not be described as a driven run.  What it
can support is exactly what the persistence filter claims: whether real
returns from one kerb recur in the same world cells across frames.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_capture_boundary.py --attach \\
        --out logs/goal_20260921/boundary_capture.npz
    .venv\\Scripts\\python.exe scripts\\m5_capture_boundary.py --attach \\
        --teleport 786.87 732.72 -14.7 --frames 12 --step-m 2.0 \\
        --out logs/goal_20260921/boundary_seq.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot import geometry as G  # noqa: E402
from beamng_autopilot.connector import BeamNGConnector  # noqa: E402
from beamng_autopilot_tech.annotations import (  # noqa: E402
    annotation_palette,
)
from beamng_autopilot.runtime import (  # noqa: E402
    build_camera_ring_provider,
    build_range_provider,
)
from beamng_autopilot.vision.hydra import FrameContext, HydraNet  # noqa: E402
from beamng_autopilot.vision.heads.semantic import SemanticHead  # noqa: E402

# The LiDAR's first poll after attach/teleport is empty (measured
# 2026-09-22: 0 points, then ~414k on the next poll); the capture warms up
# instead of recording a silent zero.
LIDAR_WARMUP_POLLS = 4
MAX_RECORDED_POINTS = 60000


def grab_lidar(range_prov, pos, warmup: int = LIDAR_WARMUP_POLLS):
    """Return ``(cloud, stride, raw_points, notes)`` from one real poll.

    The cloud is NOT part of ``RangeSample`` (which deliberately carries
    only the fused obstacles/ray hits), so the provider is asked for its
    raw payload - the same object ``scan()`` feeds to ``process()``.
    """
    notes: list[str] = []
    cloud = np.empty((0, 3), dtype=float)
    fetch = getattr(range_prov, "fetch", None)
    if callable(fetch):
        for attempt in range(max(1, int(warmup))):
            try:
                payload_raw = fetch(pos)
                cloud = np.asarray(getattr(payload_raw, "cloud",
                                           np.empty((0, 3))), dtype=float)
            except Exception as exc:
                notes.append(f"fetch failed: {exc}")
                cloud = np.empty((0, 3), dtype=float)
            if cloud.size:
                break
            if attempt == 0:
                notes.append("first poll empty (warm-up)")
            time.sleep(0.5)
    if cloud.size == 0:
        rng = range_prov.scan(pos)
        cloud = np.asarray(getattr(rng, "cloud", np.empty((0, 3))), dtype=float)
        notes.append("fell back to scan()")
    raw = int(len(cloud))
    stride = 1
    if len(cloud) > MAX_RECORDED_POINTS:
        stride = int(np.ceil(len(cloud) / float(MAX_RECORDED_POINTS)))
        cloud = cloud[::stride]
    return cloud, stride, raw, notes


def grab_mask(ring, net, pos, heading: float, ground_z: float, role_arg: str):
    """Return ``(mask, cam, role_used)`` from the requested ring mount."""
    if ring is None:
        return None, None, None
    snap = ring.grab_ring()
    if not snap:
        return None, None, None
    role = role_arg if role_arg in snap else next(iter(snap))
    frame, cam = snap[role]
    ctx = FrameContext(frame_rgb=frame, cam=cam, pos=pos, heading=heading,
                       ground_z=ground_z, role=role)
    out = net.run(ctx).get("semantic")
    mask = None
    if out is not None and "road" in out.masks:
        mask = np.asarray(out.masks["road"], dtype=bool)
    return mask, cam, role


def grab_annotation_counts(ring, palette_classes: dict, role_arg: str):
    """Per-class pixel counts from the ENGINE's annotated frame.

    This is the independent label for "what kind of edge is beside the
    road" (GUARD_RAIL / GRASS / TERRAIN ...): counting them keeps the
    acceptance-matrix row from resting on the operator's description.
    Returns ``(counts, annotated_rgb_or_None)``.
    """
    grab = getattr(ring, "grab_ring_labels", None)
    if not callable(grab):
        return {}, None
    try:
        labels = grab()
    except Exception as exc:
        print(f"[capture] annotated grab failed: {exc}")
        return {}, None
    if not labels:
        return {}, None
    role = role_arg if role_arg in labels else next(iter(labels))
    ann = np.asarray(labels[role][1])
    if ann.ndim != 3 or ann.shape[2] < 3:
        return {}, None
    rgb = ann[:, :, :3].astype(np.int16)
    counts: dict = {}
    for name, col in palette_classes.items():
        c = np.asarray(col, dtype=np.int16)
        counts[str(name)] = int(np.count_nonzero(
            (rgb[:, :, 0] == c[0]) & (rgb[:, :, 1] == c[1])
            & (rgb[:, :, 2] == c[2])))
    return {k: v for k, v in counts.items() if v > 0}, ann


def main() -> int:
    ap = argparse.ArgumentParser(description="capture boundary inputs")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="tech")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--map", type=str, default="italy")
    ap.add_argument("--vehicle", type=str, default="etk800")
    ap.add_argument("--role", type=str, default="front_main")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="drive this long before capturing (0 = capture now)")
    ap.add_argument("--speed", type=float, default=0.0)
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"),
                    help="place the car here before the first frame")
    ap.add_argument("--frames", type=int, default=1,
                    help="frames in the sequence (1 = the old single frame)")
    ap.add_argument("--step-m", type=float, default=0.0,
                    help="placement advance along the current heading [m]")
    ap.add_argument("--settle-ticks", type=int, default=4,
                    help="ticks to let pose and sensor settle after placing")
    ap.add_argument("--no-mask", action="store_true",
                    help="skip the semantic mask (no ring / no labels)")
    ap.add_argument("--annotations", action="store_true",
                    help="also grab the engine's annotated frame and count "
                         "palette classes per frame (GUARD_RAIL / GRASS / ...)"
                         " - this is the independent label for which kind of "
                         "edge a stretch has, instead of the operator's word")
    ap.add_argument("--save-ann", action="store_true",
                    help="write the annotated frame as a PNG next to --out")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    conn = BeamNGConnector(
        args.map, args.vehicle,
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    frames = max(1, int(args.frames))
    try:
        conn.open(launch=not args.attach)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            conn.load_scenario()
        ring, mode = build_camera_ring_provider(
            conn, args.runtime, 320, 240, annotations=bool(args.annotations))
        range_prov, _ = build_range_provider(conn, args.runtime)
        net = None if (args.no_mask or ring is None) else HydraNet()
        if net is not None:
            net.add(SemanticHead())
        palette_classes: dict = {}
        if args.annotations and ring is not None:
            get_ann = getattr(conn.bng, "get_annotations", None)
            tech_ann = get_ann() if callable(get_ann) else None
            try:
                palette_classes = dict(annotation_palette(tech_ann)["classes"])
            except Exception as exc:
                print(f"[capture] annotation palette unavailable: {exc}")
                palette_classes = {}
            print(f"[capture] annotation classes: "
                  f"{','.join(sorted(palette_classes)) or 'none'}")
        payload: dict = {"runtime": mode, "frames": np.array([frames]),
                         "step_m": np.array([float(args.step_m)]),
                         "role": np.array([args.role]),
                         "map": np.array([args.map]),
                         "vehicle": np.array([args.vehicle]),
                         "evidence_level": np.array([
                             "placed_step_replay" if frames > 1
                             else "recorded_sensor_replay"])}
        if frames > 1 and args.step_m <= 0.0:
            payload["evidence_level"] = np.array(["static_repeat"])
        settle = max(1, int(args.settle_ticks))
        for i in range(frames):
            if i == 0 and args.teleport:
                x, y, yaw = (float(v) for v in args.teleport)
                if not conn.safe_teleport(x, y, heading_deg=yaw):
                    print(f"[capture] teleport to {x:.1f},{y:.1f} failed")
                    return 2
            elif i > 0 and args.step_m > 0.0:
                st_prev = conn.get_state()
                hdg = float(st_prev.heading)
                p_prev = np.asarray(st_prev.pos, dtype=float)
                x = float(p_prev[0]) + float(args.step_m) * float(np.cos(hdg))
                y = float(p_prev[1]) + float(args.step_m) * float(np.sin(hdg))
                if not conn.safe_teleport(x, y,
                                          heading_deg=float(np.degrees(hdg))):
                    print(f"[capture] step {i}: teleport to "
                          f"{x:.1f},{y:.1f} failed; stopping the sequence")
                    payload["frames"] = np.array([i])
                    break
            conn.step(settle)
            st = conn.get_state()
            pos = np.asarray(st.pos, dtype=float)
            heading = float(st.heading)
            ground_z = G.ego_ground_z(pos)
            cloud, stride, raw, notes = grab_lidar(range_prov, pos)
            mask = cam = role_used = None
            if net is not None:
                mask, cam, role_used = grab_mask(ring, net, pos, heading,
                                                 ground_z, args.role)
            if palette_classes:
                counts, ann_img = grab_annotation_counts(ring, palette_classes,
                                                        args.role)
                if counts:
                    payload[f"ann_counts_{i}"] = np.array(
                        [json.dumps(counts, sort_keys=True)])
                    payload[f"ann_n_{i}"] = np.array([sum(counts.values())])
                    print(f"[capture] frame {i} ann: " + ", ".join(
                        f"{k}={v}" for k, v in sorted(counts.items(),
                                                      key=lambda kv: -kv[1])
                        [:6]), flush=True)
                if args.save_ann and ann_img is not None:
                    out_png = Path(args.out).with_name(
                        Path(args.out).stem + f"_ann_{i:02d}.png")
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_png), np.asarray(ann_img)[:, :, ::-1])
            payload[f"points_{i}"] = cloud
            payload[f"pos_{i}"] = pos
            if i == 0:
                # Legacy single-frame keys: m5_boundary_arms.py reads them,
                # and a one-frame capture is still the old artefact.
                payload["points"] = cloud
                payload["pos"] = pos
                payload["heading"] = np.array([heading])
                payload["ground_z"] = np.array([ground_z])
                if mask is not None:
                    payload["mask"] = mask
            payload[f"heading_{i}"] = np.array([heading])
            payload[f"ground_z_{i}"] = np.array([ground_z])
            payload[f"speed_{i}"] = np.array([float(getattr(st, "speed", 0.0)
                                                    or 0.0)])
            payload[f"t_{i}"] = np.array([time.time()])
            payload[f"settle_ticks_{i}"] = np.array([settle])
            payload[f"cloud_stride_{i}"] = np.array([stride])
            payload[f"cloud_raw_points_{i}"] = np.array([raw])
            if notes:
                payload[f"notes_{i}"] = np.array(["; ".join(notes)])
            if mask is not None:
                payload[f"mask_{i}"] = mask
            if cam is not None:
                payload[f"cam_role_{i}"] = np.array([str(role_used)])
                payload[f"cam_offset_{i}"] = np.asarray(cam.offset, dtype=float)
                payload[f"cam_fwd_{i}"] = np.asarray(cam.fwd_local, dtype=float)
                payload[f"cam_up_{i}"] = np.asarray(cam.up_local, dtype=float)
                payload[f"cam_fov_{i}"] = np.array([float(cam.fov_deg)])
                payload[f"cam_w_{i}"] = np.array([int(cam.width)])
                payload[f"cam_h_{i}"] = np.array([int(cam.height)])
            print(f"[capture] frame {i}: pos=({pos[0]:.2f},{pos[1]:.2f},"
                  f"{pos[2]:.2f}) hdg={heading:+.3f} points={len(cloud)}"
                  f"/{raw} mask={'yes' if mask is not None else 'no'}"
                  + (f" notes={'|'.join(notes)}" if notes else ""), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **payload)
        print(f"[capture] {args.out}: frames={frames} runtime={mode}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
