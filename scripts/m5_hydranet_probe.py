"""Probe the HydraNet shared-backbone pipeline on the camera ring.

Polls the eight-camera ring on BeamNG.tech, runs the semantic + object
heads over every frame and prints a per-head/per-role summary, showing
the FSD-style "one frame in, many task outputs out" data flow.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_hydranet_probe.py --runtime tech --attach
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.runtime import build_camera_ring_provider
from beamng_autopilot.vision.hydra import FrameContext, HydraNet
from beamng_autopilot.vision.heads.semantic import SemanticHead
from beamng_autopilot.vision.heads.object import ObjectHead


def main() -> int:
    ap = argparse.ArgumentParser(description="HydraNet multi-task probe")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="auto")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--roles", nargs="*", default=None,
                    help="ring roles to process (default: all eight)")
    ap.add_argument("--with-object", action="store_true",
                    help="run the YOLO object head (needs torch/ultralytics)")
    args = ap.parse_args()

    conn = BeamNGConnector(
        "italy", "etk800",
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    try:
        conn.open(launch=not args.attach)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            # Fresh Tech session with no scenario: load one so the ring
            # has a vehicle to attach the cameras to.
            conn.load_scenario()
        ring, mode = build_camera_ring_provider(
            conn, args.runtime, args.width, args.height)
        if ring is None:
            print(f"[hydra] runtime={mode}: no ring (front-only Steam)")
            return 0
        net = HydraNet()
        try:
            net.add(SemanticHead())
        except Exception as exc:
            print(f"[hydra] semantic head init failed: {exc}")
        if args.with_object:
            try:
                net.add(ObjectHead())
            except Exception as exc:
                print(f"[hydra] object head init failed: {exc}")

        st = conn.get_state()
        pos = np.asarray(st.pos[:2], dtype=float)
        heading = float(st.heading)

        snap = ring.grab_ring()
        roles = list(snap.keys()) if args.roles is None else args.roles
        print(f"[hydra] runtime={mode}: {len(snap)} snapshots, "
              f"{net.names()}")
        for role in roles:
            if role not in snap:
                print(f"[hydra] role {role}: no snapshot")
                continue
            frame, cam = snap[role]
            ctx = FrameContext(
                frame_rgb=frame, cam=cam, pos=pos, heading=heading,
                ground_z=float(st.pos[2]) if len(st.pos) > 2 else 0.0,
                role=role)
            results = net.run(ctx)
            parts = []
            for name, out in results.items():
                if out.masks:
                    road = bool(out.masks["road"].mean() > 0.5)
                    line_px = int(out.masks["line"].sum())
                    parts.append(f"{name}:road={road} line_px={line_px} "
                                 f"markings={len(out.meta.get('markings', []))}")
                if out.obstacles:
                    parts.append(f"{name}:obs={len(out.obstacles)}")
            print(f"[hydra] {role:16s} " + " | ".join(parts))
        if net.errors:
            print(f"[hydra] head errors: {net.errors}")
        print("[hydra] all heads forwarded one frame each")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())