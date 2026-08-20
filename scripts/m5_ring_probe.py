"""Probe the FSD-style eight-camera ring on BeamNG.tech.

Grab one multi-view snapshot from every ring camera and report each
mount's pan/FOV and a small ASCII preview of the frame, so the ring is
verifiable without opening eight windows.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_ring_probe.py --runtime tech --attach
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
from beamng_autopilot.vision.ring import CAMERA_RING, pan_deg_of


def _ascii_preview(frame, cols: int = 48, rows: int = 14) -> str:
    g = np.asarray(frame, dtype=np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    if g.max() > g.min():
        g = (g - g.min()) / (g.max() - g.min()) * 15.0
    chars = " .:-=+*#%@"
    out = []
    h, w = g.shape
    for r in range(rows):
        line = ""
        for c in range(cols):
            line += chars[int(g[int(r * h / rows), int(c * w / cols)])
                          % len(chars)]
        out.append(line)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="FSD-style camera ring probe")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="auto")
    ap.add_argument("--attach", action="store_true",
                    help="attach to a running game without launching")
    ap.add_argument("--width", type=int, default=1076)
    ap.add_argument("--height", type=int, default=806)
    ap.add_argument("--save", type=str, default=None,
                    help="save snapshots to this dir (optional)")
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
            print(f"[ring] runtime={mode}: no ring provider (front-only "
                  "Steam fallback)")
            return 0
        print(f"[ring] runtime={mode}: {len(ring.cameras)} cameras")
        snap = ring.grab_ring()
        save = Path(args.save) if args.save else None
        if save is not None:
            save.mkdir(parents=True, exist_ok=True)
        for mount in CAMERA_RING:
            if mount.role not in snap:
                continue
            frame, model = snap[mount.role]
            pan = pan_deg_of(model)
            print(f"\n[{mount.role}] pan={pan:+.0f} fov={mount.fov_deg:.0f}"
                  f" shape={frame.shape} mean={float(frame.mean()):.1f}")
            if save is not None:
                import cv2
                cv2.imwrite(str(save / f"{mount.role}.png"),
                            frame[:, :, ::-1])
        print("\n[ring] all ring cameras polled OK")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())