"""Visualise the painted-line detection chain on recorded live frames.

For sampled frames of a shadow episode this renders, on the actual
camera image:

* the DEPLOYED model's fresh line mask (green) and road mask (blue tint);
* the world-space markings the projection chain produced (their origin
  pixels, red dots);
* the own-lane centre target from ``painted_line_lane_center`` (yellow
  cross, projected back into the image);
* the live centre-line lateral read (line_lat) as text.

Output: PNGs in ``logs/m5_vis/`` (gitignored) so the detection can be
judged by eye instead of through counters.

    .venv\Scripts\python.exe scripts\m5_vis_line_debug.py --episode latest --frames 10
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from beamng_autopilot import config
from beamng_autopilot.vision.heads.semantic import SemanticHead
from beamng_autopilot.vision.hydra import FrameContext
from beamng_autopilot.vision.lanes import (
    painted_line_lane_center, painted_line_markings,
)
from beamng_autopilot.vision.ring import CAMERA_RING, FRONT_MAIN


def main() -> int:
    ap = argparse.ArgumentParser(description="paint-line detection visualiser")
    ap.add_argument("--episode", type=str, default="latest",
                    help="shadow episode npz ('latest' = newest)")
    ap.add_argument("--frames", type=int, default=10,
                    help="how many frames to render (evenly sampled)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    eps = sorted(glob.glob(str(config.LOGS_DIR / "m5_e2e" / "shadow_fsd_*.npz")),
                 key=os.path.getmtime)
    if not eps:
        print("no shadow episodes found")
        return 1
    ep = (eps[-1] if args.episode == "latest"
          else str(config.LOGS_DIR / "m5_e2e" / args.episode))
    d = np.load(ep, allow_pickle=True)
    t, xs, ys, hds, rgbs, labs = (d["t"], d["x"], d["y"], d["heading"],
                                  d["rgb"], d["label"])
    print(f"episode: {os.path.basename(ep)} ({len(t)} frames)")

    mount = next(m for m in CAMERA_RING if m.role == FRONT_MAIN)
    cam = mount.camera_model(400, 300)
    sem = SemanticHead()

    out_dir = (Path(args.out) if args.out
               else config.LOGS_DIR / "m5_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    idxs = np.linspace(0, len(t) - 1, max(1, args.frames)).astype(int)
    for k, i in enumerate(idxs):
        rgb = np.asarray(rgbs[i], dtype=np.uint8).copy()
        pos = np.array([float(xs[i]), float(ys[i]), 178.0])
        heading = float(hds[i])
        ctx = FrameContext(frame_rgb=rgb, cam=cam, pos=pos,
                           heading=heading, ground_z=float(pos[2]),
                           role="front_main")
        out = sem.run(ctx)
        road = np.asarray(out.masks.get("road"), dtype=bool)
        line = np.asarray(out.masks.get("line"), dtype=bool)

        vis = rgb.copy()
        vis[road] = (vis[road] * 0.55 + np.array([0.0, 0.0, 120.0]) * 0.45
                     ).astype(np.uint8)
        # line mask SPLIT BY IMAGE HALF: left-of-centre pixels (the
        # centre line / left boundary) green, right-of-centre pixels
        # (the right edge line) yellow - "are BOTH sides detected"
        # becomes directly visible, together with the principal-point
        # vertical line marking the camera boresight.
        cx_img = int(cam.cx)
        line_l = line.copy()
        line_l[:, cx_img:] = False
        line_r = line.copy()
        line_r[:, :cx_img] = False
        vis[line_l] = (0, 255, 0)
        vis[line_r] = (0, 230, 255)
        cv2.line(vis, (cx_img, 0), (cx_img, vis.shape[0] - 1),
                 (255, 0, 255), 1)
        # world markings: draw their origin pixels (the line-pixels that
        # survived projection) in red
        n_marks = 0
        try:
            marks = painted_line_markings(out, cam, pos, heading,
                                          ground_z=float(pos[2]))
            n_marks = len(marks) if marks else 0
        except Exception as e:
            n_marks = -1
        # own-lane centre target projected back into the image
        tgt_txt = "none"
        try:
            tgt = painted_line_lane_center(out, cam, pos, heading,
                                           ground_z=float(pos[2]))
            if tgt is not None:
                uu, vv, val = cam.project(
                    np.array([[tgt[0], tgt[1], pos[2] - 0.3]]), pos, heading)
                if bool(val[0]) and np.isfinite(uu[0]) and np.isfinite(vv[0]):
                    u, v = int(uu[0]), int(vv[0])
                    if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
                        cv2.drawMarker(vis, (u, v), (0, 255, 255),
                                       cv2.MARKER_CROSS, 21, 2)
                        tgt_txt = f"({tgt[0]:.1f}, {tgt[1]:.1f})"
        except Exception as e:
            tgt_txt = f"err:{e}"
        # line_lat: mean lateral of near world markings (left = +)
        line_lat = None
        try:
            if marks:
                left = np.array([-math.sin(heading), math.cos(heading)])
                p2 = pos[:2]
                vals = []
                for mk in marks:
                    w = np.asarray(mk.world, dtype=float)[:, :2]
                    near = w[np.linalg.norm(w - p2, axis=1) < 25.0]
                    if len(near):
                        vals.append(float(((near - p2) @ left).mean()))
                if vals:
                    line_lat = float(np.mean(vals))
        except Exception:
            pass

        line_frac = float(line.mean())
        n_l = int(line_l.sum())
        n_r = int(line_r.sum())
        txt1 = (f"t={float(t[i]):5.1f}s  line_px L/R = {n_l}/{n_r} "
                f"({line_frac:.3%})")
        txt2 = f"markings={n_marks}  line_lat={'-' if line_lat is None else f'{line_lat:+.2f}m'}"
        txt3 = f"lane_center_target: {tgt_txt}"
        for jj, txt in enumerate((txt1, txt2, txt3)):
            cv2.putText(vis, txt, (8, 18 + jj * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(vis, txt, (8, 18 + jj * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        fp = out_dir / f"line_debug_{k:02d}_t{int(float(t[i])):04d}.png"
        cv2.imwrite(str(fp), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"  [{k:02d}] t={float(t[i]):5.1f}s line_px={int(line.sum()):5d} "
              f"markings={n_marks:3d} line_lat="
              f"{'-' if line_lat is None else f'{line_lat:+.2f}m'} -> {fp.name}")
    print(f"visualisations -> {out_dir}")
    return 0


import math  # noqa: E402  (used in line_lat block)

if __name__ == "__main__":
    sys.exit(main())
