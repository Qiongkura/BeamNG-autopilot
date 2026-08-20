"""Probe the FSD-style BEV occupancy grid on BeamNG.tech.

Builds an ego-centred occupancy grid from the live sensor channels -
semantic masks (projected camera->BEV), LiDAR/ray obstacles (flooded to
cells) - and writes an ASCII rendering of the vector space so it can be
verified visually without the GUI.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_bev_probe.py --runtime tech --attach
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
from beamng_autopilot.runtime import (
    build_camera_ring_provider,
    build_range_provider,
)
from beamng_autopilot.occupancy import (
    OccupancyGrid,
    fuse_obstacles_to_grid,
    project_road_mask_to_grid,
)
from beamng_autopilot.vision.hydra import FrameContext, HydraNet
from beamng_autopilot.vision.heads.semantic import SemanticHead


def _render_ascii(grid: OccupancyGrid, width: int = 46) -> str:
    occ = grid.as_raster()
    if occ.size == 0:
        return "(empty)"
    rows = []
    n = occ.shape[0]
    stride = max(1, n // width)
    for r in range(0, n, stride):
        line = ""
        for c in range(0, n, stride):
            o = occ[r, c]
            if grid.obstacle[r, c]:
                line += "#"
            elif o > 0.4:
                line += "X"
            elif o > 0.05:
                line += ":"
            elif grid.drivable[r, c]:
                line += "."
            else:
                line += " "
        rows.append(line[::-1])  # left = +y toward reader
    cr, cc = grid.center
    out = []
    for i, row in enumerate(rows):
        if abs(i - cr // stride) <= 1:
            j = cc // stride
            out.append(row[:j] + ("^" if abs(i - cr // stride) == 0 else "|")
                       + row[j + 1:])
        else:
            out.append(row)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="BEV occupancy grid probe")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="auto")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=60, help="cells per side")
    ap.add_argument("--role", default="front_main",
                    help="ring camera to seed drivable space from")
    ap.add_argument("--save", type=str, default=None)
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
            conn.load_scenario()
        ring, mode = build_camera_ring_provider(
            conn, args.runtime, 320, 240)
        if ring is None:
            print(f"[bev] runtime={mode}: no ring (front-only Steam)")
            return 0
        range_prov, _ = build_range_provider(conn, args.runtime)

        st = conn.get_state()
        pos = np.asarray(st.pos, dtype=float)
        heading = float(st.heading)
        grid = OccupancyGrid(args.n, args.n, args.res,
                             origin=(float(pos[0]), float(pos[1])),
                             heading=heading)

        # 1) semantic head -> road mask -> BEV drivable
        net = HydraNet()
        try:
            net.add(SemanticHead())
        except Exception as exc:
            print(f"[bev] semantic head unavailable: {exc}")
        snap = ring.grab_ring()
        if args.role in snap:
            frame, cam = snap[args.role]
            ctx = FrameContext(frame_rgb=frame, cam=cam, pos=pos,
                               heading=heading, ground_z=float(pos[2]),
                               role=args.role)
            out = net.run(ctx).get("semantic")
            if out is not None and "road" in out.masks:
                project_road_mask_to_grid(grid, out.masks["road"], cam,
                                          pos, heading, step=4)

        # 2) range -> obstacles + ray hits -> BEV occupancy
        rng = range_prov.scan(pos)
        fuse_obstacles_to_grid(grid, rng.obstacles, rng.ray_hits)

        print(f"[bev] runtime={mode} grid={args.n}x{args.n}@{args.res}m "
              f"drivable={int(grid.drivable.sum())} "
              f"occupied={int((grid.obstacle > 0).sum())} "
              f"evidence={int((grid.occupancy > 0).sum())}")
        print(_render_ascii(grid))
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save, grid.as_raster())
            print(f"[bev] raster saved -> {args.save}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())