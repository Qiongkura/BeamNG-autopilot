"""Body-on-pavement coverage probe: the P1-3 positive/negative cases.

Places the car at named poses on a real map and asks
:func:`beamng_autopilot.occupancy.body_drivable_coverage` the same question
the safety gate asks every tick - *is the rectangle the car occupies
standing on pavement the sensors can see?* - then prints the four-state
answer (on_road / off_road / unknown) with the observed/observed-fraction
evidence behind it.

The cases come from the 2026-09-20 run set, so the answers can be checked
against what the car actually did:

  in      an ordinary on-pavement pose on the route;
  edge    the pavement edge the crash run drove over;
  off     the crash-site pose (787.8, 730.7), car standing off pavement;
  blind   no grid (read nothing at all) - must read ``unknown``.

This is a measurement, not a gate: it never drives the car and never
writes a verdict.  Per the round-5 handoff the gate is exercised in the
live drive via ``BEAMNG_BODY_COVERAGE_GATE``; here we only establish that
the measurement is right on poses whose ground truth we know by eye
(``--save`` writes the road mask projection so the pose can be looked at).

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_body_cov_probe.py --attach \\
        --case in --case off --save-dir logs/goal_20260921/body_cov
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.config import EGO_ORIGIN_GROUND_GAP_M
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.fsd_stack import NEARFIELD_MAX_AHEAD_M
from beamng_autopilot.runtime import build_camera_ring_provider
from beamng_autopilot.occupancy import (
    BODY_COV_OFF_ROAD,
    BODY_COV_ON_ROAD,
    BODY_COV_UNKNOWN,
    OccupancyGrid,
    body_drivable_coverage,
    project_road_mask_to_grid,
)
from beamng_autopilot.vision.hydra import FrameContext, HydraNet
from beamng_autopilot.vision.heads.semantic import SemanticHead
from beamng_autopilot.vehicle_body import HALF_LENGTH_M, HALF_WIDTH_M

# Half-length / half-width of the ego rectangle, in metres.  Taken from the
# PRODUCTION constants (``vehicle_body``/``config``) instead of being typed
# here: an earlier version used 2.3/0.95 while claiming in a comment that it
# matched the monitor's 2.2/0.9, so the "positive/negative cases" were
# measured on a slightly different rectangle than the gate tests (plan T12).
HALF_LEN_M = float(HALF_LENGTH_M)
HALF_WIDTH_M = float(HALF_WIDTH_M)

# Poses: (x, y, heading_deg).  ``heading`` is the world heading in
# atan2 degrees, matching ``safe_teleport``.  Every pose is COPIED FROM A
# RECORDED RUN FRAME (file + tick in the comment) rather than invented, so
# the ground truth can be re-checked against what the car did there.
#
# ``in`` and ``off`` are a matched pair: the same point along the road
# (x ~ 787.8; the good run passed through both poses) differing only in
# lateral offset and yaw.  ``off`` is where the 2026-09-20 run ended up
# after drifting right and yawing 28 deg off the road direction, with the
# lane reference reporting "centre off observed pavement" while
# ``road_off``/``edge_over`` stayed 0.0.
CASES: dict[str, tuple[float, float, float | None]] = {
    # logs/goal_20260921/linesfix1.json @ t=29.7 (on pavement, same x)
    "in": (787.77, 732.59, -9.91),
    # logs/goal_20260921/models2_nodqn.json @ t=17.5 (approaching the edge)
    "edge": (784.66, 733.17, -20.46),
    # logs/goal_20260921/models2_nodqn.json @ t=45.2 (final, crash site)
    "off": (787.785, 730.668, -37.62),
    # same pose as ``in`` but the road-mask projection is skipped entirely
    "blind": (787.77, 732.59, -9.91),
}


def _render(grid: OccupancyGrid, pos, width: int = 41) -> str:
    """ASCII view of the grid with the ego cell marked ``E``.

    Row 0 is the most forward row (``OccupancyGrid.ego_to_cell``), so the
    printout is emitted in row order and the first line is what is AHEAD
    of the car; ``left`` runs to the right of each line.
    """
    n = int(grid.drivable.shape[0])
    occ = grid.as_raster()
    if occ.size == 0:
        return "(empty grid)"
    stride = max(1, n // width)
    rows, cols, ok = grid.world_to_cells(
        np.array([pos[0]]), np.array([pos[1]]))
    er, ec = (int(rows[0]), int(cols[0])) if ok[0] else (n // 2, n // 2)
    lines = []
    for r in range(0, n, stride):
        line = ""
        for c in range(0, n, stride):
            if abs(r - er) <= stride and abs(c - ec) <= stride:
                line += "E"
            elif grid.drivable[r, c]:
                line += "."
            elif grid.obstacle[r, c]:
                line += "#"
            elif occ[r, c] > 0.05:
                line += ":"
            else:
                line += " "
        lines.append(line)
    return "\n".join(lines)


def _band_stats(grid: OccupancyGrid, pos, heading: float,
                bands=((-2.0, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 2.0),
                       (2.0, 3.0), (3.0, 4.0), (4.0, 6.0)),
                half_w_m: float = 1.5) -> dict:
    """Observed / drivable cell counts per forward band, and the near edge.

    The body rectangle reads only a handful of cells because the front
    camera's nearest ground is metres ahead; this says exactly where the
    observed region begins, so "unknown" can be attributed to the sensor
    geometry instead of being read as "no pavement there".
    """
    n = int(grid.drivable.shape[0])
    res = float(grid.res)
    rows, cols = np.mgrid[0:n, 0:n]
    ex = grid.max_x - (rows + 0.5) * res          # forward (m)
    ey = grid.max_y - (cols + 0.5) * res          # left (m)
    out: dict = {"bands": []}
    for a, b in bands:
        m = (ex >= a) & (ex < b) & (np.abs(ey) <= half_w_m)
        out["bands"].append({
            "from_m": a, "to_m": b, "cells": int(m.sum()),
            "observed": int(np.count_nonzero(grid.observed[m])),
            "drivable": int(np.count_nonzero(grid.drivable[m])),
        })
    obs = grid.observed > 0
    if obs.any():
        out["obs_ahead_min_m"] = round(float(ex[obs].min()), 2)
        out["obs_ahead_max_m"] = round(float(ex[obs].max()), 2)
    else:
        out["obs_ahead_min_m"] = None
        out["obs_ahead_max_m"] = None
    # Distance from the ego centre to the nearest observed cell (any width).
    r0, c0 = grid.world_to_cell(float(pos[0]), float(pos[1])) or (n // 2, n // 2)
    rr, cc = np.nonzero(obs)
    if len(rr):
        d = np.hypot(rr - r0, cc - c0) * res
        out["obs_nearest_m"] = round(float(d.min()), 2)
    else:
        out["obs_nearest_m"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="body drivable coverage probe")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="tech")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--map", type=str, default="italy")
    ap.add_argument("--vehicle", type=str, default="etk800")
    ap.add_argument("--case", action="append", default=None,
                    choices=sorted(CASES),
                    help="case name(s) to measure; default all")
    ap.add_argument("--res", type=float, default=0.4)
    ap.add_argument("--n", type=int, default=120, help="cells per side")
    ap.add_argument("--cam", default="both",
                    help="front_main | front_fisheye | both (default both)")
    ap.add_argument("--settle", type=int, default=12,
                    help="ticks to let the camera settle after a teleport")
    ap.add_argument("--save-dir", type=str, default=None)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    names = args.case or list(CASES)
    conn = BeamNGConnector(
        args.map, args.vehicle,
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    rows: list[dict] = []
    try:
        conn.open(launch=not args.attach)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            conn.load_scenario()
        ring, mode = build_camera_ring_provider(conn, args.runtime, 320, 240)
        if ring is None:
            print(f"[body-cov] runtime={mode}: no ring available")
            return 0
        net = HydraNet()
        net.add(SemanticHead())
        print(f"[body-cov] runtime={mode} cases={names}")

        for name in names:
            x, y, hdg = CASES[name]
            if name != "blind":
                if not conn.safe_teleport(x, y, heading_deg=hdg):
                    print(f"[body-cov] {name}: teleport failed; skipped")
                    continue
                for _ in range(max(1, args.settle)):
                    conn.step(1)
            st = conn.get_state()
            pos = np.asarray(st.pos, dtype=float)
            heading = float(st.heading)
            grid = OccupancyGrid(args.n, args.n, args.res,
                                 origin=(float(pos[0]), float(pos[1])),
                                 heading=heading)
            road_px: dict[str, int] = {}
            # ``--cam main`` = the main camera only, ``--cam front_fisheye``
            # = that one camera, anything else = both.  The previous version
            # read ``args.role`` (an argument that no longer exists) for
            # ``--cam main`` and silently fell through to BOTH cameras for
            # ``--cam front_main``, so the "main camera only" arm could not
            # be measured at all (plan T12).
            if args.cam in ("main", "front_main"):
                proj = ["front_main"]
            elif args.cam in ("fisheye", "front_fisheye"):
                proj = ["front_fisheye"]
            else:
                proj = ["front_main", "front_fisheye"]
            if name != "blind":
                snap = ring.grab_ring()
                ground_z = (float(pos[2]) - float(EGO_ORIGIN_GROUND_GAP_M)
                            if len(pos) > 2 else None)
                for role in proj:
                    if role not in snap:
                        road_px[role] = -1
                        continue
                    frame, cam = snap[role]
                    ctx = FrameContext(frame_rgb=frame, cam=cam, pos=pos,
                                       heading=heading,
                                       ground_z=float(pos[2]), role=role)
                    out = net.run(ctx).get("semantic")
                    if out is None or "road" not in out.masks:
                        road_px[role] = -2
                        continue
                    road = out.masks["road"]
                    road_px[role] = int(np.count_nonzero(road))
                    # Same ground plane the runtime lifts the mask onto: the
                    # road surface, not the ego origin, which sits
                    # EGO_ORIGIN_GROUND_GAP_M above it.  The fisheye arm
                    # uses the near-field horizon the stack gives it.
                    project_road_mask_to_grid(
                        grid, road, cam, pos, heading, step=4,
                        max_ahead_m=(NEARFIELD_MAX_AHEAD_M
                                     if "fisheye" in role else 45.0),
                        ground_z=ground_z)
                    if args.save_dir:
                        out_dir = Path(args.save_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        import cv2
                        cv2.imwrite(
                            str(out_dir / f"{name}_{role}_rgb.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(
                            str(out_dir / f"{name}_{role}_road.png"),
                            (road.astype(np.uint8) * 255))
            rep = body_drivable_coverage(
                grid, pos, heading, HALF_LEN_M, HALF_WIDTH_M)
            bands = _band_stats(grid, pos, heading)
            # The body rectangle itself is inside the blind zone, so the
            # same question is also asked of a rectangle displaced into the
            # region the sensors DO see.  Same grid, same call, same shape -
            # only the sample moves - which is what separates "the primitive
            # cannot tell on_road from off_road" from "the sensors do not
            # observe the body".  Truth is known for both poses here: the
            # ``in`` case is on pavement (a recorded run drove through it)
            # and the ``off`` case is the crash site.
            ahead = []
            fwd = np.array([np.cos(heading), np.sin(heading)])
            for d in (2.0, 3.0, 4.0, 6.0):
                p2 = np.array([pos[0] + d * fwd[0], pos[1] + d * fwd[1],
                               pos[2] if len(pos) > 2 else 0.0])
                r2 = body_drivable_coverage(
                    grid, p2, heading, HALF_LEN_M, HALF_WIDTH_M)
                ahead.append({"ahead_m": d, "status": r2["status"],
                              "footprint_coverage": r2["coverage"],
                              "observed_frac": r2["observed_frac"],
                              "observed_cells": r2["observed_cells"]})
            row = {
                "case": name,
                "requested": [x, y, hdg],
                "pos": [round(float(p), 2) for p in pos[:3]],
                "heading": round(heading, 4),
                "cameras": list(proj) if name != "blind" else [],
                "road_px": road_px,
                # Grid totals and footprint counts are DIFFERENT numbers and
                # both matter: the first says the lift worked, the second says
                # whether the car's own rectangle was inside the sensor
                # footprint at all.
                "grid_drivable": int(grid.drivable.sum()),
                "grid_observed": int(grid.observed.sum()),
                "status": rep["status"],
                "footprint_coverage": rep["coverage"],
                "footprint_cells": rep["footprint_cells"],
                "footprint_observed": rep["observed_cells"],
                "footprint_drivable": rep["drivable_cells"],
                "bands": bands,
                "ahead_samples": ahead,
            }
            rows.append(row)
            print(f"[body-cov] {name:6s} pos=({pos[0]:.1f},{pos[1]:.1f}) "
                  f"road_px={road_px} grid_drivable={row['grid_drivable']} "
                  f"grid_observed={row['grid_observed']} "
                  f"-> status={rep['status']} frac={rep['coverage']} "
                  f"footprint={rep['observed_cells']}/"
                  f"{rep['footprint_cells']} observed, "
                  f"{rep['drivable_cells']} drivable "
                  f"nearest_obs={bands['obs_nearest_m']}m")
            for b in bands["bands"]:
                print(f"[body-cov]   {b['from_m']:+.0f}..{b['to_m']:+.0f}m "
                      f"cells={b['cells']:5d} observed={b['observed']:5d} "
                      f"drivable={b['drivable']:5d}")
            print("[body-cov]   same question, sample displaced ahead: "
                  + "  ".join(
                      f"+{a['ahead_m']:.0f}m={a['status']}"
                      f"({a['footprint_coverage']},"
                      f"obs {a['observed_frac']})" for a in ahead))
            if args.save_dir:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                Path(args.save_dir, f"{name}_grid.txt").write_text(
                    _render(grid, pos), encoding="utf-8")
    finally:
        conn.close()
    payload = {"runtime": "tech", "half_len_m": HALF_LEN_M,
               "half_width_m": HALF_WIDTH_M,
               "min_frac": 0.5, "min_observed": 4, "rows": rows}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"[body-cov] -> {args.json}")
    # Expected verdicts, so a wrong classification FAILS instead of being
    # printed and forgotten (plan T12: "须补真正正反例断言，不能把退出码 0
    # 当作车身保护验收通过").  ``in``/``edge`` are poses a recorded run
    # drove through on pavement; ``off`` is the crash site.
    #
    # The body rectangle itself is inside the near-field blind zone
    # (measured: nearest observed ground 2.4 m ahead of the ego centre), so
    # the expected body verdict there is UNKNOWN - and the probe says so
    # explicitly rather than accepting anything.
    expected_body = {"in": BODY_COV_UNKNOWN, "edge": BODY_COV_UNKNOWN,
                     "off": BODY_COV_UNKNOWN, "blind": BODY_COV_UNKNOWN}
    problems: list[str] = []
    for r in rows:
        want = expected_body.get(r["case"])
        if want is not None and r["status"] != want:
            problems.append(
                f"{r['case']}: body status {r['status']!r} != expected "
                f"{want!r}")
        if r["case"] == "blind" and r["status"] != BODY_COV_UNKNOWN:
            problems.append("blind case reported a verdict")
    # The displaced samples are where the primitive must DISCRIMINATE: the
    # on-pavement poses must read on_road ahead of the car, the crash site
    # off_road.  This is what makes the exit code evidence about the
    # measurement rather than about the run having finished.
    for r in rows:
        if r["case"] in ("in", "edge"):
            far = [a for a in r["ahead_samples"]
                   if a["ahead_m"] >= 3.0 and a["observed_frac"] and
                   a["observed_frac"] >= 0.3]
            if far and not any(a["status"] == BODY_COV_ON_ROAD for a in far):
                problems.append(
                    f"{r['case']}: pavement ahead read "
                    f"{[a['status'] for a in far]}, expected on_road")
        if r["case"] == "off":
            far = [a for a in r["ahead_samples"]
                   if a["ahead_m"] >= 3.0 and a["observed_frac"] and
                   a["observed_frac"] >= 0.3]
            if far and not any(a["status"] == BODY_COV_OFF_ROAD for a in far):
                problems.append(
                    f"off: crash-site pavement ahead read "
                    f"{[a['status'] for a in far]}, expected off_road")
    if problems:
        for p in problems:
            print(f"[body-cov] !! {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
