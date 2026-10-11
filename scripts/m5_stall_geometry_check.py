"""Geometric confirmation for recorded non-executable-path frames, game-free.

Companion to ``m5_stall_attribution.py``: that script buckets stopped time by
recorded evidence and leaves its ``no_executable_path`` class with an open
question - "confirm a legal executable path existed".  This script answers it
per frame from the ``control-geometry-v1`` chains captured with
``--probe-geometry`` (scene raster + planner candidates + safety assessment +
verdict + command), reusing the production occupancy conversion and the
monitor's own ahead-corridor test.

Per frame it reports, without collapsing them:

* ``corridor_ahead_m``   - nearest obstacle intruding into the perception-lane
  corridor ahead (``None`` = none within ``--ahead-max-m``: clear ahead);
* planner/candidate state (candidates count, planner path, accepted path);
* the recorded verdict level and reason, verbatim.

Frame classes (taxonomy of docs/PROJECT_NEXT_DEVELOPMENT_REVIEW_20261011.md
§5.4; they refine, never replace, the recorded reason):

* ``C_perception_missing``      no perception lane, planner produced nothing
* ``A_no_candidates``           planner returned no candidate at all
* ``B1_rejected_corridor_free`` planner proposed, safety rejected, corridor
  clear ahead - the "存在空隙但被拒" population a fix would have to justify
* ``B2_rejected_blocked_ahead`` planner proposed, safety rejected, obstacle in
  the ahead-corridor
* ``D1_accepted_slow`` / ``D2_accepted_moving``
* ``E_other``                   anything else (kept visible, not hidden)

A geometrically clear corridor is NOT a safety certificate: the recorded
rejection may be a body-sweep boundary crossing rather than an obstacle, and
that distinction is kept explicit in the output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.occupancy import OccupancyGrid          # noqa: E402
from beamng_autopilot.safety_monitor import (                 # noqa: E402
    EASE_AHEAD_MIN_M, EASE_CORRIDOR_HALF_WIDTH_M, _corridor_ahead_distance,
)

SLOW_MPS = 0.5


def _dense(layer: dict | None) -> np.ndarray | None:
    if layer is None:
        return None
    arr = np.zeros(layer["shape"], dtype=np.float32)
    idx = np.asarray(layer["indices"], dtype=np.int64)
    if idx.size:
        arr.ravel()[idx] = np.asarray(layer["values"], dtype=np.float32)
    return arr


def _occupied_world_points(grid: OccupancyGrid) -> np.ndarray:
    """Occupied-cell world coordinates via the monitor's own conversion."""
    rr, cc = np.nonzero(grid.obstacle)
    if not len(rr):
        return np.zeros((0, 2), dtype=float)
    ch, sh = math.cos(grid.heading), math.sin(grid.heading)
    ex = grid.max_x - (rr + 0.5) * grid.res
    ey = grid.max_y - (cc + 0.5) * grid.res
    wx = grid.origin[0] + ex * ch - ey * sh
    wy = grid.origin[1] + ex * sh + ey * ch
    return np.stack([wx, wy], axis=1)


def corridor_ahead_m(grid: OccupancyGrid, lane_ref) -> float | None:
    if lane_ref is None or len(lane_ref) < 2:
        return None
    occ = _occupied_world_points(grid)
    if not len(occ):
        return None
    return _corridor_ahead_distance(
        occ, np.asarray(lane_ref, dtype=float),
        EASE_CORRIDOR_HALF_WIDTH_M, EASE_AHEAD_MIN_M)


def classify_frame(frame: dict, ahead_max_m: float) -> dict:
    gp = frame.get("geometry_probe")
    if not gp:
        return {"class": "E_other", "why": "no_geometry_probe"}
    grid_payload = gp.get("grid")
    lane_ref = gp.get("lane_ref")
    corridor = None
    if grid_payload is not None and lane_ref and len(lane_ref) >= 2:
        grid = OccupancyGrid(int(grid_payload["n_rows"]),
                             int(grid_payload["n_cols"]),
                             float(grid_payload["res_m"]),
                             origin=tuple(grid_payload["origin"]),
                             heading=float(grid_payload["heading_rad"]))
        for key in ("occupancy", "obstacle", "drivable", "observed"):
            dense = _dense(grid_payload.get(key))
            if dense is not None and dense.shape == getattr(grid, key).shape:
                getattr(grid, key)[:] = dense
        raw = corridor_ahead_m(grid, np.asarray(lane_ref, dtype=float))
        # Clear beyond the window reads as clear; an intrusion inside it is
        # the distance to the nearest intruder.
        corridor = (None if raw is None or raw > ahead_max_m
                    else round(float(raw), 3))
    lane_source = str(gp.get("lane_source") or "")
    planner_path = gp.get("planner_path")
    selected = gp.get("selected_path") or gp.get("control_path")
    n_cand = len(gp.get("candidates") or [])
    rejects = gp.get("planner_rejects") or {}
    speed = float(frame.get("speed") or 0.0)
    verdict = gp.get("verdict") or {}
    if lane_source == "perception-unavailable" and planner_path is None:
        cls = "C_perception_missing"
    elif n_cand == 0 and planner_path is None:
        cls = "A_no_candidates"
    elif planner_path is None and n_cand > 0 and rejects:
        # The planner generated candidates and rejected every one of them
        # internally; the reject counters (lane_cross / body_cross / ...) are
        # the reason "no drivable path" does not name.
        cls = "A2_candidates_all_rejected"
    elif planner_path is not None and not selected:
        cls = ("B1_rejected_corridor_free" if corridor is None
               else "B2_rejected_blocked_ahead")
    elif selected:
        cls = "D1_accepted_slow" if speed < SLOW_MPS else "D2_accepted_moving"
    else:
        cls = "E_other"
    return {"class": cls,
            "recorded_reason": verdict.get("reason") or frame.get("reason"),
            "recorded_level": verdict.get("level"),
            "planner_rejects": (dict(rejects) if rejects else None),
            "corridor_ahead_m": corridor,
            "n_candidates": n_cand,
            "lane_source": lane_source,
            "speed_mps": round(speed, 3),
            "throttle": frame.get("throttle"),
            "brake": frame.get("brake")}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--telemetry", required=True, nargs="+", type=Path,
                    help="telemetry.json files with geometry captures")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ahead-max-m", type=float, default=10.0,
                    help="corridor beyond this distance counts as clear")
    args = ap.parse_args()
    if args.out.exists():
        ap.error("use a fresh report path; preserve earlier evidence")

    rows: list[dict] = []
    per_file: list[dict] = []
    for tel in args.telemetry:
        frames = json.loads(tel.read_text(encoding="utf-8"))
        file_rows = []
        for i, frame in enumerate(frames):
            row = {"telemetry": str(tel), "frame_index": i,
                   "cmd_seq": frame.get("cmd_seq"),
                   **classify_frame(frame, args.ahead_max_m)}
            file_rows.append(row)
        rows.extend(file_rows)
        per_file.append({"telemetry": str(tel), "frames": len(file_rows),
                         "classes": dict(Counter(r["class"] for r in file_rows))})

    summary = Counter(r["class"] for r in rows)
    reject_split = Counter(
        f"{r['class']} | {r.get('recorded_reason')}"
        for r in rows if r["class"].startswith(("B", "A2")))
    report = {
        "schema": "stall-geometry-check-v1",
        "ahead_max_m": args.ahead_max_m,
        "slow_mps": SLOW_MPS,
        "per_file": per_file,
        "summary_classes": dict(summary),
        "rejected_frames_by_recorded_reason": dict(reject_split),
        "frames": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"frames={len(rows)} "
          f"classes={json.dumps(report['summary_classes'], ensure_ascii=False)}")
    if reject_split:
        print("rejected frames by recorded reason:")
        for k, n in reject_split.most_common():
            print(f"  {n:3d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
