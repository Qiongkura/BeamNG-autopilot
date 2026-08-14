"""Offline replay of lane-centring telemetry for M5 development.

Replays run telemetry through ``pair_lane_markings`` + ``LaneTracker`` the
same way the live loop consumes snapshots, and prints the lane metrics the
run report uses.  Marking confidence is not recorded in telemetry, so the
classic-CV kinds are given representative confidence values here.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot.lane import LaneFrame, LaneTracker, choose_sensor_lane, pair_lane_markings
from beamng_autopilot.perception import Obstacle
from beamng_autopilot.planner import LocalPlanner
from beamng_autopilot.vision.lanes import LaneMarking


CONF_BY_KIND = {"solid": 0.8, "dashed": 0.55, "unknown": 0.5}
LOOKAHEAD_M = 30.0


def boxes_from_row(row):
    out = []
    for b in row["extra"].get("boxes", []):
        if len(b) < 5:
            continue
        x, y, hw, hh, cat = b[:5]
        axis = None
        half_len = half_thick = 0.0
        if len(b) >= 8:
            ax = np.asarray(b[5], dtype=float)
            if np.linalg.norm(ax) > 1e-9 and float(b[6]) > 0.0:
                axis = ax / np.linalg.norm(ax)
                half_len = float(b[6])
                half_thick = float(b[7])
        out.append(Obstacle(
            x=float(x), y=float(y), half_w=float(hw), half_h=float(hh),
            category=str(cat), label=str(cat), axis=axis,
            half_len=half_len, half_thick=half_thick))
    return out


def lane_frame_from_telemetry(row, pos, fwd):
    """Rebuild the chosen LaneFrame from the live telemetry fields."""
    ex = row["extra"]
    src = ex.get("lane_src") or ""
    if not src:
        return None
    paired = bool(ex.get("lane_paired"))
    conf = float(ex.get("lane_conf") or 0.0)
    span = float(ex.get("lane_span") or 0.0)
    width = float(ex.get("lane_w") or 3.5)
    dbg = ex.get("lidar_dbg") or {}
    lane_lat = ex.get("lane_lat")
    left = np.array([-fwd[1], fwd[0]])
    stations = np.linspace(0.0, max(span, 6.0), 24)
    clat = 0.0 if lane_lat is None else float(lane_lat)
    center = pos + stations[:, None] * fwd + clat * left
    left_pts = right_pts = None
    side = ""
    if src == "lidar" and not paired:
        side = dbg.get("fallback", "")
        edge_med = dbg.get("edge_med")
        if side in ("left", "right") and edge_med is not None:
            edge_pts = pos + stations[:, None] * fwd + float(edge_med) * left
            if side == "left":
                left_pts = edge_pts
            else:
                right_pts = edge_pts
    sources = (src,)
    if not paired and src == "lidar":
        sources = (src, side or "right")
    return LaneFrame(center=center, left=left_pts, right=right_pts,
                     width=width, confidence=conf, span_m=span,
                     sources=sources, paired=paired)


def station_lateral_offset(drive, route, station_m: float):
    """Signed lateral offset of the drive path at ``station_m`` ahead.

    Uses the route segment at the same arclength instead of the nearest
    point on the route, so the keep-right offset is actually measured.
    Positive is left of the route.
    """
    dr = np.asarray(drive[:, :2], dtype=float)
    rt = np.asarray(route[:, :2], dtype=float)
    if len(dr) < 2 or len(rt) < 2:
        return None
    d_cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dr, axis=0), axis=1))])
    r_cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(rt, axis=0), axis=1))])
    target = min(station_m, float(d_cum[-1]) * 0.95, float(r_cum[-1]) * 0.95)
    if target < 5.0:
        return None
    di = int(np.searchsorted(d_cum, target, side="right"))
    di = max(1, min(len(dr) - 1, di))
    t_d = (target - d_cum[di - 1]) / max(1e-9, d_cum[di] - d_cum[di - 1])
    dp = dr[di - 1] + t_d * (dr[di] - dr[di - 1])
    ri = int(np.searchsorted(r_cum, target, side="right"))
    ri = max(1, min(len(rt) - 1, ri))
    t_r = (target - r_cum[ri - 1]) / max(1e-9, r_cum[ri] - r_cum[ri - 1])
    rp = rt[ri - 1] + t_r * (rt[ri] - rt[ri - 1])
    seg = rt[ri] - rt[ri - 1]
    seg /= max(1e-9, float(np.linalg.norm(seg)))
    left = np.array([-seg[1], seg[0]])
    return float((dp - rp) @ left)


def markings_from_row(row):
    out = []
    for m in row["extra"].get("markings", []):
        world = np.asarray(m["poly"], dtype=float)
        if world.ndim != 2 or world.shape[1] < 2 or len(world) < 2:
            continue
        kind = m.get("kind", "unknown")
        out.append(LaneMarking(
            world=world,
            pixels=np.zeros((len(world), 2)),
            color=m.get("color", "white"),
            kind=kind,
            confidence=CONF_BY_KIND.get(kind, 0.5),
        ))
    return out


def replay(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    planner = LocalPlanner()
    lane_lats = []
    lane_srcs = []
    path_lats = []
    path_modes = []
    for row in rows:
        ex = row["extra"]
        if ex.get("auto") != 1:
            continue
        pos = np.asarray(row["pos"], dtype=float)[:2]
        heading = float(row["heading"])
        fwd = np.asarray(row["dir_vec"], dtype=float)[:2]
        fn = float(np.linalg.norm(fwd))
        if fn > 1e-9:
            fwd = fwd / fn
        else:
            fwd = np.array([math.cos(heading), math.sin(heading)])
        route = np.asarray(ex.get("rte") or [], dtype=float)
        if len(route) < 3:
            continue
        lane_frame = lane_frame_from_telemetry(row, pos, fwd)
        obstacles = boxes_from_row(row)
        solid_lines = markings_from_row(row)
        # Telemetry's ``rte`` is already the route window starting at the
        # car, so the local window index is 0 regardless of the live
        # full-route nearest index.
        nearest = 0
        drive, _ = planner.plan(
            route, obstacles, pos, heading, nearest,
            solid_lines=solid_lines, sensor_lane=lane_frame)
        drive = np.asarray(drive, dtype=float)
        lat = station_lateral_offset(drive, route, LOOKAHEAD_M)
        if lat is not None:
            path_lats.append(lat)
            path_modes.append(getattr(planner, "last_lane_mode", "nav"))
        if lane_frame is not None:
            left = np.array([-fwd[1], fwd[0]])
            c0 = np.asarray(lane_frame.center[0], dtype=float)[:2]
            lane_lats.append(float((c0 - pos) @ left))
            lane_srcs.append(lane_frame.sources[0])
    lats = np.asarray(lane_lats, dtype=float)
    srcs = np.asarray(lane_srcs)
    plats = np.asarray(path_lats, dtype=float)
    pmodes = np.asarray(path_modes)
    return {
        "lane_frames": int(len(lats)),
        "lane_src": {s: int(np.sum(srcs == s)) for s in sorted(set(srcs))},
        "median_lane_lat": float(np.median(lats)) if len(lats) else None,
        "mean_lane_lat": float(np.mean(lats)) if len(lats) else None,
        "max_abs_lane_lat": float(np.max(np.abs(lats))) if len(lats) else None,
        "centered_ratio": float(np.mean(np.abs(lats) < 0.5)) if len(lats) else None,
        "pct_gt_1": float(np.mean(lats > 1.0)) if len(lats) else None,
        "pct_lt_neg1": float(np.mean(lats < -1.0)) if len(lats) else None,
        "path_frames": int(len(plats)),
        "path_modes": {m: int(np.sum(pmodes == m)) for m in sorted(set(pmodes))},
        "median_path_lat": float(np.median(plats)) if len(plats) else None,
        "mean_path_lat": float(np.mean(plats)) if len(plats) else None,
        "std_path_lat": float(np.std(plats)) if len(plats) else None,
        "path_centered_ratio": (
            float(np.mean(np.abs(plats) < 0.5)) if len(plats) else None),
        "path_right_band_ratio": (
            float(np.mean((plats <= -0.75) & (plats >= -2.0)))
            if len(plats) else None),
    }


def main() -> None:
    runs = [
        ("run67", str(Path(__file__).resolve().parent.parent
                      / "logs" / "live_runs" / "run_67"
                      / "telemetry_history.json")),
    ]
    for name, path in runs:
        print(f"== {name} ==")
        print(replay(path))


if __name__ == "__main__":
    main()
