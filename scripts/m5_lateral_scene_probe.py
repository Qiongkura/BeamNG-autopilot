"""One tick of lateral ground truth, with every number the gates use.

The strict lane question ("may this geometry steer the car?") is decided by
``lane.reference.select_lane_reference`` from three inputs: where the ego is
on the nav route, where the detected paint is, and what the occupancy grid
says is drivable.  Telemetry records the DECISION but not those inputs, so a
refusal cannot be told apart from a perception error - and the two need
opposite fixes.

This probe drives nothing.  It warms the perception stack the way a real run
does, then prints, per tick:

* the ego's signed lateral offset from the nav route (route = road
  centreline; the own lane under RHT sits ~half a lane to the RIGHT);
* every detected marking with its lateral offset in the ROUTE frame, kind,
  colour and world length - so "which line is this" is answerable;
* the pairing result (width, span, width spread, centre) and every pair
  candidate the scorer considered;
* the drivable/occupied lateral band at several distances ahead (car frame);
* the lane-reference decision and, when refused, the offset and limit that
  caused it.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_lateral_scene_probe.py --runtime tech \\
        --attach --teleport 779.7 735.6 -13 --goal 868.3 744.9 --ticks 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from beamng_autopilot import config, fsd_drive
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.planning import local_route

# Lateral band centres (forward distance in metres) sampled from the grid.
BAND_S_M = (3.0, 6.0, 9.0, 12.0)


def lat_in_ref(pts, ref, pos) -> float | None:
    """Median signed lateral (left +) of ``pts`` against polyline ``ref``.

    Same convention as ``planning.arbiter.lane_side_offset_m``: positive
    means the point sits LEFT of the reference direction of travel.
    """
    if pts is None or ref is None or len(ref) < 2:
        return None
    p = np.asarray(pts, dtype=float)[:, :2]
    if p.ndim != 2 or len(p) == 0:
        return None
    r = np.asarray(ref, dtype=float)[:, :2]
    if not np.isfinite(p).all():
        p = p[np.isfinite(p).all(axis=1)]
        if len(p) == 0:
            return None
    seg = r[1:] - r[:-1]
    l2 = np.maximum((seg * seg).sum(axis=1), 1e-12)
    rel = p[:, None, :] - r[None, :-1, :]
    t = np.clip(np.einsum("kmi,mi->km", rel, seg) / l2[None, :], 0.0, 1.0)
    proj = r[None, :-1, :] + t[..., None] * seg[None, :, :]
    d = np.linalg.norm(proj - p[:, None, :], axis=2)
    bi = np.argmin(d, axis=1)
    rows = np.arange(len(p))
    sy = p[:, 1] - proj[rows, bi, 1]
    sx = p[:, 0] - proj[rows, bi, 0]
    cross = seg[bi, 0] * sy - seg[bi, 1] * sx
    val = np.where(cross > 0, 1.0, -1.0) * d[rows, bi]
    keep = d[rows, bi] <= 25.0
    if not keep.any():
        return None
    return float(np.median(val[keep]))


def lat_in_car(pts, pos, heading: float) -> float | None:
    """Median signed lateral (left +, car frame) of world points."""
    if pts is None:
        return None
    p = np.asarray(pts, dtype=float)[:, :2]
    if p.ndim != 2 or len(p) == 0:
        return None
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) == 0:
        return None
    fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
    left = np.array([-fwd[1], fwd[0]])
    rel = p - np.asarray(pos[:2], dtype=float)
    near = np.linalg.norm(rel, axis=1) <= 25.0
    if not near.any():
        return None
    return float(np.median(rel[near] @ left))


def gap_runs(xs: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of the given integer indices."""
    out: list[tuple[int, int]] = []
    for x in xs:
        if out and x == out[-1][1] + 1:
            out[-1] = (out[-1][0], x)
        else:
            out.append((x, x))
    return out


def band_span(mask_row, res: float, extent: float) -> dict:
    """Lateral intervals (left +, car frame) of the nonzero cells in a row."""
    cols = np.nonzero(mask_row)[0]
    if cols.size == 0:
        return {"n": 0, "spans": [], "widest_m": 0.0}
    spans = []
    for a, b in gap_runs([int(c) for c in cols]):
        # grid.world_to_cell maps +y (car-left) to a SMALLER column, so a
        # cell's car-frame lateral is extent - (col + 0.5) * res.
        lat_a = extent - (a + 0.5) * res
        lat_b = extent - (b + 0.5) * res
        spans.append([round(float(lat_a), 2), round(float(lat_b), 2)])
    widest = max(abs(s[1] - s[0]) for s in spans)
    return {"n": int(cols.size), "spans": spans, "widest_m": round(widest, 2)}


def grid_bands(grid, ego_speed: float) -> dict:
    """Drivable / occupied lateral bands at several distances ahead."""
    if grid is None:
        return {}
    out: dict = {}
    res = float(grid.res)
    extent = float(grid.extent)
    for s in BAND_S_M:
        row = int(round((extent - s) / res - 0.5))
        if row < 0 or row >= grid.obstacle.shape[0]:
            continue
        obs = getattr(grid, "observed", None)
        out[f"s={s:.0f}m"] = {
            "drivable": band_span(grid.drivable[row] > 0, res, extent),
            "observed": (band_span(obs[row] > 0, res, extent)
                         if obs is not None and getattr(obs, "size", 0)
                         else None),
            "obstacle": band_span(grid.obstacle[row] > 0, res, extent),
        }
    return out


def mask_stats(out) -> dict:
    """Semantic mask coverage: road / line pixels and their fractions."""
    sem = (out.head_outputs or {}).get("semantic")
    masks = getattr(sem, "masks", {}) or {}
    stats: dict = {}
    total = None
    for name in ("road", "line"):
        m = masks.get(name)
        if m is None:
            stats[f"{name}_frac"] = None
            continue
        arr = np.asarray(m, dtype=bool)
        total = arr.size
        stats[f"{name}_px"] = int(arr.sum())
        stats[f"{name}_frac"] = (round(float(arr.mean()), 4) if total else None)
    return stats


def dump_tick(t: float, out, nav_route, pos, heading, v) -> dict:
    """Every lateral input and decision of one tick, JSON-safe."""
    route_local = local_route(pos, heading, nav_route)
    rec: dict = {
        "t": round(float(t), 2),
        "pos": [round(float(x), 2) for x in pos[:3]],
        "heading_deg": round(math.degrees(float(heading)), 2),
        "speed": round(float(v), 3),
        "ego_lat_route_m": (None if route_local is None
                            else _round(lat_in_ref(np.asarray(pos[:2])[None, :],
                                                   route_local, pos))),
        "route_pts": (0 if nav_route is None else int(len(nav_route))),
    }
    grid = getattr(getattr(out, "scene", None), "grid", None)
    rec["bands"] = grid_bands(grid, float(v))
    sem = (out.head_outputs or {}).get("semantic")
    marks = list(getattr(sem, "meta", {}).get("markings", []) or [])
    rec["markings"] = []
    for m in marks:
        world = np.asarray(getattr(m, "world", np.zeros((0, 2))), dtype=float)
        if world.ndim != 2 or world.shape[0] < 2:
            continue
        length = float(np.sum(np.linalg.norm(np.diff(world[:, :2], axis=0),
                                             axis=1)))
        rec["markings"].append({
            "kind": str(getattr(m, "kind", "")),
            "color": str(getattr(m, "color", "")),
            "conf": round(float(getattr(m, "confidence", 0.0) or 0.0), 2),
            "pts": int(len(world)),
            "len_m": round(length, 1),
            "lat_car_m": _round(lat_in_car(world[:, :2], pos, heading)),
            "lat_route_m": _round(lat_in_ref(world[:, :2], route_local, pos)),
        })
    meta = dict(getattr(out, "meta", {}) or {})
    rec["lane"] = {
        "src": meta.get("lane_src_sel"),
        "reject": meta.get("lane_reject_reason"),
        "side_off_m": meta.get("lane_side_off_m"),
        "side_limit_m": meta.get("lane_side_limit_m"),
        "pair_width_m": meta.get("pair_width_m"),
        "pair_span_m": meta.get("pair_span_m"),
        "pair_conf": meta.get("pair_conf"),
        "pair_paired": meta.get("pair_paired"),
        "pair_sources": meta.get("pair_sources"),
        "lane_marks_n": meta.get("lane_marks_n"),
        "lane_from": meta.get("lane_from"),
        "divider": meta.get("lane_divider"),
    }
    dbg = dict(meta.get("lane_pair_debug") or {})
    rec["pair"] = {
        "mode": dbg.get("mode"),
        "width": dbg.get("width"),
        "width_spread": dbg.get("pair_width_spread"),
        "span": dbg.get("span"),
        "center0": dbg.get("center0"),
        "left_med": dbg.get("left_med"),
        "right_med": dbg.get("right_med"),
        "candidates": dbg.get("pair_candidates"),
        "rejects": dbg.get("pair_rejects"),
    }
    env = getattr(out, "lane_envelope", None)
    if env is not None:
        rec["envelope"] = {"width_m": round(float(env.width_m), 2),
                           "conf": round(float(env.confidence), 2),
                           "paired": bool(env.paired),
                           "source": str(env.source),
                           "uncertainty_m": round(
                               float(getattr(env, "uncertainty_m", 0.0) or 0.0), 2)}
        rec["envelope"]["left_lat_route_m"] = _round(
            lat_in_ref(getattr(env, "left", None), route_local, pos))
        rec["envelope"]["right_lat_route_m"] = _round(
            lat_in_ref(getattr(env, "right", None), route_local, pos))
        rec["envelope"]["center_lat_route_m"] = _round(
            lat_in_ref(getattr(env, "center", None), route_local, pos))
    rec.update(mask_stats(out))
    rec["plan_blocked"] = meta.get("plan_blocked")
    rec["tracks"] = track_dump(out, pos, heading)
    return rec


def track_dump(out, pos, heading: float, limit: int = 8) -> dict:
    """Tracked objects graded by the risk layer, nearest first.

    The risk layer stops the car on a track inside the corridor, so when
    it stops a car standing on an open road the FIRST question is which
    track did it - and whether that track is a real object or a return
    from the ego's own footprint.
    """
    tracks = list(getattr(out, "tracks", None) or [])
    if not tracks:
        return {"n": 0, "nearest": [], "categories": {}}
    p = np.asarray(pos[:2], dtype=float)
    fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
    left = np.array([-fwd[1], fwd[0]])
    cats: dict = {}
    rows = []
    for tr in tracks:
        x = float(getattr(tr, "x", 0.0) or 0.0)
        y = float(getattr(tr, "y", 0.0) or 0.0)
        cat = str(getattr(tr, "category", "object") or "object")
        cats[cat] = cats.get(cat, 0) + 1
        rel = np.array([x, y]) - p
        rows.append({
            "cat": cat,
            "label": str(getattr(tr, "label", "") or ""),
            "dist_m": round(float(np.linalg.norm(rel)), 2),
            "along_m": round(float(rel @ fwd), 2),
            "lat_m": round(float(rel @ left), 2),
            "matches": int(getattr(tr, "matches", 0) or 0),
            "lost": int(getattr(tr, "lost", 0) or 0),
            "speed_mps": round(float(math.hypot(
                float(getattr(tr, "vx", 0.0) or 0.0),
                float(getattr(tr, "vy", 0.0) or 0.0))), 2),
        })
    rows.sort(key=lambda r: r["dist_m"])
    return {"n": len(tracks), "categories": cats, "nearest": rows[:limit]}


def _round(value):
    return None if value is None else round(float(value), 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="tech")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--map", type=str, default=None)
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"))
    ap.add_argument("--seg-model", type=str, default=None)
    ap.add_argument("--line-seg-model", type=str, default=None)
    ap.add_argument("--ticks", type=int, default=3)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    conn = BeamNGConnector(
        args.map or "italy", "etk800",
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    drive_args = SimpleNamespace(
        runtime=args.runtime, attach=args.attach, map=args.map,
        seconds=1.0, speed=3.0, steps=3, cam_w=536, cam_h=403,
        seg_model=args.seg_model, line_seg_model=args.line_seg_model,
        teleport=list(args.teleport) if args.teleport else None,
        out=None, lane_mode="sensor", strict=True, allow_unplaced=True,
        corridor_lane=False, paved_lane=False,
        e2e_model=None, no_e2e=True, bc_model=None, no_bc=True,
        dqn_model=None, no_dqn=True,
        goal=list(args.goal) if args.goal else None,
        no_signal=True, ring="front", traffic=0, vis=0, no_shadow=True)
    session = fsd_drive.FSDriveSession(drive_args)
    records = []
    try:
        conn.open(launch=not args.attach)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            conn.load_scenario()
        if drive_args.teleport:
            x, y, yaw = drive_args.teleport
            conn.safe_teleport(float(x), float(y), heading_deg=float(yaw))
        nav_route, _nav_ref, _rl, _rr = session._build_route(conn)
        (_rp, stack, _e2e, _bc, _dqn) = session._setup_runtime(conn, drive_args)
        (_pw, ok, rc) = session._prewarm_and_place(
            conn, stack, nav_route, 2)
        print(f"[probe] prewarm: perceived_lane={ok} rc={rc}", flush=True)
        if rc:
            return rc
        for i in range(max(1, int(args.ticks))):
            st = conn.get_state()
            pos = np.asarray(st.pos, dtype=float)
            heading = float(st.heading)
            route_local = local_route(pos, heading, nav_route)
            out = stack.tick(st=st, route_ref=route_local,
                             time_budget_s=1.0)
            rec = dump_tick(i, out, nav_route, pos, heading, float(st.speed))
            records.append(rec)
            print(json.dumps(rec, ensure_ascii=False, indent=1), flush=True)
    finally:
        try:
            stack.close()
        except Exception:
            pass
        conn.close()
    if args.json and records:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(records, ensure_ascii=False,
                                             indent=1), encoding="utf-8")
        print(f"[probe] wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
