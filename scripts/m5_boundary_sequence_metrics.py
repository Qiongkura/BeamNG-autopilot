"""T07: selectivity and persistence of the boundary detectors on a REAL sequence.

``m5_boundary_arms.py`` compares the arms on one frame.  This tool answers
the question the persistence filter was written for, over a multi-frame
recording of the same stretch:

* how many curb candidates the arms produce per frame, and how many of
  them recur in the SAME world cells across frames (persistence);
* whether consecutive frames' candidates agree with each other (a
  world-space recurrence rate, independent of the persistence filter);
* whether the candidates sit where a boundary should be according to a
  DIFFERENT sensor - the semantic road mask: on its edge, inside the
  pavement (a false boundary for a kerb detector) or outside it;
* how tight the candidate set is (PCA residual of the candidate cloud in
  the plane) - a detector that fires on every grass tuft spreads out.

There is no independent geometric truth in the recording, so the output
is a set of measurements with their limits, never an accuracy claim.  The
mask is a cross-sensor consistency check, not truth (plan §7.1).

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_boundary_sequence_metrics.py \\
        --seq logs/goal_20260921/boundary_seq.npz \\
        --json logs/goal_20260921/boundary_seq_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.lane.boundary_evidence import (  # noqa: E402
    CURB_HEIGHT_M,
    CurbPersistence,
    curb_candidates,
    line_consistent_candidates,
    pavement_edges,
)
from beamng_autopilot.vision.projection import CameraModel  # noqa: E402

DEFAULT_EDGE_PX = 6.0
DEFAULT_RECALL_M = 0.5


def _frame_indices(z) -> list[int]:
    i = 0
    while f"points_{i}" in z.files:
        i += 1
    return list(range(i))


def _camera_from(z, i: int):
    keys = ("cam_offset_", "cam_fwd_", "cam_up_", "cam_fov_", "cam_w_", "cam_h_")
    if any(f"{k}{i}" not in z.files for k in keys):
        return None
    return CameraModel(
        offset=np.asarray(z[f"cam_offset_{i}"], dtype=float).ravel(),
        fwd_local=np.asarray(z[f"cam_fwd_{i}"], dtype=float).ravel(),
        up_local=np.asarray(z[f"cam_up_{i}"], dtype=float).ravel(),
        fov_deg=float(np.asarray(z[f"cam_fov_{i}"]).ravel()[0]),
        width=int(np.asarray(z[f"cam_w_{i}"]).ravel()[0]),
        height=int(np.asarray(z[f"cam_h_{i}"]).ravel()[0]))


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Mask-edge pixels ``(row, col)`` via a 4-neighbour erosion."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return np.empty((0, 2), dtype=int)
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    rows, cols = np.nonzero(m & ~er)
    return np.column_stack([rows, cols]) if len(rows) else np.empty((0, 2), int)


def _min_px_distance(rows, cols, boundary: np.ndarray) -> np.ndarray:
    """Distance in pixels from each (row, col) to the nearest boundary pixel."""
    if boundary.size == 0 or len(rows) == 0:
        return np.full(len(rows), np.nan)
    out = np.full(len(rows), np.inf)
    step = 2048
    for s in range(0, len(rows), step):
        r = rows[s:s + step, None].astype(float)
        c = cols[s:s + step, None].astype(float)
        d = np.hypot(r - boundary[None, :, 0], c - boundary[None, :, 1])
        out[s:s + step] = d.min(axis=1)
    return out


def _pca_residual(points) -> float | None:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or len(p) < 4:
        return None
    q = p - p.mean(axis=0)
    try:
        _, sv, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = vt[0]
    normal = np.array([-axis[1], axis[0]])
    return float(np.median(np.abs(q @ normal)))


def _cells(points, cell_m: float) -> set[tuple[int, int]]:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or len(p) == 0:
        return set()
    return {(int(math.floor(x / cell_m)), int(math.floor(y / cell_m)))
            for x, y in p[:, :2]}


def _lateral_offset(points, pos, heading: float):
    """Ego-frame lateral offset (+ = left) of world points, one per point."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or len(p) == 0:
        return np.empty(0, dtype=float)
    h = float(heading)
    left = np.array([-math.sin(h), math.cos(h)])
    return (p[:, :2] - np.asarray(pos, dtype=float)[:2]) @ left


def _chain_points(line_ev) -> dict:
    """``{"published": ndarray, "other": [ndarray, ...]}`` for one frame."""
    other = []
    for ch in line_ev.meta.get("other_chains") or []:
        pts = np.asarray(ch.get("points") or [], dtype=float)
        if pts.ndim == 2 and len(pts):
            other.append(pts)
    return {"published": np.asarray(line_ev.points, dtype=float),
            "other": other}


def _mask_agreement(points, mask, cam, pos, heading: float, ground_z: float,
                    boundary: np.ndarray, edge_px: float) -> dict:
    """Where do these world points sit relative to the semantic road mask?

    Projects each candidate onto the ground plane into the image and
    classifies the pixel: on the mask edge (expected for a boundary), well
    inside the pavement (a false boundary for a kerb detector) or well
    outside it.  Cross-sensor CONSISTENCY, not truth.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or len(p) == 0:
        return {}
    world = np.column_stack([p[:, :2], np.full(len(p), float(ground_z))])
    u, v, valid = cam.project(world, pos, heading)
    h, w = mask.shape[:2]
    inside_img = (valid & np.isfinite(u) & np.isfinite(v)
                  & (u >= 0) & (u < w) & (v >= 0) & (v < h))
    n_proj = int(np.count_nonzero(inside_img))
    out = {"proj_n": n_proj}
    if not n_proj:
        return out
    ui = np.clip(np.round(u[inside_img]).astype(int), 0, w - 1)
    vi = np.clip(np.round(v[inside_img]).astype(int), 0, h - 1)
    on_road = mask[vi, ui]
    dist = _min_px_distance(vi, ui, boundary)
    near_edge = np.isfinite(dist) & (dist <= float(edge_px))
    out["proj_on_edge"] = int(np.count_nonzero(near_edge))
    out["proj_inside_road"] = int(np.count_nonzero(on_road & ~near_edge))
    out["proj_outside_road"] = int(np.count_nonzero(~on_road & ~near_edge))
    out["proj_edge_px_p50"] = (None if not np.isfinite(dist).any()
                               else round(float(np.nanmedian(dist)), 2))
    return out


def _recurrence(prev, cur, tol_m: float) -> tuple[int, int]:
    """(matched, total) candidates of ``cur`` within ``tol_m`` of ``prev``."""
    a = np.asarray(prev, dtype=float)
    b = np.asarray(cur, dtype=float)
    if b.ndim != 2 or len(b) == 0:
        return 0, 0
    if a.ndim != 2 or len(a) == 0:
        return 0, len(b)
    matched = 0
    for q in b[:, :2]:
        if np.min(np.linalg.norm(a[:, :2] - q, axis=1)) <= tol_m:
            matched += 1
    return matched, len(b)


def analyse(seq_path: str, *, edge_px: float = DEFAULT_EDGE_PX,
            recall_m: float = DEFAULT_RECALL_M, cell_m: float = 0.5,
            max_range_m: float = 45.0) -> dict:
    z = np.load(seq_path, allow_pickle=True)
    frames = _frame_indices(z)
    meta = {k: (np.asarray(z[k]).ravel().tolist() if k in z.files else None)
            for k in ("evidence_level", "map", "vehicle", "role", "step_m")}
    persistence = CurbPersistence()
    rows: list[dict] = []
    prev_raw = None
    for i in frames:
        pts = np.asarray(z[f"points_{i}"], dtype=float)
        pos = np.asarray(z[f"pos_{i}"], dtype=float).ravel()[:3]
        heading = float(np.asarray(z[f"heading_{i}"]).ravel()[0])
        ground_z = float(np.asarray(z[f"ground_z_{i}"]).ravel()[0])
        t_s = float(np.asarray(z[f"t_{i}"]).ravel()[0])
        mask = (np.asarray(z[f"mask_{i}"], dtype=bool)
                if f"mask_{i}" in z.files else None)
        cam = _camera_from(z, i)
        row: dict = {"frame": i, "points": int(len(pts)),
                     "pos": [round(float(v), 3) for v in pos],
                     "heading": round(heading, 5),
                     "ground_z": round(ground_z, 3),
                     "t_s": round(t_s, 3)}
        if f"ann_counts_{i}" in z.files:
            # engine annotation classes: the independent label for WHICH
            # kind of edge this stretch has (GUARD_RAIL / GRASS / SIDEWALK)
            try:
                counts = json.loads(str(np.asarray(
                    z[f"ann_counts_{i}"]).ravel()[0]))
            except Exception:
                counts = {}
            if counts:
                top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
                row["ann_top"] = {str(k): int(v) for k, v in top}
                for key in ("GUARD_RAIL", "GRASS", "NATURE", "SIDEWALK",
                            "ASPHALT", "MUD", "SAND", "ROCK"):
                    row[f"ann_{key.lower()}"] = int(counts.get(key, 0))
        if i > 0:
            p0 = np.asarray(z["pos_0"], dtype=float).ravel()[:3]
            row["travel_m"] = round(float(np.linalg.norm(pos[:2] - p0[:2])), 3)
        raw = curb_candidates(pts, ground_z=ground_z, pos=pos,
                              max_range_m=max_range_m)
        row["curb_raw_n"] = int(len(raw.points))
        row["curb_threshold_p50_m"] = raw.meta.get("threshold_p50_m")
        row["curb_local_gap_p50_m"] = raw.meta.get("local_gap_p50_m")
        row["curb_residual_p50_m"] = (None if len(raw.points) == 0
                                      else round(_pca_residual(
                                          raw.points), 4))
        kept = persistence.update(raw.points if len(raw.points) else None, t_s)
        row["curb_persisted_n"] = int(len(kept.points))
        # Arm B: the same raw candidates, filtered by "a boundary is a
        # curve beside the car", then intersected with the persistent cells.
        line = line_consistent_candidates(raw.points, pos, heading)
        line_pts = line.points
        row["curb_line_meta"] = dict(line.meta)
        row["curb_line_n"] = int(len(line_pts))
        # Two-sidedness: a stretch whose BOTH boundaries are published is the
        # case the lane reference needs; measure it from every usable chain,
        # not only from the longest one that gets published.
        chains = _chain_points(line)
        all_chains = ([chains["published"]] if len(chains["published"])
                      else []) + chains["other"]
        offs = [_lateral_offset(c, pos, heading) for c in all_chains]
        row["curb_line_chains"] = len(all_chains)
        row["curb_line_left_chains"] = sum(
            1 for o in offs if len(o) and float(np.median(o)) > 0.0)
        row["curb_line_right_chains"] = sum(
            1 for o in offs if len(o) and float(np.median(o)) < 0.0)
        row["curb_line_both_sides"] = bool(
            row["curb_line_left_chains"] and row["curb_line_right_chains"])
        row["curb_line_residual_p50_m"] = (
            None if len(line_pts) == 0 else round(_pca_residual(line_pts), 4))
        if len(line_pts) and len(kept.points):
            d = np.linalg.norm(
                line_pts[:, None, :] - kept.points[None, :, :], axis=2)
            both = line_pts[d.min(axis=1) <= cell_m]
        else:
            both = np.empty((0, 2), dtype=float)
        row["curb_line_persisted_n"] = int(len(both))
        cells_raw = _cells(raw.points, cell_m)
        cells_kept = _cells(kept.points, cell_m)
        row["curb_world_cells_raw"] = len(cells_raw)
        row["curb_world_cells_persisted"] = len(cells_kept)
        matched, total = _recurrence(prev_raw, raw.points, recall_m)
        row["curb_recur_prev_matched"] = matched
        row["curb_recur_prev_total"] = total
        row["curb_recur_prev_rate"] = (None if total == 0
                                       else round(matched / total, 4))
        prev_raw = raw.points if len(raw.points) else np.empty((0, 2))
        if mask is not None and cam is not None:
            edge = pavement_edges(mask, cam, pos, heading, ground_z=ground_z)
            row["pavement_edge_n"] = int(len(edge.points))
            boundary = _boundary_pixels(mask)
            for tag, pts_arm in (("", raw.points), ("_line", line_pts)):
                stats = _mask_agreement(pts_arm, mask, cam, pos, heading,
                                        ground_z, boundary, edge_px)
                for k, v in stats.items():
                    row[f"{k}{tag}"] = v
            # per side: the RAW candidate set (what a boundary detector has to
            # work with) ...
            lat_raw = _lateral_offset(raw.points, pos, heading)
            for side, sel in (("left", lat_raw > 0.0),
                              ("right", lat_raw < 0.0)):
                stats = _mask_agreement(np.asarray(raw.points)[sel], mask, cam,
                                        pos, heading, ground_z, boundary,
                                        edge_px)
                for k, v in stats.items():
                    row[f"{k}_raw_{side}"] = v
            # ... and the CHAINS the filter actually published
            if all_chains:
                lat_all = np.concatenate(offs) if offs else np.empty(0)
                pts_all = np.vstack(all_chains)
                for side, sel in (("left", lat_all > 0.0),
                                  ("right", lat_all < 0.0)):
                    stats = _mask_agreement(pts_all[sel], mask, cam, pos,
                                            heading, ground_z, boundary,
                                            edge_px)
                    for k, v in stats.items():
                        row[f"{k}_line_{side}"] = v
        elif mask is not None:
            row["note"] = "mask present but no camera model recorded"
        rows.append(row)
    summary = {}
    def med(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(float(np.median(vals)), 4) if vals else None
    for key in ("curb_raw_n", "curb_persisted_n", "curb_line_n",
                "curb_line_persisted_n", "curb_residual_p50_m",
                "curb_line_residual_p50_m", "curb_recur_prev_rate",
                "pavement_edge_n", "proj_n", "proj_on_edge",
                "proj_inside_road", "proj_outside_road", "proj_edge_px_p50",
                "proj_n_line", "proj_on_edge_line", "proj_inside_road_line",
                "proj_outside_road_line", "proj_edge_px_p50_line",
                "proj_n_line_left", "proj_on_edge_line_left",
                "proj_edge_px_p50_line_left", "proj_n_line_right",
                "proj_on_edge_line_right", "proj_edge_px_p50_line_right",
                "proj_n_raw_left", "proj_on_edge_raw_left",
                "proj_edge_px_p50_raw_left", "proj_n_raw_right",
                "proj_on_edge_raw_right", "proj_edge_px_p50_raw_right",
                "curb_line_chains", "curb_world_cells_raw",
                "curb_world_cells_persisted"):
        summary[f"{key}_p50"] = med(key)
    # publication rate and two-sidedness are COUNTS, not medians
    summary["frames"] = len(rows)
    summary["line_published_frames"] = sum(
        1 for r in rows if (r.get("curb_line_n") or 0) > 0)
    summary["line_published_rate"] = (
        None if not rows else round(summary["line_published_frames"]
                                    / len(rows), 4))
    summary["both_sides_frames"] = sum(
        1 for r in rows if r.get("curb_line_both_sides"))
    summary["both_sides_rate"] = (
        None if not rows else round(summary["both_sides_frames"] / len(rows), 4))
    for side in ("left", "right"):
        for label in ("raw", "line"):
            n = sum(r.get(f"proj_n_{label}_{side}", 0) or 0 for r in rows)
            e = sum(r.get(f"proj_on_edge_{label}_{side}", 0) or 0 for r in rows)
            summary[f"edge_share_{label}_{side}"] = (
                None if not n else round(e / n, 4))
    summary["side_candidates_raw"] = {
        side: sum(r.get(f"proj_on_edge_raw_{side}", 0) or 0 for r in rows)
        for side in ("left", "right")}
    # The SLOPE of the stretch, from the recorded ground height: the plan's
    # acceptance matrix has a "slope" row, and a slope claim must come from
    # the data, not from the operator remembering a hill.
    travel = 0.0
    gz = [r["ground_z"] for r in rows]
    pos_all = [r["pos"] for r in rows]
    for a, b in zip(pos_all, pos_all[1:]):
        travel += float(math.hypot(b[0] - a[0], b[1] - a[1]))
    if len(gz) > 1 and travel > 0.0:
        summary["travel_m"] = round(travel, 3)
        summary["dz_m"] = round(float(gz[-1] - gz[0]), 3)
        summary["grade_pct"] = round(abs(float(gz[-1] - gz[0])) / travel * 100.0,
                                     2)
        local = [abs(float(gz[i + 1] - gz[i]))
                 / max(1e-6, float(math.hypot(
                     pos_all[i + 1][0] - pos_all[i][0],
                     pos_all[i + 1][1] - pos_all[i][1])))
                 for i in range(len(gz) - 1)]
        summary["max_local_grade_pct"] = round(max(local) * 100.0, 2)
    # Engine-label summary: which edge does this stretch actually have?
    for key in ("guard_rail", "grass", "nature", "sidewalk", "asphalt"):
        vals = [r.get(f"ann_{key}") for r in rows if isinstance(
            r.get(f"ann_{key}"), int)]
        summary[f"ann_{key}_p50"] = (None if not vals
                                     else int(np.median(vals)))
        summary[f"ann_{key}_frames"] = sum(1 for v in vals if v > 0)
    for tag in ("", "_line"):
        n = sum(r.get(f"proj_n{tag}", 0) or 0 for r in rows)
        edge_n = sum(r.get(f"proj_on_edge{tag}", 0) or 0 for r in rows)
        summary[f"edge_share{tag or '_raw'}"] = (None if not n
                                                 else round(edge_n / n, 4))
    total_kept = int(sum(r.get("curb_persisted_n", 0) for r in rows))
    summary["curb_persisted_last"] = (rows[-1].get("curb_persisted_n")
                                      if rows else None)
    summary["curb_persisted_final_total"] = total_kept
    out = {
        "sequence": str(seq_path),
        "frames": len(rows),
        "meta": meta,
        "cell_m": cell_m,
        "recall_m": recall_m,
        "edge_px": edge_px,
        "curb_height_m": CURB_HEIGHT_M,
        "rows": rows,
        "summary": summary,
        "limits": [
            "No independent geometric boundary truth in the recording: the "
            "mask check is cross-sensor consistency, not accuracy.",
            "A placed-step sequence (evidence_level) is real sensor data at "
            "real poses, but the motion is placement - it is not a driven run.",
            "A kerb, a wall foot and a raised shoulder all produce the same "
            "height step; the detector does not classify them.",
            "Persistence counts recurrence in 0.5 m world cells; a repeated "
            "false return (e.g. one tall grass clump) persists too.",
        ],
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=str, required=True)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--edge-px", type=float, default=DEFAULT_EDGE_PX)
    ap.add_argument("--recall-m", type=float, default=DEFAULT_RECALL_M)
    ap.add_argument("--cell-m", type=float, default=0.5)
    args = ap.parse_args(argv)
    res = analyse(args.seq, edge_px=args.edge_px, recall_m=args.recall_m,
                  cell_m=args.cell_m)
    for r in res["rows"]:
        print(f"[seq] frame {r['frame']:2d} pts={r['points']:6d} "
              f"curb={r.get('curb_raw_n')} kept={r.get('curb_persisted_n')} "
              f"recur={r.get('curb_recur_prev_rate')} "
              f"resid={r.get('curb_residual_p50_m')} "
              f"edge={r.get('proj_on_edge')} in={r.get('proj_inside_road')} "
              f"out={r.get('proj_outside_road')}")
    print("[seq] summary " + json.dumps(res["summary"], ensure_ascii=False))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"[seq] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
