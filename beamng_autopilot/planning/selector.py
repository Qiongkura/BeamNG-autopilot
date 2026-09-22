"""Trajectory selector: pick the best feasible candidate.

Takes a ``CandidateSet`` and the constraint scorer, runs feasibility +
cost on every candidate and returns the lowest-cost feasible path (or
the reference, or ``None`` when nothing is feasible - the caller then
falls back to the legacy planner / stops).
"""

from __future__ import annotations

import math

import numpy as np

from .lateral_ref import lateral_reference
from .trajectory import Candidate


def _hold_heading_path(scene, length_m: float = 8.0, n: int = 9):
    """Ego-anchored straight-ahead polyline (current heading)."""
    p = np.asarray(scene.pos[:2], dtype=float)
    h = float(scene.heading)
    t = np.linspace(0.0, float(length_m), int(n))
    return np.column_stack([p[0] + t * math.cos(h),
                            p[1] + t * math.sin(h)])


def _perception_hold_path(scene, length_m: float = 8.0, n: int = 9):
    """Use the current sensor lane for a short hold when it is available.

    A straight heading hold is only a safe fallback on a straight lane.  On
    a curved sensor lane it can put the ego footprint across a boundary even
    though the lane centre itself is valid.
    """
    p = np.asarray(scene.pos[:2], dtype=float)
    ref = None
    if getattr(scene, "strict_perception", False):
        ref, _ = lateral_reference(scene)
    else:
        envelope = getattr(scene, "lane_envelope", None)
        if envelope is not None:
            ref = getattr(envelope, "center", None)
    if ref is None:
        return _hold_heading_path(scene, length_m, n)
    ref = np.asarray(ref, dtype=float)
    if ref.ndim != 2 or ref.shape[1] < 2:
        return _hold_heading_path(scene, length_m, n)
    ref = ref[:, :2]
    ref = ref[np.isfinite(ref).all(axis=1)]
    if len(ref) < 2:
        return _hold_heading_path(scene, length_m, n)
    h = float(scene.heading)
    fwd = np.array([math.cos(h), math.sin(h)])
    rel = ref - p
    ahead = rel @ fwd
    keep = (ahead >= -0.5) & (ahead <= float(length_m) + 0.5)
    ref = ref[keep]
    if len(ref) < 2:
        return _hold_heading_path(scene, length_m, n)
    order = np.argsort(np.linalg.norm(ref - p, axis=1))
    path = np.vstack([p, ref[order]])
    dedup = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-4]
    path = path[dedup]
    return path if len(path) >= 2 else _hold_heading_path(scene, length_m, n)


def _trim_before_body_cross(scene, path, margin_m: float = 0.35):
    """Shorten a fallback before its full ego footprint crosses a boundary."""
    try:
        from .constraints import body_lane_cross_dist_m
        crossing = float(body_lane_cross_dist_m(scene, path))
    except Exception:
        return path
    if crossing <= 0.0:
        return path
    limit = crossing - max(0.0, float(margin_m))
    if limit < 0.8:
        return None
    pts = [np.asarray(path[0], dtype=float)[:2]]
    travelled = 0.0
    arr = np.asarray(path, dtype=float)[:, :2]
    for a, b in zip(arr[:-1], arr[1:]):
        seg = float(np.linalg.norm(b - a))
        if seg < 1e-9:
            continue
        if travelled + seg >= limit:
            pts.append(a + (limit - travelled) / seg * (b - a))
            break
        pts.append(b)
        travelled += seg
    out = np.asarray(pts, dtype=float)
    return out if len(out) >= 2 else None


def _hold_path_occupied_frac(scene, path, skip_m: float = 2.5) -> float:
    """Fraction of path samples (beyond the bumper) in occupied cells."""
    grid = getattr(scene, "grid", None)
    if grid is None or path is None or len(path) < 2:
        return 0.0
    occ = getattr(grid, "obstacle", None)
    if occ is None or occ.size == 0:
        return 0.0
    p = np.asarray(scene.pos[:2], dtype=float)
    pts = np.asarray(path, dtype=float)[:, :2]
    d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
    sel = pts[d >= skip_m]
    if len(sel) == 0:
        return 0.0
    try:
        r, c, ok = grid.world_to_cells(sel[:, 0], sel[:, 1])
    except Exception:
        return 1.0
    if not np.any(ok):
        return 0.0
    return float(np.mean(occ[r[ok], c[ok]] > 0))


def select_trajectory(scene, candidate_set, constraints, *,
                      hysteresis=None, now_s: float | None = None,
                      emergency: bool = False):
    """Return ``(best_path, meta)``.

    ``best_path`` is the (N, 2) polyline of the lowest-cost feasible
    candidate, ``None`` when no candidate is feasible.  ``meta`` holds
    the ranking so a planner / HUD can explain the choice ("cost",
    "kind", "why"), plus the chosen candidate's speed profile (the
    matching longitudinal plan, ``meta["speed_profile"]``) when a scene
    target speed is available.

    ``hysteresis`` (a ``planning.hysteresis.CandidateHysteresis``) makes
    the choice STICKY: the previously chosen candidate is kept while it
    is still feasible this tick and no other candidate beats it by more
    than the configured margin, so two near-equal candidates cannot win
    on alternate frames.  The decision (including why it held or
    switched) is published as ``meta["hysteresis"]``.  Safety is
    untouched: only candidates the constraint scorer accepted this tick
    are eligible, and hysteresis can never revive a rejected one.

    When every fan candidate is rejected, an ego-anchored hold-heading
    path is returned if either the forward corridor is laterally free
    **or** that hold path itself is mostly not occupied (junction BEV
    clutter must not force ``no drivable path``).  The safety monitor
    re-checks it.
    """
    if candidate_set is None or len(candidate_set) == 0:
        return None, {"why": "no candidates"}
    feasible = []
    # Fresh tally per tick so the published reasons describe THIS tick.
    try:
        constraints.reject_counts = {}
    except Exception:
        pass
    for cand in candidate_set.candidates:
        cost, ok = constraints.score(scene, cand)
        if ok and np.isfinite(cost):
            feasible.append((cost, cand))
    rejects = dict(getattr(constraints, "reject_counts", {}) or {})
    if not feasible:
        # Strict FSD: no perception lane -> never invent a hold path
        # (constraints already declined every candidate on purpose).
        strict = bool(getattr(scene, "strict_perception", False))
        if strict and lateral_reference(scene)[0] is None:
            return None, {"why": "no feasible candidate",
                          "rejects": rejects}
        path = _perception_hold_path(scene)
        path = _trim_before_body_cross(scene, path)
        if path is None:
            return None, {"why": "hold_heading_crosses_lane_boundary"}
        if strict:
            hold = Candidate(path=path, meta={"kind": "lane_center"})
            cost, ok = constraints.score(scene, hold)
            if not ok or not np.isfinite(cost):
                return None, {"why": "hold_heading_constraints_rejected",
                              "n_eval": 0,
                              "rejects": dict(getattr(constraints,
                                                      "reject_counts", {}) or {})}
        try:
            from .constraints import corridor_free_band
            open_corridor = bool(corridor_free_band(scene))
        except Exception:
            open_corridor = False
        occ_frac = _hold_path_occupied_frac(scene, path)
        if open_corridor or occ_frac < 0.20:
            return path, {
                "why": ("fallback_hold_heading"
                        if open_corridor
                        else "fallback_hold_heading_low_occ"),
                "kind": "hold_heading",
                "cost": float("inf"),
                "n_eval": 0,
                "hold_occ_frac": round(occ_frac, 3),
            }
        return None, {"why": "no feasible candidate", "rejects": rejects}
    feasible.sort(key=lambda pair: pair[0])
    hyst_info = None
    if hysteresis is not None:
        picked = hysteresis.choose(
            feasible, time.time() if now_s is None else float(now_s),
            emergency=bool(emergency))
        if picked is None:
            return None, {"why": "no feasible candidate", "rejects": rejects}
        cost, best, hyst_info = picked
    else:
        cost, best = feasible[0]
    meta = {
        "cost": float(cost),
        "kind": best.meta.get("kind", "?"),
        "why": "best-of-N",
        "n_eval": len(feasible),
    }
    if hyst_info is not None:
        meta["hysteresis"] = hyst_info
        if hyst_info.get("reason") in ("min_dwell", "cost_margin"):
            meta["why"] = "hysteresis_hold"
    # attach the chosen candidate's longitudinal plan
    target = float(getattr(scene, "target_speed", 0.0))
    if best.speed_profile is not None and len(best.speed_profile):
        meta["speed_profile"] = best.speed_profile
    elif target > 0:
        try:
            from .speed_profile import speed_profile_for_path
            sp = speed_profile_for_path(best.path, scene, target_speed=target)
            best.speed_profile = sp
            meta["speed_profile"] = sp
        except Exception:
            pass
    return best.path, meta