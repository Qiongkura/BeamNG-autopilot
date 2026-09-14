"""Trajectory selector: pick the best feasible candidate.

Takes a ``CandidateSet`` and the constraint scorer, runs feasibility +
cost on every candidate and returns the lowest-cost feasible path (or
the reference, or ``None`` when nothing is feasible - the caller then
falls back to the legacy planner / stops).
"""

from __future__ import annotations

import math

import numpy as np


def _hold_heading_path(scene, length_m: float = 8.0, n: int = 9):
    """Ego-anchored straight-ahead polyline (current heading)."""
    p = np.asarray(scene.pos[:2], dtype=float)
    h = float(scene.heading)
    t = np.linspace(0.0, float(length_m), int(n))
    return np.column_stack([p[0] + t * math.cos(h),
                            p[1] + t * math.sin(h)])


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


def select_trajectory(scene, candidate_set, constraints):
    """Return ``(best_path, meta)``.

    ``best_path`` is the (N, 2) polyline of the lowest-cost feasible
    candidate, ``None`` when no candidate is feasible.  ``meta`` holds
    the ranking so a planner / HUD can explain the choice ("cost",
    "kind", "why"), plus the chosen candidate's speed profile (the
    matching longitudinal plan, ``meta["speed_profile"]``) when a scene
    target speed is available.

    When every fan candidate is rejected, an ego-anchored hold-heading
    path is returned if either the forward corridor is laterally free
    **or** that hold path itself is mostly not occupied (junction BEV
    clutter must not force ``no drivable path``).  The safety monitor
    re-checks it.
    """
    if candidate_set is None or len(candidate_set) == 0:
        return None, {"why": "no candidates"}
    feasible = []
    for cand in candidate_set.candidates:
        cost, ok = constraints.score(scene, cand)
        if ok and np.isfinite(cost):
            feasible.append((cost, cand))
    if not feasible:
        # Strict FSD: no perception lane -> never invent a hold path
        # (constraints already declined every candidate on purpose).
        if (getattr(scene, "strict_perception", False)
                and getattr(scene, "lane_ref", None) is None):
            return None, {"why": "no feasible candidate"}
        path = _hold_heading_path(scene)
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
        return None, {"why": "no feasible candidate"}
    feasible.sort(key=lambda pair: pair[0])
    cost, best = feasible[0]
    meta = {
        "cost": float(cost),
        "kind": best.meta.get("kind", "?"),
        "why": "best-of-N",
        "n_eval": len(feasible),
    }
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