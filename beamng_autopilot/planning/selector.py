"""Trajectory selector: pick the best feasible candidate.

Takes a ``CandidateSet`` and the constraint scorer, runs feasibility +
cost on every candidate and returns the lowest-cost feasible path (or
the reference, or ``None`` when nothing is feasible - the caller then
falls back to the legacy planner / stops).
"""

from __future__ import annotations

import numpy as np


def select_trajectory(scene, candidate_set, constraints):
    """Return ``(best_path, meta)``.

    ``best_path`` is the (N, 2) polyline of the lowest-cost feasible
    candidate, ``None`` when no candidate is feasible.  ``meta`` holds
    the ranking so a planner / HUD can explain the choice ("cost", 
    "kind", "why"), plus the chosen candidate's speed profile (the
    matching longitudinal plan, ``meta["speed_profile"]``) when a scene
    target speed is available.
    """
    if candidate_set is None or len(candidate_set) == 0:
        return None, {"why": "no candidates"}
    feasible = []
    for cand in candidate_set.candidates:
        cost, ok = constraints.score(scene, cand)
        if ok and np.isfinite(cost):
            feasible.append((cost, cand))
    if not feasible:
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