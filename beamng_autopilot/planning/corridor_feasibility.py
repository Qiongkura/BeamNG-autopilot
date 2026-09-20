"""Structured corridor feasibility (improvement plan P2.1).

Why this exists
---------------
``corridor_free_band`` returns a bool.  A bool cannot carry the answer the
escape hatch actually needs, and its True was being read as "I can drive
through here" when it only ever meant "somewhere across the width there is
a gap".  Concretely, on 2026-09-20 it returned True with the nearest
obstacle at 1.21 m, because eight lateral bands only need five of them to
be not-completely-closed.

This module answers a different question and returns the evidence with the
answer:

    FEASIBLE    a laterally free band exists, is wide enough for the body,
                is CONNECTED row to row (not a gap that alternates sides),
                and the car can still shift into it in the distance left
    INFEASIBLE  one of those fails - with which one in ``reason``
    UNKNOWN     the evidence needed to decide is missing, too old, or the
                grid has no layer to read

UNKNOWN is not a soft True.  Callers must not use it to resume cruise; the
only thing it may do is fall through to whatever the conservative branch
already does.

What is NOT claimed
-------------------
The lateral-reachability model below is kinematics, not a validated safety
parameter.  It exists so "there is a gap" and "I can still get into that
gap" are separate statements - the numbers it produces are for the
shadow-replay harness (P2.2) to calibrate, not for a driving decision yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FEASIBLE = "feasible"
INFEASIBLE = "infeasible"
UNKNOWN = "unknown"

ALL_STATES = (FEASIBLE, INFEASIBLE, UNKNOWN)

# Defaults so the primitive is callable.  They are NOT calibrated:
# P2.2 has to establish them, and until then every caller should pass its
# own explicitly.
DEFAULT_VEHICLE_WIDTH_M = 1.90
DEFAULT_MARGIN_M = 0.50
DEFAULT_MIN_TURN_RADIUS_M = 5.50      # bicycle-model lateral ceiling
DEFAULT_MAX_LATERAL_SPEED_MPS = 2.00
DEFAULT_MAX_EVIDENCE_AGE_S = 2.00     # the freshness contract line

# How much of the grid around the ego to skip: the car's own footprint and
# the bumper zone are not evidence about the corridor ahead.
EGO_BAND_FRAC = 0.12


@dataclass
class CorridorFeasibility:
    """The answer, plus what it was derived from."""

    state: str = UNKNOWN
    reason: str = ""
    # Geometry of the band that was chosen, in metres relative to the
    # grid centre line.  None when the answer is not FEASIBLE.
    width_m: float | None = None
    centre_lateral_m: float | None = None
    # How far the band stays connected ahead of the car.  A band that
    # ends in 3 m does not get you past an obstacle at 8 m.
    clear_distance_m: float | None = None
    # Reachability: how far sideways the car must move, how far it has to
    # do it in, and how much the model says it can do in that distance.
    lateral_shift_m: float | None = None
    available_distance_m: float | None = None
    reachable_shift_m: float | None = None
    candidate_id: str | None = None
    # Provenance of the grid the answer came from.  An answer with no
    # provenance is not an answer.
    evidence: dict = field(default_factory=dict)
    # Human-readable notes for the shadow replay - never used to decide.
    notes: list = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        """True only for FEASIBLE.

        Deliberately not ``state != INFEASIBLE``: UNKNOWN must not be able
        to open the escape hatch, which is exactly what the old bool did.
        """
        return self.state == FEASIBLE

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "reason": self.reason,
            "width_m": self.width_m,
            "centre_lateral_m": self.centre_lateral_m,
            "clear_distance_m": self.clear_distance_m,
            "lateral_shift_m": self.lateral_shift_m,
            "available_distance_m": self.available_distance_m,
            "reachable_shift_m": self.reachable_shift_m,
            "candidate_id": self.candidate_id,
            "evidence": dict(self.evidence),
            "notes": list(self.notes),
        }


# ------------------------------------------------------------- geometry

def free_intervals(mask_free, min_cells: int) -> list[tuple[int, int]]:
    """Maximal runs of free cells at least ``min_cells`` wide.

    ``mask_free`` is a 1-D boolean row (True = free).  Contiguity is the
    whole point: the old gate counted free CELLS, so a gap of three cells
    on the left and three on the right scored the same as one six-cell gap
    the car could actually fit through.
    """
    free = np.asarray(mask_free, dtype=bool)
    out: list[tuple[int, int]] = []
    n = int(free.size)
    start = None
    for i in range(n):
        if free[i]:
            if start is None:
                start = i
            continue
        if start is not None:
            if i - start >= min_cells:
                out.append((start, i - 1))
            start = None
    if start is not None and n - start >= min_cells:
        out.append((start, n - 1))
    return out


def intervals_connected(a: tuple[int, int], b: tuple[int, int],
                        tolerance_cells: int = 0) -> bool:
    """Two intervals in ADJACENT rows lead into each other.

    Overlap is measured with a tolerance because a band can step sideways
    by a cell or two per row and still be drivable; a step of half the
    width is a different band and must not be joined.
    """
    return not (a[1] + tolerance_cells < b[0]
                or b[1] + tolerance_cells < a[0])


def max_lateral_shift_m(distance_m: float, speed_mps: float,
                        min_turn_radius_m: float = DEFAULT_MIN_TURN_RADIUS_M,
                        max_lateral_speed_mps: float = (
                            DEFAULT_MAX_LATERAL_SPEED_MPS)) -> float:
    """How far sideways the car can get in ``distance_m`` of travel.

    Two independent ceilings, and the smaller one wins:

    * **Geometry** - a bicycle model with minimum turn radius R shifts at
      most ``s² / (2R)`` over a longitudinal distance s.  This is the one
      that binds at low speed, including standstill, where "I would have
      time" is meaningless because the car is not going anywhere.
    * **Lateral speed** - at speed v it only has ``s / v`` seconds, so a
      lateral rate cap gives ``v_lat_max · s / v``.  This binds at speed.

    Both are kinematics.  Neither is a validated limit.
    """
    s = max(0.0, float(distance_m))
    by_geometry = (s * s) / (2.0 * max(1e-6, float(min_turn_radius_m)))
    v = float(speed_mps)
    if v <= 1e-6:
        # Standstill: the geometry ceiling is the only honest bound.  A
        # time-based bound would return infinity, which is the "there is
        # a gap so I can take it" error in another form.
        return by_geometry
    by_rate = float(max_lateral_speed_mps) * s / v
    return min(by_geometry, by_rate)


# ------------------------------------------------------------- primitive

def corridor_feasibility(scene, *, ego_speed_mps: float = 0.0,
                         ego_lateral_m: float = 0.0,
                         vehicle_width_m: float = DEFAULT_VEHICLE_WIDTH_M,
                         margin_m: float = DEFAULT_MARGIN_M,
                         min_turn_radius_m: float = DEFAULT_MIN_TURN_RADIUS_M,
                         max_lateral_speed_mps: float = (
                             DEFAULT_MAX_LATERAL_SPEED_MPS),
                         max_evidence_age_s: float = (
                             DEFAULT_MAX_EVIDENCE_AGE_S),
                         required_distance_m: float | None = None,
                         candidate_id: str | None = None,
                         evidence: dict | None = None) -> CorridorFeasibility:
    """Can the car still get into a free lateral band ahead of it?

    ``scene`` needs a ``grid`` with an ``obstacle`` layer (0 = free), a
    ``res`` and ``n_rows`` / ``n_cols``, as the rest of ``planning`` uses.
    A ``drivable`` layer, if present, is ANDed in so a gap off the
    pavement, beyond the lane marking or on the wrong side of the road is
    not a gap - the old gate never looked at drivability at all.

    ``required_distance_m`` is how far ahead the band has to stay open
    (the obstacle's distance).  Without it the primitive reports how far
    the band goes and lets the caller decide; it does not assume "to the
    horizon" is good enough.
    """
    res = CorridorFeasibility()
    res.candidate_id = candidate_id
    res.evidence = dict(evidence or {})

    # --- evidence first: an answer with no provenance is not an answer --
    age = res.evidence.get("age_s")
    if age is not None and max_evidence_age_s is not None:
        try:
            if float(age) > float(max_evidence_age_s):
                res.state = UNKNOWN
                res.reason = "evidence too old"
                res.notes.append(
                    f"age {float(age):.2f}s > {float(max_evidence_age_s):.2f}s")
                return res
        except (TypeError, ValueError):
            res.state = UNKNOWN
            res.reason = "evidence age unreadable"
            return res

    grid = getattr(scene, "grid", None)
    if grid is None:
        res.state = UNKNOWN
        res.reason = "no grid"
        return res
    occ = getattr(grid, "obstacle", None)
    if occ is None or np.asarray(occ).size == 0:
        res.state = UNKNOWN
        res.reason = "no obstacle layer"
        return res

    occ = np.asarray(occ)
    n_rows, n_cols = int(grid.n_rows), int(grid.n_cols)
    cell = float(getattr(grid, "res", 0.0) or 0.0)
    if n_cols < 4 or cell <= 0.0:
        res.state = UNKNOWN
        res.reason = "grid too coarse to resolve a band"
        res.notes.append(f"n_cols={n_cols} res={cell}")
        return res

    drivable = getattr(grid, "drivable", None)
    if drivable is not None:
        drivable = np.asarray(drivable)
        if drivable.shape != occ.shape:
            # A mismatched mask cannot be ANDed in, and silently ignoring
            # it would put the pavement check back to "not checked".
            res.state = UNKNOWN
            res.reason = "drivable mask shape mismatch"
            return res

    need_m = float(vehicle_width_m) + float(margin_m)
    min_cells = max(1, int(math.ceil(need_m / cell)))
    res.notes.append(f"need {need_m:.2f} m = {min_cells} cells")

    # Rows: 0 is the furthest ahead, the ego sits at n_rows/2.
    ego_row = int(n_rows // 2)
    ego_band = max(1, int(n_rows * EGO_BAND_FRAC))
    start_row = ego_row - ego_band
    if start_row < 1:
        res.state = UNKNOWN
        res.reason = "grid has no rows ahead of the ego"
        return res

    def row_intervals(r: int) -> list[tuple[int, int]]:
        row = occ[r]
        free = (np.asarray(row) == 0)
        if drivable is not None:
            free = free & (np.asarray(drivable[r]) != 0)
        return free_intervals(free, min_cells)

    # --- walk forward, keeping only bands that connect to the last row --
    bands = row_intervals(start_row)
    if not bands:
        res.state = INFEASIBLE
        res.reason = "no band wide enough at the entry"
        res.notes.append(f"row {start_row} has no gap >= {min_cells} cells")
        return res

    clear_rows = 0
    for r in range(start_row - 1, -1, -1):
        nxt = row_intervals(r)
        if not nxt:
            break
        # Keep the intervals in this row that some interval in the
        # previous row actually leads into.  A gap that jumps from the
        # left edge to the right edge between two rows is not a band,
        # however many free cells each row counted.
        joined = [iv for iv in nxt
                  if any(intervals_connected(iv, pv) for pv in bands)]
        if not joined:
            break
        bands = joined
        clear_rows += 1

    clear_distance_m = (clear_rows + 1) * cell
    res.clear_distance_m = round(clear_distance_m, 3)

    # --- pick the widest band, preferring the least lateral shift -------
    centre_col = n_cols / 2.0
    ego_col = centre_col + (float(ego_lateral_m) / cell)
    scored = []
    for a, b in bands:
        width_m = (b - a + 1) * cell
        mid = (a + b) / 2.0
        scored.append((abs(mid - ego_col), width_m, mid))
    scored.sort(key=lambda t: (t[0], -t[1]))
    shift_cells, width_m, mid = scored[0]
    shift_m = abs(mid - ego_col) * cell

    res.width_m = round(float(width_m), 3)
    res.centre_lateral_m = round((mid - centre_col) * cell, 3)
    res.lateral_shift_m = round(float(shift_m), 3)

    # --- does the band go far enough? ----------------------------------
    if required_distance_m is not None:
        res.available_distance_m = round(float(required_distance_m), 3)
        if clear_distance_m < float(required_distance_m):
            res.state = INFEASIBLE
            res.reason = "band ends before the obstacle"
            res.notes.append(f"clear {clear_distance_m:.2f} m < "
                             f"required {float(required_distance_m):.2f} m")
            return res
    else:
        res.available_distance_m = res.clear_distance_m

    # --- can the car still get into it? --------------------------------
    # Distance available to make the shift: the band has to be entered
    # before the obstacle, so the shorter of "how far the band lasts" and
    # "how far away the obstacle is" is what the car actually has.
    budget_m = min(clear_distance_m,
                   float(required_distance_m) if required_distance_m
                   is not None else clear_distance_m)
    reach_m = max_lateral_shift_m(
        budget_m, ego_speed_mps, min_turn_radius_m=min_turn_radius_m,
        max_lateral_speed_mps=max_lateral_speed_mps)
    res.reachable_shift_m = round(float(reach_m), 3)
    if shift_m > reach_m + 1e-9:
        res.state = INFEASIBLE
        res.reason = "band not reachable in the distance left"
        res.notes.append(f"need {shift_m:.2f} m lateral, can do "
                         f"{reach_m:.2f} m in {budget_m:.2f} m at "
                         f"{float(ego_speed_mps):.2f} m/s")
        return res

    res.state = FEASIBLE
    res.reason = ""
    return res
