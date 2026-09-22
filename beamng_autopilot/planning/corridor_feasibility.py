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

    FEASIBLE    a geometric band passes width, row connectivity and the
                lateral-reachability necessary conditions; this is NOT
                proof that a particular trajectory is safe to execute
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

from beamng_autopilot.vehicle_body import HALF_LENGTH_M

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

@dataclass
class CorridorFeasibility:
    """The answer, plus what it was derived from."""

    state: str = UNKNOWN
    reason: str = ""
    # Geometry of the last band examined, ego metres with left positive.
    # Infeasible results may retain it to explain the rejected transition.
    width_m: float | None = None
    centre_lateral_m: float | None = None
    # Forward distance from ego to the checked band's far edge.  The
    # current footprint is excluded; scanning begins at its front bumper.
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
    s, v, radius, lateral_speed = map(float, (
        distance_m, speed_mps, min_turn_radius_m, max_lateral_speed_mps))
    if not all(math.isfinite(x) for x in (s, v, radius, lateral_speed)) \
            or v < 0.0 or radius <= 0.0 or lateral_speed < 0.0:
        raise ValueError("invalid lateral-reachability input")
    s = max(0.0, s)
    by_geometry = (s / radius) * (0.5 * s)
    if not math.isfinite(by_geometry):
        raise ValueError("lateral-reachability overflow")
    if v <= 1e-6:
        # Standstill: the geometry ceiling is the only honest bound.  A
        # time-based bound would return infinity, which is the "there is
        # a gap so I can take it" error in another form.
        return by_geometry
    by_rate = lateral_speed * (s / v)
    if not math.isfinite(by_rate):
        raise ValueError("lateral-reachability overflow")
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

    ``scene`` needs a ``grid`` with ``obstacle`` (0 = free), positive
    ``drivable`` evidence, ``res`` and ``n_rows`` / ``n_cols``.  When an
    ``observed`` layer exists, unobserved cells cannot authorise a band.
    Missing road evidence is UNKNOWN, not free space.  Drivable road alone
    does not establish lane legality; the candidate's detected-boundary
    and swept-body checks remain mandatory.

    ``required_distance_m`` is the forward distance from ego to the last
    row that must be checked.  Without it every row through the grid's
    forward horizon is checked.  A request beyond that horizon is UNKNOWN.
    Near-field lateral deadlines subtract the authoritative front bumper
    distance, so distant free space cannot license a last-moment shift.
    """
    res = CorridorFeasibility(candidate_id=candidate_id)
    res.evidence = dict(evidence or {})
    res.notes.append("geometry only; candidate sweep, lane legality and "
                     "dynamic feasibility require separate validation")

    try:
        speed, ego_lat, width, margin, radius, lateral_speed = map(float, (
            ego_speed_mps, ego_lateral_m, vehicle_width_m, margin_m,
            min_turn_radius_m, max_lateral_speed_mps))
        required = (None if required_distance_m is None
                    else float(required_distance_m))
        max_age = (None if max_evidence_age_s is None
                   else float(max_evidence_age_s))
        values = (speed, ego_lat, width, margin, radius, lateral_speed)
        if not all(math.isfinite(v) for v in values) \
                or speed < 0.0 or width <= 0.0 or margin < 0.0 \
                or radius <= 0.0 or lateral_speed < 0.0 \
                or (required is not None and (not math.isfinite(required)
                                             or required <= 0.0)) \
                or (max_age is not None and (not math.isfinite(max_age)
                                            or max_age < 0.0)):
            raise ValueError("invalid geometry parameter")
    except (TypeError, ValueError, OverflowError):
        res.reason = "invalid geometry parameter"
        return res

    age = res.evidence.get("age_s")
    if age is not None:
        try:
            age = float(age)
            if not math.isfinite(age) or age < 0.0:
                raise ValueError("invalid age")
        except (TypeError, ValueError, OverflowError):
            res.reason = "evidence age unreadable"
            return res
        if max_age is not None and age > max_age:
            res.reason = "evidence too old"
            return res

    grid = getattr(scene, "grid", None)
    if grid is None:
        res.reason = "no grid"
        return res
    occ = getattr(grid, "obstacle", None)
    if occ is None:
        res.reason = "no obstacle layer"
        return res
    try:
        occ = np.asarray(occ, dtype=float)
    except (TypeError, ValueError):
        res.reason = "invalid obstacle layer"
        return res
    if occ.size == 0:
        res.reason = "no obstacle layer"
        return res
    try:
        n_rows, n_cols = int(grid.n_rows), int(grid.n_cols)
        cell = float(grid.res)
        max_x = float(getattr(grid, "max_x", n_rows * cell / 2.0))
        max_y = float(getattr(grid, "max_y", n_cols * cell / 2.0))
    except (AttributeError, TypeError, ValueError, OverflowError):
        res.reason = "invalid grid geometry"
        return res
    if occ.ndim != 2 or occ.shape != (n_rows, n_cols) \
            or not np.isfinite(occ).all() or bool((occ < 0).any()):
        res.reason = "invalid obstacle layer"
        return res
    if n_cols < 4 or not all(math.isfinite(v) for v in (cell, max_x, max_y)) \
            or cell <= 0.0 or max_x <= HALF_LENGTH_M \
            or not math.isfinite(max_x / cell):
        res.reason = "grid too coarse to resolve a band"
        return res

    layers = {}
    for name in ("drivable", "observed"):
        layer = getattr(grid, name, None)
        if layer is None:
            if name == "drivable":
                res.reason = "no drivable layer"
                return res
            continue
        try:
            layer = np.asarray(layer, dtype=float)
        except (TypeError, ValueError):
            res.reason = f"invalid {name} layer"
            return res
        if layer.shape != occ.shape:
            res.reason = f"{name} mask shape mismatch"
            return res
        if not np.isfinite(layer).all() or bool((layer < 0).any()):
            res.reason = f"invalid {name} layer"
            return res
        layers[name] = layer
    drivable = layers["drivable"]
    observed = layers.get("observed")

    need_m = width + margin
    if not math.isfinite(need_m / cell):
        res.reason = "invalid geometry parameter"
        return res
    min_cells = max(1, int(math.ceil(need_m / cell)))
    res.notes.append(f"need {need_m:.2f} m = {min_cells} cells")
    goal = max_x if required is None else required
    res.available_distance_m = round(goal, 3)
    if goal < HALF_LENGTH_M:
        res.reason = "required distance lies inside the current footprint"
        return res

    # Include the cell touching the nose, then visit EVERY cell row ahead.
    # A grid-size fraction skipped real obstacles outside the bumper.
    start_row = min(n_rows - 1,
                    int(math.floor((max_x - HALF_LENGTH_M) / cell)))
    if start_row < 0:
        res.reason = "grid has no rows ahead of the ego"
        return res

    bands = None
    clear_distance = HALF_LENGTH_M
    for row in range(start_row, -1, -1):
        near_edge = max_x - (row + 1) * cell
        far_edge = max_x - row * cell
        free = (occ[row] == 0) & (drivable[row] > 0)
        unknown = drivable[row] <= 0
        if observed is not None:
            free &= observed[row] > 0
            unknown = observed[row] <= 0
        intervals = free_intervals(free, min_cells)
        if not intervals:
            possible = free_intervals(
                (occ[row] == 0) & (free | unknown), min_cells)
            res.state = UNKNOWN if possible else INFEASIBLE
            res.reason = ("road evidence missing in the required band"
                          if possible else "no band wide enough at the entry"
                          if bands is None else "band ends before the obstacle")
            res.clear_distance_m = round(clear_distance, 3)
            return res

        # Intervals describe admissible BODY CENTRES, not raw free cells.
        # Overlapping a single cell cannot connect two car-width passages.
        centers = []
        for a, b in intervals:
            lo = max_y - (b + 1) * cell + need_m / 2.0
            hi = max_y - a * cell - need_m / 2.0
            if hi >= lo:
                centers.append((lo, hi, (b - a + 1) * cell))
        if bands is not None:
            centers = [iv for iv in centers
                       if any(iv[0] <= old[1] and old[0] <= iv[1]
                              for old in bands)]
        if not centers:
            res.state = INFEASIBLE
            res.reason = "vehicle-centre bands disconnected"
            res.clear_distance_m = round(clear_distance, 3)
            return res

        # Each row has its own entry deadline, measured from the bumper.
        # A distant horizon must never authorise an impossible near shift.
        budget = max(0.0, min(near_edge, goal) - HALF_LENGTH_M)
        try:
            reach = max_lateral_shift_m(
                budget, speed, min_turn_radius_m=radius,
                max_lateral_speed_mps=lateral_speed)
        except ValueError:
            res.reason = "invalid lateral reachability"
            return res
        nearest = min(centers, key=lambda iv: (
            abs(min(max(ego_lat, iv[0]), iv[1]) - ego_lat), -iv[2]))
        center = min(max(ego_lat, nearest[0]), nearest[1])
        shift = abs(center - ego_lat)
        res.width_m = round(nearest[2], 3)
        res.centre_lateral_m = round(center, 3)
        res.lateral_shift_m = round(shift, 3)
        res.reachable_shift_m = round(reach, 3)
        res.notes.append(f"row {row}: lateral budget {budget:.3f} m")
        bands = [(max(lo, ego_lat - reach), min(hi, ego_lat + reach), w)
                 for lo, hi, w in centers
                 if lo <= ego_lat + reach and hi >= ego_lat - reach]
        if not bands:
            res.state = INFEASIBLE
            res.reason = "band not reachable in the distance left"
            res.clear_distance_m = round(clear_distance, 3)
            return res
        clear_distance = far_edge
        res.clear_distance_m = round(clear_distance, 3)
        if clear_distance >= goal:
            res.state = FEASIBLE
            return res

    res.reason = "required distance exceeds observed grid horizon"
    return res
