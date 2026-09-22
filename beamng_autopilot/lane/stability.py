"""Cross-tick stability of the accepted lateral reference (P1-2).

The round-5 review measured the same stretch of road flipping between a
paired own-lane reference, an oncoming-lane read, the whole-road fused
centre and a single-edge mirror - and the crash run steered with a
full-lock wheel while the reference was doing it.  This module answers
exactly one question per tick:

    **has this reference been the same thing, on the same side of the car,
    for long enough to deserve full steering authority?**

Rules (from the handoff):

* the side and the near-field centre must agree for ``need_ticks`` (2-3)
  consecutive ticks before normal steering authority is restored;
* a reference that is not a PAIRED perception read (single-edge mirror,
  divider fallback, hold) never earns full authority, however long it
  lasts - it may only make small corrections;
* a tick without fresh perception evidence resets the streak: an old
  reference is never upgraded to a new legal one on the strength of its
  own age;
* nothing here reads the nav route or any map offset.  The stabiliser is
  purely geometric (reference polyline + ego pose), which is what AGENTS.md
  requires of anything that can influence lateral control.

The tracker is pure state + pure functions so the behaviour is pinned by
unit tests instead of by a live run; the drive loop consumes the verdict
only when its switch is on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Near-field window used to characterise the reference next to the car.
NEAR_WINDOW_M = 15.0
# |lat| below this counts as "the reference runs through the car", which is
# the normal state when the car is centred on it - not a side.
SIDE_DEAD_BAND_M = 0.3
# Allowed tick-to-tick change of the near-field centre.
CENTRE_BAND_M = 0.5
# Consecutive agreeing ticks required for full authority.
NEED_TICKS = 2

AUTHORITY_FULL = "full"
AUTHORITY_LIMITED = "limited"

SIDE_LEFT = "left"
SIDE_RIGHT = "right"
SIDE_CENTER = "center"
SIDE_UNKNOWN = "unknown"


def near_reference_lat(ref, pos, heading: float,
                       within_m: float = NEAR_WINDOW_M) -> float | None:
    """Median lateral offset (+ = left) of ``ref`` beside the car, or None.

    Measured in the CAR frame at the current pose, so normal forward
    motion does not change it; a reference that jumped to the other side
    of the car shows up as a sign change.
    """
    if ref is None:
        return None
    arr = np.asarray(ref, dtype=float)
    if arr.ndim != 2 or len(arr) < 2 or arr.shape[1] < 2:
        return None
    p = np.asarray(pos, dtype=float)[:2]
    if p.size < 2 or not np.isfinite(p).all():
        return None
    rel = arr[:, :2] - p
    near = arr[np.linalg.norm(rel, axis=1) <= float(within_m)][:, :2]
    if len(near) == 0:
        return None
    fwd = np.array([np.cos(float(heading)), np.sin(float(heading))])
    left = np.array([-fwd[1], fwd[0]])
    val = float(((near - p) @ left).mean())
    return val if np.isfinite(val) else None


def reference_side(lat_m: float | None,
                   dead_band_m: float = SIDE_DEAD_BAND_M) -> str:
    """``left`` / ``right`` / ``center`` / ``unknown`` for a near-field lat."""
    if lat_m is None:
        return SIDE_UNKNOWN
    if abs(float(lat_m)) <= float(dead_band_m):
        return SIDE_CENTER
    return SIDE_LEFT if float(lat_m) > 0.0 else SIDE_RIGHT


@dataclass
class ReferenceStability:
    """One tick's stability verdict plus the state it was derived from."""

    authority: str = AUTHORITY_LIMITED
    reason: str = "no reference yet"
    side: str = SIDE_UNKNOWN
    lat_m: float | None = None
    stable_ticks: int = 0
    flip: bool = False
    paired: bool = False
    flips_total: int = 0


@dataclass
class ReferenceStabilityTracker:
    """Carry the previous tick's side/centre and grade this tick's read."""

    band_m: float = CENTRE_BAND_M
    need_ticks: int = NEED_TICKS
    side: str = SIDE_UNKNOWN
    lat_m: float | None = None
    ticks: int = 0
    flips: int = 0
    history: list = field(default_factory=list)

    def reset(self) -> None:
        self.side = SIDE_UNKNOWN
        self.lat_m = None
        self.ticks = 0

    def update(self, *, ref, pos, heading: float, paired: bool,
               fresh: bool) -> ReferenceStability:
        """Grade the reference accepted for THIS tick.

        ``paired`` is True only for a two-sided perception read;
        ``fresh`` is False when the reference was served from a hold /
        stale path, which resets the streak (an old reference must not
        become legal just by surviving).
        """
        lat = near_reference_lat(ref, pos, heading)
        side = reference_side(lat)
        out = ReferenceStability(paired=bool(paired), side=side,
                                 lat_m=(None if lat is None else round(lat, 3)))
        if ref is None or lat is None:
            self.reset()
            out.reason = "no measurable reference"
            out.flips_total = self.flips
            return out
        if not fresh:
            self.reset()
            out.reason = "reference is not fresh perception evidence"
            out.flips_total = self.flips
            return out
        continuous = (
            self.side == side and self.lat_m is not None
            and abs(float(lat) - float(self.lat_m)) <= float(self.band_m))
        out.flip = bool(
            not continuous and self.side not in (SIDE_UNKNOWN,)
            and side not in (SIDE_UNKNOWN,))
        if out.flip:
            self.flips += 1
        self.ticks = (self.ticks + 1) if continuous else 1
        self.side = side
        self.lat_m = float(lat)
        out.stable_ticks = int(self.ticks)
        out.flips_total = self.flips
        if not paired:
            out.authority = AUTHORITY_LIMITED
            out.reason = ("unpaired reference (single edge / divider / "
                          "hold): small corrections only")
            return out
        if self.ticks < max(1, int(self.need_ticks)):
            out.authority = AUTHORITY_LIMITED
            out.reason = (f"reference not yet stable "
                          f"({self.ticks}/{int(self.need_ticks)} ticks)")
            return out
        out.authority = AUTHORITY_FULL
        out.reason = f"paired and stable for {self.ticks} ticks"
        return out
