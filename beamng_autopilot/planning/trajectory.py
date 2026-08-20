"""Candidate trajectory sampling (FSD-style trajectory layer).

Generates a fan of candidate paths wrapped in ``Candidate`` objects with
their own (``path``, ``lateral``, ``curvature``) samples so the selector
can rank them.  Two samplers are provided:

* ``sample_arc`` - a fan of steering arcs at different curvatures (the
  classic dynamical-window / bicycle-model candidate family).
* ``sample_lane_shift`` - lateral shifts of a reference path (lane
  change / obstacle-bypass candidates around the route center).

A ``Candidate`` is a thin wrapper: path + descriptive scalar so cost
functions and the selector stay decoupled from the sampling method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Candidate:
    """One sampled trajectory.

    ``path`` is the world polyline (N, 2).  ``speed_profile`` (optional,
    length N in m/s) is the matching longitudinal plan - FSD plans path
    and speed together.
    """

    path: np.ndarray          # (N, 2) world polyline
    meta: dict = field(default_factory=dict)
    speed_profile: np.ndarray | None = None

    @property
    def length(self) -> float:
        d = np.diff(np.asarray(self.path, dtype=float), axis=0)
        return float(np.sum(np.linalg.norm(d, axis=1))) if len(d) else 0.0

    def speed_at_idx(self, i: int) -> float:
        """Max allowed speed at path index ``i`` (falls back to 0)."""
        if self.speed_profile is None or len(self.speed_profile) == 0:
            return 0.0
        return float(self.speed_profile[max(0, min(len(self.speed_profile) - 1,
                                                   int(i)))])


class CandidateSet:
    """Ordered list of candidate trajectories with a shared reference path.

    ``reference`` is the legacy planner / route polyline used to build
    shift-based candidates and to measure lateral alignment; it is also
    included as a candidate itself so the proven path is always in the
    set.
    """

    def __init__(self, reference: np.ndarray | None):
        self.reference = reference
        self.candidates: list[Candidate] = []
        if reference is not None and len(reference) >= 2:
            self.candidates.append(
                Candidate(path=np.asarray(reference, dtype=float),
                          meta={"kind": "reference"}))

    def add(self, path, kind: str, **meta) -> None:
        self.candidates.append(
            Candidate(path=np.asarray(path, dtype=float),
                      meta={"kind": kind, **meta}))

    def __len__(self) -> int:
        return len(self.candidates)


def sample_arc(start_pos, heading: float, speed: float,
               max_steer: float, dt: float = 0.25, n_steps: int = 24,
               n_curv: int = 7, max_curv: float = 0.10) -> CandidateSet:
    """Bicycle-model arc fan from the ego.

    Samples `n_curv` curvatures between -max_curv..max_curv (rad/m, a
    real sedan tops out around 0.1-0.15 rad/m ~ 7-10 m radius) and
    integrates forward with a constant steer, producing `n_curv` arc
    paths of ``n_steps`` points spaced `dt` seconds apart.  ``heading``
    accumulation is capped at 90 deg so no candidate loops back on
    itself.  A ``CandidateSet``.
    """
    pos = np.asarray(start_pos, dtype=float)[:2]
    h = float(heading)
    # physical curvature bound (rad/m); do not scale by speed (curvature
    # is a geometry property, speed only limits it via lateral accel which
    # is handled by the speed layer, not the steering fan).
    max_curv = max(0.02, min(float(max_curv), 0.20))
    curvatures = np.linspace(-max_curv, max_curv, n_curv)
    ds = max(0.5, float(speed) * dt)
    head_cap = math.pi / 2.0
    set_ = CandidateSet(reference=None)
    for kc in curvatures:
        x, y, th = pos[0], pos[1], h
        pts = [(x, y)]
        for _ in range(n_steps):
            th += kc * ds
            if abs(th - h) > head_cap:
                break
            x += ds * math.cos(th)
            y += ds * math.sin(th)
            pts.append((x, y))
        set_.add(np.asarray(pts), "arc", steer=float(kc))
    return set_


def sample_lane_shift(reference, offsets=(0.0,), blend_m: float = 8.0,
                      ahead_m: float = 30.0) -> CandidateSet:
    """Lateral-shift candidates of a reference path.

    Shifts the reference path by each lateral offset, blending from the
    ego (no shift at the start) to the full offset after ``blend_m`` -
    a lane-change / shoulder-bypass candidate family around the route.
    ``offsets`` in metres, positive = left of travel.
    """
    ref = np.asarray(reference, dtype=float)[:, :2] if reference is not None \
        else None
    if ref is None or len(ref) < 4:
        set_ = CandidateSet(reference=None)
        set_.candidates = []
        return set_
    set_ = CandidateSet(reference=ref)
    dx = np.diff(ref, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(dx, axis=1))])
    # per-point left normal of the reference
    norm = np.zeros_like(ref)
    for i in range(len(ref)):
        i0 = max(0, i - 2)
        i1 = min(len(ref) - 1, i + 2)
        tv = ref[i1] - ref[i0]
        L = float(np.linalg.norm(tv))
        if L < 1e-9:
            continue
        norm[i] = np.array([-tv[1] / L, tv[0] / L])
    for off in offsets:
        if abs(off) < 1e-9:
            continue
        shift = np.zeros(len(ref))
        for i in range(len(ref)):
            f = min(1.0, max(0.0, (cum[i] - cum[0]) / max(1e-9, blend_m)))
            # ramp back to 0 beyond ahead_m so the path rejoins the route
            g = min(1.0, max(0.0, (ahead_m - (cum[i] - cum[0]))
                             / max(1e-9, ahead_m * 0.5)))
            shift[i] = f * g * off
        shifted = ref + norm * shift[:, None]
        set_.add(shifted, "lane_shift", offset=float(off))
    return set_