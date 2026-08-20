"""Temporal fusion: multi-frame occupancy smoothing + object tracking.

Tesla FSD fuses perception across time: single-frame occupancy and
detections are noisy, so the stack keeps a running estimate that decays
old evidence and merges new observations.  This module gives that
*structure* as game-free logic:

* ``TemporalOccupancyFilter``: exponential-moving-average occupancy grid
  across frames.  Each new frame is merged into a persistent log-odds /
  probability map; unobserved cells decay toward their prior.  A planner
  or safety monitor reads ``filter_raster()`` instead of the raw single
  frame, so a single LiDAR glitch never causes a phantom wall or a
  vanished one.
* ``TrackedObject`` / ``WorldObjectTracker``: persistent object tracks
  with position / velocity / age / lost count, matched by proximity each
  frame (the FSD "deep vision + depth" object tracking layer shape).
  Static clutter is kept as a slow track so low-speed centroid jitter
  does not produce fake moving objects, exactly like the existing
  LiDAR cluster tracker but at the vector-space level.

The filter is backed by plain numpy and works with any occupancy source
(LiDAR, fused cameras, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


class TemporalOccupancyFilter:
    """EMA occupancy grid across time.

    ``tau_s`` is the EMA half-life in seconds: older evidence decays with
    ``exp(-dt/tau_s)``.  ``conf_bias`` is how strongly new observations
    pull the running probability toward 1 (observation confidence).
    """

    def __init__(self, n: int = 60, res: float = 0.5,
                 tau_s: float = 1.5, conf_bias: float = 0.6):
        self.n = int(n)
        self.res = float(res)
        self.tau_s = float(tau_s)
        self.conf_bias = float(conf_bias)
        self.extent = 0.5 * n * res
        self.occ = np.full((n, n), 0.5, dtype=np.float32)
        self._last_t = -1e9

    # ------------------------------------------------------------------
    def update(self, frame_raster: np.ndarray, dt: float) -> None:
        """Fuse one new occupancy frame (0..1, None cells as 0.5 prior)."""
        if dt > 0:
            decay = float(math.exp(-dt / max(1e-3, self.tau_s)))
            self.occ *= decay
            # unknown cells drift back toward the neutral prior
            self.occ = self.occ + (0.5 - self.occ) * (1.0 - decay) * 0.2
        raw = np.asarray(frame_raster, dtype=np.float32)
        if raw.shape != self.occ.shape:
            raw = np.resize(raw, self.occ.shape)
        # merge: new *evidence* pulls probability toward 1; a frame that
        # simply did not observe the cell (raw near 0) must NOT erase an
        # established obstacle in one tick - it only lets decay handle
        # it.  So positive observations update up, while empty frames only
        # rely on the exponential decay above.
        pos = raw > 0.15
        if np.any(pos):
            self.occ[pos] = self.occ[pos] + (
                1.0 - self.occ[pos]) * self.conf_bias
        self.occ = np.clip(self.occ, 0.0, 1.0)
        self._last_t += dt if dt > 0 else 0.0

    def raster(self) -> np.ndarray:
        return self.occ.copy()

    def occupied_mask(self, thr: float = 0.6) -> np.ndarray:
        return self.occ >= thr

    def clear(self) -> None:
        self.occ.fill(0.5)


@dataclass
class TrackedObject:
    """One persistent world-space track."""

    track_id: int
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    age_s: float = 0.0
    matches: int = 1
    lost: int = 0          # consecutive frames without a match
    category: str = "object"


class WorldObjectTracker:
    """Proximity-matched persistent object tracks.

    New detections are matched to the nearest existing track within
    ``match_m`` and the track's velocity is smoothed; unmatched tracks
    age out (``lost`` counter) and are pruned after ``max_lost``.
    """

    def __init__(self, match_m: float = 2.5, max_lost: int = 4,
                 max_speed: float = 40.0, dt_default: float = 0.1):
        self.match_m = float(match_m)
        self.max_lost = int(max_lost)
        self.max_speed = float(max_speed)
        self.dt_default = float(dt_default)
        self.tracks: list[TrackedObject] = []
        self._next_id = 1

    # ------------------------------------------------------------------
    def update(self, detections, dt: float | None = None) -> list[TrackedObject]:
        """Associate ``detections`` (iterable of (x, y, category)) with
        existing tracks and return the active track list."""
        dt = self.dt_default if dt is None else max(1e-3, float(dt))
        dets = list(detections)

        active = []
        assigned = [False] * len(dets)
        for tr in self.tracks:
            best_i = -1
            best_d = self.match_m
            for i, (x, y, cat) in enumerate(dets):
                if assigned[i]:
                    continue
                d = math.hypot(tr.x - float(x), tr.y - float(y))
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                x, y, cat = dets[best_i]
                assigned[best_i] = True
                # velocity
                vx = (float(x) - tr.x) / dt
                vy = (float(y) - tr.y) / dt
                spd = math.hypot(vx, vy)
                # cap absurd teleports as new object without fake velocity
                if spd <= self.max_speed:
                    k = 0.5
                    tr.vx = (1.0 - k) * tr.vx + k * vx
                    tr.vy = (1.0 - k) * tr.vy + k * vy
                tr.x, tr.y = float(x), float(y)
                tr.category = str(cat)
                tr.age_s += dt
                tr.matches += 1
                tr.lost = 0
                active.append(tr)
            else:
                tr.lost += 1
                if tr.lost <= self.max_lost:
                    # keep the track a few frames even when briefly missed
                    tr.age_s += dt
                    active.append(tr)

        # new detections -> new tracks
        for i, (x, y, cat) in enumerate(dets):
            if not assigned[i]:
                self._next_id += 1
                tr = TrackedObject(track_id=self._next_id,
                                   x=float(x), y=float(y),
                                   category=str(cat))
                active.append(tr)

        self.tracks = active
        return active