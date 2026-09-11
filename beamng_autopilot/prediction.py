"""Motion prediction: where each tracked object WILL be, not just where it is.

The stack tracks objects - ``temporal.WorldObjectTracker`` gives every track a
smoothed ``vx``/``vy`` - but nothing consumes that velocity.  Planning scores
candidate trajectories against the occupancy grid as it is THIS tick, so a
moving obstacle is treated as if it stood still, and a vehicle that will cross
the corridor in two seconds is only avoided once it is already inside it.

A real stack inserts a prediction layer between perception and planning:
perception says where things are, prediction says where they will be over the
planning horizon, and planning scores against that.  This module is that layer,
written as pure game-free logic so the models and their bounds are unit-tested
rather than living inside the planner.

Deliberately conservative, because a prediction nobody can bound is worse than
none:

* constant velocity is the default and needs only the tracker's own state;
* an optional constant-turn-rate variant uses a supplied yaw rate;
* speeds are clamped and sub-threshold motion is reported as stationary, so
  tracking jitter cannot manufacture a moving obstacle;
* stale tracks (``lost`` beyond ``max_lost``) produce no prediction at all;
* the horizon is capped, because linear extrapolation is meaningless far out.

Nothing here steers the car.  The first consumer should be the planner's
collision cost, gated by live A/B (the same discipline every perception change
in this repo has needed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Planning horizon and sampling.  3 s covers a junction crossing at town
# speeds; 0.5 s is the resolution the collision cost needs.
PREDICT_HORIZON_S = 3.0
PREDICT_DT_S = 0.5
PREDICT_MAX_HORIZON_S = 8.0
# Below this the object counts as stationary: a parked car's centroid jitter
# must not read as motion (the tracker already smooths velocity, this is the
# second guard).
PREDICT_MIN_SPEED_MPS = 0.3
# Above this the velocity estimate is not believable for a road object.
PREDICT_MAX_SPEED_MPS = 40.0
# Tracks unseen for longer than this contribute nothing.
PREDICT_MAX_LOST = 2


@dataclass
class TrackPrediction:
    """One track's predicted world path over the horizon."""

    track_id: int
    category: str
    t: np.ndarray            # (N,) seconds from now, t[0] == 0
    points: np.ndarray       # (N, 2) world xy
    speed_mps: float
    moving: bool

    @property
    def final(self) -> tuple[float, float]:
        p = self.points[-1]
        return float(p[0]), float(p[1])


def _clamp_speed(vx: float, vy: float) -> tuple[float, float, float]:
    """Clamp the velocity magnitude, returning ``(vx, vy, speed)``."""
    speed = math.hypot(float(vx), float(vy))
    if not math.isfinite(speed):
        return 0.0, 0.0, 0.0
    if speed > PREDICT_MAX_SPEED_MPS:
        k = PREDICT_MAX_SPEED_MPS / speed
        return float(vx) * k, float(vy) * k, PREDICT_MAX_SPEED_MPS
    return float(vx), float(vy), speed


def predict_track(track, horizon_s: float = PREDICT_HORIZON_S,
                  dt: float = PREDICT_DT_S,
                  yaw_rate_rad_s: float | None = None) -> TrackPrediction | None:
    """Predict one track's path.  ``None`` when the track is not predictable.

    ``yaw_rate_rad_s`` switches to constant turn rate (arc of radius
    ``v / |w|``); without it the model is constant velocity, which is the
    honest default for a system whose only motion evidence is the tracker's
    smoothed ``vx``/``vy``.
    """
    if track is None:
        return None
    if int(getattr(track, "lost", 0) or 0) > PREDICT_MAX_LOST:
        return None
    x = float(getattr(track, "x", 0.0) or 0.0)
    y = float(getattr(track, "y", 0.0) or 0.0)
    vx, vy, speed = _clamp_speed(getattr(track, "vx", 0.0),
                                 getattr(track, "vy", 0.0))
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    horizon = max(0.0, min(float(horizon_s), PREDICT_MAX_HORIZON_S))
    step = max(1e-3, float(dt))
    n = int(math.floor(horizon / step)) + 1
    t = np.arange(n, dtype=float) * step
    moving = speed >= PREDICT_MIN_SPEED_MPS
    if not moving:
        pts = np.column_stack([np.full(n, x), np.full(n, y)])
        return TrackPrediction(int(getattr(track, "track_id", -1)),
                               str(getattr(track, "category", "object")),
                               t, pts, speed, False)
    w = float(yaw_rate_rad_s or 0.0)
    if abs(w) < 1e-6:
        pts = np.column_stack([x + vx * t, y + vy * t])
    else:
        # Constant turn rate: integrate from the current heading, which the
        # velocity vector already gives (atan2(vy, vx)).
        h0 = math.atan2(vy, vx)
        r = speed / abs(w)
        sgn = 1.0 if w > 0 else -1.0
        cx = x - sgn * r * math.sin(h0)
        cy = y + sgn * r * math.cos(h0)
        ang = h0 + w * t
        pts = np.column_stack([cx + sgn * r * np.sin(ang),
                              cy - sgn * r * np.cos(ang)])
    return TrackPrediction(int(getattr(track, "track_id", -1)),
                           str(getattr(track, "category", "object")),
                           t, pts, speed, True)


def predict_tracks(tracks, horizon_s: float = PREDICT_HORIZON_S,
                   dt: float = PREDICT_DT_S,
                   yaw_rates: dict[int, float] | None = None
                   ) -> list[TrackPrediction]:
    """Predict every track in a list, dropping the ones that cannot be."""
    out: list[TrackPrediction] = []
    rates = yaw_rates or {}
    for tr in tracks or ():
        pred = predict_track(tr, horizon_s=horizon_s, dt=dt,
                             yaw_rate_rad_s=rates.get(
                                 int(getattr(tr, "track_id", -1))))
        if pred is not None:
            out.append(pred)
    return out


def predicted_points_at(preds, t_s: float) -> np.ndarray:
    """``(M, 2)`` world positions of all predictions at time ``t_s``."""
    rows = []
    for p in preds or ():
        if len(p.t) == 0:
            continue
        k = int(np.argmin(np.abs(p.t - float(t_s))))
        rows.append(p.points[k])
    return (np.asarray(rows, dtype=float) if rows
            else np.empty((0, 2)))


def nearest_predicted_gap_m(preds, pos, heading: float,
                            corridor_half_m: float,
                            horizon_s: float | None = None) -> float | None:
    """Closest approach of any prediction to the ego's forward corridor.

    A cheap, planner-shaped consumer: for each predicted point inside the
    forward corridor (lateral within ``corridor_half_m``) return the smallest
    longitudinal distance.  ``None`` when nothing is predicted to enter the
    corridor - which lets a caller distinguish "clear ahead" from "no data".
    """
    if not preds:
        return None
    p = np.asarray(pos[:2], dtype=float)
    fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
    left = np.array([-fwd[1], fwd[0]])
    best: float | None = None
    for pred in preds:
        t_max = (PREDICT_MAX_HORIZON_S if horizon_s is None
                 else float(horizon_s))
        pts = pred.points[pred.t <= t_max]
        if len(pts) == 0:
            continue
        rel = pts - p
        lon = rel @ fwd
        lat = rel @ left
        inside = (lon >= 0.0) & (np.abs(lat) <= float(corridor_half_m))
        if not inside.any():
            continue
        d = float(lon[inside].min())
        best = d if best is None else min(best, d)
    return best


def prediction_digest(preds) -> dict:
    """JSON-safe summary for telemetry (no arrays)."""
    ps = list(preds or ())
    moving = [p for p in ps if p.moving]
    travel = [float(np.hypot(p.final[0] - float(p.points[0][0]),
                             p.final[1] - float(p.points[0][1])))
              for p in ps]
    return {
        "n": len(ps),
        "n_moving": len(moving),
        "max_speed_mps": (round(max((p.speed_mps for p in ps), default=0.0), 2)
                          if ps else 0.0),
        "max_travel_m": round(max(travel, default=0.0), 2),
    }
