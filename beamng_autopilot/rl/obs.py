"""Decision-layer observation: a compact vector from the FSD stack.

The DQN decides HOW FAST to go, not where to steer - its observation is
the decision-level summary of the stack's own perception, normalized to
[0, 1] (unknown / sensor-off values map to a neutral 1.0 = "no
constraint"):

  0 speed / cruise target
  1 forward clearance / 30 m
  2 closest tracked obstacle distance / 30 m
  3 lane deviation / 2 m
  4 road-edge overshoot / 2 m
  5 tracked-object count / 10

`decision_observation` is pure so the offline env, the live drive loop
and the tests share one definition.
"""

from __future__ import annotations

import numpy as np

DECISION_OBS_SIZE = 6

CLEARANCE_NORM_M = 30.0
LANE_DEV_NORM_M = 2.0
ROAD_OFF_NORM_M = 2.0
TRACKS_NORM = 10.0


def decision_observation(speed: float, target_speed: float,
                         fwd_clearance, closest_obs,
                         lane_dev, road_off, n_tracks) -> np.ndarray:
    """Build the normalized decision vector (float32, size 6)."""

    def _clamp(x, scale):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(xf):
            return 1.0
        return float(max(0.0, min(1.0, xf / scale)))

    speed = max(0.0, float(speed))
    target = max(0.1, float(target_speed))
    return np.asarray([
        _clamp(speed, target),
        _clamp(fwd_clearance, CLEARANCE_NORM_M),
        _clamp(closest_obs, CLEARANCE_NORM_M),
        _clamp(lane_dev, LANE_DEV_NORM_M),
        _clamp(road_off, ROAD_OFF_NORM_M),
        _clamp(n_tracks, TRACKS_NORM),
    ], dtype=np.float32)
