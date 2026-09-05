"""Serve a trained SB3 DQN decision policy inside the FSD drive loop.

``DQNRuntime`` wraps a Stable-Baselines3 DQN zip (M4 decision layer):
one ``decision_observation`` vector in, one discrete action out, mapped
onto a cap for the plan's target speed (``action_to_target``).  The
decision layer can only SLOW the plan down - the layered planner, the
safety monitor and the no-cross rules stay authoritative, so a bad DQN
action degrades comfort, never safety.
"""

from __future__ import annotations

import time
from pathlib import Path

from beamng_autopilot.rl.obs import decision_observation

DEFAULT_DQN_WEIGHTS = "logs/m4_dqn/dqn_decision.zip"


def action_to_target(action: int, target_speed: float,
                     min_speed: float = 0.5) -> float:
    """Map a discrete decision onto a capped plan target speed."""
    mult = {0: 1.0, 1: 0.6, 2: 0.25}.get(int(action), 1.0)
    return max(float(min_speed), float(target_speed) * mult)


class DQNRuntime:
    """Load + serve one trained SB3 DQN decision policy."""

    def __init__(self, weights=None, device: str | None = None) -> None:
        self.weights = Path(weights) if weights else None
        self.device = device
        self.model = None
        self._err: str | None = None
        if self.weights is not None and self.weights.exists():
            try:
                from stable_baselines3 import DQN
                self.model = DQN.load(str(self.weights),
                                      device=self.device or "auto")
            except Exception as exc:
                self._err = str(exc)
                self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def error(self) -> str | None:
        return self._err

    def predict(self, speed: float, target_speed: float,
                fwd_clearance, closest_obs, lane_dev, road_off,
                n_tracks) -> tuple[int, float]:
        """Decision observation -> ``(action, inference ms)``."""
        if self.model is None:
            return 0, 0.0
        obs = decision_observation(
            speed=speed, target_speed=target_speed,
            fwd_clearance=fwd_clearance, closest_obs=closest_obs,
            lane_dev=lane_dev, road_off=road_off, n_tracks=n_tracks)
        t0 = time.time()
        action, _ = self.model.predict(obs, deterministic=True)
        ms = (time.time() - t0) * 1000.0
        return int(action), ms
