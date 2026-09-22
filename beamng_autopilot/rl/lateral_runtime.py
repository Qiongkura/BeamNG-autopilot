"""Serve the trained lateral-residual policy inside the drive loop.

``m5_train_lateral_dqn.py`` trains a DQN (``logs/m5_rl/dqn_lateral.zip``)
to choose a BOUNDED lateral offset that is added to whatever lateral
reference the base controller is already tracking.  The training
environment is a procedural road, and its report measured the policy at
mean |lateral error| 0.076 m against 0.128 m for the zero-residual
baseline (40k steps, seed 7, both zero off-road) - i.e. in its own
distribution the residual decision reduces lane-keeping error by about
40% without leaving the road.

That checkpoint had no runtime: nothing outside the environment, the
trainer and its tests loaded it, so the policy was trained and never
used.  This module is the missing consumer.  Design rules, all of which
exist because the alternative is a learned component with unbounded
authority:

* the residual is BOUNDED to +-0.5 m and DECAYS toward zero every step,
  exactly as in training, so the policy can never fling the car off the
  road in one step;
* it is a RESIDUAL on a lateral reference the perception already trusts -
  with no trustworthy reference the caller passes ``reference_ok=False``
  and the policy is not consulted at all (a learned component must never
  invent where the lane is);
* every returned offset is a PROPOSAL: the drive loop shifts the steering
  path, re-runs the safety monitor on the shifted path and drops the
  shift when the monitor refuses - the same contract the painted-line
  corrector uses;
* loading is fail-closed: a missing or unloadable checkpoint leaves
  ``available`` False and every call returning 0.0, so enabling the
  switch without the artefact is a no-op, not a crash or a guess.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .lateral_env import ACTION_OFFSETS, CURV_MAX, DECAY, ROAD_HALF_M

DEFAULT_LATERAL_WEIGHTS = "logs/m5_rl/dqn_lateral.zip"

# The training observations are normalised by these denominators; the
# runtime must use the same numbers or the policy sees out-of-distribution
# inputs (the DQN checkpoint contract check exists for the same reason).
HEADING_NORM_RAD = 0.5
SPEED_NORM_MPS = 14.0
RESIDUAL_NORM_M = 0.5


def lateral_observation(lat_err_m: float, heading_err_rad: float,
                        curvature: float, speed_mps: float,
                        applied_m: float) -> np.ndarray:
    """The 5-dim normalised observation the policy was trained on."""
    return np.array([
        float(lat_err_m) / ROAD_HALF_M,
        float(heading_err_rad) / HEADING_NORM_RAD,
        float(curvature) / CURV_MAX,
        float(speed_mps) / SPEED_NORM_MPS,
        float(applied_m) / RESIDUAL_NORM_M,
    ], dtype=np.float32)


class LateralRLRuntime:
    """Load and serve the lateral-residual DQN (fail-closed)."""

    def __init__(self, weights=None, device: str | None = None) -> None:
        self.weights = Path(weights) if weights else Path(DEFAULT_LATERAL_WEIGHTS)
        self.device = device
        self.model = None
        self.error: str | None = None
        self.applied = 0.0
        self.last_action: int | None = None
        if not self.weights.exists():
            self.error = f"weights not found: {self.weights}"
            return
        try:
            from stable_baselines3 import DQN
            self.model = DQN.load(str(self.weights), device=device or "auto")
        except Exception as exc:
            self.model = None
            self.error = f"load failed: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        self.applied = 0.0
        self.last_action = None

    def act(self, *, lat_err_m: float, heading_err_rad: float,
            curvature: float, speed_mps: float,
            reference_ok: bool = True,
            max_offset_m: float = 0.5) -> float:
        """Bounded lateral offset proposal for this tick (0.0 when unusable).

        ``reference_ok`` False means the base controller has no lateral
        reference it trusts this tick: the policy is then not consulted and
        the residual decays, so a learned component can never be the thing
        that decides where the lane is.
        """
        if not self.available or not reference_ok:
            self.applied *= DECAY
            self.last_action = None
            return float(np.clip(self.applied, -max_offset_m, max_offset_m))
        obs = lateral_observation(lat_err_m, heading_err_rad, curvature,
                                  speed_mps, self.applied)
        try:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)
        except Exception as exc:
            self.error = f"predict failed: {exc}"
            self.applied *= DECAY
            self.last_action = None
            return float(np.clip(self.applied, -max_offset_m, max_offset_m))
        if not (0 <= action < len(ACTION_OFFSETS)):
            self.applied *= DECAY
            self.last_action = None
            return float(np.clip(self.applied, -max_offset_m, max_offset_m))
        self.last_action = action
        target = float(ACTION_OFFSETS[action])
        # Same decay-toward-the-target update as the training env: the
        # applied residual is a first-order lag on the chosen offset, so a
        # single decision can move the reference by at most (1-DECAY)*0.5 m.
        self.applied = float(self.applied * DECAY + target * (1.0 - DECAY))
        return float(np.clip(self.applied, -max_offset_m, max_offset_m))
