"""Gym-style decision env: discrete cruise/ease/slow speed decisions.

``DecisionSpeedEnv`` is the M4 training environment.  The agent owns the
LONGITUDINAL DECISION only - steering, lane keeping and the plan's
geometry stay with the FSD stack; the decision caps the plan's target
speed:

* action 0 = cruise    (keep the plan target)
* action 1 = ease      (target x 0.6)
* action 2 = slow      (target x 0.1 - near-stop, matches a lead
  vehicle crawling ahead; without a low enough floor the ego can never
  match a slow lead and every episode ends in a guaranteed rear-end)

Two modes share one contract:

* ``mode="offline"`` (default): a game-free car-following simulator -
  the ego chases its commanded target with bounded acceleration while
  lead vehicles appear ahead at random speeds; a collision terminates
  the episode with a large penalty.  This is what the offline training
  + evaluation loop runs on (no BeamNG needed).
* ``mode="sim"``: every step polls ``FSDStack``/connector state and
  applies the decision to the live target speed (the same observation
  builder, so a policy trained offline transfers).

Reward per step: forward progress, minus a large collision penalty,
minus a small penalty for driving slower than the plan target (so
"slow forever" is not a winning strategy), minus a tiny action-cost.
"""

from __future__ import annotations

import math
import random

import gymnasium as gym
import numpy as np

from beamng_autopilot.rl.obs import (
    decision_observation, DECISION_OBS_SIZE,
)

ACTION_CRUISE, ACTION_EASE, ACTION_SLOW = 0, 1, 2
ACTION_MULT = {ACTION_CRUISE: 1.0, ACTION_EASE: 0.6, ACTION_SLOW: 0.1}

DT = 0.25                 # decision step (s), matches the control cadence
EPISODE_S = 40.0
MAX_ACCEL = 1.5           # m/s^2 the ego can chase its target with
MAX_BRAKE = 3.0
COLLISION_PENALTY = 10.0
SLOW_PENALTY = 0.6        # per (m/s) below the plan target, per step
ACTION_COST = 0.01
GAP_MIN_M = 2.0           # bumper-to-bumper distance at contact


class DecisionSpeedEnv(gym.Env):
    """Discrete speed-decision env (offline car-following by default)."""

    metadata = {"render_modes": []}

    def __init__(self, mode: str = "offline", seed: int | None = None,
                 episode_s: float = EPISODE_S,
                 cruise_speed: float = 8.0):
        super().__init__()
        self.mode = mode
        self.cruise_speed = float(cruise_speed)
        self.episode_steps = int(episode_s / DT)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(DECISION_OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.rng = random.Random(seed)
        self._reset_state()

    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        self.v = 0.0
        self.step_i = 0
        self.plan_target = self.cruise_speed
        self.eff_target = self.cruise_speed
        self.collided = False
        # Live scenes carry constant roadside clutter (trees / walls sit
        # 3-10 m away, ~60-120 tracked clusters) that must NOT read as
        # urgency - the policy learns to react to the LEAD gap dynamics
        # and ignore the clutter channels, matching the live
        # observation distribution.
        self.clutter_d = float(self.rng.uniform(3.0, 10.0))
        self.clutter_tracks = float(self.rng.uniform(60.0, 120.0))
        # lead vehicle: (distance ahead m, speed m/s); respawns when it
        # drives out of range
        self.lead_d = float(self.rng.uniform(15.0, 60.0))
        self.lead_v = float(self.rng.uniform(0.1, 0.6 * self.cruise_speed))

    def _respawn_lead(self) -> None:
        self.lead_d = float(self.rng.uniform(35.0, 80.0))
        self.lead_v = float(self.rng.uniform(0.1, 0.8 * self.cruise_speed))

    def _apply_action(self, action: int) -> None:
        mult = ACTION_MULT.get(int(action), 1.0)
        self.eff_target = self.plan_target * mult

    def _step_dynamics(self) -> float:
        """Advance the offline car-following sim; returns the reward."""
        # ego chases the commanded target with bounded acceleration
        dv = self.eff_target - self.v
        rate = MAX_ACCEL if dv > 0 else MAX_BRAKE
        self.v += max(-rate * DT, min(rate * DT, dv))
        self.v = max(0.0, self.v)
        # lead vehicle moves; gap closes at (v - lead_v)
        self.lead_d += (self.lead_v - self.v) * DT
        reward = self.v * DT                      # progress
        # Slowness only counts against the decision when the road ahead
        # is actually CLEAR - trailing a lead vehicle at its speed is the
        # CORRECT decision, and punishing it teaches the policy to ram
        # the lead (first training round: collision_rate 0.83 because
        # slowing near the lead kept paying the penalty).
        if self.lead_d > 25.0:
            reward -= SLOW_PENALTY * max(
                0.0, self.plan_target - self.v) * DT
        reward -= ACTION_COST
        if self.lead_d <= GAP_MIN_M:
            self.collided = True
            reward -= COLLISION_PENALTY
        elif self.lead_d > 100.0:
            self._respawn_lead()
        return reward

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self._reset_state()
        return self._obs(), {"mode": self.mode}

    def step(self, action):
        self._apply_action(action)
        if self.mode == "offline":
            reward = self._step_dynamics()
            terminated = self.collided
        else:
            raise NotImplementedError(
                "sim mode steps through BeamNGConnector; use the live "
                "m4 training entry with --sim (not part of the offline "
                "closed loop)")
        self.step_i += 1
        truncated = self.step_i >= self.episode_steps
        return self._obs(), float(reward), terminated, truncated, {
            "collided": self.collided,
            "speed": self.v,
        }

    def _obs(self) -> np.ndarray:
        return decision_observation(
            speed=self.v,
            target_speed=self.plan_target,
            fwd_clearance=self.lead_d,
            closest_obs=min(self.lead_d, self.clutter_d),
            lane_dev=0.0,
            road_off=0.0,
            n_tracks=self.clutter_tracks + (1.0 if self.lead_d < 60.0
                                            else 0.0),
        )
