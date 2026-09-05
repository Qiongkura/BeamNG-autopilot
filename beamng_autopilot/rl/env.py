"""Gym-style decision env: discrete cruise/ease/slow speed decisions.

``DecisionSpeedEnv`` is the M4 training environment.  The agent owns the
LONGITUDINAL DECISION only - steering, lane keeping and the plan's
geometry stay with the FSD stack; the decision caps the plan's target
speed:

* five speed levels over the plan target (1.0 / 0.8 / 0.6 / 0.35 /
  0.1): cruise keeps the plan, the middle levels trim speed for
  nearby obstacles, and the near-stop floor matches a lead vehicle
  crawling ahead (without a low enough floor the ego can never match a
  slow lead and every episode ends in a guaranteed rear-end)

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

ACTION_CRUISE, ACTION_EASE, ACTION_SLOW = 0, 2, 4
ACTION_MULT = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.35, 4: 0.1}

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
                 cruise_speed: float = 8.0,
                 conn=None, stack=None, route_provider=None,
                 sim_steps: int = 3):
        super().__init__()
        self.mode = mode
        if mode == "sim" and (conn is None or stack is None):
            raise ValueError("sim mode needs conn + stack")
        self.conn = conn
        self.stack = stack
        self.route_provider = route_provider
        self.sim_steps = max(1, int(sim_steps))
        self.cruise_speed = float(cruise_speed)
        self.episode_steps = int(episode_s / DT)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(DECISION_OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(ACTION_MULT))
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
        self._action = int(action)
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
        if self.mode == "sim" and self.conn is not None:
            try:
                self.v = max(0.0, float(self.conn.get_state().speed))
            except Exception:
                self.v = 0.0
        return self._obs(), {"mode": self.mode}

    def step(self, action):
        self._apply_action(action)
        if self.mode == "offline":
            reward = self._step_dynamics()
            terminated = self.collided
            info = {"collided": self.collided, "speed": self.v}
        else:
            reward, terminated, info = self._step_sim()
        self.step_i += 1
        truncated = self.step_i >= self.episode_steps
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def _step_sim(self) -> tuple[float, bool, dict]:
        """One live decision: steer via the stack, longitudinal by the
        decision, step the sim, observe through the stack's perception.

        The layered stack stays authoritative for WHERE to drive (its
        best_path is pursued); the DQN decision only caps the plan's
        target speed.  A low forward clearance terminates the episode
        with the collision penalty (the emergency brake is applied so a
        training run can never grind into an obstacle).
        """
        import math as _math

        import numpy as _np

        from beamng_autopilot.control.pure_pursuit import PurePursuit
        from beamng_autopilot.control.speed import SpeedController

        if not hasattr(self, "_pp"):
            self._pp = PurePursuit(lookahead=5.0)
            self._spd = SpeedController(deadband=0.2, hyst_mps=0.25)
            self._fwd_gear = None
        st = self.conn.get_state()
        heading = float(st.heading)
        pos = _np.asarray(st.pos, dtype=float)
        route_ref = (self.route_provider(pos, heading)
                     if self.route_provider else None)
        out = self.stack.tick(st=st, route_ref=route_ref)
        plan_target = float(out.best_speed or self.eff_target
                            or self.cruise_speed)
        self.plan_target = max(0.5, plan_target)
        self._last_tracks = list(getattr(out, "tracks", []) or [])
        eff = max(0.5, plan_target * ACTION_MULT.get(int(self._action), 1.0))
        steer = 0.0
        if out.best_path is not None and len(out.best_path) >= 2:
            try:
                steer, _, _ = self._pp.steering(
                    pos, heading,
                    _np.asarray(out.best_path, dtype=float)[:, :2],
                    speed=float(st.speed))
            except Exception:
                steer = 0.0
        # NB: update(target_speed, speed) - reversed args would invert
        # the whole longitudinal control (throttle on slow, brake on go)
        thr, brk = self._spd.update(eff, float(st.speed))
        if self._fwd_gear is None:
            from beamng_autopilot.control import gearbox
            self._fwd_gear = gearbox.forward_gear_input(self.conn)
        self.conn.control(throttle=float(thr), brake=float(brk),
                          steering=float(steer), gear=self._fwd_gear)
        self.conn.step(int(self.sim_steps))
        st2 = self.conn.get_state()
        self.v = max(0.0, float(st2.speed))
        clear = float(out.forward_clearance)             if out.forward_clearance is not None             and _math.isfinite(out.forward_clearance) else 30.0
        self.lead_d = clear
        collided = clear < 1.2
        if collided:
            try:
                self.conn.control(throttle=0.0, brake=1.0, steering=0.0,
                                  gear=self._fwd_gear)
                self.conn.step(5)
            except Exception:
                pass
        reward = self.v * DT - ACTION_COST
        if clear > 25.0:
            reward -= SLOW_PENALTY * max(0.0, plan_target - self.v) * DT
        if collided:
            reward -= COLLISION_PENALTY
        info = {"collided": collided, "speed": self.v,
                "clearance": clear}
        return reward, collided, info

    def _obs(self) -> np.ndarray:
        if self.mode == "sim":
            return decision_observation(
                speed=self.v,
                target_speed=self.plan_target,
                fwd_clearance=self.lead_d,
                closest_obs=self.lead_d,
                lane_dev=0.0,
                road_off=0.0,
                n_tracks=len(getattr(self, "_last_tracks", [])) or
                self.clutter_tracks,
            )
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
