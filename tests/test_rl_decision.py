"""Offline tests for the M4 DQN decision pipeline (obs / env / runtime)."""

from __future__ import annotations

import math

import json

import numpy as np
import pytest

from beamng_autopilot.rl import (
    DecisionSpeedEnv, DQNRuntime, action_to_target, decision_observation,
    DECISION_OBS_SIZE,
)


# --- observation builder -------------------------------------------------
def test_obs_vector_size_and_range() -> None:
    obs = decision_observation(
        speed=4.0, target_speed=6.0, fwd_clearance=15.0, closest_obs=None,
        lane_dev=0.4, road_off=0.2, n_tracks=5)
    assert obs.shape == (DECISION_OBS_SIZE,)
    assert obs.dtype == np.float32
    assert (obs >= 0.0).all() and (obs <= 1.0).all()


def test_obs_none_and_garbage_map_neutral() -> None:
    obs = decision_observation(
        speed=float("nan"), target_speed=6.0, fwd_clearance=None,
        closest_obs="x", lane_dev=None, road_off=float("inf"),
        n_tracks=-3)
    # unknown / off values collapse to the neutral 1.0 (no constraint)
    assert obs[1] == 1.0 and obs[2] == 1.0 and obs[3] == 1.0
    assert obs[4] == 1.0
    assert np.isfinite(obs).all()


# --- env contract ---------------------------------------------------------
def test_env_reset_step_contract() -> None:
    env = DecisionSpeedEnv(mode="offline", seed=3)
    obs, info = env.reset(seed=3)
    assert obs.shape == (DECISION_OBS_SIZE,)
    assert env.action_space.contains(0)
    obs2, reward, terminated, truncated, info = env.step(1)
    assert obs2.shape == (DECISION_OBS_SIZE,)
    assert isinstance(reward, float)
    assert terminated is False and truncated is False


def test_env_collision_terminates() -> None:
    env = DecisionSpeedEnv(mode="offline", seed=0)
    env.reset(seed=0)
    # full speed into the lead vehicle: brake nothing, ease nothing
    done = False
    steps = 0
    while not done and steps < env.episode_steps:
        _, _, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        steps += 1
        if terminated:
            assert info["collided"] is True
            break
    assert done, "an always-cruise policy must terminate eventually"


def test_env_slow_action_avoids_collision_better_than_cruise() -> None:
    def run(actions):
        env = DecisionSpeedEnv(mode="offline", seed=11)
        env.reset(seed=11)
        total, done, i = 0.0, False, 0
        info = {}
        while not done and i < 400:
            a = actions(i, env)
            _, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
            i += 1
        return total, info.get("collided", False)

    total_cruise, crashed_cruise = run(lambda i, e: 0)
    total_slow, crashed_slow = run(
        lambda i, e: 2 if e.lead_d < 25.0 else 0)
    assert crashed_slow is False or total_slow > total_cruise


# --- action mapping -------------------------------------------------------
def test_action_to_target_only_slows() -> None:
    # five levels: 1.0 / 0.8 / 0.6 / 0.35 / 0.1 of the plan target
    assert action_to_target(0, 6.0) == pytest.approx(6.0)
    assert action_to_target(1, 6.0) == pytest.approx(4.8)
    assert action_to_target(2, 6.0) == pytest.approx(3.6)
    assert action_to_target(3, 6.0) == pytest.approx(2.1)
    assert action_to_target(4, 6.0) == pytest.approx(0.6)
    for a in range(5):
        assert action_to_target(a, 6.0) <= 6.0 + 1e-9
        assert action_to_target(a, 6.0) >= 0.5 - 1e-9
    assert action_to_target(99, 6.0) == pytest.approx(6.0)  # unknown=cruise


# --- runtime with a real trained model ------------------------------------
def test_dqn_runtime_load_and_predict(tmp_path) -> None:
    from stable_baselines3 import DQN

    from beamng_autopilot.rl.env import DecisionSpeedEnv
    env = DecisionSpeedEnv(mode="offline", seed=1)
    model = DQN("MlpPolicy", env, seed=1, verbose=0, buffer_size=500,
                train_freq=4, learning_starts=0)
    model.learn(total_timesteps=64)
    p = tmp_path / "tiny_dqn.zip"
    model.save(str(p))
    rt = DQNRuntime(p)
    assert rt.loaded
    action, ms = rt.predict(speed=3.0, target_speed=6.0,
                            fwd_clearance=10.0, closest_obs=10.0,
                            lane_dev=0.1, road_off=0.0, n_tracks=1)
    assert action in (0, 1, 2, 3, 4)
    assert ms >= 0.0


def test_dqn_runtime_missing_weights_disabled(tmp_path) -> None:
    rt = DQNRuntime(tmp_path / "nope.zip")
    assert rt.loaded is False
    assert rt.predict(3.0, 6.0, 10.0, 10.0, 0.0, 0.0, 1) == (0, 0.0)


def test_trained_decision_policy_exists_and_evaluates() -> None:
    """The real offline training loop must have produced weights + report
    with a policy that beats the always-cruise baseline on collisions."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    weights = root / "logs" / "m4_dqn" / "dqn_decision.zip"
    if not weights.exists():
        pytest.skip("offline DQN training has not run on this machine")
    report = json.loads((weights.parent / "report.json").read_text(
        encoding="utf-8"))
    assert report["policy"]["collision_rate"] <= \
        report["baseline_cruise"]["collision_rate"]


# --- sim mode wiring (stubbed conn/stack, no game) ------------------------
class _StubState:
    def __init__(self, speed=2.0):
        self.pos = np.array([0.0, 0.0, 0.0])
        self.heading = 0.0
        self.speed = speed
        self.vel = None
        self.dir = None


class _StubVehicle:
    vid = "ego"


class _StubConn:
    vehicle = _StubVehicle()

    def __init__(self):
        self.speed = 2.0
        self.controls = []

    def get_state(self):
        self.speed += 0.2
        return _StubState(self.speed)

    def control(self, **kw):
        self.controls.append(kw)

    def step(self, n):
        self.steps = n


class _StubOut:
    best_speed = 4.0
    forward_clearance = 30.0
    best_path = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    tracks = []


class _StubStack:
    def tick(self, st=None, route_ref=None, **kw):
        return _StubOut()


def test_sim_mode_requires_conn_and_stack() -> None:
    with pytest.raises(ValueError):
        DecisionSpeedEnv(mode="sim")


def test_sim_step_runs_and_controls_car() -> None:
    conn = _StubConn()
    env = DecisionSpeedEnv(mode="sim", conn=conn, stack=_StubStack(),
                           cruise_speed=4.0)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(2)
    assert obs.shape == (DECISION_OBS_SIZE,)
    # at least one control frame with a steering command went out
    assert any("steering" in c for c in conn.controls)
    assert info["speed"] > 0.0
    assert terminated in (True, False)
