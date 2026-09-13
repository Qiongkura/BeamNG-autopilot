"""Lateral residual env regressions: shapes, bounds, seed determinism."""

from __future__ import annotations

import numpy as np

from beamng_autopilot.rl.lateral_env import LateralResidualEnv


def test_obs_shape_and_finite():
    env = LateralResidualEnv(mode="offline", seed=1)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (5,) and np.isfinite(obs).all()
    for _ in range(50):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(r) and np.isfinite(obs).all()
        assert not term


def test_seed_determinism():
    a = LateralResidualEnv(mode="offline", seed=42)
    b = LateralResidualEnv(mode="offline", seed=42)
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    assert np.allclose(oa, ob)
    rewards_a, rewards_b = [], []
    rng = np.random.default_rng(0)
    for k in range(100):
        act = int(rng.integers(0, 5))
        obs_a, r1, _, _, _ = a.step(act)
        obs_b, r2, _, _, _ = b.step(act)
        rewards_a.append(r1)
        rewards_b.append(r2)
    assert np.allclose(rewards_a, rewards_b)


def test_residual_stays_bounded():
    env = LateralResidualEnv(mode="offline", seed=3)
    env.reset(seed=3)
    max_applied = 0.0
    for _ in range(300):
        env.step(4)                        # 一直压最大残差
        max_applied = max(max_applied, abs(env.applied))
    assert max_applied <= 0.5 + 1e-9       # 自回中限幅：残差永不越界
