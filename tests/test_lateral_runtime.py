"""The trained lateral policy must be bounded, fail-closed and optional."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.rl.lateral_env import ACTION_OFFSETS, DECAY
from beamng_autopilot.rl.lateral_runtime import (
    LateralRLRuntime,
    lateral_observation,
)


class _StubModel:
    """Stands in for the SB3 DQN: always picks the given action."""

    def __init__(self, action: int):
        self.action = int(action)
        self.seen: list = []

    def predict(self, obs, deterministic=True):
        self.seen.append(np.asarray(obs, dtype=float))
        return np.array(self.action), None


def _runtime(action: int, weights="ignored") -> LateralRLRuntime:
    rt = LateralRLRuntime.__new__(LateralRLRuntime)
    rt.weights = weights
    rt.device = None
    rt.error = None
    rt.applied = 0.0
    rt.last_action = None
    rt.model = _StubModel(action)
    return rt


def test_observation_matches_the_training_normalisation() -> None:
    obs = lateral_observation(0.9, 0.25, 0.01, 7.0, 0.25)
    assert obs.dtype == np.float32
    assert obs[0] == pytest.approx(0.9 / 1.8)
    assert obs[1] == pytest.approx(0.5)
    assert obs[3] == pytest.approx(0.5)
    assert obs[4] == pytest.approx(0.5)


def test_the_residual_lags_toward_the_chosen_offset_and_is_bounded() -> None:
    rt = _runtime(action=4)                     # +0.5 m target
    first = rt.act(lat_err_m=0.0, heading_err_rad=0.0, curvature=0.0,
                   speed_mps=5.0)
    assert first == pytest.approx(ACTION_OFFSETS[4] * (1.0 - DECAY))
    for _ in range(50):
        value = rt.act(lat_err_m=0.0, heading_err_rad=0.0, curvature=0.0,
                       speed_mps=5.0)
    # First-order lag: it approaches the chosen offset, never overshoots it.
    assert value == pytest.approx(ACTION_OFFSETS[4], abs=0.01)
    assert value < ACTION_OFFSETS[4]
    assert abs(value) <= 0.5


def test_an_untrusted_reference_decays_and_is_never_consulted() -> None:
    rt = _runtime(action=4)
    rt.act(lat_err_m=0.0, heading_err_rad=0.0, curvature=0.0, speed_mps=5.0)
    rt.act(lat_err_m=0.0, heading_err_rad=0.0, curvature=0.0, speed_mps=5.0)
    calls = len(rt.model.seen)
    out = rt.act(lat_err_m=9.9, heading_err_rad=9.9, curvature=9.9,
                 speed_mps=9.9, reference_ok=False)
    assert len(rt.model.seen) == calls                # not consulted
    assert abs(out) < abs(ACTION_OFFSETS[4])          # decaying
    assert rt.last_action is None


def test_a_missing_checkpoint_is_a_no_op_not_a_crash() -> None:
    rt = LateralRLRuntime(weights="logs/m5_rl/does_not_exist.zip")
    assert rt.available is False
    assert rt.error and "not found" in rt.error
    assert rt.act(lat_err_m=1.0, heading_err_rad=0.1, curvature=0.0,
                  speed_mps=5.0) == 0.0


def test_a_predict_failure_degrades_to_zero() -> None:
    rt = _runtime(action=4)
    rt.model = _StubModel(4)

    def boom(obs, deterministic=True):
        raise RuntimeError("model gone")

    rt.model.predict = boom
    assert rt.act(lat_err_m=0.0, heading_err_rad=0.0, curvature=0.0,
                  speed_mps=5.0) == 0.0
    assert rt.error and "predict failed" in rt.error


def test_the_shipped_checkpoint_loads_and_stays_bounded() -> None:
    """Real artefact: it must load, and 200 steps must stay within +-0.5 m."""
    rt = LateralRLRuntime()
    if not rt.available:
        pytest.skip(f"lateral checkpoint unavailable: {rt.error}")
    worst = 0.0
    for i in range(200):
        value = rt.act(lat_err_m=0.4 * np.sin(i / 5.0), heading_err_rad=0.05,
                       curvature=0.004, speed_mps=6.0)
        worst = max(worst, abs(value))
    assert worst <= 0.5 + 1e-9
