"""Offline tests for the DAVE-2 BC steering candidate wiring."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from beamng_autopilot.neural.bc_runtime import (
    BCRuntime, DEFAULT_BC_WEIGHTS, steer_to_path,
)
from beamng_autopilot.planning.arbiter import arbitrate


# --- steer_to_path geometry --------------------------------------------
def test_steer_zero_is_straight() -> None:
    path = steer_to_path(0.0, [10.0, 20.0], 0.0, length_m=18.0, n=13)
    assert path is not None and path.shape == (13, 2)
    # heading 0 (east): the arc runs +x, y unchanged
    assert np.allclose(path[:, 1], 20.0, atol=1e-9)
    assert path[0, 0] > 10.0 and path[-1, 0] <= 10.0 + 18.0 + 1e-6


def test_left_steer_bends_left() -> None:
    # normalized LEFT (negative) at heading 0 must curve toward +y (left)
    path = steer_to_path(-0.5, [0.0, 0.0], 0.0, length_m=18.0, n=13)
    assert path[-1, 1] > 0.5, "left steer must bend left (+y)"
    assert path[-1, 0] > 0.0


def test_right_steer_bends_right() -> None:
    path = steer_to_path(0.5, [0.0, 0.0], 0.0, length_m=18.0, n=13)
    assert path[-1, 1] < -0.5


def test_steer_path_is_monotone_forward() -> None:
    # a MODERATE steering input must produce a forward-only arc; an
    # extreme full-lock rollout legitimately curves past 90 deg and is
    # the forward-progress gate's job to reject, not the rollout's
    path = steer_to_path(0.4, [0.0, 0.0], math.pi / 4)
    d = np.diff(path, axis=0)
    fwd = np.cos(math.pi / 4) * d[:, 0] + np.sin(math.pi / 4) * d[:, 1]
    assert (fwd > 0).all(), "the rollout must never go backwards"


def test_steer_path_rejects_garbage() -> None:
    assert steer_to_path(float("nan"), [0, 0], 0.0) is None
    assert steer_to_path(0.0, None, 0.0) is None
    assert steer_to_path(0.0, [float("inf"), 0.0], 0.0) is None
    # extreme inputs clamp, not explode
    p = steer_to_path(5.0, [0.0, 0.0], 0.0)
    assert p is not None and np.isfinite(p).all()


# --- BCRuntime -----------------------------------------------------------
def _tiny_ckpt(tmp_path, name="bc_test.pt"):
    """Save a real Dave2 checkpoint (random weights, training format)."""
    import torch

    from beamng_autopilot.bc import Dave2, conv_feature_size
    net = Dave2(feat_in=conv_feature_size(66, 200))
    p = tmp_path / name
    torch.save({"state_dict": net.state_dict(),
                "resize": (200, 66),
                "val_mae": 0.05}, str(p))
    return p


def test_bc_runtime_loads_training_format_checkpoint(tmp_path) -> None:
    p = _tiny_ckpt(tmp_path)
    rt = BCRuntime(p, device="cpu")
    assert rt.loaded is True
    assert (rt.img_w, rt.img_h) == (200, 66)
    steer, ms = rt.predict_steer(
        np.zeros((66, 200, 3), dtype=np.uint8))
    assert math.isfinite(steer) and abs(steer) <= 1.0
    assert ms >= 0.0


def test_bc_runtime_missing_weights_is_disabled(tmp_path) -> None:
    rt = BCRuntime(tmp_path / "nope.pt", device="cpu")
    assert rt.loaded is False
    assert rt.predict_steer(np.zeros((66, 200, 3), dtype=np.uint8)) == (0.0, 0.0)


def test_bc_runtime_real_checkpoint_if_present() -> None:
    root = Path(__file__).resolve().parents[1]
    p = root / DEFAULT_BC_WEIGHTS
    if not p.exists():
        pytest.skip("trained BC weights not present on this machine")
    rt = BCRuntime(p, device="cpu")
    assert rt.loaded
    steer, _ = rt.predict_steer(np.zeros((66, 200, 3), dtype=np.uint8))
    assert abs(steer) <= 1.0


# --- arbitration ranking -------------------------------------------------
def test_arbitrate_bc_ranks_between_e2e_and_rule() -> None:
    rule = np.array([[0.0, 0.0], [1.0, 0.0]])
    out = arbitrate(None, rule, bc_path=np.array([[0.0, 0.0], [1.0, 1.0]]),
                    bc_safe=True)
    assert out.source == "bc"
    # e2e still wins over bc
    out2 = arbitrate(None, rule, e2e_path=np.array([[0.0, 0.0], [2.0, 0.0]]),
                     e2e_safe=True, bc_path=np.array([[0.0, 0.0], [1.0, 1.0]]),
                     bc_safe=True)
    assert out2.source == "e2e"
    # an unsafe BC candidate falls through to the rule path
    out3 = arbitrate(None, rule, bc_path=np.array([[0.0, 0.0], [1.0, 1.0]]),
                     bc_safe=False)
    assert out3.source == "rule"
    # fsd still first
    out4 = arbitrate(np.array([[0.0, 0.0], [3.0, 0.0]]), rule,
                     bc_path=np.array([[0.0, 0.0], [1.0, 1.0]]), bc_safe=True)
    assert out4.source == "fsd"
