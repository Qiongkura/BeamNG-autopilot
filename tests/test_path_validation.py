"""Offline tests for the learned-path runtime acceptance contract."""

from __future__ import annotations

import numpy as np

from beamng_autopilot.planning.path_validation import (
    validate_learned_path,
)


def _straight(length_m: float = 20.0, n: int = 10) -> np.ndarray:
    x = np.linspace(0.0, length_m, n)
    return np.column_stack([x, np.zeros(n)])


def test_straight_path_is_accepted() -> None:
    out = validate_learned_path(
        _straight(), origin=(0.0, 0.0), forward=(1.0, 0.0))
    assert out.ok
    assert out.reason == ""
    assert out.extent_m > 0.0


def test_world_frame_path_uses_ego_forward_axis() -> None:
    # A path starting at the ego and advancing along +y in a heading=90°
    # frame must validate even though its world x is constant.
    path = np.column_stack([np.zeros(8), np.linspace(0.0, 16.0, 8)])
    out = validate_learned_path(path, origin=(0.0, 0.0),
                                forward=(0.0, 1.0))
    assert out.ok
    assert out.backstep_m == 0.0


def test_bad_shapes_and_nonfinite_are_rejected() -> None:
    assert validate_learned_path(None).reason == "shape"
    assert validate_learned_path(np.zeros((1, 2))).reason == "shape"
    bad = _straight()
    bad[4, 0] = np.nan
    assert validate_learned_path(bad).reason == "nonfinite"


def test_extent_lateral_backstep_and_curvature_gates() -> None:
    assert validate_learned_path(
        _straight(0.5), min_extent_m=1.0).reason == "extent_low"
    assert validate_learned_path(
        _straight(120.0), max_extent_m=80.0).reason == "extent_high"

    lateral = _straight(20.0)
    lateral[5, 1] = 40.0
    assert validate_learned_path(
        lateral, max_lateral_m=25.0).reason == "lateral"

    back = _straight(20.0)
    back[5:, 0] -= 10.0
    assert validate_learned_path(
        back, max_backstep_m=1.5).reason == "backstep"

    # A right-angle turn over 1 m is far above the road-vehicle
    # curvature bound.
    tight = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert validate_learned_path(
        tight, max_extent_m=80.0, max_curvature=0.35).reason == "curvature"


def test_reason_is_stable_and_metrics_are_finite() -> None:
    out = validate_learned_path(
        np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.2]]))
    assert out.ok
    for value in (out.extent_m, out.lateral_m, out.backstep_m,
                  out.max_curvature):
        assert np.isfinite(value)
