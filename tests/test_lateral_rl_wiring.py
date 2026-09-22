"""The learned lateral residual is bounded, optional and monitor-gated."""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_autopilot import fsd_drive as fd


def _path(y: float = 0.0, n: int = 31, step: float = 1.0):
    xs = np.arange(n, dtype=float) * step
    return np.column_stack([xs, np.full(n, y)])


def test_the_shift_is_full_at_the_ego_and_fades_by_the_horizon() -> None:
    path = _path()
    out = fd.shift_path_right(path, np.zeros(3), 0.0, 0.4,
                              horizon_m=12.0)
    # heading 0 -> "right" is -y
    assert out[0, 1] == pytest.approx(-0.4)
    assert abs(out[6, 1]) < abs(out[0, 1])
    assert out[-1, 1] == pytest.approx(0.0)          # beyond the horizon
    # the far geometry is untouched
    assert np.allclose(path[-1], out[-1])


def test_a_zero_shift_is_a_no_op() -> None:
    path = _path()
    assert fd.shift_path_right(path, np.zeros(3), 0.0, 0.0) is path


def test_a_shift_rotates_with_the_heading() -> None:
    """Heading east, "right" is -y; heading north, "right" is +x."""
    path = np.column_stack([np.zeros(11), np.arange(11, dtype=float)])
    out = fd.shift_path_right(path, np.zeros(3), np.pi / 2, 0.3)
    assert out[0, 0] == pytest.approx(0.3)
    assert out[0, 1] == pytest.approx(0.0, abs=1e-9)


class TestSteerAwayFromRoad:
    """The 2026-09-21 crash: full right lock while creeping rotated the
    nose 23 deg off the road and the car rolled into a roadside tree."""

    def test_a_command_that_turns_further_away_is_clamped(self):
        steer = fd.clamp_steer_away_from_road(
            0.55, math.radians(-36.3), -13.0)      # nose already right
        assert steer == pytest.approx(fd.STEER_AWAY_CLAMP)

    def test_the_mirror_case_is_clamped_too(self):
        steer = fd.clamp_steer_away_from_road(
            -0.55, math.radians(20.0), -13.0)      # nose already left
        assert steer == pytest.approx(-fd.STEER_AWAY_CLAMP)

    def test_turning_back_toward_the_road_is_never_clamped(self):
        assert fd.clamp_steer_away_from_road(
            -0.55, math.radians(-36.3), -13.0) == pytest.approx(-0.55)
        assert fd.clamp_steer_away_from_road(
            0.55, math.radians(20.0), -13.0) == pytest.approx(0.55)

    def test_inside_the_band_nothing_changes(self):
        assert fd.clamp_steer_away_from_road(
            0.55, math.radians(-5.0), -13.0) == pytest.approx(0.55)

    def test_a_missing_bearing_disables_the_check_instead_of_guessing(self):
        assert fd.clamp_steer_away_from_road(0.55, 0.0, None) == 0.55
        assert fd.clamp_steer_away_from_road(0.55, 0.0, "bad") == 0.55

    def test_the_wrap_around_boundary_is_handled(self):
        # nose +170 deg, road -170 deg -> 340 deg -> wraps to -20 deg, i.e.
        # the nose is 20 deg to the RIGHT: steering right is the away
        # command, steering left turns back toward the road.
        assert fd.clamp_steer_away_from_road(
            0.4, math.radians(170.0), -170.0) == pytest.approx(
            fd.STEER_AWAY_CLAMP)
        assert fd.clamp_steer_away_from_road(
            -0.4, math.radians(170.0), -170.0) == pytest.approx(-0.4)


def test_the_switch_is_off_by_default() -> None:
    assert fd.LATERAL_RL_ENABLED is False


def test_the_residual_is_bounded_by_the_runtime_contract() -> None:
    """Whatever the policy picks, the applied offset stays within +-0.5 m."""
    from beamng_autopilot.rl.lateral_runtime import LateralRLRuntime
    from beamng_autopilot.rl.lateral_env import ACTION_OFFSETS

    rt = LateralRLRuntime()
    if not rt.available:
        pytest.skip(f"lateral checkpoint unavailable: {rt.error}")
    worst = 0.0
    for i in range(60):
        worst = max(worst, abs(rt.act(lat_err_m=0.5, heading_err_rad=0.1,
                                      curvature=0.0, speed_mps=8.0)))
    assert worst <= 0.5 + 1e-9
    assert max(abs(o) for o in ACTION_OFFSETS) == pytest.approx(0.5)
