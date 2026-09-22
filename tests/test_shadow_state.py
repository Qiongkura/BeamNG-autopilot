"""T06: shadow lateral state posterior - (e, e_dot, theta, theta_dot) + cov.

The plan asks for the SIMPLE rung of the ladder first: an identity-gated
one-model filter whose measurement and process models are written down,
published with its covariance and its failure reasons - not an IMM, and
not a second tracker competing with the existing lane modules.

These tests pin the properties that make it evidence rather than a
smoother:

* the sign conventions of the measurement (e, theta, kappa);
* the covariance shrinks with measurements and grows without them;
* a hold is NOT a measurement (T02/T03 rule, in the shadow too);
* an identity change resets the state instead of differentiating across
  two lanes, and the event is RECORDED, not smoothed away;
* a single-sided read propagates the width prior's uncertainty;
* the low-slip consistency check ``e_dot ~= v*sin(theta)`` reports the
  model error it sees;
* the filter's own LAG is measured, because a smoother curve is not a
  better estimate (the plan: "一个永远不动的错误参考也很平滑");
* nothing in the control path can read it.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from beamng_autopilot.lane.shadow_state import (
    IDENTITY_JUMP_M,
    LateralShadowEstimator,
    read_reference,
)

ROOT = Path(__file__).resolve().parents[1]


def _straight_ref(lat_m: float = 0.0, n: int = 41, length_m: float = 40.0):
    return np.column_stack([np.linspace(0.0, length_m, n),
                            np.full(n, float(lat_m))])


def _bend_ref(radius_m: float = 100.0, n: int = 61, length_m: float = 60.0):
    """A left-curving reference (positive curvature in this convention)."""
    s = np.linspace(0.0, length_m, n)
    ang = s / float(radius_m)
    return np.column_stack([radius_m * np.sin(ang), radius_m * (1 - np.cos(ang))])


class TestMeasurementConventions:
    def test_ego_left_of_the_reference_is_positive_e(self):
        read = read_reference(_straight_ref(0.0), np.array([3.0, 1.2, 0.0]),
                              0.0)
        assert read.ok and read.e_m == pytest.approx(1.2, abs=1e-9)

    def test_ego_right_of_the_reference_is_negative_e(self):
        read = read_reference(_straight_ref(0.0), np.array([3.0, -0.8, 0.0]),
                              0.0)
        assert read.e_m == pytest.approx(-0.8, abs=1e-9)

    def test_heading_left_of_the_reference_is_positive_theta(self):
        read = read_reference(_straight_ref(), np.array([3.0, 0.0, 0.0]), 0.1)
        assert read.theta_rad == pytest.approx(0.1, abs=1e-9)

    def test_a_left_bend_has_positive_curvature(self):
        read = read_reference(_bend_ref(100.0), np.array([3.0, 0.0, 0.0]), 0.0)
        assert read.ok
        assert read.kappa_1pm > 0.0
        assert read.kappa_1pm == pytest.approx(1.0 / 100.0, rel=0.35)

    def test_no_reference_is_reported_not_assumed(self):
        read = read_reference(None, np.zeros(3), 0.0)
        assert read.ok is False and read.reason
        read2 = read_reference(_straight_ref(), np.array([500.0, 0.0, 0.0]),
                               0.0)
        assert read2.ok is False and "far" in read2.reason


class TestFilterBehaviour:
    def _feed(self, est, n=6, dt=0.5, e=-0.4, theta=0.0, speed=2.0,
              t0=100.0, **kw):
        st = None
        for i in range(n):
            st = est.update(ref=_straight_ref(), pos=np.array([2.0 + i, e, 0.0]),
                            heading=theta, speed_mps=speed, now=t0 + i * dt,
                            **kw)
        return st

    def test_a_measurement_converges_and_shrinks_the_covariance(self):
        est = LateralShadowEstimator()
        st = self._feed(est, two_sided=True, inferred=False, fresh_obs=True)
        assert st.mode == "measured"
        assert st.e_m == pytest.approx(-0.4, abs=0.02)
        assert st.sigma_e_m is not None and st.sigma_e_m < 0.15
        P = np.asarray(st.cov)
        assert P.shape == (4, 4)
        np.testing.assert_allclose(P, P.T, atol=1e-12)   # symmetric
        assert np.all(np.linalg.eigvalsh(P) > -1e-12)    # PSD

    def test_a_hold_is_not_a_measurement(self):
        est = LateralShadowEstimator()
        self._feed(est, two_sided=True, inferred=False, fresh_obs=True)
        n = est.n_updates
        sig = est.P[0, 0]
        st = est.update(ref=_straight_ref(), pos=np.array([9.0, -0.4, 0.0]),
                        heading=0.0, speed_mps=2.0, now=104.0,
                        two_sided=True, inferred=False, fresh_obs=False)
        assert st.mode == "predicted"
        assert est.n_updates == n
        assert est.P[0, 0] > sig, "covariance must grow while held"
        assert st.age_since_update_s is not None

    def test_single_sided_reads_carry_the_width_uncertainty(self):
        a = LateralShadowEstimator()
        b = LateralShadowEstimator()
        sa = self._feed(a, n=1, two_sided=True, inferred=False,
                        fresh_obs=True)
        sb = self._feed(b, n=1, two_sided=False, inferred=True, fresh_obs=True)
        assert sb.sigma_e_m > sa.sigma_e_m
        # the inferred case adds the width prior's half-width uncertainty
        assert sb.sigma_e_m >= math.hypot(sa.sigma_e_m, 0.5 * 0.30) * 0.95

    def test_the_yaw_rate_source_is_labelled(self):
        est = LateralShadowEstimator()
        self._feed(est, two_sided=True, inferred=False, fresh_obs=True)
        # _feed's last update is at t0 + 5*dt = 102.5: stay inside the
        # "a gap is not a rate" window so the difference is meaningful
        st = est.update(ref=_straight_ref(), pos=np.array([9.0, -0.4, 0.0]),
                        heading=0.0, speed_mps=2.0, now=103.0,
                        two_sided=True, inferred=False, fresh_obs=True)
        assert st.yaw_rate_source == "finite_difference"
        st2 = est.update(ref=_straight_ref(), pos=np.array([9.5, -0.4, 0.0]),
                         heading=0.0, speed_mps=2.0, now=103.5,
                         two_sided=True, inferred=False, fresh_obs=True,
                         yaw_rate=0.05)
        assert st2.yaw_rate_source == "supplied"


class TestIdentityGating:
    def test_a_new_identity_resets_instead_of_differentiating(self):
        est = LateralShadowEstimator()
        for i in range(5):
            st = est.update(ref=_straight_ref(), pos=np.array([2.0 + i, -0.3, 0.0]),
                            heading=0.0, speed_mps=2.0, now=100.0 + i,
                            two_sided=True, inferred=False, fresh_obs=True,
                            lane_id="sensor|left")
        assert st.identity == "sensor|left"
        st = est.update(ref=_straight_ref(2.0), pos=np.array([7.0, -0.3, 0.0]),
                        heading=0.0, speed_mps=2.0, now=105.0,
                        two_sided=True, inferred=False, fresh_obs=True,
                        lane_id="sensor|right")
        assert st.identity == "sensor|right"
        assert st.identity_event == "identity"
        assert any(e["reason"] == "identity" for e in est.identity_events)

    def test_a_jump_in_the_same_identity_is_recorded_and_reset(self):
        est = LateralShadowEstimator()
        for i in range(5):
            st = est.update(ref=_straight_ref(), pos=np.array([2.0 + i, 0.0, 0.0]),
                            heading=0.0, speed_mps=2.0, now=100.0 + i,
                            two_sided=True, inferred=False, fresh_obs=True,
                            lane_id="sensor|left")
        assert st.e_m == pytest.approx(0.0, abs=0.05)
        big = IDENTITY_JUMP_M + 0.5
        st = est.update(ref=_straight_ref(big), pos=np.array([7.0, 0.0, 0.0]),
                        heading=0.0, speed_mps=2.0, now=105.0,
                        two_sided=True, inferred=False, fresh_obs=True,
                        lane_id="sensor|left")
        assert st.identity_event == "jump"
        assert any(e["reason"] == "jump" for e in est.identity_events)
        # the new lane is measured fresh: no derivative from the old one
        assert st.e_m == pytest.approx(-big, abs=0.1)
        assert st.e_dot_mps == pytest.approx(0.0, abs=1e-6)


class TestModelConsistency:
    def test_the_residual_reports_the_low_slip_model_error(self):
        """Ego crossing left at 0.5 m/s with theta = 0.3 rad:

        the model expects e_dot = v*sin(theta) = 0.296 m/s, the measurement
        says 0.5 m/s, so the residual must report the difference - a filter
        that hid it would look better and mean less.
        """
        est = LateralShadowEstimator()
        st = None
        for i, (x, y) in enumerate(zip([2.0, 2.5, 3.0, 3.5, 4.0],
                                       [-1.0, -0.75, -0.5, -0.25, 0.0])):
            st = est.update(ref=_straight_ref(0.0),
                            pos=np.array([x, y, 0.0]), heading=0.3,
                            speed_mps=1.0, now=200.0 + 0.5 * i,
                            two_sided=True, inferred=False, fresh_obs=True)
        assert st.consistency_residual_mps is not None
        assert st.consistency_residual_mps == pytest.approx(0.5 - math.sin(0.3),
                                                            abs=0.05)

    def test_a_consistent_track_has_a_small_residual(self):
        est = LateralShadowEstimator()
        # ego moving left at exactly v*sin(theta)
        v, theta = 2.0, 0.1
        rate = v * math.sin(theta)
        st = None
        for i in range(6):
            y = -1.0 + rate * (0.5 * i)
            st = est.update(ref=_straight_ref(0.0),
                            pos=np.array([2.0 + 1.0 * i, y, 0.0]),
                            heading=theta, speed_mps=v, now=300.0 + 0.5 * i,
                            two_sided=True, inferred=False, fresh_obs=True)
        assert abs(st.consistency_residual_mps) < 0.1


class TestLagIsMeasuredNotAssumed:
    """The plan's acceptance: compare response delay, not just smoothness."""

    def test_the_filter_lags_a_step_and_the_lag_is_reported(self):
        # the reference steps 1.0 m to the LEFT of the ego, so the
        # MEASUREMENT (ego relative to reference, + = ego left) steps to
        # -1.0 m; the filter must follow it, and its lag is the number the
        # plan wants measured rather than assumed
        truth = -1.0
        raw_lag = None
        filt_lag = None
        est = LateralShadowEstimator()
        for i in range(10):
            pos = np.array([2.0 + i, 0.0, 0.0])
            est.update(ref=_straight_ref(), pos=pos, heading=0.0,
                       speed_mps=2.0, now=400.0 + 0.5 * i,
                       two_sided=True, inferred=False, fresh_obs=True)
        # the reference steps 1.0 m to the left
        for i in range(10):
            t = 405.0 + 0.5 * i
            pos = np.array([12.0 + i, 0.0, 0.0])
            # reference 1.0 m to the LEFT of the ego -> measurement -1.0
            st = est.update(ref=_straight_ref(1.0), pos=pos, heading=0.0,
                            speed_mps=2.0, now=t,
                            two_sided=True, inferred=False, fresh_obs=True)
            if raw_lag is None:
                raw_lag = 0.0 if abs(truth - st.e_m) < 0.1 else None
            if raw_lag is not None and filt_lag is None \
                    and abs(st.e_m - truth) < 0.1:
                filt_lag = 0.5 * i
        # the filtered estimate needs at least one update to move, and it
        # must eventually get there (no frozen reference)
        assert filt_lag is not None and filt_lag <= 2.0
        assert abs(st.e_m - truth) < 0.15

    def test_a_frozen_estimate_is_not_a_good_estimate(self):
        """A filter that never moves scores perfectly on smoothness."""
        est = LateralShadowEstimator()
        for i in range(4):
            est.update(ref=_straight_ref(), pos=np.array([2.0 + i, 0.0, 0.0]),
                       heading=0.0, speed_mps=2.0, now=500.0 + i,
                       two_sided=True, inferred=False, fresh_obs=True)
        # the car genuinely drifts 0.8 m: the estimate must follow it
        for i in range(6):
            st = est.update(ref=_straight_ref(), pos=np.array([6.0 + i, -0.8, 0.0]),
                            heading=0.0, speed_mps=2.0, now=504.0 + i,
                            two_sided=True, inferred=False, fresh_obs=True)
        assert abs(st.e_m - (-0.8)) < 0.2, st.e_m


class TestShadowOnly:
    """The estimate must not be able to steer the car."""

    def test_no_control_module_imports_the_shadow_state(self):
        control = (ROOT / "beamng_autopilot" / "control")
        for path in control.rglob("*.py"):
            src = path.read_text(encoding="utf-8")
            assert "shadow_state" not in src, path

    def test_the_estimator_has_no_actuator_output(self):
        methods = [n for n, _ in inspect.getmembers(
            LateralShadowEstimator, inspect.isfunction)]
        assert not any("steer" in m or "command" in m or "act" == m
                       for m in methods), methods

    def test_the_drive_loop_only_records_it(self):
        src = (ROOT / "beamng_autopilot" / "fsd_drive.py").read_text("utf-8")
        # only the telemetry assignment and no arithmetic on it
        assert '"lane_shadow": out.meta.get("lane_shadow")' in src
        for line in src.splitlines():
            if "lane_shadow" in line and "out.meta.get" not in line:
                assert "shadow" not in line.lower() or "steer" not in line, line


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
