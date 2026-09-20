"""The control heartbeat must not depend on the loop that stalls (P3).

A watchdog driven by tick completion goes quiet exactly when it is
needed, because the perception loop is what blocks.  These measure the
gap between commands instead, and cover the case where the gap cannot be
measured at all.
"""

from beamng_autopilot import fsd_drive as fd
from beamng_autopilot.fsd_drive import (
    CTRL_WATCHDOG_MAX_GAP_S,
    substep_digest,
    watchdog_verdict,
)


class TestDefaultOff:
    def test_the_watchdog_does_not_brake_by_default(self):
        assert fd.CTRL_WATCHDOG_ENABLED is False

    def test_the_threshold_is_above_the_measured_worst_gap(self):
        # Measured command gap on town_1789890111: p50 0.666 s, max 0.801 s.
        # A threshold under that would fire on every slow tick.
        assert CTRL_WATCHDOG_MAX_GAP_S > 0.801


class TestWatchdogVerdict:
    def test_a_short_gap_is_ok(self):
        v = watchdog_verdict(10.0, 9.5, 1.5)
        assert v["action"] == "ok"
        assert v["due"] is False
        assert v["gap_s"] == 0.5

    def test_a_long_gap_is_due(self):
        v = watchdog_verdict(10.0, 8.0, 1.5)
        assert v["action"] == "brake"
        assert v["due"] is True
        assert "no command for" in v["reason"]

    def test_the_boundary_is_not_due(self):
        # Strictly greater: exactly at the threshold is still on time.
        assert watchdog_verdict(10.0, 8.5, 1.5)["due"] is False

    def test_cold_start_is_unknown_not_ok(self):
        """Nothing has been sent yet.  Calling that "on time" would be the
        default-value-as-healthy error again."""
        v = watchdog_verdict(10.0, None, 1.5)
        assert v["action"] == "unknown"
        assert v["due"] is False
        assert v["gap_s"] is None

    def test_a_missing_now_is_unknown(self):
        assert watchdog_verdict(None, 1.0, 1.5)["action"] == "unknown"

    def test_time_going_backwards_is_unknown_not_negative_ok(self):
        v = watchdog_verdict(5.0, 9.0, 1.5)
        assert v["action"] == "unknown"
        assert v["due"] is False

    def test_unreadable_timestamps_are_unknown(self):
        v = watchdog_verdict("x", 1.0, 1.5)
        assert v["action"] == "unknown"

    def test_it_reports_a_verdict_not_a_pedal(self):
        """P3 forbids a second control channel: the caller brakes."""
        v = watchdog_verdict(10.0, 8.0, 1.5)
        assert set(v) == {"action", "gap_s", "due", "reason"}


class TestSubstepDigest:
    def test_nominal_versus_actual_are_separate_numbers(self):
        # 15 Hz requested over a 0.666 s frame is 10 sub-steps; 2 ran.
        d = substep_digest(15.0, 0.666, 2)
        assert d["expected"] == 10
        assert d["executed"] == 2
        assert d["shortfall"] == 8

    def test_a_full_frame_has_no_shortfall(self):
        assert substep_digest(15.0, 0.666, 10)["shortfall"] == 0

    def test_more_than_expected_is_not_a_negative_shortfall(self):
        assert substep_digest(15.0, 0.1, 9)["shortfall"] == 0

    def test_an_unmeasured_interval_does_not_claim_zero_expected(self):
        d = substep_digest(15.0, None, 3)
        assert d["expected"] is None
        assert d["shortfall"] is None

    def test_a_skip_reason_is_carried(self):
        # "Nothing to send" and "did not get round to it" are different.
        d = substep_digest(15.0, 0.666, 0, "stale plan")
        assert d["skip_reason"] == "stale plan"

    def test_no_reason_is_none_not_empty_string(self):
        assert substep_digest(15.0, 0.666, 3)["skip_reason"] is None
