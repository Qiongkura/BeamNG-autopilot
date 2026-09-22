"""Command receipts and deadline enforcement on the existing control channel.

Python enforces the overdue stop when execution resumes; the game-side
watchdog remains responsible while Python is blocked.
"""

from __future__ import annotations

import pytest

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

    @pytest.mark.parametrize("interval", [float("inf"), float("nan"), -1.0])
    def test_invalid_interval_is_unknown(self, interval):
        assert substep_digest(15.0, interval, 3)["shortfall"] is None

    def test_cumulative_count_cannot_hide_current_frame_shortfall(self):
        before, after = 100, 102
        d = substep_digest(15.0, 0.8, after - before)
        assert d["executed"] == 2
        assert d["expected"] == 12
        assert d["shortfall"] == 10


@pytest.mark.parametrize("now,last,bound", [
    (float("nan"), 1.0, 1.5), (2.0, float("nan"), 1.5),
    (2.0, 1.0, float("nan")), (2.0, 1.0, float("inf")),
    (2.0, 1.0, 0.0), (2.0, 1.0, "bad"),
])
def test_invalid_clock_or_threshold_cannot_be_ok(now, last, bound):
    assert watchdog_verdict(now, last, bound)["action"] == "unknown"


class _Clock:
    def __init__(self, value=10.0):
        self.value = value

    def __call__(self):
        return self.value


class _Connection:
    def __init__(self):
        self.controls = []
        self.fail = False

    def control(self, **controls):
        if self.fail:
            raise RuntimeError("send failed")
        self.controls.append(controls)


def _stream(enabled=True):
    conn, clock, wall = _Connection(), _Clock(), _Clock(100.0)
    stream = fd._CommandStream(conn, enabled=enabled, max_gap_s=1.5,
                               clock=clock, wall_clock=wall)
    return stream, conn, clock, wall


def _drive(stream):
    return stream.send(throttle=0.7, brake=0.0, steering=0.1,
                       gear=2, parkingbrake=0.0)


def test_slow_tick_brakes_on_main_send_without_any_substep():
    stream, conn, clock, _ = _stream()
    _drive(stream)
    clock.value += 2.0
    sent = _drive(stream)
    assert sent == conn.controls[-1]
    assert sent["throttle"] == 0.0
    assert sent["brake"] == 1.0
    assert sent["steering"] == 0.0
    assert sent["parkingbrake"] == 1.0
    assert stream.braked
    assert stream.verdict["gap_s"] == 2.0
    assert stream.seq == 2


def test_substep_send_refreshes_the_shared_deadline():
    stream, _, clock, _ = _stream()
    _drive(stream)
    clock.value += 1.0
    _drive(stream)
    clock.value += 1.0
    _drive(stream)
    assert stream.seq == 3
    assert not stream.braked
    assert stream.verdict["gap_s"] == 1.0


def test_wall_clock_jump_does_not_change_command_deadline():
    stream, _, clock, wall = _stream()
    _drive(stream)
    clock.value += 0.1
    wall.value -= 50.0
    _drive(stream)
    assert stream.verdict["action"] == "ok"
    assert stream.verdict["gap_s"] == 0.1
    assert stream.sent_t == 50.0


def test_disabled_watchdog_reports_lateness_without_overriding():
    stream, _, clock, _ = _stream(enabled=False)
    _drive(stream)
    clock.value += 2.0
    sent = _drive(stream)
    assert stream.verdict["due"]
    assert not stream.braked
    assert sent["throttle"] == 0.7


def test_failed_send_does_not_refresh_the_receipt():
    stream, conn, clock, _ = _stream()
    _drive(stream)
    prior = stream.monotonic_t
    clock.value += 2.0
    conn.fail = True
    with pytest.raises(RuntimeError, match="send failed"):
        _drive(stream)
    assert stream.seq == 1
    assert stream.monotonic_t == prior
    conn.fail = False
    assert _drive(stream)["brake"] == 1.0


def test_substep_receipt_contains_actual_controls_and_source_age():
    stream, _, _, wall = _stream()
    wall.value = 101.25
    _drive(stream)
    sched = {"semantic": {"source_t": 100.0, "publish_t": 101.0,
                           "source_seq": 5, "result_seq": 7}}
    receipt = stream.receipt(sched, 3.3)
    assert receipt["cmd_seq"] == 1
    assert receipt["cmd_t"] == 101.25
    assert receipt["throttle"] == 0.7
    assert receipt["target_speed"] == 3.3
    assert receipt["consumed"]["semantic"]["age_s"] == 1.25
    assert receipt["consumed"]["semantic"]["publish_age_s"] == 0.25


def test_braked_receipt_never_claims_the_requested_cruise_target():
    stream, _, clock, _ = _stream()
    _drive(stream)
    clock.value += 2.0
    _drive(stream)
    assert stream.receipt({}, 6.0)["target_speed"] == 0.0


def test_send_latency_is_part_of_receipt_gap_not_the_pre_send_check():
    stream, conn, clock, _ = _stream()
    _drive(stream)
    original_control = conn.control

    def slow_send(**controls):
        clock.value += 0.4
        original_control(**controls)

    conn.control = slow_send
    clock.value += 0.2
    _drive(stream)
    receipt = stream.receipt({}, 3.0)
    assert receipt["watchdog_gap_s"] == pytest.approx(0.2)
    assert receipt["cmd_gap_s"] == pytest.approx(0.6)
    assert receipt["cmd_send_ms"] == pytest.approx(400.0)
