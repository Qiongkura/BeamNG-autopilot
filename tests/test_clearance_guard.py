"""Forward-clearance temporal guard regression (pure logic, no game).

The guard exists because the raw clearance reading is *instantaneous*:
town run 2026-09-20 (`town_1789886413.json`) accelerated through a 1.20 m
reading and hit at -0.25 m, and 31 recorded single-frame steps take the
reading from below 2 m to above 8 m.  These tests pin the properties that
make the reserve trustworthy: a low reading is never cleared by one
anomalous frame, "not measured" never reads as clear, and the guard can
only ever lower the value.
"""
from __future__ import annotations

import math

from beamng_autopilot.planning.clearance_guard import (
    CLEARANCE_CONFIRM_N,
    CLEARANCE_HOLD_S,
    CLEARANCE_JUMP_M,
    CLEARANCE_MAX_AGE_S,
    REASON_DECREASE,
    REASON_FIRST,
    REASON_INCREASE,
    REASON_JUMP_HELD,
    REASON_STALE,
    REASON_UNMEASURED,
    REASON_WINDOW,
    UNMEASURED_CLEARANCE_M,
    ClearanceGuard,
)

DT = 0.7  # the FSD control loop's wall-time tick in the town runs


def _feed(guard, values, *, dt=DT, start=0.0, **kw):
    """Feed one reading per tick; returns [(t, ClearanceReading), ...]."""
    out = []
    t = float(start)
    for v in values:
        out.append((t, guard.update(v, t, **kw)))
        t += dt
    return out


class TestBasicSemantics:
    def test_first_reading_is_trusted(self):
        g = ClearanceGuard()
        r = g.update(9.0, 0.0)
        assert r.valid is True
        assert r.value == 9.0
        assert r.raw == 9.0
        assert r.held is False
        assert r.reason == REASON_FIRST

    def test_decrease_is_trusted_immediately(self):
        g = ClearanceGuard()
        _feed(g, [20.0, 12.0])
        r = g.update(1.2, 2.0)
        assert r.reason == REASON_DECREASE
        assert r.value == 1.2
        assert r.held is False

    def test_negative_reading_is_preserved_not_clamped(self):
        # A negative value means the car already overlaps a hit: it is
        # evidence of a real intrusion and must survive verbatim.
        g = ClearanceGuard()
        r = g.update(-0.44, 0.0)
        assert r.value == -0.44
        assert r.raw == -0.44

    def test_measured_clear_without_history_stays_clear(self):
        g = ClearanceGuard()
        r = g.update(float("inf"), 0.0)
        assert math.isinf(r.value)
        assert r.valid is True
        assert r.held is False

    def test_value_never_exceeds_raw(self):
        g = ClearanceGuard()
        seq = [12.0, 3.0, 1.2, 19.01, 0.4, 16.9, 8.0, 0.2]
        for _, r in _feed(g, seq):
            assert r.value <= r.raw

    def test_reset_clears_state(self):
        g = ClearanceGuard()
        _feed(g, [1.2, 19.01])
        g.reset()
        r = g.update(19.01, 5.0)
        assert r.reason == REASON_FIRST
        assert r.value == 19.01
        assert r.n_jump == 0


class TestJumpCannotClearRisk:
    def test_single_upward_jump_is_held(self):
        g = ClearanceGuard()
        _feed(g, [1.2])
        r = g.update(19.01, DT)
        assert r.jumped is True
        assert r.held is True
        assert r.reason == REASON_JUMP_HELD
        assert r.value <= 1.2

    def test_jump_threshold_is_inclusive_below(self):
        g = ClearanceGuard()
        _feed(g, [1.0])
        r = g.update(1.0 + CLEARANCE_JUMP_M, DT)
        assert r.jumped is False  # exactly at the threshold is not a jump

    def test_jump_just_above_threshold_flags(self):
        g = ClearanceGuard()
        _feed(g, [1.0])
        r = g.update(1.0 + CLEARANCE_JUMP_M + 0.01, DT)
        assert r.jumped is True

    def test_latch_releases_only_after_confirm_frames(self):
        g = ClearanceGuard()
        _feed(g, [1.2])
        r = g.update(19.01, DT)
        assert r.jumped is True
        t = DT
        # confirm_n corroborating frames are required before the reserve
        # lets go of the low reading
        for _ in range(CLEARANCE_CONFIRM_N - 1):
            t += DT
            r = g.update(19.01, t)
            assert r.value <= 1.2, "released too early"
        t += DT
        r = g.update(19.01, t)
        assert r.value == 19.01, "never released after corroboration"

    def test_latch_drops_early_when_reading_comes_back_down(self):
        g = ClearanceGuard()
        _feed(g, [1.2])
        g.update(19.01, DT)          # jump -> latched at 1.2
        r = g.update(0.8, 2 * DT)    # reading itself is now low again
        assert r.value == 0.8
        assert r.held is False

    def test_reported_anomaly_sequence_never_reads_clear(self):
        # The exact sequence from the collision report: the risk must not
        # be cleared by either of the two upward spikes.
        g = ClearanceGuard()
        rows = _feed(g, [1.18, -0.44, 19.01, -0.22, 16.9])
        for _, r in rows:
            assert r.value <= 1.18
        assert rows[2][1].reason == REASON_JUMP_HELD
        assert rows[4][1].reason == REASON_JUMP_HELD

    def test_collision_run_ramp_is_not_outrun(self):
        # town_1789886413: 3.44 -> 1.20 -> -0.25 while the car was still
        # accelerating.  The guard keeps the minimum from the moment the
        # reserve was entered.
        g = ClearanceGuard()
        rows = _feed(g, [8.42, 5.56, 4.31, 3.44, 1.20, -0.25])
        assert rows[-1][1].value == -0.25
        assert rows[-2][1].value == 1.20


class TestUnmeasuredIsNotClear:
    def test_no_reading_with_no_history_fails_closed(self):
        g = ClearanceGuard()
        r = g.update(float("inf"), 0.0, valid=False)
        assert r.valid is False
        assert r.value == UNMEASURED_CLEARANCE_M
        assert r.reason == REASON_UNMEASURED
        assert r.held is True

    def test_no_reading_keeps_last_known_minimum(self):
        g = ClearanceGuard()
        _feed(g, [4.0, 1.5])
        r = g.update(float("inf"), DT, valid=False)
        assert r.valid is False
        assert r.value == 1.5
        assert r.reason == REASON_UNMEASURED

    def test_unmeasured_history_holds_last_known_minimum(self):
        # "Not measured" is not evidence of "clear": the last known
        # minimum keeps the braking authority for as long as the silence
        # lasts.  Deciding whether stale sensors may drive at all is the
        # safety monitor's job, not this guard's.
        g = ClearanceGuard()
        _feed(g, [4.0, 1.5])
        t = DT
        seen = []
        for _ in range(8):
            r = g.update(float("inf"), t, valid=False)
            seen.append(r.value)
            t += DT
        assert seen[0] == 1.5
        assert seen[-1] == 1.5
        assert all(r == 1.5 for r in seen)

    def test_frozen_window_is_replaced_by_fresh_evidence(self):
        g = ClearanceGuard()
        _feed(g, [1.5])
        t = DT
        for _ in range(4):
            g.update(float("inf"), t, valid=False)
            t += DT
        # a fresh reading long after the silence is what counts now
        r = g.update(9.0, t + 5.0)
        assert r.valid is True
        assert r.jumped is True          # ... but it still has to earn it
        assert r.value <= 1.5

    def test_stale_reading_is_not_fresh_evidence(self):
        g = ClearanceGuard()
        _feed(g, [2.0])
        r = g.update(30.0, DT, age_s=CLEARANCE_MAX_AGE_S + 0.5)
        assert r.valid is False
        assert r.reason == REASON_STALE
        assert r.value <= 2.0

    def test_stale_cannot_clear_a_frozen_low(self):
        g = ClearanceGuard()
        _feed(g, [0.9])
        t = DT
        for _ in range(5):
            r = g.update(float("inf"), t,
                         age_s=CLEARANCE_MAX_AGE_S + 10.0)
            assert r.value <= 0.9
            t += DT

    def test_age_at_threshold_is_still_usable(self):
        g = ClearanceGuard()
        r = g.update(30.0, 0.0, age_s=CLEARANCE_MAX_AGE_S)
        assert r.valid is True
        assert r.reason == REASON_FIRST

    def test_age_is_reported(self):
        g = ClearanceGuard()
        r = g.update(5.0, 0.0, age_s=0.31)
        assert r.age_s == 0.31


class TestWindowMinimum:
    def test_small_rise_is_bounded_by_window_minimum(self):
        g = ClearanceGuard()
        _feed(g, [1.2])
        r = g.update(2.0, DT)
        assert r.jumped is False
        assert r.reason == REASON_WINDOW
        assert r.value == 1.2
        assert r.min_recent == 1.2

    def test_rising_reading_lags_one_tick(self):
        # With a 0.7 s tick and a 1.2 s window the previous sample is
        # always still inside it, so a rising reading is bounded by the
        # reading before it: the reserve releases one tick late, never
        # instantly.
        g = ClearanceGuard()
        rows = _feed(g, [1.2, 2.0, 3.0, 4.0])
        assert [r.value for _, r in rows] == [1.2, 1.2, 2.0, 3.0]
        assert all(r.reason == REASON_WINDOW for _, r in rows[1:])

    def test_window_minimum_expires_after_a_long_gap(self):
        # A tick gap longer than the window drops the old sample, so a
        # small rise is then released without waiting for corroboration.
        g = ClearanceGuard()
        g.update(1.2, 0.0)
        r = g.update(4.0, CLEARANCE_HOLD_S + 1.8)
        assert r.jumped is False
        assert r.held is False
        assert r.value == 4.0
        assert r.reason == REASON_INCREASE

    def test_measured_clear_cannot_beat_a_windowed_low(self):
        g = ClearanceGuard()
        _feed(g, [1.2])
        r = g.update(float("inf"), DT)
        assert math.isinf(r.raw)
        assert r.value <= 1.2

    def test_steady_readings_pass_through(self):
        g = ClearanceGuard()
        rows = _feed(g, [7.0] * 4)
        for _, r in rows:
            assert r.value == 7.0
        assert rows[-1][1].reason == REASON_INCREASE


class TestDigest:
    def test_digest_is_json_safe_for_infinity(self):
        import json

        g = ClearanceGuard()
        r = g.update(float("inf"), 0.0)
        d = r.digest()
        assert d["raw"] == "inf"
        assert d["value"] == "inf"
        json.dumps(d)  # must not raise

    def test_digest_reports_the_decision(self):
        g = ClearanceGuard()
        _feed(g, [1.2])
        d = g.update(19.01, DT).digest()
        assert d["jumped"] is True
        assert d["held"] is True
        assert d["reason"] == REASON_JUMP_HELD
        assert d["min_recent"] == 1.2
        assert d["n_jump"] == 1
        assert d["valid"] is True

    def test_counters_accumulate(self):
        g = ClearanceGuard()
        _feed(g, [1.2, 19.01, 19.01, 1.0, 19.01])
        r = g.update(1.2, 5 * DT)
        assert r.n_jump == 2
        assert r.n_hold >= 2
