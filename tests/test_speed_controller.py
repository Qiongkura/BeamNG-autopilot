"""SpeedController hysteresis / deadband unit tests (no game needed)."""

from __future__ import annotations

from beamng_autopilot.control.speed import SpeedController


def _fresh(hyst: float = 0.0, **kw):
    c = SpeedController(hyst_mps=hyst, **kw)
    c.update(5.0, 5.0, dt=0.1)  # settle at target, coast
    return c


def _drive(c, target, speed, n=1, dt=0.1):
    """Run n updates; return the final (thr, brk) pair."""
    out = (0.0, 0.0)
    for _ in range(n):
        out = c.update(target, speed, dt=dt)
    return out


class TestDeadband:
    def test_coast_inside_band(self):
        c = _fresh()
        thr, brk = c.update(5.2, 5.0, dt=0.1)  # err=0.2 < deadband
        assert thr == 0.0 and brk == 0.0

    def test_throttle_above_band(self):
        c = _fresh()
        thr, brk = _drive(c, 6.0, 5.0, n=3)  # err=+1.0
        assert thr > 0.0 and brk == 0.0

    def test_brake_below_band(self):
        c = _fresh()
        thr, brk = _drive(c, 4.0, 5.0, n=3)  # err=-1.0
        assert brk > 0.0 and thr == 0.0

    def test_never_ride_both(self):
        c = _fresh()
        for target in (3.0, 5.0, 7.0, 4.0, 6.0):
            thr, brk = _drive(c, target, 5.0, n=4)
            assert not (thr > 0.05 and brk > 0.05)

    def test_pedal_ramp_increment_is_bounded(self):
        c = SpeedController()
        thr1, _ = _drive(c, 20.0, 5.0)
        thr2, _ = _drive(c, 20.0, 5.0)
        assert 0.0 < thr2 - thr1 <= 0.1 * c.thr_up + 1e-9


class TestHysteresis:
    def test_brake_does_not_flip_to_throttle_inside_hyst(self):
        # h=0.4, deadband=0.35 -> throttle needs err > 0.75
        c = _fresh(hyst=0.4)
        _drive(c, 3.5, 5.0, n=2)          # err=-1.5 -> brake mode
        thr, brk = _drive(c, 5.4, 5.0, n=2)  # err=+0.4 still < 0.75
        assert thr == 0.0

    def test_brake_flips_to_throttle_after_hyst(self):
        c = _fresh(hyst=0.4)
        _drive(c, 3.5, 5.0, n=2)          # brake mode
        thr, brk = _drive(c, 5.9, 5.0, n=4)  # err=+0.9 > 0.75
        assert thr > 0.0 and brk == 0.0
        assert c._mode == 1

    def test_throttle_does_not_flip_to_brake_inside_hyst(self):
        c = _fresh(hyst=0.4)
        _drive(c, 6.5, 5.0, n=2)          # err=+1.5 -> throttle mode
        thr, brk = _drive(c, 4.6, 5.0, n=2)  # err=-0.4 still > -0.75
        assert brk == 0.0

    def test_throttle_flips_to_brake_after_hyst(self):
        c = _fresh(hyst=0.4)
        _drive(c, 6.5, 5.0, n=2)          # throttle mode
        thr, brk = _drive(c, 4.1, 5.0, n=4)  # err=-0.9 < -0.75
        assert brk > 0.0 and thr == 0.0
        assert c._mode == -1

    def test_default_hyst_flips_immediately(self):
        # h=0: err=+0.4 (past deadband) flips straight to throttle
        c = _fresh(hyst=0.0)
        _drive(c, 3.5, 5.0, n=2)          # brake mode
        thr, brk = _drive(c, 5.4, 5.0, n=4)
        assert thr > 0.0 and brk == 0.0
        assert c._mode == 1
