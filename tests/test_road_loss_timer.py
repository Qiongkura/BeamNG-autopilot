"""The road-loss clock must survive a flickering reader (plan P4).

The band is a PERCEIVED band read 2-12 m AHEAD, so it flickers.  With the
old rule - one ON frame clears the clock - the clock can never reach the
degrade or stop threshold no matter how bad coverage is, and a sustained
loss looks identical to a flickering one.  These are the injected
sequences P4 asks for: sustained unknown, confirmed off_road, intermittent
recovery, and a gap so long a tick was clearly missed.
"""

from beamng_autopilot.safety_monitor import (
    ROAD_SURFACE_OFF,
    ROAD_SURFACE_ON,
    ROAD_SURFACE_UNKNOWN,
    SafetyMonitor,
    blind_drive_distance_m,
    road_loss_timer,
)


class TestTimer:
    def test_a_sustained_unknown_accumulates(self):
        since, _, lost = road_loss_timer(ROAD_SURFACE_UNKNOWN, 10.0, None)
        assert since == 10.0 and lost == 0.0
        since, _, lost = road_loss_timer(ROAD_SURFACE_UNKNOWN, 14.0, since)
        assert lost == 4.0

    def test_confirmed_off_road_accumulates(self):
        since, _, lost = road_loss_timer(ROAD_SURFACE_OFF, 10.0, None)
        since, _, lost = road_loss_timer(ROAD_SURFACE_OFF, 12.5, since)
        assert lost == 2.5

    def test_the_default_clears_on_one_frame(self):
        """confirm_s = 0 reproduces today's behaviour exactly, so turning
        the hysteresis on is a separate decision, not a side effect."""
        since, _, _ = road_loss_timer(ROAD_SURFACE_UNKNOWN, 10.0, None)
        since, _, lost = road_loss_timer(ROAD_SURFACE_ON, 10.1, since)
        assert since is None and lost == 0.0

    def test_with_hysteresis_a_single_on_frame_does_not_clear_it(self):
        since, _, _ = road_loss_timer(ROAD_SURFACE_UNKNOWN, 10.0, None)
        since, on_s, lost = road_loss_timer(
            ROAD_SURFACE_ON, 13.0, since, confirm_s=2.0)
        assert since == 10.0          # clock still running
        assert lost == 3.0
        assert on_s == 13.0           # the recovery window started

    def test_a_held_on_clears_it(self):
        since, on_s, _ = road_loss_timer(
            ROAD_SURFACE_ON, 13.0, 10.0, confirm_s=2.0)
        since, _, lost = road_loss_timer(
            ROAD_SURFACE_ON, 15.0, since, on_s, confirm_s=2.0)
        assert since is None and lost == 0.0

    def test_an_intermittent_reader_never_reaches_the_threshold(self):
        """The failure: ON every third second, so with no hysteresis the
        clock resets forever and the 8 s stop can never fire."""
        since, on_s, lost = None, None, 0.0
        worst_no_hyst = 0.0
        for i, t in enumerate(range(0, 30)):
            state = ROAD_SURFACE_ON if t % 3 == 0 else ROAD_SURFACE_UNKNOWN
            since, on_s, lost = road_loss_timer(
                state, float(t), since, on_s, confirm_s=0.0)
            worst_no_hyst = max(worst_no_hyst, lost)
        assert worst_no_hyst < 3.0      # never gets near 8 s

        since, on_s, worst_hyst = None, None, 0.0
        for t in range(0, 30):
            state = ROAD_SURFACE_ON if t % 3 == 0 else ROAD_SURFACE_UNKNOWN
            since, on_s, lost = road_loss_timer(
                state, float(t), since, on_s, confirm_s=2.0)
            worst_hyst = max(worst_hyst, lost)
        assert worst_hyst > 8.0         # and with hysteresis it does

    def test_a_missed_tick_does_not_look_like_a_short_gap(self):
        # A long gap between readings is elapsed time, and the stop rule
        # must see it as elapsed time.
        since, _, _ = road_loss_timer(ROAD_SURFACE_UNKNOWN, 10.0, None)
        _, _, lost = road_loss_timer(ROAD_SURFACE_UNKNOWN, 20.0, since)
        assert lost == 10.0

    def test_the_clock_does_not_go_negative(self):
        since, _, lost = road_loss_timer(ROAD_SURFACE_UNKNOWN, 10.0, None)
        _, _, lost = road_loss_timer(ROAD_SURFACE_UNKNOWN, 9.0, since)
        assert lost == 0.0


class TestGateIntegration:
    def _mon(self, confirm_s=0.0):
        mon = SafetyMonitor()
        mon.road_recover_confirm_s = confirm_s
        return mon

    class _Scene:
        def __init__(self, grid):
            self.grid = grid

    def test_a_gridless_scene_does_not_start_or_clear_the_clock(self):
        mon = self._mon()
        sc = self._Scene(None)
        for t in (10.0, 11.0, 12.0):
            _, lost, checked = mon._road_surface_gate(sc, t)
            assert lost == 0.0 and checked is False

    def test_an_intermittent_grid_does_not_restart_the_clock(self):
        """Grid present, then absent, then present: the absent frames say
        nothing, so they must not silently reset a running loss."""
        mon = self._mon()
        _, lost, _ = mon._road_surface_gate(self._Scene(object()), 10.0)
        assert lost == 0.0
        _, lost, _ = mon._road_surface_gate(self._Scene(object()), 14.0)
        assert lost == 4.0
        # grid disappears for one frame
        _, lost2, checked = mon._road_surface_gate(self._Scene(None), 15.0)
        assert checked is False
        # ... and the loss clock is where it was left, not restarted
        assert mon._road_lost_since == 10.0


class TestThresholds:
    def test_the_stop_threshold_is_a_distance_not_a_duration(self):
        """P4: 8 s at 5 m/s is 40 m of driving with no road evidence.
        Nobody validated 40 m as an acceptable blind-driving distance."""
        assert blind_drive_distance_m(5.0, 8.0) == 40.0

    def test_the_degrade_threshold_distance(self):
        assert blind_drive_distance_m(5.0, 4.0) == 20.0

    def test_standing_still_makes_the_duration_harmless(self):
        assert blind_drive_distance_m(0.0, 8.0) == 0.0

    def test_the_distance_scales_with_speed(self):
        assert blind_drive_distance_m(10.0, 8.0) == 80.0
