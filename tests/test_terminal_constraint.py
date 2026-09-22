"""T04: the shaper may not exceed the authority it was given.

The plan (FINAL_INTEGRATED_PLAN §2.3/T04) measured the failure as a
component counterexample: previous command 0.55, this tick's limit 0.15,
``dt = 0.1`` -> the shaper emitted 0.49, i.e. a reference the stabiliser
had just limited to "small corrections only" kept a full correction on the
wire.  These tests pin the fix in the shaper and, through the drive-loop
harness, on the command that is actually SENT.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.control.steering import SteeringShaper


class TestAuthorityCap:
    def test_the_measured_counterexample_is_closed(self):
        sh = SteeringShaper()
        for _ in range(12):
            sh.update(0.55, dt=0.1)
        assert sh.value == pytest.approx(0.55, abs=1e-3)
        out = sh.update(0.15, dt=0.1, cap=0.15)
        assert abs(out) <= 0.15 + 1e-9

    def test_the_state_is_re_based_so_it_cannot_keep_pushing_outward(self):
        sh = SteeringShaper()
        for _ in range(12):
            sh.update(0.55, dt=0.1)
        sh.update(0.15, dt=0.1, cap=0.15)
        assert abs(sh.value) <= 0.15 + 1e-9
        # the next step must not resume ramping toward the old 0.55
        out = sh.update(0.15, dt=0.1, cap=0.15)
        assert abs(out) <= 0.15 + 1e-9
        assert abs(sh.rate) <= sh.max_rate_per_s + 1e-9

    def test_a_cap_only_bounds_the_magnitude_it_does_not_flip_the_sign(self):
        sh = SteeringShaper()
        out = sh.update(-0.4, dt=0.1, cap=0.2)
        assert -0.2 - 1e-9 <= out <= 0.0

    def test_the_request_is_clipped_before_shaping(self):
        """A 0.55 request under a 0.15 cap must not be *approached* slowly
        from a 0.55 start only to be clipped at the end - the shape target
        is inside the cap from the first step."""
        sh = SteeringShaper()
        for _ in range(12):
            sh.update(0.55, dt=0.1)
        out = sh.update(0.55, dt=0.1, cap=0.15)
        assert abs(out) <= 0.15 + 1e-9

    def test_no_cap_keeps_the_previous_behaviour(self):
        """The positive control: without a cap the shaper still ramps."""
        sh = SteeringShaper()
        for _ in range(12):
            sh.update(0.55, dt=0.1)
        out = sh.update(0.15, dt=0.1)
        assert out > 0.2          # rate/jerk limited, NOT teleported

    def test_a_wide_cap_changes_nothing(self):
        a, b = SteeringShaper(), SteeringShaper()
        seq = [0.1, 0.3, 0.6, 0.4, 0.0, -0.25]
        for v in seq:
            assert a.update(v, dt=0.1) == pytest.approx(
                b.update(v, dt=0.1, cap=1.0))

    def test_capped_counter_is_evidence_that_it_fired(self):
        sh = SteeringShaper()
        for _ in range(12):
            sh.update(0.55, dt=0.1)
        assert sh.capped == 0
        sh.update(0.15, dt=0.1, cap=0.15)
        assert sh.capped == 1
        assert sh.digest()["capped"] == 1

    def test_force_still_owns_the_tick(self):
        """A safety action is not delayed by the comfort shaper."""
        sh = SteeringShaper()
        for _ in range(12):
            sh.update(0.55, dt=0.1)
        out = sh.update(0.0, dt=0.1, force=True, cap=0.15)
        assert out == pytest.approx(0.0)

    def test_force_state_adopts_an_externally_vetoed_command(self):
        sh = SteeringShaper()
        sh.update(0.5, dt=0.1)
        sh.force_state(0.0)
        assert sh.value == 0.0 and sh.rate == 0.0
        # the next step starts from the vetoed value, so the veto holds
        # jerk-limited ramp: 0.6 rate * 0.1 s
        assert sh.update(0.5, dt=0.1) == pytest.approx(0.06)

    def test_a_negative_or_absurd_cap_is_ignored_as_unbounded(self):
        sh = SteeringShaper()
        out = sh.update(0.4, dt=0.1, cap=float("nan"))
        assert out > 0.0


class TestDriveLoopWire:
    """The plan's acceptance: assert the command that was SENT."""

    def test_limited_authority_reaches_the_wire(self, monkeypatch, tmp_path):
        """The plan's scenario: the wheel is already outside the new limit.

        Tick 1 drives at full authority; tick 2 the reference stabiliser
        withdraws authority to 0.15.  Before the fix the shaped command was
        still well above the limit on the wire.
        """
        import test_fsd_drive_pipeline as pipe
        from beamng_autopilot import fsd_drive as fd
        monkeypatch.setattr(fd, "REF_STABILITY_ENABLED", True)
        frames, conn = pipe.drive(
            monkeypatch, tmp_path, caps=(6.0, 6.0), delays=(0.5, 0.5),
            speed=4.0, lateral=6.0, ref_authority=[None, "limited"])
        limit = fd.REF_STABILITY_LIMITED_STEER
        # the fixture MUST have commanded more than the new limit before
        # the drop, or the test proves nothing about the drop
        assert abs(float(frames[0]["steer"])) > limit, frames[0]["steer"]
        assert float(frames[1]["steer_authority_cap"]) == pytest.approx(limit)
        for frame in frames:
            cap = float(frame["steer_authority_cap"])
            sent = [c for t, c in conn.commands if t == frame["cmd_t"]]
            assert sent, "no command reached the wire"
            # every command sent in this step answers to the authority
            # published for this step (plan T04 acceptance)
            for c in sent:
                assert abs(float(c["steering"])) <= cap + 1e-6, (
                    f"authority {cap} exceeded on the wire: "
                    f"steering={c['steering']}")
            assert abs(float(frame["steer"])) <= cap + 1e-6
        # ...and the drop itself was honoured (the counterexample)
        assert abs(float(frames[1]["steer"])) <= limit + 1e-6
        assert int(frames[1]["steer_shaper"]["capped"]) >= 1

    def test_full_authority_is_not_restricted(self, monkeypatch, tmp_path):
        """Positive control: the same demand with full authority stays large.

        Without this, "everything is clamped" would pass the test above.
        """
        import test_fsd_drive_pipeline as pipe
        from beamng_autopilot import fsd_drive as fd
        monkeypatch.setattr(fd, "REF_STABILITY_ENABLED", True)
        frames, _ = pipe.drive(
            monkeypatch, tmp_path, caps=(6.0, 6.0), delays=(0.5, 0.5),
            speed=4.0, lateral=6.0, ref_authority=[None, None])
        peak = max(abs(float(f["steer"])) for f in frames)
        assert peak > fd.REF_STABILITY_LIMITED_STEER


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
