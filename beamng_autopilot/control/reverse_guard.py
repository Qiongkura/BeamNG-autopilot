"""Reverse guard: the FSD drive must never drive backwards.

Real factory stacks (and the project's m5_autopilot) keep the car
pointing forward: they run a realistic gearbox in D and brake the moment
the signed speed along the ego heading turns negative - after a wall
impact the car bounces backward and an unguarded throttle would keep
driving into reverse ("dumb reversing" seen on FSD-mode probes).

This module is the *pure, game-free* state machine so it is unit-testable
without a game:

* ``ReverseGuard.decide(signed_mps) -> (brake, clear)``: given the
  signed forward speed (vel projected on the ego heading), returns how
  hard to brake and whether the reverse condition has cleared.

The caller (m5_fsd_drive / autopilot) also must keep the gearbox in D -
that part is mechanical, not state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReverseGuard:
    """Brake while the car is moving backwards, release once forward again.

    ``threshold_mps``: signed speed below which reverse is detected
    (negative = travelling backward).  ``clear_mps``: signed speed that
    must be re-crossed (with hysteresis) before the guard releases, so a
    bounce that settles near 0 does not flap brake on/off.  A car that
    has been braked to a true standstill (``|signed| <= settled_mps`` for
    ``settle_s`` seconds) is released again: otherwise a wall impact that
    leaves the car stopped at 0 would lock the brakes forever because the
    hysteresis release point (+0.2) is never reached.
    """

    threshold_mps: float = -0.35
    clear_mps: float = 0.2
    settled_mps: float = 0.08
    settle_s: float = 0.6
    braking: bool = False
    _settle_t: float = 0.0

    def decide(self, signed_mps: float,
               brake_hold: float = 1.0,
               dt: float = 0.0) -> tuple[float, bool]:
        """Return ``(brake_command, reverse_detected)``.

        ``brake_command`` is the brake to apply (1.0 while reversing,
        0.0 when safe to drive); ``reverse_detected`` is the raw flag so
        the caller can zero the steering / clear sensor ghosts.
        ``dt`` is the wall-clock seconds since the previous call; pass a
        real frame interval so the settle-timer can release a stopped
        car (callers that omit it keep the pure hysteresis behaviour).
        """
        s = float(signed_mps)
        if s < self.threshold_mps:
            self.braking = True
            self._settle_t = 0.0
            return float(brake_hold), True
        if self.braking and s < self.clear_mps:
            # still below the release point after a detected reverse:
            # keep braking (hysteresis; do not flap brake on/off), but a
            # car that has come to a true standstill may re-arm the drive
            # after settle_s - otherwise a bounce that leaves the car at
            # 0 would brake it forever.
            if abs(s) <= self.settled_mps:
                self._settle_t += max(0.0, float(dt))
                if self._settle_t >= self.settle_s:
                    self.braking = False
                    self._settle_t = 0.0
                    return 0.0, False
            else:
                self._settle_t = 0.0
            return float(brake_hold), True
        self.braking = False
        self._settle_t = 0.0
        return 0.0, False

    def reset(self) -> None:
        self.braking = False
        self._settle_t = 0.0
