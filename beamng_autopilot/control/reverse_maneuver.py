"""Controlled reverse escape: back out of a dead-end, then re-plan forward.

The FSD drive must never reverse by accident (see ``reverse_guard.py``) -
but a dead-end is a different case: a real stack detects "no drivable
forward path", checks the space behind, and reverses a bounded distance to
re-position itself, then re-plans forward.  This module is the pure,
game-free state machine so it is unit-testable without a game:

* ``ReverseManeuver.decide(...)`` returns a ``ReverseCommand``: whether a
  controlled reverse is active, which gear input to use, and the signed
  reverse target speed.

States: ``idle -> reversing -> paused -> (reversing | give_up)``.

* idle: no forward path for ``stall_s`` at (near) standstill with at
  least ``min_rear_clear_m`` of space behind arms the reverse.
* reversing: reverse at ``target_reverse_mps`` until the car has moved
  back ``max_reverse_m`` (or ``max_reverse_s`` elapsed, or the rear
  clearance drops below the minimum, or a forward path reappears).
* paused: standstill, re-check the forward planner; a fresh forward path
  returns to idle (normal forward driving), otherwise after ``pause_s``
  the next attempt starts (up to ``max_attempts``).
* give_up: no more attempts; the caller simply keeps the car stopped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ReverseCommand:
    """What the caller should do this frame."""

    active: bool = False
    gear: int = 2
    target_speed_mps: float = 0.0
    reason: str = ""


@dataclass
class ReverseManeuver:
    """State machine for a bounded, forward-afterwards reverse escape."""

    # --- arming --------------------------------------------------------
    stall_s: float = 1.5          # no-forward-path time before arming
    stall_speed_mps: float = 0.5  # |signed speed| must be at/under this
    min_rear_clear_m: float = 4.0  # straight-line space behind the car
    # --- execution -----------------------------------------------------
    target_reverse_mps: float = -0.4
    max_reverse_m: float = 1.5    # distance budget for one attempt
    max_reverse_s: float = 2.5    # time budget for one attempt
    # A reverse attempt that runs away (backward speed far beyond the
    # target, e.g. down a steep slope) must terminate the whole maneuver:
    # retrying would only roll the car further backwards.  The caller holds
    # the car with brake + handbrake after this.
    # A 0.10-throttle R-gear escape on a mild slope legitimately reaches
    # ~-2.5 m/s before the frame-rate-limited brake catches it (fix38 run
    # 2026-08-27); only a real roll-away past -3.0 m/s is terminal.
    runaway_mps: float = -3.0
    # --- between attempts ----------------------------------------------
    pause_s: float = 1.0
    max_attempts: int = 1
    # After a non-runaway give-up the car may retry later: the blockage
    # can clear (or the extra metres of backup can open a rejoin path),
    # and every attempt stays bounded by distance/time/rear-clearance.
    retry_cooldown_s: float = 8.0
    # --- gear inputs ----------------------------------------------------
    fwd_gear: int = 2
    rev_gear: int = -1

    state: str = "idle"
    _stall_t: float = 0.0
    _rev_t: float = 0.0
    _rev_dist: float = 0.0
    _pause_t: float = 0.0
    _attempts: int = 0
    _giveup_t: float = 0.0
    _runaway: bool = False
    _start: tuple[float, float] | None = None

    def decide(self, has_forward_path: bool, rear_clear_m: float | None,
               signed_speed: float, pos2d, dt: float = 0.0,
               ) -> ReverseCommand:
        """Advance the state machine and return this frame's command.

        ``rear_clear_m`` is the straight-line clearance behind the car;
        ``None`` means no rear sensor data (never reverse blind).
        ``pos2d`` is the current (x, y); ``dt`` the seconds since the
        previous call (0 keeps the pure hysteresis behaviour for tests).
        """
        dt = max(0.0, float(dt))
        pos = (float(pos2d[0]), float(pos2d[1]))
        s = float(signed_speed)

        if self.state == "idle":
            if has_forward_path:
                self._stall_t = 0.0
                return ReverseCommand(reason="forward")
            self._stall_t += dt
            if (self._stall_t >= self.stall_s
                    and abs(s) <= self.stall_speed_mps
                    and rear_clear_m is not None
                    and rear_clear_m >= self.min_rear_clear_m):
                return self._arm(pos, "stalled")
            return ReverseCommand(reason="waiting")

        if self.state == "reversing":
            self._rev_t += dt
            if self._start is not None:
                self._rev_dist = max(
                    self._rev_dist,
                    math.hypot(pos[0] - self._start[0],
                               pos[1] - self._start[1]))
            # A forward path reappeared: stop reversing and let the normal
            # drive take over (the caller shifts back to D next frame).
            if has_forward_path and s >= -0.1:
                return self._pause("forward_free")
            # The attempt ran away backwards (downhill / brake failure):
            # stop reversing for good - a retry would roll further back.
            if s < self.runaway_mps:
                self._runaway = True
                self._giveup_t = 0.0
                self.state = "give_up"
                return ReverseCommand(reason="runaway")
            # Rear space disappeared or the budgets ran out: stop.
            if (rear_clear_m is not None
                    and rear_clear_m < self.min_rear_clear_m):
                return self._pause("rear_blocked")
            if self._rev_dist >= self.max_reverse_m \
                    or self._rev_t >= self.max_reverse_s:
                return self._pause("budget_done")
            return ReverseCommand(active=True, gear=self.rev_gear,
                                  target_speed_mps=self.target_reverse_mps,
                                  reason="reversing")

        if self.state == "paused":
            if has_forward_path:
                return self._idle()
            self._pause_t += dt
            if self._pause_t >= self.pause_s:
                if self._attempts >= self.max_attempts:
                    self.state = "give_up"
                    return ReverseCommand(reason="give_up")
                return self._arm(pos, "retry")
            return ReverseCommand(reason="paused")

        # give_up
        # A reappearing forward path always recovers normal driving (the
        # caller re-verifies it and shifts back to D); the caller's passive
        # reverse guard brakes any remaining backward roll.
        if has_forward_path:
            return self._idle()
        if self._runaway:
            # Runaway is terminal: a retry would only roll further back.
            return ReverseCommand(reason="give_up")
        # Non-runaway give-up: after a cooldown at standstill with clear
        # rear space, re-arm another bounded attempt.  Each attempt is
        # distance/time/rear-guarded, so retrying cannot grind into the
        # obstacle ahead or roll far backwards.
        self._giveup_t += dt
        if (self._giveup_t >= self.retry_cooldown_s
                and abs(s) <= self.stall_speed_mps
                and rear_clear_m is not None
                and rear_clear_m >= self.min_rear_clear_m):
            return self._arm(pos, "cooldown_retry")
        return ReverseCommand(reason="give_up")

    def _arm(self, pos, reason: str) -> ReverseCommand:
        self.state = "reversing"
        self._attempts += 1
        self._rev_t = 0.0
        self._rev_dist = 0.0
        self._pause_t = 0.0
        self._giveup_t = 0.0
        self._start = pos
        return ReverseCommand(active=True, gear=self.rev_gear,
                              target_speed_mps=self.target_reverse_mps,
                              reason=reason)

    def _pause(self, reason: str) -> ReverseCommand:
        self.state = "paused"
        self._pause_t = 0.0
        return ReverseCommand(reason=reason)

    def _idle(self) -> ReverseCommand:
        self.state = "idle"
        self._stall_t = 0.0
        self._pause_t = 0.0
        self._giveup_t = 0.0
        self._runaway = False
        return ReverseCommand(reason="forward")

    def reset(self) -> None:
        self.state = "idle"
        self._stall_t = 0.0
        self._rev_t = 0.0
        self._rev_dist = 0.0
        self._pause_t = 0.0
        self._attempts = 0
        self._giveup_t = 0.0
        self._runaway = False
        self._start = None

