"""Smooth longitudinal control: bounded acceleration and rate-limited
actuator output.

The old throttle logic fed a PID directly to the pedal, which produced
sudden throttle jumps (especially the "creep" hack that snapped to 0.6
from standstill).  This controller instead turns a target speed into a
desired acceleration with comfort limits, maps it to throttle/brake and
then rate-limits the pedals so every change is a gentle linear ramp.
"""

from __future__ import annotations

import numpy as np


def _ramp(current: float, target: float, up: float, down: float) -> float:
    """Move ``current`` toward ``target`` by at most ``up``/``down``."""
    if target > current:
        return current + min(up, target - current)
    return current - min(down, current - target)


class SpeedController:
    """Convert target speed to smooth throttle/brake commands."""

    def __init__(
        self,
        max_accel: float = 2.2,      # m/s^2 comfort acceleration
        max_decel: float = 4.0,      # m/s^2 emergency-ish braking
        kp: float = 0.6,             # accel per m/s of speed error
        deadband: float = 0.35,      # m/s error below which we coast
        creep_throttle: float = 0.34,
        creep_speed: float = 0.7,    # m/s below which creep engages
        thr_up: float = 1.6,         # pedal rate when pressing throttle (1/s)
        thr_down: float = 2.4,       # pedal rate when releasing throttle
        brk_up: float = 3.0,         # pedal rate when pressing brake
        brk_down: float = 1.2,       # pedal rate when releasing brake
        dt: float = 1.0 / 60.0,
    ):
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.kp = kp
        self.deadband = deadband
        self.creep_throttle = creep_throttle
        self.creep_speed = creep_speed
        self.thr_up = thr_up
        self.thr_down = thr_down
        self.brk_up = brk_up
        self.brk_down = brk_down
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self._throttle = 0.0
        self._brake = 0.0
        self.slip_active = False

    @property
    def throttle(self) -> float:
        return self._throttle

    @property
    def brake(self) -> float:
        return self._brake

    def update(self, target_speed: float, speed: float,
               dt: float | None = None,
               wheel_speed: float | None = None) -> tuple[float, float]:
        """Return smoothed (throttle, brake) for this control step."""
        dt = dt or self.dt
        err = float(target_speed - speed)
        if abs(err) < self.deadband:
            err = 0.0
        accel = float(np.clip(self.kp * err, -self.max_decel, self.max_accel))
        if accel >= 0.0:
            thr_req = float(np.clip(accel / self.max_accel, 0.0, 1.0))
            brk_req = 0.0
        else:
            thr_req = 0.0
            brk_req = float(np.clip(-accel / self.max_decel, 0.0, 1.0))

        # Gentle creep to get moving from standstill (ramped, not a jump).
        if speed < self.creep_speed and target_speed > 1.0:
            thr_req = max(thr_req, self.creep_throttle)

        # Wheel-spin guard: when the drive wheels turn far faster than the
        # vehicle actually moves (sand, gravel, ice), full throttle just
        # digs in.  Cut the throttle (and dab the brake once rolling) so
        # the tyres re-grip; the demand ramps back in on the next step.
        # The guard only engages at a real driving speed: at 1-3 m/s the
        # wheel-speed / body-speed sensors differ by more than the ratio
        # threshold on ordinary asphalt (etk800 live runs), so without a
        # speed floor the car would never accelerate past a crawl.
        self.slip_active = False
        if (wheel_speed is not None and speed > 3.5
                and wheel_speed > max(1.0, speed * 1.35)
                and wheel_speed - speed > 1.2):
            thr_req = min(thr_req, 0.08)
            if speed > 1.0:
                brk_req = max(brk_req, 0.12)
            self.slip_active = True

        self._throttle = _ramp(self._throttle, thr_req,
                               self.thr_up * dt, self.thr_down * dt)
        self._brake = _ramp(self._brake, brk_req,
                            self.brk_up * dt, self.brk_down * dt)
        # Never ride throttle and brake at the same time.
        if self._brake > 0.05:
            self._throttle = _ramp(self._throttle, 0.0, 0.0,
                                   self.thr_down * dt)
        return self._throttle, self._brake
