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


def rate_limit_pedal(throttle: float, brake: float,
                     prev_throttle: float, prev_brake: float,
                     dt: float,
                     thr_up: float = 0.8, thr_down: float = 1.2,
                     brk_up: float = 1.2, brk_down: float = 0.8,
                     ) -> tuple[float, float]:
    """Clamp commanded pedals toward the previous tick's at bounded rates.

    ``SpeedController`` already ramps its own outputs, but a driving loop
    overrides them (downhill cap, overspeed taper, bend governor, climb /
    reverse / hard stop) and feeds the result straight to the vehicle - an
    override step (0 -> 0.8 throttle after a relaunch) reads as a visible
    speed kick.  Applying this AFTER all overrides makes every *commanded*
    change a linear ramp; safety branches bypass it on purpose (hard stop,
    climb, reverse escape, end-zone hold).
    """
    dt = max(0.01, float(dt))
    thr = float(np.clip(throttle, prev_throttle - thr_down * dt,
                        prev_throttle + thr_up * dt))
    brk = float(np.clip(brake, prev_brake - brk_down * dt,
                        prev_brake + brk_up * dt))
    return thr, brk


class SpeedController:
    """Convert target speed to smooth throttle/brake commands."""

    def __init__(
        self,
        max_accel: float = 2.2,      # m/s^2 comfort acceleration
        max_decel: float = 4.0,      # m/s^2 emergency-ish braking
        kp: float = 0.6,             # accel per m/s of speed error
        deadband: float = 0.35,      # m/s error below which we coast
        hyst_mps: float = 0.0,       # extra m/s before brake<->throttle flips
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
        self.hyst_mps = float(hyst_mps)
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
        # Pedal mode for the brake/throttle hysteresis: -1 brake, 0 coast,
        # +1 throttle.  Keeps the pedal state across frames so a target
        # speed that jitters around the deadband edge does not slam the
        # brake on one tick and the throttle on the next.
        self._mode = 0

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
        # Hysteresis: enter a pedal mode only when the error clearly
        # crosses the (deadband + hyst) edge, and hold that mode until it
        # crosses the OPPOSITE edge.  Inside the band the controller
        # coasts (both pedals ramp to 0).  With hyst=0 the behaviour is
        # the classic deadband controller; hyst>0 stops a jittering
        # target speed from flicking brake<->throttle frame to frame.
        mode = self._mode
        if mode <= 0 and err > self.deadband + self.hyst_mps:
            mode = 1
        elif mode >= 0 and err < -(self.deadband + self.hyst_mps):
            mode = -1
        accel = float(np.clip(self.kp * err, -self.max_decel, self.max_accel))
        if mode == 1:
            thr_req = float(np.clip(accel / self.max_accel, 0.0, 1.0))
            brk_req = 0.0
        elif mode == -1:
            thr_req = 0.0
            brk_req = float(np.clip(-accel / self.max_decel, 0.0, 1.0))
        else:
            thr_req = 0.0
            brk_req = 0.0
        self._mode = mode

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
