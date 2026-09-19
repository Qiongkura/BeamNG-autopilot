"""Explicit driving modes (plan phase D5), opt-in and reversible.

The loop has grown a lot of situational behaviour - launch, lane
alignment at low speed, cruising, corner entry, obstacle braking, the
commanded stop and the recovery after one - but it is expressed as
ad-hoc branches spread through the tick.  Plan phase D5 asks for those
situations to be explicit MODES with their own policy, so the shaped
command can differ by situation (a launch must not yank the wheel, a
stop must not fight throttle against brake) and telemetry can say which
situation the car believed it was in.

This module is that classification and policy as pure logic: every input
is a scalar the caller already computed (speed, target, the C3 risk
flag, the lane lateral error, the near-ahead radius, whether perception
is usable).  It reads no sensor, no map and no route.

Two behaviours the plan calls out explicitly are encoded here:

* STARTING reduces the steering authority, because a large correction
  during the launch is what makes the car lurch off the line;
* CONTROLLED_STOP and OBSTACLE_BRAKING forbid throttle, so the pedals
  cannot fight each other while the car is being brought down.

Escalation is immediate and de-escalation waits for ``dwell_s``: leaving
"stopping" for "cruising" is a judgement call, entering "obstacle
braking" is not.
"""

from __future__ import annotations

from dataclasses import dataclass

MODE_STARTING = "starting"
MODE_ALIGN = "low_speed_alignment"
MODE_CRUISE = "cruising"
MODE_CURVE = "curve_entry"
MODE_OBSTACLE = "obstacle_braking"
MODE_STOP = "controlled_stop"
MODE_RECOVERY = "recovery"

MODES = (MODE_STARTING, MODE_ALIGN, MODE_CRUISE, MODE_CURVE,
         MODE_OBSTACLE, MODE_STOP, MODE_RECOVERY)

# Urgency: a higher rank may take over immediately; a lower one must wait
# for the dwell window.
_PRIORITY = {
    MODE_STOP: 6,
    MODE_OBSTACLE: 5,
    MODE_RECOVERY: 4,
    MODE_STARTING: 3,
    MODE_ALIGN: 2,
    MODE_CURVE: 1,
    MODE_CRUISE: 0,
}


@dataclass
class ModePolicy:
    """What each mode allows this tick."""

    mode: str = MODE_CRUISE
    steer_gain: float = 1.0
    max_speed_mps: float = float("inf")
    allow_throttle: bool = True
    reason: str = ""
    changed: bool = False

    def digest(self) -> dict:
        return {
            "mode": self.mode,
            "steer_gain": round(float(self.steer_gain), 3),
            "cap_mps": (None if self.max_speed_mps == float("inf")
                        else round(float(self.max_speed_mps), 2)),
            "throttle": int(bool(self.allow_throttle)),
            "why": self.reason,
        }


def _policy(mode: str, reason: str) -> ModePolicy:
    if mode == MODE_STOP:
        return ModePolicy(mode=mode, steer_gain=1.0, max_speed_mps=0.0,
                          allow_throttle=False, reason=reason)
    if mode == MODE_OBSTACLE:
        # the C3 risk cap already carries the bound; this mode's job is
        # to stop the throttle from fighting the brake
        return ModePolicy(mode=mode, steer_gain=1.0,
                          max_speed_mps=float("inf"),
                          allow_throttle=False, reason=reason)
    if mode == MODE_RECOVERY:
        return ModePolicy(mode=mode, steer_gain=1.0, max_speed_mps=1.5,
                          allow_throttle=True, reason=reason)
    if mode == MODE_STARTING:
        return ModePolicy(mode=mode, steer_gain=0.6,
                          max_speed_mps=float("inf"),
                          allow_throttle=True, reason=reason)
    if mode == MODE_ALIGN:
        return ModePolicy(mode=mode, steer_gain=0.85, max_speed_mps=3.0,
                          allow_throttle=True, reason=reason)
    if mode == MODE_CURVE:
        return ModePolicy(mode=mode, steer_gain=1.0,
                          max_speed_mps=float("inf"),
                          allow_throttle=True, reason=reason)
    return ModePolicy(mode=MODE_CRUISE, steer_gain=1.0,
                      max_speed_mps=float("inf"),
                      allow_throttle=True, reason=reason)


@dataclass
class DriveModeClassifier:
    """Classify the tick into a mode and hand back its policy."""

    start_window_s: float = 8.0
    start_speed_mps: float = 2.5
    align_speed_mps: float = 3.0
    align_lat_band_m: float = 0.35
    creep_speed_mps: float = 1.5
    stopped_speed_mps: float = 0.3
    curve_radius_m: float = 25.0
    stop_target_mps: float = 0.05
    dwell_s: float = 0.5
    # state
    mode: str = MODE_CRUISE
    since: float | None = None
    saw_stop: bool = False
    switches: int = 0

    def classify(self, *, now: float, speed_mps: float, target_speed: float,
                 stop_commanded: bool, risk_braking: bool = False,
                 radius_m: float | None = None,
                 lateral_error_m: float | None = None,
                 perception_ok: bool = True,
                 elapsed_s: float = 1e9) -> ModePolicy:
        """The mode this tick belongs to, after the dwell rule."""
        v = max(0.0, float(speed_mps))
        target = float(target_speed)
        if stop_commanded or target <= self.stop_target_mps:
            self.saw_stop = True
            mode, why = MODE_STOP, "stop commanded"
        elif risk_braking:
            mode, why = MODE_OBSTACLE, "obstacle risk braking"
        elif v < self.stopped_speed_mps or (self.saw_stop
                                            and v < self.creep_speed_mps):
            if perception_ok:
                mode, why = MODE_RECOVERY, "stopped; re-validated, creep"
            else:
                mode, why = MODE_STOP, "stopped; perception not usable"
        elif elapsed_s < self.start_window_s and v < self.start_speed_mps:
            mode, why = MODE_STARTING, "launch"
        elif (v < self.align_speed_mps and lateral_error_m is not None
              and abs(float(lateral_error_m)) > self.align_lat_band_m):
            mode, why = MODE_ALIGN, "low-speed lane alignment"
        elif radius_m is not None and 0.0 < float(radius_m) < self.curve_radius_m:
            mode, why = MODE_CURVE, "bend ahead"
        else:
            mode, why = MODE_CRUISE, "cruising"
        # recovered: once we are moving again, the stop is behind us
        if mode in (MODE_CRUISE, MODE_CURVE, MODE_ALIGN, MODE_STARTING) \
                and v > self.creep_speed_mps:
            self.saw_stop = False
        # de-escalation waits for the dwell window
        if (mode != self.mode and self.since is not None
                and _PRIORITY[mode] < _PRIORITY[self.mode]
                and float(now) - float(self.since) < self.dwell_s):
            return _policy(self.mode, f"hold {self.mode} (dwell)")
        policy = _policy(mode, why)
        if mode != self.mode:
            policy.changed = True
            self.mode = mode
            self.since = float(now)
            self.switches += 1
        elif self.since is None:
            self.since = float(now)
        return policy

    def reset(self) -> None:
        self.mode = MODE_CRUISE
        self.since = None
        self.saw_stop = False
        self.switches = 0
