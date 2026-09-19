"""Control sub-step policy: high-rate control between perception ticks.

The improvement plan's phase A2/A4 asks for a control loop that runs at
10-20 Hz while perception (semantic segmentation, LiDAR, YOLO) refreshes
on its own slower cadence, and for a slow perception head to never block
the control command.

This module holds that policy as pure logic.  The drive loop keeps the
full perception + planning tick at its natural rate (~2 Hz) and, between
ticks, re-issues the control command from the LAST VERIFIED plan with a
fresh pose and speed.  A sub-step never plans and never sees new sensor
data, so it may only:

* keep driving the cached plan, with the cached target speed capped by
  the freshest obstacle-risk bound;
* hand the decision BACK to the next full tick (``ok=False``) when a
  condition needs the full verdict - a body crossing has a convergence
  recovery rule that only the safety monitor owns, and a sub-step must
  not fight it by braking;
* brake, and only for two reasons: the cached plan has gone stale
  (perception has not refreshed it inside ``stale_plan_s``), or a
  contacted obstacle risk says stop.

Everything else - sensor freshness, lane validation, planning - stays in
the tick, exactly as before: decoupling changes WHEN a command is
issued, never what evidence may authorize it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SubstepDecision:
    """What one control sub-step is allowed to do."""

    ok: bool                  # False -> hand back to the next full tick
    stop: bool = False        # -> brake and hold until the next tick
    reason: str = ""
    target_speed: float = 0.0

    @property
    def drive(self) -> bool:
        return bool(self.ok and not self.stop)


class ControlSubstep:
    """Rules for re-issuing control between two perception ticks."""

    def __init__(self, stale_plan_s: float, contact_band_m: float = 0.0):
        self.stale_plan_s = float(stale_plan_s)
        self.contact_band_m = float(contact_band_m)

    def decide(self, *, plan_age_s: float, target_speed: float,
               pose_crosses: bool = False, risk_stop: bool = False,
               risk_cap: float | None = None) -> SubstepDecision:
        """Decide this sub-step's action.

        ``plan_age_s`` is the age of the cached, verified plan;
        ``target_speed`` its cached target; ``pose_crosses`` whether the
        CURRENT body crosses a detected boundary (needs the monitor's
        recovery rule, so it defers); ``risk_stop`` / ``risk_cap`` come
        from the obstacle risk model re-evaluated at the fresh pose with
        the cached tracks.
        """
        if float(plan_age_s) > self.stale_plan_s:
            return SubstepDecision(ok=True, stop=True,
                                   reason="stale plan",
                                   target_speed=0.0)
        if risk_stop:
            return SubstepDecision(ok=True, stop=True,
                                   reason="obstacle contact risk",
                                   target_speed=0.0)
        if pose_crosses:
            # Never resolved here: the monitor owns the convergence
            # recovery for a body already over the line, so braking in a
            # sub-step would fight the very path that brings the car back.
            return SubstepDecision(ok=False, stop=False,
                                   reason="body crosses lane boundary",
                                   target_speed=0.0)
        target = max(0.0, float(target_speed))
        if risk_cap is not None:
            target = min(target, max(0.0, float(risk_cap)))
        return SubstepDecision(ok=True, stop=False, reason="",
                               target_speed=target)
