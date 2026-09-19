"""Candidate trajectory hysteresis: stop the planner flapping between paths.

The layered planner re-scores its whole candidate fan every tick, so two
candidates whose costs are within noise of each other win on alternate
frames.  The car then follows a different trajectory every 0.5 s - the
"steering intent changes every tick" behaviour the improvement plan
(phase D1) asks to remove - even though both candidates are safe.

Hysteresis fixes the *decision*, never the safety: a candidate may only
be kept while the constraint scorer still accepts it THIS tick, and every
keep/switch decision is recorded for telemetry
(``candidate_switch_count`` / ``candidate_switch_reason`` /
``current_candidate_age``).

Switch rules (first match wins), with ``feasible`` sorted by cost:

1. no previous choice                 -> switch ``first_choice``
2. the previous candidate is no
   longer feasible this tick           -> switch ``previous_infeasible``
   (it disappeared or the scorer declined it: an immediate hazard or a
   genuinely unusable path, never held)
3. ``emergency`` (caller-declared
   immediate collision risk)           -> switch ``emergency``
4. dwell time not yet served          -> hold   ``min_dwell``
5. the previous candidate's own cost
   exploded vs. when it was chosen     -> switch ``previous_degraded``
6. a new candidate beats it by more
   than ``cost_margin``                -> switch ``cost_improved``
7. otherwise                          -> hold   ``cost_margin``

Pure logic: it takes the per-tick feasible list and returns which entry
to drive.  It never builds a path and never relaxes a constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Minimum time a chosen candidate is kept before cost alone may replace
# it (plan's suggested 0.5-1.0 s band; the control loop is ~2 Hz).
HYSTERESIS_MIN_DWELL_S = 0.7
# How much cheaper a new candidate must be: absolute cost units.  The
# selector's costs are weighted sums (collision weight 5.0), so this is
# well below one collision-penalty step but above per-tick jitter.
HYSTERESIS_COST_MARGIN = 0.6
# The kept candidate's own cost growing by this factor (plus the margin)
# counts as degradation: it is still feasible, but it got much worse.
HYSTERESIS_DEGRADE_RATIO = 1.5

REASON_FIRST = "first_choice"
REASON_GONE = "previous_infeasible"
REASON_EMERGENCY = "emergency"
REASON_DWELL = "min_dwell"
REASON_DEGRADED = "previous_degraded"
REASON_IMPROVED = "cost_improved"
REASON_MARGIN = "cost_margin"

_SWITCH_REASONS = (REASON_FIRST, REASON_GONE, REASON_EMERGENCY,
                   REASON_DEGRADED, REASON_IMPROVED)


def candidate_key(candidate) -> tuple:
    """Stable identity of a candidate across ticks.

    The fan is rebuilt every tick, so identity must come from the
    generator's parameters, not from the path array: arcs are keyed by
    curvature, lane shifts by their offset, everything else by kind.
    """
    meta = getattr(candidate, "meta", None) or {}
    kind = str(meta.get("kind", "?"))
    for field_name in ("steer", "offset"):
        if field_name in meta:
            try:
                return (kind, round(float(meta[field_name]), 4))
            except (TypeError, ValueError):
                return (kind,)
    return (kind,)


@dataclass
class HysteresisState:
    """Bookkeeping for the current choice (telemetry + decisions)."""

    key: tuple | None = None
    cost: float = float("inf")
    chosen_at: float | None = None
    switches: int = 0
    last_reason: str = ""
    last_switch_at: float | None = None
    holds: int = 0
    last_switch_age_s: float | None = None

    def digest(self, now_s: float | None = None) -> dict:
        age = None
        if self.chosen_at is not None and now_s is not None:
            age = max(0.0, float(now_s) - float(self.chosen_at))
        return {
            "kind": (self.key[0] if self.key else None),
            "key": (None if self.key is None
                    else [str(self.key[0]),
                          (None if len(self.key) < 2 else self.key[1])]),
            "switch": int(self.last_reason in _SWITCH_REASONS),
            "reason": self.last_reason,
            "age_s": (None if age is None else round(float(age), 2)),
            "n_switch": int(self.switches),
            "n_hold": int(self.holds),
            "last_switch_age_s": (
                None if self.last_switch_age_s is None
                else round(float(self.last_switch_age_s), 2)),
        }


@dataclass
class CandidateHysteresis:
    """Keep the current candidate unless a rule says otherwise."""

    min_dwell_s: float = HYSTERESIS_MIN_DWELL_S
    cost_margin: float = HYSTERESIS_COST_MARGIN
    degrade_ratio: float = HYSTERESIS_DEGRADE_RATIO
    state: HysteresisState = field(default_factory=HysteresisState)

    def choose(self, feasible, now_s: float, *,
               emergency: bool = False):
        """Pick one entry from ``feasible`` (``[(cost, candidate), ...]``).

        ``feasible`` must already be this tick's constraint-approved
        list, sorted ascending by cost.  Returns ``(cost, candidate,
        info)`` where ``info`` describes the decision for telemetry.
        """
        entries = list(feasible or ())
        if not entries:
            return None
        best_cost, best = entries[0]
        prev = None
        if self.state.key is not None:
            for cost, cand in entries:
                if candidate_key(cand) == self.state.key:
                    prev = (float(cost), cand)
                    break

        reason = None
        if self.state.key is None:
            reason = REASON_FIRST
        elif prev is None:
            reason = REASON_GONE
        elif emergency:
            reason = REASON_EMERGENCY
        elif (self.state.chosen_at is not None
              and now_s - self.state.chosen_at < self.min_dwell_s):
            reason = REASON_DWELL
        elif (float(prev[0]) > self.state.cost * self.degrade_ratio
              + self.cost_margin):
            reason = REASON_DEGRADED
        elif float(best_cost) < float(prev[0]) - self.cost_margin:
            reason = REASON_IMPROVED
        else:
            reason = REASON_MARGIN

        if reason in _SWITCH_REASONS:
            cost, cand = float(best_cost), best
            if self.state.chosen_at is not None:
                self.state.last_switch_age_s = max(
                    0.0, float(now_s) - float(self.state.chosen_at))
            self.state.key = candidate_key(cand)
            self.state.cost = cost
            self.state.chosen_at = float(now_s)
            self.state.switches += 1
            self.state.last_switch_at = float(now_s)
        else:
            cost, cand = float(prev[0]), prev[1]
            # refresh the recorded cost so "previous_degraded" measures
            # growth against the last accepted value, not the first one
            self.state.cost = cost
            self.state.holds += 1
        self.state.last_reason = reason
        return cost, cand, self.state.digest(now_s)

    def reset(self) -> None:
        self.state = HysteresisState()
