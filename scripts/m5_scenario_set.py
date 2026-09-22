"""The minimum closed-loop scenario set and the paired-run protocol (P5).

Two things go wrong without this.

First, "it drove fine" is not a test.  A run in an empty town says
nothing about the escape hatch, and a run that never triggers the guard
cannot fail it - three zero-trigger rounds were once offered as a pass.
So each scenario names the QUESTION it answers and what has to be
observed for the answer to count.

Second, order confounds.  Session drift over a long benchmarking session
is real (the town arms differ by more within an arm than between arms),
so A/B has to be run in pairs with the order randomised, and the effect
estimated from the PAIRWISE difference - not from the mean of arm A minus
the mean of arm B.

The unit of statistics is the RUN.  Adjacent frames are not independent
samples; treating 500 frames of one run as 500 observations invents
significance.

Nothing here drives the car.  It is the protocol and the gate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.eval import STATUS_FAIL, STATUS_PASS, STATUS_UNKNOWN
from beamng_autopilot.fsd_realism import SRC_UNAVAILABLE

# Every scenario: the question, what must be measured, and what releases.
SCENARIOS: dict[str, dict] = {
    "A": {
        "name": "empty legal lane",
        "question": "does it stop for no reason, and does control stay continuous",
        "requires": ["damage_total", "reason", "cmd_seq", "cmd_gap_s"],
        "release": {"collisions": 0, "unjustified_stops": 0},
    },
    "B": {
        "name": "roadside clutter, legally passable",
        "question": "does it pass without hitting, crossing, or leaving pavement",
        "requires": ["damage_total", "corridor_state", "body_cross_current",
                     "road_surface", "road_checked"],
        "release": {"collisions": 0, "line_crossings": 0,
                    "pavement_exits": 0, "unjustified_stops": 0},
    },
    "C": {
        "name": "gap exists but is unreachable or illegal",
        "question": "does it refuse the escape speed-up and stop instead",
        "requires": ["corridor_state", "corridor_reason", "mon_target",
                     "closest_obs_m", "damage_total"],
        "release": {"collisions": 0, "escape_speedups": 0},
    },
    "D": {
        "name": "known collision precursor",
        "question": "does the whole chain hold, perception through final pedal",
        "requires": ["damage_total", "mon_target", "target_sm", "throttle",
                     "brake", "head_sched", "cmd_t"],
        "release": {"collisions": 0},
    },
    "E": {
        "name": "range/object stalled, cold start, out of order",
        "question": "are the failure modes distinguished, and is old output not "
                    "passed off as fresh",
        "requires": ["head_sched", "head_age_s", "range_sched", "consumed",
                     "cmd_seq"],
        "release": {"fault_handling_triggered": True,
                    "stale_presented_as_fresh": 0},
    },
    "F": {
        "name": "semantic / boundary loss and edge-hugging",
        "question": "does it degrade on time and react laterally",
        "requires": ["road_surface", "road_checked", "road_lost_s",
                     "lat_left", "lat_right", "body_cross_current"],
        "release": {"crossings_detected": True},
    },
    "G": {
        "name": "ring over budget, long tick, worker blocked",
        "question": "do the watchdog and the scheduling floor engage",
        "requires": ["tick_ms", "budget_s", "watchdog", "cmd_gap_s",
                     "range_sched"],
        "release": {"watchdog_triggered": True},
    },
    "H": {
        "name": "dynamic intrusion",
        "question": "does closing speed and predicted path risk close it down",
        "requires": ["risk_closest_m", "min_ttc_s", "mon_target",
                     "damage_total"],
        "release": {"collisions": 0},
    },
}

ALL_SCENARIOS = tuple(sorted(SCENARIOS))

# Exclusion is allowed but must be declared up front and recorded.
VALID_EXCLUSIONS = (
    "scenario_setup_failed",       # placement / spawn never came up
    "contaminated",                # another controller on the same port
    "measurement_missing",         # a required column never landed
    "aborted_by_harness",
)


def pair_order(n_pairs: int, seed: int | None = None) -> list[str]:
    """Randomised AB/BA order for ``n_pairs`` pairs, balanced.

    Balanced on purpose: every pair contributes one A-first and one
    B-first across the set, so session drift that favours whichever ran
    first does not become the effect.  Seeded, so a set is reproducible.
    """
    if n_pairs <= 0:
        return []
    orders = ["AB"] * (n_pairs // 2) + ["BA"] * (n_pairs // 2)
    if n_pairs % 2:
        orders.append("AB")
    rng = random.Random(seed)
    rng.shuffle(orders)
    return orders


def pairwise_deltas(pairs: list[dict]) -> list[float]:
    """Effect per pair: A minus B within the pair, order-independent.

    ``pairs`` entries: {"a": value, "b": value}.  A run that produced no
    value contributes nothing rather than a zero - that would invent an
    agreement that was never measured.
    """
    out = []
    for p in pairs:
        a, b = p.get("a"), p.get("b")
        if a is None or b is None:
            continue
        out.append(float(a) - float(b))
    return out


def summarise_effect(pairs: list[dict]) -> dict:
    """Repeatability and effect size, at the level of the RUN.

    Reports the spread as well as the mean, because a mean difference
    smaller than the within-pair spread is noise, and a set of two pairs
    cannot support a claim either way.
    """
    d = pairwise_deltas(pairs)
    if not d:
        return {"n_pairs": 0, "mean_delta": None, "spread": None,
                "note": "no complete pairs"}
    mean = sum(d) / len(d)
    spread = max(d) - min(d)
    return {
        "n_pairs": len(d),
        "mean_delta": round(mean, 4),
        "spread": round(spread, 4),
        "sign_consistent": (all(x > 0 for x in d) or all(x < 0 for x in d)),
        "note": ("effect smaller than the spread - indistinguishable from "
                 "noise" if abs(mean) < spread else ""),
    }


def _finite(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _flag(value):
    """Only JSON booleans and exact binary numeric scalars are flags."""
    if isinstance(value, bool):
        return value
    if _finite(value) and value in (0, 1):
        return bool(value)
    return None


def _no_head_result(record) -> bool:
    if not isinstance(record, dict) or "result_seq" not in record:
        return False
    if record["result_seq"] is not None:
        return False
    if "result_available" in record:
        return _flag(record["result_available"]) is False
    return record.get("state") == "async_idle_no_output"


def _has_measurement(frame, col: str, *, allow_unavailable: bool = False) -> bool:
    if not isinstance(frame, dict):
        return False
    # Keep the protocol name, accepting the actual fsd_drive JSON spelling.
    # An explicitly invalid primary value must not be hidden by the alias.
    key = "min_ttc" if col == "min_ttc_s" and col not in frame else col
    if key not in frame:
        return False
    value = frame[key]
    if col in ("road_checked", "body_cross_current"):
        return _flag(value) is not None
    if col in ("road_surface", "road_lost_s"):
        if _flag(frame.get("road_checked")) is not True:
            return False
    if col in ("lat_left", "lat_right"):
        if frame.get("lane_sel") == SRC_UNAVAILABLE:
            return allow_unavailable and value is None
    if col == "road_surface":
        return isinstance(value, str) and value in (
            "on_road", "off_road", "unknown")
    if col == "corridor_state":
        return isinstance(value, str) and value in (
            "feasible", "infeasible", "unknown")
    if col in ("reason", "corridor_reason"):
        return isinstance(value, str)
    if col == "watchdog":
        return isinstance(value, str) and value in ("ok", "brake", "unknown")
    if col in ("head_sched", "range_sched"):
        if not isinstance(value, dict) or not value:
            return False
        records = value.values() if col == "head_sched" else [value]
        return all(isinstance(rec, dict)
                   and isinstance(rec.get("state"), str)
                   and bool(rec["state"].strip())
                   and ("result_available" not in rec
                        or _flag(rec["result_available"]) is not None)
                   for rec in records)
    if col in ("head_age_s", "consumed"):
        sched = frame.get("head_sched")
        if (not isinstance(value, dict) or not value
                or not isinstance(sched, dict) or not sched
                or not set(sched).issubset(value)):
            return False
        for name, sample in value.items():
            unavailable = _no_head_result(sched.get(name))
            if unavailable and not allow_unavailable:
                return False
            if col == "head_age_s":
                if not (sample is None if unavailable else _finite(sample)):
                    return False
            elif not isinstance(sample, dict):
                return False
            elif unavailable:
                if (not all(k in sample for k in (
                        "result_seq", "source_seq", "age_s"))
                        or sample["result_seq"] is not None
                        or sample["age_s"] is not None
                        or not (sample["source_seq"] is None
                                or _finite(sample["source_seq"]))):
                    return False
            elif not all(_finite(sample.get(k)) for k in (
                    "result_seq", "source_seq", "age_s")):
                return False
        return True
    if col == "tick_ms":
        return isinstance(value, dict) and _finite(value.get("total"))
    return _finite(value)


def measurement_coverage(runs: list[dict], scenario: str) -> dict:
    """Required evidence, separating readings from recorded unavailability.

    Critical numeric channels must be measured throughout a run.  F/E may
    record expected boundary/output loss using existing explicit states;
    those frames are NOT numeric readings.  Bare nulls remain UNKNOWN.
    ``runs_with`` counts complete evidence, not mere key presence.
    """
    req = SCENARIOS.get(scenario, {}).get("requires", [])
    missing = {}
    columns = {}
    for col in req:
        n = 0
        counts = {"frames": 0, "measured_frames": 0, "unavailable_frames": 0,
                  "unknown_frames": 0, "missing_column": 0}
        for run in runs:
            frames = run.get("frames")
            if not isinstance(frames, list) or not frames:
                continue
            complete = True
            for frame in frames:
                counts["frames"] += 1
                key = ("min_ttc" if col == "min_ttc_s"
                       and isinstance(frame, dict) and col not in frame else col)
                if not isinstance(frame, dict) or key not in frame:
                    counts["missing_column"] += 1
                if _has_measurement(frame, col):
                    counts["measured_frames"] += 1
                elif scenario in ("E", "F") and _has_measurement(
                        frame, col, allow_unavailable=True):
                    counts["unavailable_frames"] += 1
                else:
                    counts["unknown_frames"] += 1
                    complete = False
            n += int(complete)
        counts["coverage"] = (counts["measured_frames"] / counts["frames"]
                              if counts["frames"] else None)
        columns[col] = counts
        if not runs or n < len(runs):
            missing[col] = {"runs_with": n, "runs": len(runs)}
    return {"required": req, "missing": missing, "columns": columns,
            "complete": not missing}


def gate(results: dict) -> dict:
    """Does the set release?  Every criterion, every scenario.

    As in eval.score_run, measured violations are FAIL; missing or invalid
    evidence without a measured violation is UNKNOWN.  Only complete PASS
    evidence releases, and declared exclusions supply no evidence.
    """
    checks = {}
    for sid in ALL_SCENARIOS:
        scen = SCENARIOS[sid]
        # Declared exclusions are allowed; an undeclared one is a FAIL
        # before anything else, because "we dropped it" with no reason is
        # how a bad run quietly leaves the sample.
        unrecorded = [r for r in results.get(sid, [])
                      if _flag(r.get("excluded", False)) is True
                      and r.get("exclusion_reason") not in VALID_EXCLUSIONS]
        if unrecorded:
            checks[sid] = {"state": STATUS_FAIL,
                           "reason": "excluded without a declared reason"}
            continue
        runs = [r for r in results.get(sid, [])
                if _flag(r.get("excluded", False)) is not True]
        if not runs:
            checks[sid] = {"state": STATUS_UNKNOWN,
                           "reason": "no valid runs"}
            continue
        measured = measurement_coverage(runs, sid)
        unknown = [f"{col}: required evidence missing or incomplete"
                   for col in measured["missing"]]
        if any(_flag(r.get("excluded", False)) is None for r in runs):
            unknown.append("excluded: invalid flag")
        failures = []
        for key, want in scen["release"].items():
            for r in runs:
                metrics = r.get("metrics")
                got = metrics.get(key) if isinstance(metrics, dict) else None
                valid = (_flag(got) is not None if isinstance(want, bool)
                         else _finite(got))
                if not valid:
                    unknown.append(f"{key}: not measured or invalid")
                elif got != want:
                    failures.append(f"{key}: {got} != {want}")
        state = (STATUS_FAIL if failures else
                 STATUS_UNKNOWN if unknown else STATUS_PASS)
        checks[sid] = {"state": state, "measurement_coverage": measured}
        if failures or unknown:
            checks[sid]["reason"] = "; ".join(sorted(set(failures + unknown)))

    released = all(c["state"] == STATUS_PASS for c in checks.values())
    return {
        "scenarios": checks,
        "released": released,
        "note": ("zero collisions here is a result ABOUT THIS SET, not a "
                 "safety guarantee about any environment"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=0,
                    help="generate a randomised AB/BA order")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--results", default=None,
                    help="results JSON to gate")
    args = ap.parse_args(argv)

    if args.pairs:
        print(" ".join(pair_order(args.pairs, args.seed)))
        return 0

    if args.results:
        p = Path(args.results)
        if not p.exists():
            print(f"[scenarios] missing: {p}")
            return 1
        g = gate(json.loads(p.read_text(encoding="utf-8")))
        for sid, c in g["scenarios"].items():
            print(f"  {sid} {SCENARIOS[sid]['name']:42s} {c['state']}"
                  + (f"  {c['reason']}" if c.get("reason") else ""))
        print(f"released: {g['released']}")
        print(g["note"])
        return 0 if g["released"] else 1

    print("Minimum closed-loop scenario set:")
    for sid in ALL_SCENARIOS:
        s = SCENARIOS[sid]
        print(f"  {sid}  {s['name']}")
        print(f"      question : {s['question']}")
        print(f"      requires : {', '.join(s['requires'])}")
        print(f"      release  : {s['release']}")
    print("\nUnit of statistics: the run.  Adjacent frames are not "
          "independent samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
