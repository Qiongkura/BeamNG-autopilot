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
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def measurement_coverage(runs: list[dict], scenario: str) -> dict:
    """Which required columns the runs for a scenario actually carry."""
    req = SCENARIOS.get(scenario, {}).get("requires", [])
    missing = {}
    for col in req:
        n = sum(1 for r in runs
                if isinstance(r.get("frames"), list)
                and any(col in f for f in r["frames"]))
        if n < len(runs):
            missing[col] = {"runs_with": n, "runs": len(runs)}
    return {"required": req, "missing": missing,
            "complete": not missing}


def gate(results: dict) -> dict:
    """Does the set release?  Every criterion, every scenario.

    A scenario with no valid runs is not a pass and not a fail - it is
    UNKNOWN, and UNKNOWN does not release.
    """
    checks = {}
    for sid in ALL_SCENARIOS:
        scen = SCENARIOS[sid]
        # Declared exclusions are allowed; an undeclared one is a FAIL
        # before anything else, because "we dropped it" with no reason is
        # how a bad run quietly leaves the sample.
        unrecorded = [r for r in results.get(sid, [])
                      if r.get("excluded")
                      and r.get("exclusion_reason") not in VALID_EXCLUSIONS]
        if unrecorded:
            checks[sid] = {"state": "FAIL",
                           "reason": "excluded without a declared reason"}
            continue
        runs = [r for r in results.get(sid, [])
                if not r.get("excluded")]
        if not runs:
            checks[sid] = {"state": "UNKNOWN",
                           "reason": "no valid runs"}
            continue
        failures = []
        for key, want in scen["release"].items():
            for r in runs:
                got = r.get("metrics", {}).get(key)
                if got is None:
                    failures.append(f"{key}: not measured")
                elif isinstance(want, bool):
                    if bool(got) != want:
                        failures.append(f"{key}: {got} != {want}")
                elif got != want:
                    failures.append(f"{key}: {got} != {want}")
        checks[sid] = ({"state": "PASS"} if not failures
                       else {"state": "FAIL", "reason": "; ".join(
                           sorted(set(failures)))})

    released = all(c["state"] == "PASS" for c in checks.values())
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
