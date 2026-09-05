"""FSD benchmark: named scenarios -> fsd_drive.run -> hard-target scorecard.

The driving quality itself is judged by ``beamng_autopilot.eval``
(assess_run / score_run); this script adds the two pieces a *benchmark*
needs beyond a single report:

* a scenario registry - each scenario is a fixed starting state (teleport
  / goal / duration) so runs are comparable across commits, mirroring the
  real-vehicle verification records in the README;
* a one-command loop - run each selected scenario through
  ``beamng_autopilot.fsd_drive.run`` (no game changes needed here: the
  drive entry is a library call), assess + score the telemetry against
  the hard targets (0 crossings / 0 off-road / 0 reversing / 0 stalls,
  goal reached for goal scenarios) and write one scorecard JSON.

Modes::

    # drive + score (needs BeamNG.tech running / --attach)
    .venv\\Scripts\\python.exe scripts\\m5_fsd_benchmark.py --attach --runtime tech \\
        --scenarios mountain

    # score existing telemetry exports only (no game)
    .venv\\Scripts\\python.exe scripts\\m5_fsd_benchmark.py --score logs\\fsd_benchmark\\mountain_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.eval import assess_run, score_run

# Scenario registry: fixed starting states mirroring the README real-
# vehicle verification records so runs are comparable across commits.
# ``require_goal`` scenarios must be given --goal (a point ON the road
# graph - a grass goal makes the A* tail a straight line across terrain).
SCENARIOS: dict[str, dict] = {
    "mountain": {
        "seconds": 90.0,
        "speed": 6.0,
        "teleport": (729.6, 763.9, 45.0),
        "goal": (616.2, 894.5),
        "require_goal": False,
        "note": "mountain hairpin start (README real-vehicle record); "
                "goal ~200 m along the road graph so a nav route exists "
                "on a fresh game (no-route = known crawl behaviour)",
    },
    "town": {
        "seconds": 120.0,
        "speed": 6.0,
        "teleport": (779.7, 735.6, -13.0),
        "goal": (868.3, 744.9),
        "require_goal": True,
        "note": "town route (start node 22209, goal ~90 m along the "
                "road graph); --traffic adds parked NPC vehicles for "
                "YOLO / obstacle-fusion verification",
    },
    "free": {
        "seconds": 60.0,
        "speed": 6.0,
        "teleport": None,
        "goal": None,
        "require_goal": False,
        "note": "drive from the current pose on the active nav route",
    },
}

# fsd_drive.run(args) namespace fields the scenario layer may override;
# defaults mirror scripts/m5_fsd_drive.py's argparse.
_DRIVE_ARG_DEFAULTS = {
    "runtime": "auto",
    "attach": False,
    "seconds": 20.0,
    "speed": 6.0,
    "steps": 3,
    "cam_w": 400,
    "cam_h": 300,
    "teleport": None,
    "out": None,
    "lane_mode": "map",
    "strict": False,
    "e2e_model": None,
    "no_e2e": False,
    "bc_model": None,
    "no_bc": False,
    "traffic": 0,
    "goal": None,
    "no_signal": False,
    "ring": "front",
    "no_shadow": False,
}


def scenario_args(name: str, base: dict, out_path: Path):
    """Namespace for one scenario: CLI base overridden by the scenario."""
    scen = SCENARIOS[name]
    vals = dict(_DRIVE_ARG_DEFAULTS)
    vals.update(base)
    vals["seconds"] = float(scen["seconds"])
    vals["speed"] = float(scen["speed"])
    tp = scen.get("teleport")
    vals["teleport"] = [float(tp[0]), float(tp[1]), float(tp[2])] if tp else None
    vals["out"] = str(out_path)
    # scenario-embedded goal when the CLI did not supply one
    if vals.get("goal") is None and scen.get("goal") is not None:
        g = scen["goal"]
        vals["goal"] = [float(g[0]), float(g[1])]
    from types import SimpleNamespace
    return SimpleNamespace(**vals)


def score_telemetry(path: Path, require_goal: bool, goal=None) -> dict:
    """Assess + score one ``--out`` telemetry JSON file.

    ``settle_s=3.0``: the first seconds after a teleport spawn are
    settling (semantic head warm-up + perception placement onto the own
    lane), not driving - discipline counts exclude them while the raw
    full-run metrics stay visible in ``assessed``.
    """
    hist = json.loads(Path(path).read_text(encoding="utf-8"))
    # settle_s=8.0 mirrors the drive's own WARMUP_S phase: the first
    # seconds are cold-launch + perception placement, not driving.
    assessed = assess_run(hist, goal=goal, settle_s=8.0)
    verdict = score_run(assessed, require_goal=require_goal)
    return {"file": str(path), "assessed": assessed, **verdict}


def _print_row(name: str, r: dict) -> None:
    a = r["assessed"]
    failed = [k for k, ok in r["checks"].items() if not ok]
    print(f"  {name:10s} {'PASS' if r['pass'] else 'FAIL':4s} "
          f"frames={a.get('frames', 0):4d} "
          f"rev={a.get('reversing_frames', 0):3d} "
          f"crossC={a.get('cross_centre_frames', 0):3d} "
          f"crossR={a.get('cross_right_frames', 0):3d} "
          f"off={a.get('off_road_frames', 0):3d} "
          f"stall={a.get('stall_frames', 0):3d} "
          f"dist={a.get('travelled_m', 0.0):6.1f}m"
          + (f"  goal={a.get('goal_dist_m')}m" if a.get("goal_dist_m")
             is not None else "")
          + (f"  FAILED: {','.join(failed)}" if failed else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="FSD benchmark runner/scorer")
    ap.add_argument("--list", action="store_true",
                    help="list the scenario registry and exit")
    ap.add_argument("--scenarios", type=str, default="mountain",
                    help="comma-separated scenario names to run")
    ap.add_argument("--score", nargs="*", default=None,
                    help="score existing --out telemetry JSONs, no driving")
    # scenario overrides for the drive layer
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="auto")
    ap.add_argument("--lane-mode", choices=("map", "auto", "sensor"),
                    default="map")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-e2e", action="store_true")
    ap.add_argument("--no-bc", action="store_true")
    ap.add_argument("--traffic", type=int, default=0, metavar="N",
                    help="park N NPC vehicles along the route")
    ap.add_argument("--no-signal", action="store_true")
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"))
    args = ap.parse_args()

    if args.list:
        for name, scen in SCENARIOS.items():
            print(f"{name:10s} {scen['note']}  "
                  f"(seconds={scen['seconds']} speed={scen['speed']} "
                  f"teleport={scen['teleport']} "
                  f"requires_goal={scen['require_goal']})")
        return 0

    base = {
        "attach": args.attach,
        "runtime": args.runtime,
        "lane_mode": args.lane_mode,
        "strict": args.strict,
        "no_e2e": args.no_e2e,
        "no_bc": args.no_bc,
        "traffic": int(args.traffic),
        "no_signal": args.no_signal,
        "goal": (list(args.goal) if args.goal is not None else None),
    }

    if args.score is not None:
        results = []
        for p in (args.score or []):
            path = Path(p)
            if not path.exists():
                print(f"[benchmark] missing {p}, skipped")
                continue
            results.append(score_telemetry(path, require_goal=False))
        all_pass = bool(results) and all(r["pass"] for r in results)
        for r in results:
            _print_row(Path(r["file"]).stem, r)
        print(f"[benchmark] {len(results)} file(s), "
              f"{'ALL PASS' if all_pass else 'FAILURES PRESENT'}")
        return 0 if all_pass else 1

    from beamng_autopilot import fsd_drive

    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for name in names:
        if name not in SCENARIOS:
            print(f"[benchmark] unknown scenario '{name}' "
                  f"(known: {', '.join(SCENARIOS)})")
            return 2
        if SCENARIOS[name]["require_goal"] and base["goal"] is None \
                and SCENARIOS[name].get("goal") is None:
            print(f"[benchmark] scenario '{name}' requires --goal "
                  f"(a road-graph point)")
            return 2

    out_dir = config.LOGS_DIR / "fsd_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    results = []
    for name in names:
        out_path = out_dir / f"{name}_{ts}.json"
        print(f"[benchmark] === {name}: {SCENARIOS[name]['note']} ===")
        ns = scenario_args(name, base, out_path)
        rc = fsd_drive.run(ns)
        if rc != 0 or not out_path.exists():
            print(f"[benchmark] scenario '{name}' produced no telemetry "
                  f"(rc={rc})")
            results.append({"scenario": name, "pass": False,
                            "checks": {"produced_telemetry": False}})
            continue
        _eff_goal = base["goal"] or (
            list(SCENARIOS[name]["goal"])
            if SCENARIOS[name].get("goal") else None)
        r = score_telemetry(
            out_path,
            require_goal=SCENARIOS[name]["require_goal"],
            goal=_eff_goal)
        r["scenario"] = name
        results.append(r)
        _print_row(name, r)

    all_pass = bool(results) and all(r["pass"] for r in results)
    scorecard = {"scenarios": names, "results": results,
                 "pass": all_pass}
    card_path = out_dir / f"scorecard_{ts}.json"
    card_path.write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[benchmark] {'ALL PASS' if all_pass else 'FAILURES PRESENT'}"
          f" -> {card_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
