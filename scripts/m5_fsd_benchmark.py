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
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.eval import STATUS_FAIL, STATUS_UNKNOWN, assess_run, score_run

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
        # Explicit pin.  Without one this scenario silently used the
        # deployed default, which is byte-identical to seg_model_v12 - and
        # v8 beats v12 on the objective metric (annotation-GT line IoU) on
        # EVERY collected dataset, this one included: v8 0.2092 vs v12
        # 0.1313 vs v13b 0.0208 (2026-09-11 matrix).  The user's manual
        # stroke gate rates v13b 72.9% here, but v13b has a 0.02 line IoU on
        # this domain, so that metric is a sanity check, not a selector.
        "seg_model": "logs/m5_seg/seg_model_v8_backup/best.pt",
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
        # The iron rule: the lateral reference comes from PERCEPTION
        # ONLY (painted lines + LiDAR corridor) - the map answers "where
        # to", never "where the lane is".  sensor+strict = perception
        # lane leads, no paired perception lane -> minimal-risk stop,
        # never a map-lane drive.
        "lane_mode": "sensor",
        "strict": True,
        # Domain-specific segmentation.  v13b (the user's manual town line
        # annotations, 85.1% tolerance recall) paired most often but its
        # centres landed left of the ego lane; the 2026-09-11 full recipe +
        # hand labels + per-epoch task-metric selection produced
        # seg_model_hand (paired 16.2%, in-lane 57.2% vs v13b's 24.3% /
        # 25.8%), and on 5 live town arms each - same session, alternating -
        # it passed off/crossC/crossR = 0 on 5/5 (v13b 2/5 for off-road,
        # one crossC) with lower stall (p50 115 vs 137) and higher
        # availability (p50 27% vs 19%).  Mountain keeps its own pin.
        "seg_model": "logs/m5_seg/seg_model_hand/best_task.pt",
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
    "cam_w": 536,
    "cam_h": 403,
    "seg_model": None,
    "line_seg_model": None,
    "teleport": None,
    "out": None,
    "lane_mode": "map",
    "strict": False,
    "e2e_model": None,
    "no_e2e": False,
    "bc_model": None,
    "no_bc": False,
    "dqn_model": None,
    "no_dqn": False,
    "traffic": 0,
    "goal": None,
    "no_signal": False,
    "ring": "front",
    "no_shadow": False,
    "vis": 0,
    "corridor_lane": False,
    "paved_lane": False,
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
    # scenario-owned lane policy: the scenario IS the specification
    # (town runs perception-led per the iron rule unless the CLI
    # explicitly overrides)
    if scen.get("lane_mode") is not None:
        vals["lane_mode"] = scen["lane_mode"]
    if scen.get("strict") is not None:
        vals["strict"] = bool(scen["strict"])
    if scen.get("seg_model") is not None and not vals.get("seg_model"):
        vals["seg_model"] = scen["seg_model"]
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


def write_manifest(out_dir: Path, ts: int, names: list[str], base: dict,
                   env: dict | None = None, procs=None) -> dict | None:
    """Record what produced this batch of runs (plan P0-1/P0-2).

    Written BEFORE the first run, so it describes the state the runs started
    from.  Without it "same configuration" is an assumption: a run left a
    telemetry JSON and nothing said which code, which switch values, which
    model and which process produced it.

    ``env``/``procs`` are injectable so this is testable without a game.
    Returns the manifest, or None when the module is unavailable (a missing
    record must never be the reason a run fails).
    """
    try:
        from beamng_autopilot.run_manifest import build_manifest
    except Exception as exc:                    # pragma: no cover - import guard
        print(f"[benchmark] manifest unavailable: {exc}")
        return None
    try:
        from beamng_autopilot.fsd_drive import WARMUP_S as _warmup
        warmup = float(_warmup)
    except Exception:                       # pragma: no cover - import guard
        warmup = None
    run = {
        "scenarios": list(names),
        "goal": base.get("goal"),
        # Plan P0-1 asks for the warmup phase, the random seed and the vehicle
        # id.  Two of the three are NOT knowable here and are recorded as
        # unknown rather than filled in with something plausible:
        # - vehicle id (vid) only exists once the connector is attached, which
        #   happens inside fsd_drive, after this manifest is written.
        # - the seed: BeamNG.tech owns weather/traffic randomness and the stack
        #   exposes no seed for it, so "same configuration" cannot control it.
        #   That is part of why town mileage varies 30.9-92.9 m within one arm.
        "warmup_s": warmup,
        "vehicle_id": None,
        "seed": None,
        "seed_controlled": False,
        "teleport": {n: list(SCENARIOS[n].get("teleport") or ()) for n in names},
        "seconds": {n: SCENARIOS[n].get("seconds") for n in names},
        "speed_mps": {n: SCENARIOS[n].get("speed") for n in names},
        "strict": base.get("strict"),
        "lane_mode": base.get("lane_mode"),
        "runtime": base.get("runtime"),
        "attach": base.get("attach"),
        "traffic": base.get("traffic"),
        "no_signal": base.get("no_signal"),
        "seg_model": base.get("seg_model"),
        "line_seg_model": base.get("line_seg_model"),
    }
    # Tech version: the connector exposes no version API, so the install
    # path is the only place it is written down.  A run from another Tech
    # build is a different experiment and nothing said which build it was.
    home = (env or {}).get("BEAMNG_TECH_HOME") \
        or os.environ.get("BEAMNG_TECH_HOME") \
        or getattr(config, "BEAMNG_TECH_HOME", None)
    man = build_manifest(config.PROJECT_ROOT, run=run, env=env, procs=procs,
                         tech_home=home)
    path = out_dir / f"manifest_{ts}.json"
    path.write_text(json.dumps(man, ensure_ascii=False, indent=1),
                    encoding="utf-8")

    git = man.get("git") or {}
    switches = man.get("switches") or {}
    unset = [k for k, v in switches.items() if v is None]
    print(f"[benchmark] manifest -> {path}")
    print(f"[benchmark]   commit {git.get('commit_short')} "
          f"branch {git.get('branch')} "
          f"dirty={git.get('dirty')} ({git.get('dirty_count')} path(s))")
    print(f"[benchmark]   switches: {len(switches) - len(unset)} set, "
          f"{len(unset)} unset (unset = module default, NOT '0')")
    for art in (man.get("artifacts") or {}).values():
        if not art.get("present"):
            print(f"[benchmark]   !! artifact missing: {art.get('path')}")
    return man


def check_exclusivity(man: dict | None) -> bool:
    """True when this is the only controller on the machine (plan P0-2).

    Two controllers on one port drive the same vehicle through two teleports
    and the run belongs to neither experiment - that already cost four runs
    (town_1789889000/_9006/_9171/_9172).  The earlier detector compared log
    mtimes, which only fires when the runs overlap to the second.
    """
    if not man:
        return True                    # no record -> nothing to contradict
    exc = man.get("exclusivity") or {}
    if exc.get("ok", True):
        return True
    others = exc.get("others") or []
    print(f"[benchmark] !! {len(others)} OTHER controller process(es) running "
          f"- a run started now is contaminated:")
    for o in others:
        print(f"[benchmark]      pid={o.get('pid')} "
              f"{' '.join(str(c) for c in (o.get('cmdline') or ()))}")
    return False


def _print_row(name: str, r: dict) -> None:
    a = r["assessed"]
    # Three-state: a check that could not be measured is UNKNOWN, not FAIL -
    # and it must not be printed as a pass either (plan P0-3).
    status = r.get("status") or ("PASS" if r["pass"] else "FAIL")
    unknown = list(r.get("unknown") or ())
    failed = [k for k, ok in r["checks"].items() if not ok and k not in unknown]
    print(f"  {name:10s} {status:7s} "
          f"frames={a.get('frames', 0):4d} "
          f"lane={a.get('lane_sensor_rate', 0.0):4.0%} "
          f"rev={a.get('reversing_frames', 0):3d} "
          f"crossC={a.get('cross_centre_frames', 0):3d} "
          f"crossR={a.get('cross_right_frames', 0):3d} "
          f"off={a.get('off_road_frames', 0):3d} "
          f"stall={a.get('stall_frames', 0):3d} "
          f"dist={a.get('travelled_m', 0.0):6.1f}m"
          + (f"  goal={a.get('goal_dist_m')}m" if a.get("goal_dist_m")
             is not None else "")
          + (f"  FAILED: {','.join(failed)}" if failed else "")
          + (f"  UNKNOWN: {','.join(unknown)}" if unknown else ""))


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
    ap.add_argument("--no-dqn", action="store_true")
    ap.add_argument("--seg-model", type=str, default=None,
                    help="segmentation checkpoint override")
    ap.add_argument("--line-seg-model", type=str, default=None,
                    help="separate painted-line checkpoint")
    ap.add_argument("--traffic", type=int, default=0, metavar="N",
                    help="park N NPC vehicles along the route")
    ap.add_argument("--no-signal", action="store_true")
    ap.add_argument("--corridor-lane", action="store_true",
                    help="REFUTED LIVE (town_1789142315: candidate 1.24 m "
                         "off lane centre on 52.7%% of frames; kept only as "
                         "the recorded negative result)")
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"))
    ap.add_argument("--allow-contaminated", action="store_true",
                    help="drive even when another controller process is live "
                         "(the manifest still records the conflict)")
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
        "no_dqn": args.no_dqn,
        "seg_model": args.seg_model,
        "line_seg_model": args.line_seg_model,
        "traffic": int(args.traffic),
        "no_signal": args.no_signal,
        "corridor_lane": bool(getattr(args, "corridor_lane", False)),
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
        n_fail = sum(1 for r in results if r.get("status") == STATUS_FAIL)
        n_unk = sum(1 for r in results if r.get("status") == STATUS_UNKNOWN)
        print(f"[benchmark] {len(results)} file(s): "
              f"{len(results) - n_fail - n_unk} PASS / {n_fail} FAIL / "
              f"{n_unk} UNKNOWN"
              + ("" if all_pass else " - UNKNOWN does not release the gate"))
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

    # Plan P0-1/P0-2: record the provenance BEFORE the first run, and refuse
    # to drive while another controller is live - a contaminated run belongs
    # to no experiment (four runs were lost that way on 2026-09-20).
    man = write_manifest(out_dir, ts, names, base)
    if not check_exclusivity(man) and not args.allow_contaminated:
        print("[benchmark] refusing to drive; pass --allow-contaminated to "
              "override (the manifest still records the conflict)")
        return 3

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
