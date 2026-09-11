"""Same-session, interleaved live A/B for one binary env switch.

Every arm is a full ``m5_fsd_benchmark`` scenario run against the SAME
running game, taken alternately (A, B, A, B, ...) so slow drift of the
session is shared by both conditions instead of landing on one of them.
Each arm records the scorecard metrics, the stale-sensor frame count and
a hash of the perception/planning files, so a mid-run edit or a degraded
arm is visible instead of being silently averaged in.

Why this exists: the town scenario's off-road / crossing counts vary
0-22 between runs of ONE configuration, so a 2-arm comparison cannot
separate an effect from noise.  Five arms per condition is the smallest
run set that has been able to settle a question here (2026-09-11,
BEAMNG_LANE_REF_SLEW), and any future lateral/planning A/B should use it
rather than trusting a single before/after pair.

    .venv\\Scripts\\python.exe scripts\\m5_live_ab.py ^
        --factor BEAMNG_LANE_REF_SLEW --arms 5 ^
        --pin BEAMNG_DASHED_RECOVERY=1 --scenario town

The switch is compared at "0" (off) vs "1" (on); ``--pin`` fixes any
other switch for every arm so only the factor moves.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYEXE = ROOT / ".venv" / "Scripts" / "python.exe"
# Files whose change invalidates a run set: a mid-A/B edit must be visible.
WATCH = (
    "beamng_autopilot/planning/lateral_ref.py",
    "beamng_autopilot/fsd_stack.py",
    "beamng_autopilot/vision/lanes.py",
    "beamng_autopilot/vision/segmentation.py",
    "beamng_autopilot/safety_monitor.py",
)
# Metrics pulled off the scorecard's assessed block.  The crossing / off
# road entries are frame COUNTS: report them per arm, never only a median.
METRICS = ("lane_sensor_rate", "stall_frames", "cross_centre_frames",
           "cross_right_frames", "off_road_frames", "reversing_frames",
           "travelled_m", "goal_dist_m")


def _watch_hashes() -> dict[str, str]:
    out = {}
    for rel in WATCH:
        path = ROOT / rel
        if path.is_file():
            out[rel] = hashlib.md5(path.read_bytes()).hexdigest()[:8]
    return out


def _newest_scorecard() -> Path | None:
    files = glob.glob(str(ROOT / "logs" / "fsd_benchmark"
                          / "scorecard_*.json"))
    return Path(max(files, key=os.path.getmtime)) if files else None


def _stale_frames(card: Path) -> int | None:
    telemetry = card.parent / f"town_{card.stem.split('_')[-1]}.json"
    if not telemetry.is_file():
        return None
    try:
        return sum(1 for f in json.loads(telemetry.read_text("utf-8"))
                   if f.get("reason") == "stale sensor")
    except (OSError, json.JSONDecodeError):
        return None


def _run_arm(scenario: str, env: dict[str, str]) -> dict | None:
    subprocess.run([str(PYEXE), "scripts/m5_fsd_benchmark.py", "--attach",
                    "--runtime", "tech", "--scenarios", scenario],
                   cwd=str(ROOT), env=env, capture_output=True, text=True)
    card = _newest_scorecard()
    if card is None:
        return None
    try:
        payload = json.loads(card.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for result in payload.get("results", []):
        assessed = result.get("assessed")
        if assessed:
            return {"card": card.name, "stale": _stale_frames(card),
                    **{k: assessed.get(k) for k in METRICS}}
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="same-session live A/B")
    ap.add_argument("--factor", required=True,
                    help="env switch compared at 0 (off) vs 1 (on)")
    ap.add_argument("--arms", type=int, default=5,
                    help="runs PER condition (default 5)")
    ap.add_argument("--scenario", default="town")
    ap.add_argument("--pin", action="append", default=[],
                    metavar="NAME=VALUE",
                    help="another env switch fixed for every arm")
    args = ap.parse_args()
    if args.arms < 1:
        print("[ab] --arms must be >= 1")
        return 2

    base = _watch_hashes()
    print(f"[ab] factor={args.factor} arms/condition={args.arms} "
          f"pinned={args.pin}")
    print(f"[ab] code: {base}")
    rows: list[dict] = []
    for i in range(args.arms):
        for value in ("0", "1"):          # interleaved, drift is shared
            env = dict(os.environ, **{args.factor: value})
            for item in args.pin:
                name, _, val = item.partition("=")
                env[name.strip()] = val.strip()
            t0 = time.time()
            rec = _run_arm(args.scenario, env)
            changed = _watch_hashes() != base
            base = _watch_hashes()
            if changed:
                print("[ab] !! watched code changed during this arm - "
                      "run set invalidated")
            if rec is None:
                print(f"[ab] arm {args.factor}={value}: no scorecard")
                continue
            rec["value"] = value
            rows.append(rec)
            print(f"[ab] arm {args.factor}={value} "
                  f"lane={rec['lane_sensor_rate'] or 0:.0%} "
                  f"stall={rec['stall_frames']} "
                  f"crossC={rec['cross_centre_frames']} "
                  f"crossR={rec['cross_right_frames']} "
                  f"off={rec['off_road_frames']} "
                  f"dist={rec['travelled_m']:.1f} "
                  f"stale={rec['stale']} ({time.time() - t0:.0f}s)")

    if not rows:
        print("[ab] no usable arms")
        return 1
    print("\n[ab] === per condition ===")
    for value in ("0", "1"):
        sel = [r for r in rows if r["value"] == value]
        if not sel:
            continue
        stall = sorted(r["stall_frames"] for r in sel)
        lane = sorted(r["lane_sensor_rate"] or 0.0 for r in sel)
        dist = sorted(r["travelled_m"] for r in sel)
        print(f"[ab] {args.factor}={value} n={len(sel)} | "
              f"lane p50={statistics.median(lane):.0%} | "
              f"stall p50={statistics.median(stall):.0f} "
              f"range={stall[0]}-{stall[-1]} | "
              f"dist p50={statistics.median(dist):.1f} | "
              f"off={[r['off_road_frames'] for r in sel]} "
              f"crossC={[r['cross_centre_frames'] for r in sel]} "
              f"crossR={[r['cross_right_frames'] for r in sel]}")
    print("[ab] crossing / off-road figures are frame COUNTS per arm: "
          "compare the lists, not a median.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
