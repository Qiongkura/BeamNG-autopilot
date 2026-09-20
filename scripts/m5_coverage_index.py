"""Which telemetry columns actually exist in the recorded runs (plan P0-4/P0-5).

The round reports quote numbers like ``off_road_frames`` and ``road_lost_s`` as
if every run carried the column.  Some did not - the road-surface columns landed
mid-round - and a missing column read through ``dict.get()`` returns ``None``,
which ``str()`` turns into ``'None'``, which a filter can mistake for a valid
value.  (That near-miss produced a false "8 runs never lost the band" reading.)

So a claim has to be scoped to the runs that can actually support it.  This
script answers, per column, HOW MANY of the selected runs carry it, and how many
frames are UNKNOWN for the three-state safety fields.

Usage:
    python logs/_coverage_index.py logs/fsd_benchmark/town_*.json
    python logs/_coverage_index.py --json out.json logs/fsd_benchmark/town_*.json
    python logs/_coverage_index.py --glob "logs/fsd_benchmark/town_1789890*.json"
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Columns whose presence/absence decides whether a claim is measurable at all.
KEY_COLUMNS = (
    "road_surface", "road_lost_s", "road_checked", "road_off",
    "damage_total",
    "range_sched", "range_worker", "head_worker", "head_age_s",
    "head_sched", "head_errors",
    "tick_ms", "tick_wall_ms", "frame_ms", "budget_s", "budget_skips",
    "fwd_clear", "fwd_clear_guarded", "clear_guard", "clear_src",
    "corridor_open", "closest_obs_m", "path_occ_frac",
    "mon_target", "target_sm", "reason", "level", "emergency",
    "errors", "freshness", "source",
)

# Frames are UNKNOWN for the road-surface gate when the reader did not run.
# ``road_checked`` landed after the first runs, so an absent column counts as
# "cannot tell", not as "checked".
ROAD_UNKNOWN_STATES = ("unknown", "none", "")


def load_run(path: str) -> list[dict] | None:
    """Frames of one run, or None when the file is not a frame list."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if isinstance(data, list):
        return [f for f in data if isinstance(f, dict)]
    if isinstance(data, dict):
        for key in ("frames", "hist", "history", "log"):
            v = data.get(key)
            if isinstance(v, list):
                return [f for f in v if isinstance(f, dict)]
    return None


def road_accounting(frames: list[dict]) -> dict:
    """Split "this frame does not say confirmed on_road" into its real causes.

    Collapsing them hides the useful reading: "the column does not exist in this
    run" and "the gate ran and answered unknown" are different problems, and
    only the second one is a statement about the road.
    """
    acc = {
        "total": len(frames),
        "no_state_col": 0,     # road_surface column absent -> not instrumented
        "no_checked_col": 0,   # road_surface present, road_checked absent
        "checked_false": 0,    # road_checked present and 0 -> reader did not run
        "state_unknown": 0,    # gate ran, answer is unknown
        "state_on_road": 0,
        "state_off_road": 0,
        "state_other": 0,
    }
    for f in frames:
        if "road_surface" not in f:
            acc["no_state_col"] += 1
            continue
        if "road_checked" not in f:
            acc["no_checked_col"] += 1
        elif int(f.get("road_checked") or 0) == 0:
            acc["checked_false"] += 1
        state = str(f.get("road_surface", "")).strip().lower()
        if state in ROAD_UNKNOWN_STATES:
            acc["state_unknown"] += 1
        elif state == "on_road":
            acc["state_on_road"] += 1
        elif state == "off_road":
            acc["state_off_road"] += 1
        else:
            acc["state_other"] += 1
    return acc


def main() -> int:
    ap = argparse.ArgumentParser(description="telemetry column coverage index")
    ap.add_argument("paths", nargs="*", help="run JSON files")
    ap.add_argument("--glob", default=None, help="glob pattern instead of paths")
    ap.add_argument("--json", default=None, help="also write the index here")
    args = ap.parse_args()

    paths: list[str] = list(args.paths)
    if args.glob:
        paths += _glob.glob(args.glob)
    if not paths:
        paths = _glob.glob(os.path.join(ROOT, "logs", "fsd_benchmark", "town_*.json"))
    paths = sorted({os.path.abspath(p) for p in paths})

    runs: list[tuple[str, list[dict]]] = []
    skipped: list[str] = []
    for p in paths:
        frames = load_run(p)
        if frames:
            runs.append((p, frames))
        else:
            skipped.append(p)

    if not runs:
        print("no run JSON with a frame list found")
        return 1

    n_runs = len(runs)
    n_frames = sum(len(f) for _, f in runs)
    print(f"runs: {n_runs}   frames: {n_frames}   "
          f"skipped (not frame lists): {len(skipped)}")
    print()

    # ---- per-column coverage -------------------------------------------
    coverage: dict[str, int] = {c: 0 for c in KEY_COLUMNS}
    all_cols: dict[str, int] = {}
    for _, frames in runs:
        present = set()
        for f in frames:
            present |= set(f.keys())
        for c in KEY_COLUMNS:
            if c in present:
                coverage[c] += 1
        for c in present:
            all_cols[c] = all_cols.get(c, 0) + 1

    print(f"{'column':<22} {'runs':>6}  status")
    print("-" * 52)
    for c in KEY_COLUMNS:
        n = coverage[c]
        if n == n_runs:
            status = "all runs"
        elif n == 0:
            status = "ABSENT everywhere"
        else:
            status = f"PARTIAL - {n_runs - n} run(s) cannot support this"
        print(f"{c:<22} {n:>3}/{n_runs}  {status}")
    print()

    partial = sorted(c for c, n in all_cols.items()
                     if 0 < n < n_runs and c not in KEY_COLUMNS)
    if partial:
        print(f"other partially-present columns ({len(partial)}):")
        for c in partial:
            print(f"  {c:<22} {all_cols[c]:>3}/{n_runs}")
        print()

    # ---- UNKNOWN accounting for the three-state fields ------------------
    road: dict[str, int] = {}
    for _, frames in runs:
        for k, v in road_accounting(frames).items():
            road[k] = road.get(k, 0) + v
    tot = max(1, road.get("total", 0))
    print("road-surface gate (three-state):")
    print(f"  frames                                   : {road['total']}")
    print(f"  road_surface column absent (uninstrumented): "
          f"{road['no_state_col']}")
    print(f"  state == on_road                         : {road['state_on_road']}")
    print(f"  state == off_road                        : {road['state_off_road']}")
    print(f"  state == unknown                         : {road['state_unknown']}")
    if road.get("state_other"):
        print(f"  state == <other>                         : {road['state_other']}")
    print(f"  road_checked column absent (cannot tell) : {road['no_checked_col']}")
    print(f"  road_checked == 0 (reader did not run)   : {road['checked_false']}")
    uninstrumented = road["no_state_col"]
    if uninstrumented == road["total"]:
        print("  -> the gate is NOT instrumented in this selection at all")
    print()

    no_damage = [os.path.relpath(p, ROOT) for p, frames in runs
                 if not any("damage_total" in f for f in frames)]
    print("collision observability:")
    print(f"  runs with a damage channel   : {n_runs - len(no_damage)}/{n_runs}")
    if no_damage:
        print(f"  runs UNKNOWN for collision   : {len(no_damage)}")
        for p in no_damage[:10]:
            print(f"    {p}")
        if len(no_damage) > 10:
            print(f"    ... and {len(no_damage) - 10} more")

    if args.json:
        payload = {
            "n_runs": n_runs,
            "n_frames": n_frames,
            "runs": [os.path.relpath(p, ROOT) for p, _ in runs],
            "key_column_runs_present": coverage,
            "partial_columns": {c: all_cols[c] for c in partial},
            "road_accounting": road,
            "runs_unknown_for_collision": no_damage,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
