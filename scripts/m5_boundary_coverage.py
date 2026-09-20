"""How much of the boundary the car actually sees (plan P4).

P4 is about NOT answering "am I still on the road" with something that
cannot see the road.  The 2026-09-20 finding was that the boundaries the
car receives (``lat_left`` / ``lat_right``) were absent on 82-99% of town
frames, so "how far past a DETECTED boundary" read 0.0 m while the car
finished 6.96 m past the pavement edge.

This measures, per run:

- how often each boundary channel carries a reading at all
- how often the road-surface band was actually CHECKED versus merely
  reported (``road_checked`` is the flag that separates them)
- the longest continuous stretch with no boundary reading, as BOTH a
  duration and a distance - a duration hides what it costs
- whether that clears the bar for a lateral safety claim at all

It does not propose a fix.  If the sensor layout cannot see the region
that matters, that is a sensor-coverage blocker and has to be recorded as
one rather than tuned away with a threshold further downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Boundary channels the lateral guards can use.
BOUNDARY_CHANNELS = ("lat_left", "lat_right")

# Below this, a lateral claim is not supportable: the guard is reading a
# number that is absent most of the time.
MIN_COVERAGE_FOR_A_CLAIM = 0.50


def _num(frame: dict, key: str):
    v = frame.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _has(frame: dict, key: str) -> bool:
    return key in frame and _num(frame, key) is not None


def longest_gap_runs(frames: list[dict], key: str) -> list[int]:
    """Lengths of consecutive runs with no reading in ``key``."""
    runs = []
    cur = 0
    for f in frames:
        if _has(f, key):
            if cur:
                runs.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        runs.append(cur)
    return runs


def gap_duration_s(frames: list[dict], start: int, length: int):
    """Wall seconds spanned by a run of frames, or None if not measurable."""
    ts = [_num(f, "t") for f in frames]
    lo = ts[start] if start < len(ts) else None
    hi = ts[min(start + length, len(ts) - 1)] if ts else None
    if lo is None or hi is None:
        return None
    return max(0.0, hi - lo)


def coverage(frames: list[dict]) -> dict:
    n = len(frames)
    out = {"frames": n}
    for key in BOUNDARY_CHANNELS:
        present = sum(1 for f in frames if _has(f, key))
        missing_col = sum(1 for f in frames if key not in f)
        runs = longest_gap_runs(frames, key)
        out[key] = {
            "frames_with_reading": present,
            "coverage": (present / n) if n else None,
            "missing_column": missing_col,
            "longest_gap_frames": max(runs) if runs else 0,
        }

    # Road-surface band: reported vs actually checked.
    states: dict[str, int] = {}
    checked = 0
    for f in frames:
        st = f.get("road_surface")
        states[str(st)] = states.get(str(st), 0) + 1
        if f.get("road_checked") is True:
            checked += 1
    out["road_surface"] = {
        "states": states,
        "checked_frames": checked,
        "checked_coverage": (checked / n) if n else None,
    }

    # Cost of the longest boundary gap, in metres.
    speeds = [_num(f, "speed") for f in frames]
    known = [s for s in speeds if s is not None]
    mean_speed = (sum(known) / len(known)) if known else None
    worst = max((out[k]["longest_gap_frames"] for k in BOUNDARY_CHANNELS),
                default=0)
    dur = None
    if worst:
        for i, f in enumerate(frames):
            if not _has(f, "lat_left") and not _has(f, "lat_right"):
                dur = gap_duration_s(frames, i, worst)
                break
    out["no_boundary_gap"] = {
        "longest_frames": worst,
        "duration_s": dur,
        "mean_speed_mps": mean_speed,
        "distance_m": (None if (dur is None or mean_speed is None)
                       else dur * mean_speed),
    }

    cov = [out[k]["coverage"] for k in BOUNDARY_CHANNELS
           if out[k]["coverage"] is not None]
    worst_cov = min(cov) if cov else None
    out["verdict"] = {
        "min_boundary_coverage": worst_cov,
        "threshold": MIN_COVERAGE_FOR_A_CLAIM,
        "supports_a_lateral_claim": (
            None if worst_cov is None
            else worst_cov >= MIN_COVERAGE_FOR_A_CLAIM),
    }
    return out


def print_coverage(name: str, c: dict) -> None:
    print(f"--- {name} ---")
    print(f"  frames: {c['frames']}")
    for key in BOUNDARY_CHANNELS:
        d = c[key]
        cov = ("n/a" if d["coverage"] is None
               else f"{d['coverage']:.1%}")
        print(f"  {key:10s}: readings {d['frames_with_reading']}/"
              f"{c['frames']} ({cov})  "
              f"longest gap {d['longest_gap_frames']} frames")
        if d["missing_column"]:
            print(f"    !! {d['missing_column']} frames lack the column")
    rs = c["road_surface"]
    cc = ("n/a" if rs["checked_coverage"] is None
          else f"{rs['checked_coverage']:.1%}")
    print(f"  road_surface: {rs['states']}")
    print(f"  road_checked True on {rs['checked_frames']}/{c['frames']} "
          f"({cc}) - the rest cannot support a road claim")
    g = c["no_boundary_gap"]
    dist = "n/a" if g["distance_m"] is None else f"{g['distance_m']:.1f} m"
    print(f"  longest stretch with neither boundary: {g['longest_frames']} "
          f"frames, {g['duration_s']} s, ~{dist} at "
          f"{(g['mean_speed_mps'] if g['mean_speed_mps'] is None else round(g['mean_speed_mps'], 2))} m/s mean")
    v = c["verdict"]
    if v["supports_a_lateral_claim"] is None:
        print("  lateral claim: NO DATA - cannot judge")
    elif v["supports_a_lateral_claim"]:
        print(f"  lateral claim: supported (min coverage "
              f"{v['min_boundary_coverage']:.1%})")
    else:
        print(f"  lateral claim: NOT SUPPORTED (min coverage "
              f"{v['min_boundary_coverage']:.1%} < "
              f"{v['threshold']:.0%})")
        print("    -> sensor-coverage blocker: the region that matters is")
        print("       not being observed.  Tuning a downstream threshold")
        print("       would not make it observed.")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="run JSON files")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    results = []
    for path in args.files:
        p = Path(path)
        if not p.exists():
            print(f"[coverage] missing: {p}")
            continue
        try:
            frames = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[coverage] unreadable {p.name}: {exc}")
            continue
        if not isinstance(frames, list) or not frames:
            print(f"[coverage] {p.name}: no frames")
            continue
        c = coverage(frames)
        c["run"] = p.stem
        results.append(c)
        print_coverage(p.stem, c)

    if not results:
        print("[coverage] no usable runs")
        return 1
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2),
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
