"""Shadow-replay the escape hatch against recorded runs (plan P2.2).

WHAT THIS CAN AND CANNOT DO
---------------------------
P2.2 asks for a shadow replay of the new feasibility primitive on the
2026-09-20 collision run.  Before writing it I checked what that run
actually holds: `town_1789886413.json` has 160 frames of SCALARS -
`closest_obs_m`, `path_occ_frac`, `corridor_open`, `mon_target`, `pos`,
`speed`, `damage_total` - and no grid, no obstacle layer, no candidate
trajectory, no per-modality version.

So the grid-level primitive CANNOT be replayed on it.  Reconstructing a
BEV from `closest_obs_m` would be inventing the input, and a primitive
scored on invented input proves nothing.  What this script does instead
is the replay that the recorded data does support: it reads what the
escape hatch actually DID on those frames, so the claim "corridor_open
was True" can be checked against what the car did next.

Concretely, per run:

1. how often `corridor_open` was True at all, and how close the nearest
   obstacle was when it was;
2. the frames where it was True with an obstacle NEARER than the two
   reference distances - those are the frames the escape hatch authorised
   a speed on, and they are the ones to look at;
3. what target speed the monitor offered on those frames versus what the
   car was doing;
4. where damage appears, and what the hatch said in the frames leading
   up to it.

It prints numbers and does not declare pass/fail: with one run and no
grid there is nothing to declare.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The two reference distances from the 2026-09-20 analysis.  They anchor
# the buckets; they are NOT the only frames worth looking at, and the
# script always reports the full distribution as well.
PROTECT_M = 3.750     # the scene the hatch must not damage
PROBE_M = 3.789       # the collision run's approach distance

# The logs write 999.0 for "no obstacle detected".  It is a sentinel, not
# a distance: averaging it or comparing it as one puts a 999 m phantom
# into every statistic it touches.
NO_OBSTACLE_SENTINEL = 999.0


def closest_of(frame: dict):
    """Nearest obstacle, or None when there is no measurement.

    None covers both a missing column and the 999 sentinel - "nothing
    detected" and "not measured" are different facts, but neither is a
    distance, and both must stay out of the min/median.
    """
    c = _f(frame, "closest_obs_m")
    if c is None or c >= NO_OBSTACLE_SENTINEL:
        return None
    return c


def _f(frame: dict, key: str):
    v = frame.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def open_frames(frames: list[dict]) -> list[dict]:
    """Frames where the escape hatch said the corridor was open."""
    return [f for f in frames if f.get("corridor_open") is True]


def nearer_than(frames: list[dict], dist_m: float) -> list[dict]:
    """Open frames whose nearest obstacle is closer than ``dist_m``.

    A frame with no `closest_obs_m` is NOT counted as close and NOT
    counted as far - it is unknown, and folding it into either bucket is
    how "no measurement" turns into "no problem".
    """
    out = []
    for f in frames:
        c = closest_of(f)
        if c is not None and c < dist_m:
            out.append(f)
    return out


def damage_onsets(frames: list[dict]) -> list[int]:
    """Indices where cumulative damage increased."""
    out = []
    prev = None
    for i, f in enumerate(frames):
        d = _f(f, "damage_total")
        if d is None:
            continue
        if prev is not None and d > prev + 1e-9:
            out.append(i)
        prev = d
    return out


def summarise(frames: list[dict], label: str = "") -> dict:
    opens = open_frames(frames)
    closes = [closest_of(f) for f in opens]
    known = [c for c in closes if c is not None]
    res = {
        "frames": len(frames),
        "corridor_open_frames": len(opens),
        "corridor_open_missing_column": sum(
            1 for f in frames if "corridor_open" not in f),
        "closest_missing_column": sum(
            1 for f in frames if "closest_obs_m" not in f),
        "closest_sentinel_frames": sum(
            1 for f in frames if (_f(f, "closest_obs_m") is not None
                                  and _f(f, "closest_obs_m")
                                  >= NO_OBSTACLE_SENTINEL)),
        "open_closest_min_m": min(known) if known else None,
        "open_closest_median_m": (
            sorted(known)[len(known) // 2] if known else None),
        "open_nearest_than_protect": len(nearer_than(opens, PROTECT_M)),
        "open_nearest_than_probe": len(nearer_than(opens, PROBE_M)),
        "open_nearer_than_2m": len(nearer_than(opens, 2.0)),
        "damage_onsets": damage_onsets(frames),
    }
    if label:
        res["run"] = label
    return res


def print_run(res: dict, frames: list[dict], lookback: int = 6) -> None:
    print(f"--- {res.get('run', 'run')} ---")
    print(f"  frames                       : {res['frames']}")
    print(f"  corridor_open == True        : {res['corridor_open_frames']}")
    if res["corridor_open_missing_column"]:
        print(f"  !! frames with no corridor_open column: "
              f"{res['corridor_open_missing_column']}")
    if res["closest_missing_column"]:
        print(f"  !! frames with no closest_obs_m column: "
              f"{res['closest_missing_column']}")
    if res["closest_sentinel_frames"]:
        print(f"  frames with the 999 sentinel (no obstacle detected): "
              f"{res['closest_sentinel_frames']} - excluded from min/median")
    print(f"  when open, closest_obs_m min : {res['open_closest_min_m']}")
    print(f"  when open, closest_obs_m med : {res['open_closest_median_m']}")
    print(f"  open and nearer than {PROTECT_M} m (protect): "
          f"{res['open_nearest_than_protect']}")
    print(f"  open and nearer than {PROBE_M} m (probe)  : "
          f"{res['open_nearest_than_probe']}")
    print(f"  open and nearer than 2.0 m       : "
          f"{res['open_nearer_than_2m']}")

    near = nearer_than(open_frames(frames), 2.0)
    if near:
        print("  the frames the hatch opened with an obstacle under 2 m:")
        for f in near[:8]:
            print(f"    t={f.get('t')} closest={f.get('closest_obs_m')} "
                  f"mon_target={f.get('mon_target')} "
                  f"speed={f.get('speed')} reason={f.get('reason')!r}")
        if len(near) > 8:
            print(f"    ... and {len(near) - 8} more")

    onsets = res["damage_onsets"]
    if onsets:
        print(f"  damage onsets at frame index : {onsets}")
        for i in onsets[:3]:
            lo = max(0, i - lookback)
            print(f"    before onset {i}:")
            for j in range(lo, i + 1):
                f = frames[j]
                print(f"      t={f.get('t')} "
                      f"corridor_open={f.get('corridor_open')} "
                      f"closest={f.get('closest_obs_m')} "
                      f"mon_target={f.get('mon_target')} "
                      f"speed={f.get('speed')}")
    else:
        print("  damage onsets                : none in this run")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="run JSON files")
    ap.add_argument("--json", default=None, help="write the summary here")
    args = ap.parse_args(argv)

    print("SHADOW REPLAY - scalar only")
    print("This run's log carries no grid, no obstacle layer and no")
    print("candidate trajectory, so the P2.1 primitive cannot be replayed")
    print("on it.  What follows is what the OLD bool did on these frames.")
    print()

    all_res = []
    for path in args.files:
        p = Path(path)
        if not p.exists():
            print(f"[replay] missing: {p}")
            continue
        try:
            frames = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[replay] unreadable {p.name}: {exc}")
            continue
        if not isinstance(frames, list) or not frames:
            print(f"[replay] {p.name}: no frames")
            continue
        res = summarise(frames, p.stem)
        all_res.append(res)
        print_run(res, frames)

    if not all_res:
        print("[replay] no usable runs")
        return 1

    if args.json:
        Path(args.json).write_text(json.dumps(all_res, indent=2),
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
