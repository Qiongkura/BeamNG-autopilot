"""Attribute every stop before optimising any of them (plan P6).

P6 is last on purpose, and its first move is classification, not tuning.
The current counters call a stop "degraded" and stop there, which lumps
together "an obstacle is in the way" and "we stopped for no reason" -
and those have opposite fixes.  Raising a speed floor to remove the
second one removes the first one's protection too.

So: find the stops, describe each by its own evidence, and only then say
which are candidates for optimisation.  A stop is unjustified ONLY when a
legal executable path existed AND no reason to stop was present.  Every
other stop is a stop that happened, and "make it go" is not its fix.

Classes, in the order they are tested:

    no_executable_path     no drivable path was produced
    stale_sensor           the perception feeding the decision was stale
    obstacle_or_boundary   something was there, or the road was unknown
    control_timeout        commands stopped going out
    near_goal              the run had arrived
    no_route_config        there was no route to follow
    control_oscillation    it stopped and restarted repeatedly
    unjustified            CANDIDATE for optimisation - see above
    unknown                the evidence to classify it is missing

The report gives each class's share of stopped time and the longest
continuous stop, because a class that owns 2% of stops but 40% of
stopped time is a different problem from the one that owns the count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STOP_SPEED_MPS = 0.30          # below this the car is not moving
MIN_STOP_FRAMES = 3            # a single frame at 0 is a sample, not a stop
NEAR_GOAL_M = 5.0              # within this of the goal, arriving is the point
CONTROL_TIMEOUT_S = 1.5        # matches the watchdog threshold

# A monitor-imposed stop counts as protecting something only if it names
# it.  These are the markers that name it.
SAFE_REASON_MARKERS = (
    "obstacle", "lane", "road", "perception", "boundary", "contact",
    "stale", "path", "crossing", "risk",
)


def _num(frame: dict, key: str):
    v = frame.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def find_stops(frames: list[dict]) -> list[dict]:
    """Contiguous runs of frames where the car was not moving."""
    stops = []
    cur: list[dict] = []
    for f in frames:
        v = _num(f, "speed")
        moving = (v is None) or (v > STOP_SPEED_MPS)
        if moving:
            if len(cur) >= MIN_STOP_FRAMES:
                stops.append(cur)
            cur = []
        else:
            cur.append(f)
    if len(cur) >= MIN_STOP_FRAMES:
        stops.append(cur)
    return stops


def stop_segment(frames: list[dict]) -> dict:
    """Describe one stop by its own evidence."""
    dur = None
    ts = [_num(f, "t") for f in frames]
    known = [t for t in ts if t is not None]
    if len(known) >= 2:
        dur = max(0.0, max(known) - min(known))
    reasons = [str(f.get("reason") or "") for f in frames]
    levels = [str(f.get("level") or "") for f in frames]
    closest = [_num(f, "closest_obs_m") for f in frames]
    closest = [c for c in closest if c is not None and c < 900.0]
    gaps = [_num(f, "cmd_gap_s") for f in frames]
    gaps = [g for g in gaps if g is not None]
    return {
        "frames": len(frames),
        "duration_s": dur,
        "reasons": sorted({r for r in reasons if r}),
        "levels": sorted({lv for lv in levels if lv}),
        "min_closest_obs_m": min(closest) if closest else None,
        "road_states": sorted({str(f.get("road_surface"))
                               for f in frames
                               if f.get("road_surface") is not None}),
        "road_checked_any": any(f.get("road_checked") is True
                                for f in frames),
        "stale_any": any(f.get("stale_sensor") or f.get("stale_planner")
                         for f in frames),
        "max_cmd_gap_s": max(gaps) if gaps else None,
        "goal_remaining_m": _num(frames[-1], "rem_end"),
        "has_route": any(f.get("route_dist") is not None for f in frames),
        # Whether the column was there at all.  Only 35 of 131 town runs
        # carry route_dist, so "no route" inferred from an absent column
        # would be a measurement gap reported as a fact.
        "route_column_seen": any("route_dist" in f for f in frames),
    }


def classify_stop(seg: dict) -> str:
    """Which class this stop belongs to.

    Order matters: the earliest class whose evidence is present wins, so a
    stop with an obstacle is never counted as unjustified just because it
    also had no route.  A stop with NO evidence at all is unknown, not
    unjustified - absence of a reason is not proof of a bad stop.
    """
    reasons = " ".join(seg.get("reasons") or []).lower()
    levels = " ".join(seg.get("levels") or []).lower()

    if "no drivable path" in reasons or "path hold" in reasons:
        return "no_executable_path"
    if seg.get("stale_any"):
        return "stale_sensor"
    gap = seg.get("max_cmd_gap_s")
    if gap is not None and gap > CONTROL_TIMEOUT_S:
        return "control_timeout"
    if seg.get("min_closest_obs_m") is not None:
        return "obstacle_or_boundary"
    if "unknown" in (seg.get("road_states") or []) \
            or "off_road" in (seg.get("road_states") or []):
        return "obstacle_or_boundary"
    if "minimal_risk" in levels or "degraded" in levels:
        # The monitor asked for the stop.  That is not enough on its own:
        # it has to say WHAT it was protecting.  Without a named reason
        # the stop is unclassified, and lumping every monitor stop into
        # "obstacle" would make the no_stall problem look already solved.
        if any(m in reasons for m in SAFE_REASON_MARKERS):
            return "obstacle_or_boundary"
        return "unknown"
    rem = seg.get("goal_remaining_m")
    if rem is not None and rem <= NEAR_GOAL_M:
        return "near_goal"
    if not seg.get("has_route"):
        # No route VALUE is a configuration finding; no route COLUMN is a
        # measurement gap, and calling it configuration would invent 26%
        # of stopped time out of nothing.
        if seg.get("route_column_seen"):
            return "no_route_config"
        return "unknown"
    if len(seg.get("levels") or []) > 1 and "safe" in levels:
        return "control_oscillation"
    # Nothing says it had to stop.  Candidate for optimisation - and only
    # a candidate: P6 wants the legal-path check confirmed before the
    # floor is touched.
    if seg.get("reasons") or seg.get("levels"):
        return "unjustified"
    return "unknown"


def classify_stops(frames: list[dict]) -> list[dict]:
    out = []
    for run in find_stops(frames):
        seg = stop_segment(run)
        seg["class"] = classify_stop(seg)
        out.append(seg)
    return out


def attribute(frames: list[dict]) -> dict:
    stops = classify_stops(frames)
    total_frames = len(frames)
    stopped_frames = sum(s["frames"] for s in stops)
    by_class: dict[str, dict] = {}
    for s in stops:
        c = by_class.setdefault(
            s["class"], {"count": 0, "frames": 0, "duration_s": 0.0,
                         "longest_s": 0.0})
        c["count"] += 1
        c["frames"] += s["frames"]
        if s["duration_s"] is not None:
            c["duration_s"] += s["duration_s"]
            c["longest_s"] = max(c["longest_s"], s["duration_s"])
    for c in by_class.values():
        c["share_of_stopped_frames"] = (
            c["frames"] / stopped_frames if stopped_frames else None)
    return {
        "frames": total_frames,
        "stopped_frames": stopped_frames,
        "stopped_share": (stopped_frames / total_frames
                          if total_frames else None),
        "n_stops": len(stops),
        "by_class": by_class,
        "stops": stops,
    }


def print_attribute(name: str, a: dict) -> None:
    print(f"--- {name} ---")
    share = ("n/a" if a["stopped_share"] is None
             else f"{a['stopped_share']:.1%}")
    print(f"  frames {a['frames']}, stopped frames {a['stopped_frames']} "
          f"({share}), {a['n_stops']} stop(s)")
    if not a["by_class"]:
        print("  no stops")
        print()
        return
    print(f"  {'class':22s} {'n':>3s} {'stopped':>8s} {'share':>7s} "
          f"{'time':>8s} {'longest':>8s}")
    for k in sorted(a["by_class"],
                    key=lambda c: -a["by_class"][c]["frames"]):
        c = a["by_class"][k]
        sh = ("n/a" if c["share_of_stopped_frames"] is None
              else f"{c['share_of_stopped_frames']:.1%}")
        print(f"  {k:22s} {c['count']:3d} {c['frames']:8d} {sh:>7s} "
              f"{c['duration_s']:7.1f}s {c['longest_s']:7.1f}s")
    unj = a["by_class"].get("unjustified")
    if unj:
        print(f"  -> {unj['count']} stop(s) are candidates for "
              f"optimisation ({unj['duration_s']:.1f}s); confirm a legal "
              f"executable path existed before touching any speed floor")
    if a["by_class"].get("unknown"):
        print("  -> some stops could not be classified: the frames carry "
              "no reason, no level and no proximity")
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
            print(f"[stall] missing: {p}")
            continue
        try:
            frames = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[stall] unreadable {p.name}: {exc}")
            continue
        if not isinstance(frames, list) or not frames:
            print(f"[stall] {p.name}: no frames")
            continue
        a = attribute(frames)
        a["run"] = p.stem
        results.append(a)
        print_attribute(p.stem, a)

    if not results:
        print("[stall] no usable runs")
        return 1
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2),
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
