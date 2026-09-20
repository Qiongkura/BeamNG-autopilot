"""Where the tick time actually goes (plan P3).

P3 asks for the critical path split BEFORE any optimisation is chosen,
because the obvious target (the async switch) only helps if the blocking
item is the thing it makes async.  It also asks for P50/P95/P99/max and
the over-limit counts, and says plainly that a target which is not met
stays not met.

What can be split from the recorded runs:

    tick_ms.total / ring / range / plan   -- the tick's own phases
    frame_ms.local / tick / grid_mon/rest -- the drive loop around it
    head_sched[*].compute_ms              -- per head: semantic / traffic
                                             / object, when the head ran
    t between frames                      -- the ACTUAL control interval

What CANNOT be split, and is therefore missing from the profile:

    ring is one number.  It covers capture, pre-processing and result
    handling, and the ring is not instrumented inside, so "the ring is
    91% of the tick" cannot yet be narrowed to which part.  That is the
    blocker for choosing an optimisation, and it is reported as such
    rather than guessed at.

Nominal substeps (15 Hz) is not the achieved control rate; the achieved
rate is measured from the interval between commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# P3's release criterion for the tick.
TARGET_TICK_P95_MS = 150.0

# The stages read out of tick_ms / frame_ms, in the order they run.
TICK_STAGES = ("ring", "range", "plan", "total")
FRAME_STAGES = ("local", "tick", "grid_mon", "rest")


def pct(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, or None for no data."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def describe(values: list[float], *, over_ms: float | None = None) -> dict:
    out = {
        "n": len(values),
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "p99": pct(values, 99),
        "max": max(values) if values else None,
        "mean": (sum(values) / len(values)) if values else None,
    }
    if over_ms is not None:
        out["over_limit_ms"] = over_ms
        out["over_limit_count"] = sum(1 for v in values if v > over_ms)
    return out


def _num(frame: dict, *path):
    """Read a nested numeric field; None when absent or not a number.

    A missing stage is not a zero-ms stage.
    """
    cur = frame
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if cur is None:
        return None
    try:
        v = float(cur)
    except (TypeError, ValueError):
        return None
    return v


def collect(frames: list[dict]) -> dict:
    tick = {s: [] for s in TICK_STAGES}
    frame = {s: [] for s in FRAME_STAGES}
    heads: dict[str, list[float]] = {}
    budgets: list[float] = []
    over_budget = 0

    for f in frames:
        for s in TICK_STAGES:
            v = _num(f, "tick_ms", s)
            if v is not None:
                tick[s].append(v)
        for s in FRAME_STAGES:
            v = _num(f, "frame_ms", s)
            if v is not None:
                frame[s].append(v)
        hs = f.get("head_sched")
        if isinstance(hs, dict):
            for name, rec in hs.items():
                if not isinstance(rec, dict):
                    continue
                v = rec.get("compute_ms")
                if v is None:
                    continue
                try:
                    heads.setdefault(name, []).append(float(v))
                except (TypeError, ValueError):
                    continue
        b = _num(f, "budget_s")
        if b is not None:
            budgets.append(b)
            t = _num(f, "tick_ms", "total")
            if t is not None and t > b * 1000.0:
                over_budget += 1

    return {
        "tick_ms": {
            s: describe(tick[s],
                        over_ms=(TARGET_TICK_P95_MS if s == "total" else None))
            for s in TICK_STAGES},
        "frame_ms": {s: describe(frame[s]) for s in FRAME_STAGES},
        "head_compute_ms": {k: describe(v) for k, v in heads.items()},
        "budget_s": {"n": len(budgets),
                     "values": sorted(set(budgets))[:8],
                     "over_budget_frames": over_budget},
        "control_interval_s": describe(control_intervals(frames)),
    }


def control_intervals(frames: list[dict]) -> list[float]:
    """Seconds between consecutive frames - the achieved control rate."""
    out = []
    prev = None
    for f in frames:
        t = _num(f, "t")
        if t is None:
            continue
        if prev is not None:
            d = t - prev
            if d > 0:
                out.append(d)
        prev = t
    return out


def print_profile(name: str, prof: dict) -> None:
    print(f"--- {name} ---")
    print("  tick_ms:")
    for s in TICK_STAGES:
        d = prof["tick_ms"][s]
        if not d["n"]:
            print(f"    {s:8s}: no data")
            continue
        print(f"    {s:8s}: n={d['n']:4d} p50={d['p50']:7.1f} "
              f"p95={d['p95']:7.1f} p99={d['p99']:7.1f} "
              f"max={d['max']:7.1f} mean={d['mean']:7.1f}")
    print("  frame_ms:")
    for s in FRAME_STAGES:
        d = prof["frame_ms"][s]
        if not d["n"]:
            print(f"    {s:8s}: no data")
            continue
        print(f"    {s:8s}: n={d['n']:4d} p50={d['p50']:7.1f} "
              f"p95={d['p95']:7.1f} max={d['max']:7.1f}")
    if prof["head_compute_ms"]:
        print("  head compute_ms (only frames where the head ran):")
        for k, d in sorted(prof["head_compute_ms"].items()):
            print(f"    {k:10s}: n={d['n']:4d} p50={d['p50']:7.1f} "
                  f"p95={d['p95']:7.1f} max={d['max']:7.1f}")
    b = prof["budget_s"]
    print(f"  budget: values={b['values']} "
          f"frames over budget={b['over_budget_frames']}/{b['n']}")
    ci = prof["control_interval_s"]
    if ci["n"]:
        print(f"  achieved control interval: p50={ci['p50']:.3f}s "
              f"p95={ci['p95']:.3f}s max={ci['max']:.3f}s "
              f"(= {1.0 / ci['p50']:.1f} Hz at p50)")

    total = prof["tick_ms"]["total"]
    if total["n"] and total["p95"] is not None:
        verdict = ("MEETS" if total["p95"] <= TARGET_TICK_P95_MS
                   else "NOT MET")
        print(f"  tick p95 {total['p95']:.1f} ms vs target "
              f"{TARGET_TICK_P95_MS:.0f} ms: {verdict}")
        print(f"  frames over {TARGET_TICK_P95_MS:.0f} ms: "
              f"{total.get('over_limit_count')}/{total['n']}")
    else:
        print("  tick p95: no data - cannot judge the target")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="run JSON files")
    ap.add_argument("--json", default=None, help="write the profile here")
    args = ap.parse_args(argv)

    print("PERFORMANCE PROFILE - what the recorded runs can resolve")
    print("ring is ONE number: capture / pre-processing / result handling")
    print("are not instrumented separately, so the blocking item inside it")
    print("cannot be named from this data yet.")
    print()

    profiles = []
    for path in args.files:
        p = Path(path)
        if not p.exists():
            print(f"[profile] missing: {p}")
            continue
        try:
            frames = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[profile] unreadable {p.name}: {exc}")
            continue
        if not isinstance(frames, list) or not frames:
            print(f"[profile] {p.name}: no frames")
            continue
        prof = collect(frames)
        prof["run"] = p.stem
        profiles.append(prof)
        print_profile(p.stem, prof)

    if not profiles:
        print("[profile] no usable runs")
        return 1

    if args.json:
        Path(args.json).write_text(json.dumps(profiles, indent=2),
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
