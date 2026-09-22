"""P1-6: where the tick time goes, and which modality goes stale.

The round-5 handoff asks for the performance question to be answered by
decomposition instead of a single "the pipeline is slow" number: the tick
cadence, the per-stage cost (ring / range / heads / plan), the budget
skips, the ages each head and modality is actually consumed at, the
command gap, the watchdog state, and - the part that decides what to fix -
WHICH modality is the one that goes stale.

Everything here is read from the telemetry ``m5_fsd_drive --out`` already
writes; nothing new is measured, so a run set can be re-decomposed after
the fact (``p95`` over one run is a distribution, not an anecdote).

Usage::
    .venv\\Scripts\\python.exe scripts/m5_perf_decompose.py \\
        --hist logs/goal_20260921/phase1/A0.json \\
        --hist logs/goal_20260921/phase1/B0.json --json logs/perf.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    k = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[k]


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _series(hist: list[dict], getter) -> list[float]:
    out = []
    for f in hist:
        v = getter(f)
        v = _num(v)
        if v is not None:
            out.append(v)
    return out


def _stage_stats(hist: list[dict], key: str) -> dict:
    """p50/p95/p99/max per sub-key of a nested timing dict."""
    acc: dict[str, list[float]] = collections.defaultdict(list)
    for f in hist:
        d = f.get(key)
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            v = _num(v)
            if v is not None:
                acc[k].append(v)
    return {k: {"n": len(v), "p50": round(_pct(v, .5), 1),
                "p95": round(_pct(v, .95), 1), "p99": round(_pct(v, .99), 1),
                "max": round(max(v), 1)} for k, v in sorted(acc.items())}


def _nested_stage_stats(hist: list[dict], key: str) -> dict:
    """Same, one level deeper (e.g. segmentation_ms[road][inference_decode])."""
    acc: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for f in hist:
        d = f.get(key)
        if not isinstance(d, dict):
            continue
        for outer, inner in d.items():
            if not isinstance(inner, dict):
                continue
            for k, v in inner.items():
                v = _num(v)
                if v is not None:
                    acc[outer][k].append(v)
    return {o: {k: {"n": len(v), "p50": round(_pct(v, .5), 1),
                    "p95": round(_pct(v, .95), 1),
                    "max": round(max(v), 1)}
                for k, v in sorted(inner.items())}
            for o, inner in sorted(acc.items())}


def decompose(hist: list[dict]) -> dict:
    n = len(hist)
    if not n:
        return {"frames": 0, "status": "UNKNOWN", "reason": "no frames"}
    t = [float(f.get("t") or 0.0) for f in hist]
    dur = t[-1] - t[0] if n > 1 else 0.0
    out: dict = {
        "frames": n,
        "duration_s": round(dur, 2),
        "tick_hz": round((n - 1) / dur, 2) if dur > 0 else None,
        "tick_ms": _stage_stats(hist, "tick_ms"),
        "frame_ms": _stage_stats(hist, "frame_ms"),
        "perception_ms": _stage_stats(hist, "perception_ms"),
        "semantic_ms": _stage_stats(hist, "semantic_ms"),
        "segmentation_ms": _nested_stage_stats(hist, "segmentation_ms"),
        "head_age_s": _stage_stats(hist, "head_age_s"),
        "cmd_gap_s": None,
        "watchdog": None,
        "stale_owner": {},
        "budget_skips": {},
        "budget_s": None,
    }
    wall = _series(hist, lambda f: f.get("tick_wall_ms"))
    out["tick_wall_ms"] = ({"p50": round(_pct(wall, .5), 1),
                            "p95": round(_pct(wall, .95), 1),
                            "p99": round(_pct(wall, .99), 1),
                            "max": round(max(wall), 1)} if wall else None)
    gap = _series(hist, lambda f: f.get("cmd_gap_s"))
    out["cmd_gap_s"] = ({"p50": round(_pct(gap, .5), 3),
                         "p95": round(_pct(gap, .95), 3),
                         "max": round(max(gap), 3), "n": len(gap)}
                        if gap else None)
    wd = collections.Counter(str(f.get("watchdog")) for f in hist)
    braked = sum(1 for f in hist if f.get("watchdog_braked"))
    out["watchdog"] = {"states": dict(wd), "braked_frames": braked}

    # Which modality holds the max age on each frame.  This is an
    # ATTRIBUTION AID, not the stale decision: SafetyMonitor separates the
    # live modalities, the reusable range and the composite pipeline
    # latency, so "who has the biggest age" is not by itself "who fired
    # the stale rule" (plan T12).  Read it with the recorded reasons.
    owner = collections.Counter()
    for f in hist:
        fr = f.get("freshness")
        if not isinstance(fr, dict):
            continue
        cand = {k: _num(v) for k, v in fr.items() if k != "max_s"}
        cand = {k: v for k, v in cand.items() if v is not None}
        if cand:
            owner[max(cand, key=lambda k: cand[k])] += 1
    out["stale_owner"] = dict(owner)

    skips = collections.Counter()
    skip_frames = 0
    for f in hist:
        sk = f.get("budget_skips")
        if isinstance(sk, list) and sk:
            skip_frames += 1
            for s in sk:
                skips[str(s)] += 1
    out["budget_skips"] = {"frames_with_skip": skip_frames,
                           "by_head": dict(skips.most_common(8))}
    bs = _series(hist, lambda f: f.get("budget_s"))
    out["budget_s"] = ({"p50": round(_pct(bs, .5), 3),
                        "max": round(max(bs), 3)} if bs else None)

    # Consumed-result age per head: published but not consumed is a
    # different failure from slow compute.
    cons: dict[str, list[float]] = collections.defaultdict(list)
    pub: dict[str, list[float]] = collections.defaultdict(list)
    for f in hist:
        c = f.get("consumed")
        if not isinstance(c, dict):
            continue
        for head, d in c.items():
            if not isinstance(d, dict):
                continue
            a = _num(d.get("age_s"))
            if a is not None:
                cons[head].append(a)
            b = _num(d.get("publish_age_s"))
            if b is not None:
                pub[head].append(b)
    out["consumed_age_s"] = {h: {"n": len(v),
                                 "p50": round(_pct(v, .5), 3),
                                 "p95": round(_pct(v, .95), 3),
                                 "max": round(max(v), 3)}
                             for h, v in sorted(cons.items())}
    out["publish_age_s"] = {h: {"n": len(v), "p50": round(_pct(v, .5), 3),
                                "max": round(max(v), 3)}
                            for h, v in sorted(pub.items())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="P1-6 performance decomposition")
    ap.add_argument("--hist", action="append", default=[],
                    help="telemetry JSON export(s); repeat per file")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--label", action="append", default=[],
                    help="label per --hist, in the same order")
    args = ap.parse_args()
    if not args.hist:
        print("[perf] pass at least one --hist")
        return 2
    labels = list(args.label) + [Path(p).stem for p in args.hist][len(
        args.label):]
    payload = {}
    for i, p in enumerate(args.hist):
        path = Path(p)
        if not path.is_file():
            print(f"[perf] {p}: not found")
            continue
        hist = json.loads(path.read_text(encoding="utf-8"))
        rep = decompose(hist)
        payload[labels[i] if i < len(labels) else path.stem] = rep
        tk = rep["tick_ms"]
        _tot = tk.get("total", {})
        print(f"[perf] {path.name}: frames={rep['frames']} "
              f"hz={rep['tick_hz']} tick p50={_tot.get('p50')}ms "
              f"p95={_tot.get('p95')}ms p99={_tot.get('p99')}ms | "
              f"ring={tk.get('ring', {}).get('p50')} "
              f"range={tk.get('range', {}).get('p50')} "
              f"plan={tk.get('plan', {}).get('p50')}"
              + ("" if tk else "  (no tick_ms fields in this export)"))
        for stage in ("frame_ms", "perception_ms", "semantic_ms"):
            if rep[stage]:
                print(f"[perf]   {stage}: "
                      + ", ".join(f"{k}={v['p50']}/{v['p95']}ms"
                                  for k, v in rep[stage].items()))
        if rep["stale_owner"]:
            print(f"[perf]   stale owner frames: {rep['stale_owner']}")
        if rep["consumed_age_s"]:
            print("[perf]   consumed age p50/p95: "
                  + ", ".join(f"{h}={v['p50']}/{v['p95']}s"
                              for h, v in rep["consumed_age_s"].items()))
        print(f"[perf]   cmd_gap p50={rep['cmd_gap_s']['p50'] if rep['cmd_gap_s'] else None}s "
              f"watchdog={rep['watchdog']['states']} "
              f"budget_skips={rep['budget_skips']['frames_with_skip']} frames")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"[perf] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
