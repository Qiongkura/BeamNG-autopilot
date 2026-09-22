"""The single recompute command for a run's headline numbers (P0-1/P0-2).

Three inputs, one output contract:

* ``--hist run.json`` - a ``m5_fsd_drive --out`` telemetry export.  This is
  the canonical mode: it prints the unified **stop digest** (one stop
  definition, seconds per cause, UNKNOWN for missing columns) and the
  **lateral digest** (per field: coverage + spread + the field contract),
  so any number a report quotes about stopping or lateral position can be
  recomputed from the JSON alone.
* ``--log run.txt`` - the old console-log mode (kept unchanged).
* ``--shadow episode.npz`` - the old shadow-episode mode (kept unchanged).

Usage:
    .venv\\Scripts\\python.exe scripts\\m5_run_metrics.py --hist logs\\...\\run.json
    .venv\\Scripts\\python.exe scripts\\m5_run_metrics.py --hist a.json b.json --table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def metrics_from_text(t: str) -> dict:
    sensor = t.count("lane=sensor")
    unav = t.count("lane=perception-unavailable")
    n = sensor + unav
    return {
        "lane_ticks": n,
        "sensor_ticks": sensor,
        "sensor_rate": round(sensor / n, 4) if n else 0.0,
        "unavailable_ticks": unav,
        "no_drivable_path": t.count("no drivable path"),
        "grazes_obstacle": t.count("path grazes obstacle"),
        "placed": "placed=True" in t,
        "unplaced": "UNPLACED" in t or "placed=False" in t,
    }


def metrics_from_shadow(path: Path) -> dict:
    import numpy as np
    d = np.load(path, allow_pickle=True)
    lane = [str(x) for x in d["lane_src"].tolist()]
    sensor = sum(1 for s in lane if s == "sensor")
    n = len(lane)
    return {
        "frames": n,
        "sensor_rate": round(sensor / n, 4) if n else 0.0,
        "trajectory_ok_rate": round(float(d["trajectory_ok"].mean()), 4)
        if "trajectory_ok" in d else None,
    }


def ref_stability_digest(hist) -> dict:
    """Per-run reference-stability metrics (handoff P1-2 acceptance).

    Counts what the handoff asks the A/B to report instead of里程: side
    flips, the tick-to-tick centre jump distribution, how often the
    reference earned full authority, and the provenance mix (paired /
    single-side / divider / unavailable).
    """
    rows = list(hist or [])
    flips = 0
    jumps: list[float] = []
    prev: float | None = None
    authority: dict = {}
    provenance: dict = {}
    paired = pair_paired = 0
    for row in rows:
        if row.get("ref_flip"):
            flips += 1
        val = row.get("ref_lat_m")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if prev is not None:
                jumps.append(abs(float(val) - prev))
            prev = float(val)
        else:
            prev = None
        auth = str(row.get("ref_authority") or "absent")
        authority[auth] = authority.get(auth, 0) + 1
        src = str(row.get("lane_src_sel") or row.get("lane_sel") or "?")
        lfrom = row.get("lane_from")
        key = f"{src}" + (f"|{lfrom}" if lfrom else "")
        provenance[key] = provenance.get(key, 0) + 1
        if int(bool(row.get("lane_paired") or 0)):
            paired += 1
        if int(bool(row.get("pair_paired") or 0)):
            pair_paired += 1

    def _pct(values, q):
        if not values:
            return None
        vals = sorted(values)
        k = min(len(vals) - 1, max(0, int(round(q / 100.0 * (len(vals) - 1)))))
        return round(vals[k], 3)

    return {
        "frames": len(rows),
        "side_flip_frames": int(flips),
        "ref_side_flips_total": (max((int(r.get("ref_side_flips") or 0)
                                      for r in rows), default=0)),
        "centre_jump_m": {"n": len(jumps), "p50": _pct(jumps, 50),
                          "p95": _pct(jumps, 95),
                          "max": (round(max(jumps), 3) if jumps else None)},
        "authority_frames": authority,
        "provenance_frames": provenance,
        "lane_paired_frames": paired,
        "pair_paired_frames": pair_paired,
        "note": ("centre_jump is |delta ref_lat_m| between CONSECUTIVE "
                 "frames; a gap (missing value) restarts the comparison"),
    }


def digest_from_hist(paths, settle_s: float = 8.0) -> dict:
    """Canonical per-run digest(s) for telemetry JSON exports."""
    from beamng_autopilot.eval import assess_run, score_run, stop_digest
    from beamng_autopilot.telemetry_contract import lateral_digest

    out: dict = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            out[str(path)] = {"status": "UNKNOWN",
                              "reason": "file not found"}
            continue
        hist = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(hist, list) or not hist:
            out[str(path)] = {"status": "UNKNOWN", "reason": "no frames"}
            continue
        out[str(path)] = {
            "stop_digest": stop_digest(hist, settle_s=settle_s),
            "lateral_digest": lateral_digest(hist),
            "stop_by_lane_drivable": stop_by_lane_drivable(
                hist, settle_s=settle_s),
            "ref_stability": ref_stability_digest(hist),
            "assessed": {"frames": len(hist)},
            "verdict": score_run(assess_run(hist, settle_s=settle_s),
                                 require_goal=False),
        }
    return out


def stop_by_lane_drivable(hist, *, settle_s: float = 8.0) -> dict:
    """Stop seconds split by WHY the lane reference was withdrawn (P1-5).

    The drivable gate can withdraw a lane for three very different
    reasons - not enough observed samples ("we did not see"), the centre
    off observed pavement ("we saw no road there"), or nothing recorded -
    and the handoff asks for the cost of a withdrawal to be measured
    rather than assumed.  Counted only on frames that are also stops.
    """
    t = [(row.get("t") or 0.0) for row in hist]
    settled = [i for i in range(len(hist)) if float(t[i]) >= float(settle_s)]
    out: dict = {}
    for k, i in enumerate(settled):
        row = hist[i]
        spd = row.get("speed")
        if isinstance(spd, bool) or not isinstance(spd, (int, float)) \
                or spd >= 0.3:
            continue
        reason = None
        ld = row.get("lane_drivable")
        if isinstance(ld, dict):
            reason = ld.get("reason") or None
            if reason == "":
                reason = "checked_and_passed"
        src = str(row.get("lane_src_sel") or row.get("lane_sel") or "?")
        key = f"{src} | {reason or 'no_record'}"
        slot = out.setdefault(key, {"stop_frames": 0, "stop_s": 0.0})
        slot["stop_frames"] += 1
        if k + 1 < len(settled):
            slot["stop_s"] += max(0.0, float(t[settled[k + 1]]) - float(t[i]))
    for slot in out.values():
        slot["stop_s"] = round(slot["stop_s"], 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument("--shadow", type=str, default=None)
    ap.add_argument("--hist", nargs="*", default=None,
                    help="telemetry JSON export(s) from m5_fsd_drive --out")
    ap.add_argument("--settle-s", type=float, default=8.0,
                    help="settling window excluded from the counts")
    ap.add_argument("--table", action="store_true",
                    help="one line per run instead of the full JSON")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    out: dict = {}
    if args.hist:
        out.update(digest_from_hist(args.hist, settle_s=args.settle_s))
        if args.table:
            print(f"{'run':28s} {'settled':>7s} {'stop_f':>6s} {'stop_s':>7s} "
                  f"{'longest':>7s} {'hard_f':>6s} {'main cause':24s} "
                  f"{'lat cov':>8s}")
            for path, rec in out.items():
                d = rec.get("stop_digest")
                if not d:
                    print(f"{Path(path).stem:28s} {rec.get('reason', '?')}")
                    continue
                causes = sorted((d.get("by_reason") or {}).items(),
                                key=lambda kv: -kv[1]["s"])
                top = (f"{causes[0][0]} {causes[0][1]['s']:.1f}s"
                       if causes else "-")
                lat = rec["lateral_digest"]["fields"]
                cov = []
                for name in ("line_lat", "lat_left", "lane_dev_m"):
                    f = lat[name]
                    cov.append(f"{name}={f['measured_frames']}"
                               if f["status"] == "measured"
                               else f"{name}=UNKNOWN")
                print(f"{Path(path).stem:28s} {d['settled_frames']:7d} "
                      f"{str(d['stop_frames']):>6s} {str(d['stop_s']):>7s} "
                      f"{str(d['stop_longest_s']):>7s} "
                      f"{str(d['hard_stop_frames']):>6s} {top:24s} "
                      f"{' '.join(cov)}")
            if args.json:
                Path(args.json).write_text(
                    json.dumps(out, indent=2, ensure_ascii=False),
                    encoding="utf-8")
                print(f"wrote {args.json}")
            return 0
    if args.log:
        raw = Path(args.log).read_bytes()
        text = (raw.decode("utf-16") if raw[:2] == b"\xff\xfe"
                else raw.decode("utf-8", errors="replace"))
        out.update(metrics_from_text(text))
    if args.shadow:
        out.update(metrics_from_shadow(Path(args.shadow)))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
