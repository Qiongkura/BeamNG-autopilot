"""Offline lane metrics report (plan phase E6) - no game, no network.

Reads a drive telemetry JSON (the ``--out`` history of ``m5_fsd_drive`` /
``m5_fsd_benchmark`` runs) and prints the metric set the plan asks for,
instead of a single average speed or a successful screenshot:

    paired_rate, false_pair_rate, in_lane_rate, false_boundary_rate,
    line_lat_mean_error, line_lat_std, lane_width_mean, lane_width_std,
    near/mid/far line recall, future 1 s / 2 s cross rates

The observational half comes from the telemetry itself.  The reference
half (recall, boundary errors, lateral error) needs labels, so it is
reported as ``n/a`` unless ``--labels`` supplies a per-frame JSON with
the ground truth: a metric that grades the prediction against itself is
exactly what the plan's E6 warning is about.

Labels file format::

    [{"t": 1.2, "pair_real": true, "boundary_published": true,
      "boundary_real": false, "line_lat_gt_m": -0.8,
      "lane_width_gt_m": 3.5,
      "zones": {"near": {"present": true, "detected": true}, ...}}, ...]

Rows are matched to telemetry frames by order (and by ``t`` when both
sides have it).

Usage::

    python scripts/m5_lane_metrics.py --hist logs/fsd_eval_*.json
    python scripts/m5_lane_metrics.py --hist t.json --labels gt.json \
        --out logs/m5_lane_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.lane.metrics import (  # noqa: E402
    ZONES,
    compute_lane_metrics,
    format_report,
    records_from_hist,
)


def load_json(path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_labels(records, labels) -> int:
    """Merge a labels list into the records; returns how many matched.

    Matching is by ``t`` when both sides carry it (within 0.05 s), else by
    position - telemetry and labels come from the same frame order in the
    collection pipeline, and a silent misalignment would corrupt exactly
    the metrics this script exists to produce, so the caller sees the
    match count.
    """
    if not isinstance(labels, list):
        return 0
    matched = 0
    by_t = {}
    for lab in labels:
        if isinstance(lab, dict) and lab.get("t") is not None:
            by_t[round(float(lab["t"]), 2)] = lab
    for i, rec in enumerate(records):
        lab = None
        if by_t:
            for key in (round(rec.t, 2), round(rec.t, 2) - 0.01,
                        round(rec.t, 2) + 0.01):
                if key in by_t:
                    lab = by_t[key]
                    break
        if lab is None and not by_t and i < len(labels):
            cand = labels[i]
            lab = cand if isinstance(cand, dict) else None
        if lab is None:
            continue
        matched += 1
        for key in ("pair_real", "boundary_published", "boundary_real"):
            if key in lab:
                setattr(rec, key, bool(lab[key]))
        for key in ("line_lat_gt_m", "lane_width_gt_m"):
            if lab.get(key) is not None:
                setattr(rec, key, float(lab[key]))
        zones = lab.get("zones") or {}
        for zone in ZONES:
            z = zones.get(zone) or {}
            if z.get("present"):
                rec.zone_present[zone] = True
                rec.zone_detected[zone] = bool(z.get("detected"))
    return matched


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hist", required=True,
                    help="drive telemetry JSON (a list of frame rows)")
    ap.add_argument("--labels", default=None,
                    help="optional per-frame ground truth JSON")
    ap.add_argument("--out", default=None,
                    help="write the metric digest as JSON here")
    args = ap.parse_args(argv)

    rows = load_json(args.hist)
    if not isinstance(rows, list):
        print("telemetry file is not a frame list; nothing to score")
        return 2
    records = records_from_hist(rows)
    matched = 0
    if args.labels:
        matched = apply_labels(records, load_json(args.labels))
    metrics = compute_lane_metrics(records)
    print(f"telemetry: {args.hist}  ({len(records)} frames scored)")
    if args.labels:
        print(f"labels:    {args.labels}  ({matched} frames matched)")
    print("-" * 46)
    print(format_report(metrics))
    out = (Path(args.out) if args.out
           else config.LOGS_DIR / "m5_lane_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics.digest(), indent=1,
                              ensure_ascii=False), encoding="utf-8")
    print(f"digest -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
