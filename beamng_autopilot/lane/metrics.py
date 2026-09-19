"""Lane / lane-marking evaluation metrics (plan phase E6).

The plan is explicit that a model must not be chosen on ``val_mIoU``
alone: what matters is whether the car stayed in its lane - so the
metrics are the lane outcomes and the marking geometry's stability:

paired_rate, false_pair_rate, in_lane_rate, false_boundary_rate,
line_lat_mean_error, line_lat_std, lane_width_mean, lane_width_std,
near/mid/far line recall, future 1 s / 2 s cross rates.

Two kinds of metric live here and they are NOT mixed:

* **observational** metrics are computed from what the run recorded
  (was a lane published, where was the car inside it, how did the
  painted-line offset and the lane width behave) - no ground truth
  needed;
* **reference** metrics need labels (was the published boundary real,
  was a line present in this zone, what was the true lateral offset).
  When a record carries no reference value they are reported as ``None``
  with a count of how many frames had one, instead of being silently
  computed from the prediction itself - a metric that grades the model
  against the model is the failure mode this whole section exists to
  avoid (see the README's v13b incident: an 85% "recall" for a model whose
  line IoU was 0.02).

Pure logic: records in, metrics out.  The telemetry adapter
(:func:`records_from_hist`) maps the drive loop's frame rows onto the
record shape and leaves anything it cannot know as None.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Zones for the recall split (the same names the E2 gate uses).
ZONES = ("near", "mid", "far")


@dataclass
class LaneRecord:
    """One frame's lane evidence, as measured or as recorded."""

    t: float = 0.0
    # observational
    paired: bool | None = None
    lane_src: str = ""
    in_lane: bool | None = None
    line_lat_m: float | None = None
    lane_width_m: float | None = None
    cross_1s: bool | None = None
    cross_2s: bool | None = None
    # reference (labels); None when the run/record has no ground truth
    boundary_published: bool | None = None
    boundary_real: bool | None = None
    pair_real: bool | None = None
    line_lat_gt_m: float | None = None
    lane_width_gt_m: float | None = None
    zone_present: dict = field(default_factory=dict)
    zone_detected: dict = field(default_factory=dict)


def _rate(values) -> tuple[float | None, int]:
    """``(mean of the known booleans, n)``; None when nothing is known."""
    known = [bool(v) for v in values if v is not None]
    if not known:
        return None, 0
    return float(np.mean(known)), len(known)


def _finite(values) -> np.ndarray:
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return np.asarray(out, dtype=float)


@dataclass
class LaneMetrics:
    """The plan's metric list plus the sample counts behind each one."""

    frames: int = 0
    paired_rate: float | None = None
    false_pair_rate: float | None = None
    in_lane_rate: float | None = None
    false_boundary_rate: float | None = None
    line_lat_mean_error: float | None = None
    line_lat_std: float | None = None
    lane_width_mean: float | None = None
    lane_width_std: float | None = None
    zone_recall: dict = field(default_factory=dict)
    future_1s_cross_rate: float | None = None
    future_2s_cross_rate: float | None = None
    counts: dict = field(default_factory=dict)

    def digest(self) -> dict:
        def _r(v):
            if v is None:
                return None
            return round(float(v), 4) if isinstance(v, float) else v
        return {
            "frames": int(self.frames),
            "paired_rate": _r(self.paired_rate),
            "false_pair_rate": _r(self.false_pair_rate),
            "in_lane_rate": _r(self.in_lane_rate),
            "false_boundary_rate": _r(self.false_boundary_rate),
            "line_lat_mean_error": _r(self.line_lat_mean_error),
            "line_lat_std": _r(self.line_lat_std),
            "lane_width_mean": _r(self.lane_width_mean),
            "lane_width_std": _r(self.lane_width_std),
            "zone_recall": {k: _r(v) for k, v in self.zone_recall.items()},
            "future_1s_cross_rate": _r(self.future_1s_cross_rate),
            "future_2s_cross_rate": _r(self.future_2s_cross_rate),
            "counts": dict(self.counts),
        }


def compute_lane_metrics(records) -> LaneMetrics:
    """Compute the plan's metric set from per-frame records."""
    recs = list(records or ())
    m = LaneMetrics(frames=len(recs))
    if not recs:
        return m

    m.paired_rate, n_paired = _rate(r.paired for r in recs)

    # false_pair_rate: of the frames that published a pair, how many were
    # labelled not-a-real-pair (needs the reference; None without it)
    pub = [r for r in recs if r.pair_real is not None or r.paired]
    m.false_pair_rate, n_fp = _rate(
        (not bool(r.pair_real) for r in recs if r.pair_real is not None))
    m.in_lane_rate, n_in = _rate(r.in_lane for r in recs)

    fb = [(not bool(r.boundary_real), bool(r.boundary_published))
          for r in recs if r.boundary_real is not None]
    if fb:
        # a boundary is "false" when it was published and the label says
        # it is not real; frames without a published boundary carry no
        # error either way
        known = [bool(not real) for real, published in fb if published]
        m.false_boundary_rate = (float(np.mean(known)) if known else None)
        m.counts["false_boundary_frames"] = len(known)

    lat = _finite(r.line_lat_m for r in recs)
    if len(lat):
        m.line_lat_std = float(np.std(lat))
        m.counts["line_lat_frames"] = int(len(lat))
    errs = []
    for r in recs:
        if r.line_lat_m is None or r.line_lat_gt_m is None:
            continue
        try:
            a, b = float(r.line_lat_m), float(r.line_lat_gt_m)
        except (TypeError, ValueError):
            continue
        if math.isfinite(a) and math.isfinite(b):
            errs.append(abs(a - b))
    if errs:
        m.line_lat_mean_error = float(np.mean(errs))
        m.counts["line_lat_error_frames"] = len(errs)

    widths = _finite(r.lane_width_m for r in recs)
    if len(widths):
        m.lane_width_mean = float(np.mean(widths))
        m.lane_width_std = float(np.std(widths))
        m.counts["lane_width_frames"] = int(len(widths))

    for zone in ZONES:
        present = 0
        detected = 0
        for r in recs:
            if not r.zone_present.get(zone):
                continue
            present += 1
            if r.zone_detected.get(zone):
                detected += 1
        m.zone_recall[zone] = (detected / present) if present else None
        m.counts[f"{zone}_present_frames"] = present

    m.future_1s_cross_rate, n1 = _rate(r.cross_1s for r in recs)
    m.future_2s_cross_rate, n2 = _rate(r.cross_2s for r in recs)
    m.counts.update({"paired_frames": n_paired, "in_lane_frames": n_in,
                     "cross_1s_frames": n1, "cross_2s_frames": n2})
    return m


def records_from_hist(hist) -> list[LaneRecord]:
    """Adapt the drive loop's telemetry rows onto :class:`LaneRecord`.

    Only what a row can actually say is filled in: the reference fields
    (labels) stay None, so the metrics that need ground truth report None
    rather than grading the prediction against itself.
    """
    out: list[LaneRecord] = []
    for row in hist or ():
        if not isinstance(row, dict):
            continue
        lat_left = row.get("lat_left")
        lat_right = row.get("lat_right")
        in_lane = None
        if lat_left is not None or lat_right is not None:
            ok = True
            if lat_left is not None:
                ok = ok and float(lat_left) <= 0.05      # + = over the line
            if lat_right is not None:
                ok = ok and float(lat_right) >= -0.05    # - = off the road
            in_lane = bool(ok)
        speed = row.get("speed")
        first_cross = row.get("first_cross_m")
        cross_1s = cross_2s = None
        if first_cross is not None and speed is not None:
            try:
                reach_1 = float(speed) * 1.0
                reach_2 = float(speed) * 2.0
                cross_1s = bool(float(first_cross) <= reach_1)
                cross_2s = bool(float(first_cross) <= reach_2)
            except (TypeError, ValueError):
                cross_1s = cross_2s = None
        # body_cross_current is an observed crossing, not a prediction:
        # it counts as "crossed now" for both horizons when the row has
        # no planned-crossing distance at all
        if cross_1s is None and row.get("body_cross_current"):
            cross_1s = cross_2s = True
        out.append(LaneRecord(
            t=float(row.get("t") or 0.0),
            paired=(None if row.get("lane_paired") is None
                    else bool(row.get("lane_paired"))),
            lane_src=str(row.get("lane_src") or row.get("lane_sel") or ""),
            in_lane=in_lane,
            line_lat_m=(None if row.get("line_lat") is None
                        else float(row["line_lat"])),
            cross_1s=cross_1s, cross_2s=cross_2s))
    return out


def format_report(m: LaneMetrics) -> str:
    """Human-readable report; reference metrics print "n/a" when absent."""
    def _p(v, pct: bool = True, *, reference: bool = False):
        if v is None:
            # the two absences mean different things: "no labels" is a
            # missing reference, "no data" is a run that never recorded
            # the quantity at all
            return ("n/a (no labels)" if reference else "n/a (no data)")
        return f"{100.0 * v:.1f}%" if pct else f"{v:.3f}"
    lines = [f"frames: {m.frames}"]
    lines.append(f"paired_rate:            {_p(m.paired_rate)}")
    lines.append(f"false_pair_rate:        "
                 f"{_p(m.false_pair_rate, reference=True)}")
    lines.append(f"in_lane_rate:           {_p(m.in_lane_rate)}")
    lines.append(f"false_boundary_rate:    "
                 f"{_p(m.false_boundary_rate, reference=True)}")
    lines.append(f"line_lat_mean_error:    "
                 f"{_p(m.line_lat_mean_error, reference=True, pct=False)} m")
    lines.append(f"line_lat_std:           {_p(m.line_lat_std, False)} m")
    lines.append(f"lane_width_mean:        {_p(m.lane_width_mean, False)} m")
    lines.append(f"lane_width_std:         {_p(m.lane_width_std, False)} m")
    for zone in ZONES:
        lines.append(f"{zone}_line_recall:        "
                     f"{_p(m.zone_recall.get(zone), reference=True)}")
    lines.append(f"future_1s_cross_rate:   {_p(m.future_1s_cross_rate)}")
    lines.append(f"future_2s_cross_rate:   {_p(m.future_2s_cross_rate)}")
    return "\n".join(lines)
