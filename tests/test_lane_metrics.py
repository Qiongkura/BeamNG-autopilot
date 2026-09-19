"""Lane evaluation metrics (plan phase E6)."""

from __future__ import annotations

import json

import pytest

from beamng_autopilot.lane.metrics import (
    ZONES,
    LaneMetrics,
    LaneRecord,
    compute_lane_metrics,
    format_report,
    records_from_hist,
)


# ---------------------------------------------------------------------------
# observational metrics
# ---------------------------------------------------------------------------

def test_paired_and_in_lane_rates() -> None:
    recs = [LaneRecord(paired=True, in_lane=True),
            LaneRecord(paired=True, in_lane=True),
            LaneRecord(paired=False, in_lane=False),
            LaneRecord(paired=False)]
    m = compute_lane_metrics(recs)
    assert m.paired_rate == pytest.approx(0.5)
    assert m.in_lane_rate == pytest.approx(2.0 / 3.0)
    assert m.counts["in_lane_frames"] == 3


def test_line_lat_std_and_width_spread() -> None:
    recs = [LaneRecord(line_lat_m=v, lane_width_m=w)
            for v, w in ((-1.0, 3.5), (-0.5, 3.6), (-0.75, 3.4))]
    m = compute_lane_metrics(recs)
    assert m.line_lat_std == pytest.approx(0.2041, abs=1e-3)
    assert m.lane_width_mean == pytest.approx(3.5, abs=1e-6)
    assert m.lane_width_std == pytest.approx(0.0816, abs=1e-3)


def test_cross_rates() -> None:
    recs = [LaneRecord(cross_1s=True, cross_2s=True),
            LaneRecord(cross_1s=False, cross_2s=True),
            LaneRecord(cross_1s=False, cross_2s=False),
            LaneRecord()]
    m = compute_lane_metrics(recs)
    assert m.future_1s_cross_rate == pytest.approx(1 / 3)
    assert m.future_2s_cross_rate == pytest.approx(2 / 3)


def test_empty_input_is_all_none_not_zero() -> None:
    m = compute_lane_metrics([])
    assert m.frames == 0
    for name in ("paired_rate", "in_lane_rate", "line_lat_std",
                 "lane_width_mean", "future_1s_cross_rate"):
        assert getattr(m, name) is None


# ---------------------------------------------------------------------------
# reference metrics need labels - and say so when they have none
# ---------------------------------------------------------------------------

def test_reference_metrics_are_none_without_labels() -> None:
    """The whole point of E6: never grade the prediction against itself."""
    recs = [LaneRecord(paired=True, line_lat_m=-1.0, lane_width_m=3.5)
            for _ in range(5)]
    m = compute_lane_metrics(recs)
    assert m.false_pair_rate is None
    assert m.false_boundary_rate is None
    assert m.line_lat_mean_error is None
    assert all(m.zone_recall[z] is None for z in ZONES)
    report = format_report(m)
    # the two absences must not be confused: a missing reference says so,
    # a quantity the run never recorded says "no data"
    # 3 reference metrics + 3 zone recalls lack labels; the observational
    # metrics this fixture does not record (in-lane, both cross horizons)
    # say "no data" instead - the two absences must stay distinguishable
    assert report.count("n/a (no labels)") == 6
    assert report.count("n/a (no data)") == 3
    assert "line_lat_std:           0.000 m" in report


def test_false_pair_rate_uses_the_labels() -> None:
    recs = [LaneRecord(paired=True, pair_real=True),
            LaneRecord(paired=True, pair_real=False),
            LaneRecord(paired=False, pair_real=False)]
    m = compute_lane_metrics(recs)
    assert m.false_pair_rate == pytest.approx(2 / 3)


def test_false_boundary_rate_counts_only_published_boundaries() -> None:
    recs = [LaneRecord(boundary_published=True, boundary_real=True),
            LaneRecord(boundary_published=True, boundary_real=False),
            # a frame with no published boundary cannot be a false one
            LaneRecord(boundary_published=False, boundary_real=True),
            LaneRecord(boundary_published=False, boundary_real=False)]
    m = compute_lane_metrics(recs)
    assert m.false_boundary_rate == pytest.approx(0.5)
    assert m.counts["false_boundary_frames"] == 2


def test_line_lat_error_needs_both_sides() -> None:
    recs = [LaneRecord(line_lat_m=-1.0, line_lat_gt_m=-0.8),
            LaneRecord(line_lat_m=-0.5, line_lat_gt_m=-0.7),
            LaneRecord(line_lat_m=-1.0)]        # no reference
    m = compute_lane_metrics(recs)
    assert m.line_lat_mean_error == pytest.approx(0.2)
    assert m.counts["line_lat_error_frames"] == 2


def test_zone_recall_per_zone() -> None:
    recs = [
        LaneRecord(zone_present={"near": True, "mid": True, "far": False},
                   zone_detected={"near": True, "mid": False, "far": False}),
        LaneRecord(zone_present={"near": True},
                   zone_detected={"near": True}),
    ]
    m = compute_lane_metrics(recs)
    assert m.zone_recall["near"] == pytest.approx(1.0)
    assert m.zone_recall["mid"] == pytest.approx(0.0)
    assert m.zone_recall["far"] is None          # nothing was present
    assert m.counts["far_present_frames"] == 0


def test_digest_is_json_safe() -> None:
    m = compute_lane_metrics([LaneRecord(paired=True, in_lane=True)])
    text = json.dumps(m.digest())
    assert "paired_rate" in text and "nan" not in text.lower()


# ---------------------------------------------------------------------------
# telemetry adapter
# ---------------------------------------------------------------------------

def _row(**kw):
    base = {"t": 1.0, "lane_paired": 1, "line_lat": -0.9, "speed": 6.0,
            "lat_left": -1.2, "lat_right": -1.4}
    base.update(kw)
    return base


def test_adapter_maps_the_recorded_evidence() -> None:
    recs = records_from_hist([_row(), _row(lane_paired=0)])
    assert len(recs) == 2
    assert recs[0].paired is True and recs[1].paired is False
    assert recs[0].line_lat_m == pytest.approx(-0.9)
    # inside the lane: left of the line (lat_left <= 0) and inside the
    # right boundary (lat_right >= 0 - here -1.4 means... negative is off
    # the edge, so this frame is OUTSIDE)
    assert recs[0].in_lane is False
    recs2 = records_from_hist([_row(lat_left=-1.2, lat_right=0.4)])
    assert recs2[0].in_lane is True


def test_adapter_leaves_reference_fields_empty() -> None:
    recs = records_from_hist([_row()])
    assert recs[0].boundary_real is None
    assert recs[0].line_lat_gt_m is None
    assert recs[0].zone_present == {}


def test_adapter_derives_the_cross_horizons_from_the_crossing_distance() -> None:
    # 3 m ahead at 6 m/s: inside 1 s (6 m) and 2 s (12 m)
    recs = records_from_hist([_row(first_cross_m=3.0, speed=6.0)])
    assert recs[0].cross_1s is True and recs[0].cross_2s is True
    # 8 m ahead at 6 m/s: inside 2 s only
    recs2 = records_from_hist([_row(first_cross_m=8.0, speed=6.0)])
    assert recs2[0].cross_1s is False and recs2[0].cross_2s is True
    # an observed current crossing counts for both horizons
    recs3 = records_from_hist([_row(first_cross_m=None,
                                    body_cross_current=1)])
    assert recs3[0].cross_1s is True and recs3[0].cross_2s is True


def test_adapter_tolerates_junk_rows() -> None:
    recs = records_from_hist([None, "not a row", {}, _row()])
    assert len(recs) == 2          # the two dicts; junk is skipped
    assert isinstance(compute_lane_metrics(recs), LaneMetrics)
