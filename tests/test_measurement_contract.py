"""T01: measurement contract and scoring must not clear what was not measured.

The plan lists four defects here (FINAL_INTEGRATED_PLAN §2.5), all of them
"a field reads as a clean result because its column was missing":

1. a run with no lateral samples at all still cleared the crossing checks;
2. one missing side was reported as "no crossing" for BOTH sides;
3. ``by_reason[*].s`` is reason-ACTIVE time, not stop time, and was quoted
   as stop seconds;
4. the field table described ``lat_left/right`` as body-corner distances
   while the producer measures the vehicle centre point.

These tests pin the fixes at the level a reviewer consumes (assess/score),
not at the level of the function that happens to compute the number.
"""

from __future__ import annotations

import pytest

from beamng_autopilot.eval import assess_run, score_run, stop_digest
from beamng_autopilot.telemetry_contract import LATERAL_FIELD_SPECS


def _rows(n=10, **over):
    rows = []
    for i in range(n):
        row = {
            "t": i * 0.5, "pos": [float(i), 0.0, 0.0], "speed": 3.0,
            "reversing": 0, "stuck": 0, "emergency": 0,
            "lat_left": -1.75, "lat_right": 1.75, "road_off": 0.0,
            "rem_end": 100.0, "damage_total": 0.0,
        }
        row.update(over)
        rows.append(row)
    return rows


class TestMissingLateralNeverPasses:
    def test_a_run_without_lateral_columns_is_not_a_clean_run(self):
        rows = _rows()
        for r in rows:
            r.pop("lat_left")
            r.pop("lat_right")
        a = assess_run(rows)
        v = score_run(a)
        assert v["pass"] is False
        assert v["status"] == "UNKNOWN"
        assert "no_centre_crossing" in v["unknown"]
        assert "no_edge_crossing" in v["unknown"]

    def test_one_missing_side_only_unknowns_that_side(self):
        """The measured side still reports; the missing one is UNKNOWN."""
        rows = _rows()
        for r in rows:
            r.pop("lat_left")           # centre/oncoming side unmeasured
        a = assess_run(rows)
        assert a["lat_left_frames"] == 0
        assert a["lat_right_frames"] > 0
        v = score_run(a)
        assert "no_centre_crossing" in v["unknown"]
        assert "no_edge_crossing" not in v["unknown"]
        assert v["checks"]["no_edge_crossing"] is True

    def test_side_coverage_is_published(self):
        a = assess_run(_rows())
        assert a["lat_left_frames"] > 0 and a["lat_right_frames"] > 0
        assert a["lat_left_coverage"] == pytest.approx(1.0)
        assert a["lat_right_coverage"] == pytest.approx(1.0)

    def test_a_measured_crossing_still_fails(self):
        """The positive control: the gate must not have gone vacuous."""
        a = assess_run(_rows(lat_left=0.5))
        v = score_run(a)
        assert v["checks"]["no_centre_crossing"] is False
        assert v["pass"] is False
        assert "no_centre_crossing" not in v["unknown"]


class TestStopReasonAccounting:
    def _hist(self, speeds, rules):
        return [{"t": 0.5 * i, "pos": [float(i), 0.0, 0.0], "speed": sp,
                 "effective_rule": ru, "reason": ru, "rem_end": 100.0,
                 "damage_total": 0.0}
                for i, (sp, ru) in enumerate(zip(speeds, rules))]

    def test_active_time_is_not_stop_time(self):
        # rule active the whole run; the car only stops in the middle
        speeds = [3.0, 3.0, 0.2, 0.2, 0.2, 3.0, 3.0, 3.0]
        rules = ["blocked"] * len(speeds)
        d = stop_digest(self._hist(speeds, rules), settle_s=0.0)
        slot = d["by_reason"]["blocked"]
        assert slot["s"] == pytest.approx(slot["active_s"])
        assert slot["stop_s"] < slot["active_s"]
        assert slot["stop_frames"] == 3
        assert slot["stop_longest_s"] < slot["longest_s"]

    def test_a_reason_that_never_stopped_has_zero_stop_seconds(self):
        speeds = [3.0] * 8
        d = stop_digest(self._hist(speeds, ["cruising"] * 8), settle_s=0.0)
        slot = d["by_reason"]["cruising"]
        assert slot["stop_s"] == 0.0
        assert slot["stop_frames"] == 0
        assert slot["active_s"] > 0.0

    def test_the_spec_describes_both_numbers(self):
        d = stop_digest(self._hist([3.0] * 8, ["x"] * 8), settle_s=0.0)
        fields = d["spec"]["by_reason_fields"]
        assert "active" in fields and "stop_frames" in fields


class TestFieldContract:
    def test_lat_left_is_a_centre_point_measurement(self):
        """Plan §2.5-1: the table used to say "body corners" for a field the
        producer computes from the vehicle centre point."""
        for name in ("lat_left", "lat_right"):
            spec = LATERAL_FIELD_SPECS[name]
            assert "centre point" in spec["frame"], spec["frame"]
            assert "corner" not in spec["frame"]
        for name in ("body_lat_left", "body_lat_right"):
            assert "corner" in LATERAL_FIELD_SPECS[name]["reference"].lower()

    def test_lane_dev_is_metric_not_dimensionless(self):
        """Plan §2.5-2: it is a metre-valued median distance."""
        spec = LATERAL_FIELD_SPECS["lane_dev_m"]
        assert spec["unit"] == "m"
        assert "unsigned" in spec["sign"].lower()


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
