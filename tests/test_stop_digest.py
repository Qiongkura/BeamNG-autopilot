"""P0-1: one stop definition, and missing columns stay UNKNOWN."""

from __future__ import annotations

import pytest

from beamng_autopilot.eval import (
    CREEP_SPEED_MPS,
    STOP_SPEED_MPS,
    stop_digest,
)


def _run(n: int | None = None, dt: float = 1.0, **cols):
    """Frames at t=0,1,2...; every column may be overridden per frame.

    ``n`` defaults to the longest supplied column so a caller can pass a
    four-frame speed list without repeating the length.
    """
    if n is None:
        n = max([len(v) for v in cols.values()] or [6])
    out = []
    for i in range(n):
        row = {"t": i * dt}
        for key, val in cols.items():
            row[key] = val(i) if callable(val) else val[i]
        out.append(row)
    return out


def test_the_spec_states_every_threshold():
    d = stop_digest(_run(speed=[0.0] * 6), settle_s=0.0)
    spec = d["spec"]
    assert spec["stop"] == f"speed < {STOP_SPEED_MPS} m/s"
    assert f"{STOP_SPEED_MPS} <= speed < {CREEP_SPEED_MPS} m/s" == spec["creep"]
    assert "interval to the next FRAME" in spec["interval_rule"]
    assert "final_stop" in spec["hard_stop"]


def test_stop_and_creep_are_separate_categories():
    speeds = [0.0, 0.2, 0.35, 0.9, 1.5, 0.0]
    d = stop_digest(_run(speed=speeds), settle_s=0.0)
    assert d["stop_frames"] == 3          # 0.0, 0.2, 0.0
    assert d["creep_frames"] == 2         # 0.35, 0.9
    assert d["speed_lt_0_3_frames"] == 3
    assert d["speed_lt_0_1_frames"] == 2


def test_settle_window_is_excluded_from_counts_but_not_from_raw_frames():
    d = stop_digest(_run(n=6, speed=[0.0] * 6), settle_s=2.0)
    assert d["raw_frames"] == 6
    assert d["settled_frames"] == 4
    assert d["stop_frames"] == 4


def test_duration_uses_the_frame_interval_rule():
    # stop at t=0,1 then moving at t=2: the stop owns [0,1) and [1,2) = 2 s
    d = stop_digest(_run(n=4, speed=[0.0, 0.0, 2.0, 2.0]), settle_s=0.0)
    assert d["stop_s"] == 2.0
    assert d["stop_longest_s"] == 2.0
    assert d["stop_episodes"] == 1


def test_hard_stop_is_reported_separately_from_speed():
    d = stop_digest(_run(speed=[0.0] * 4, final_stop=[True, False, True, False],
                         emergency=[0, 0, 1, 0],
                         level=["minimal_risk", "safe", "safe",
                                "minimal_risk"]),
                    settle_s=0.0)
    assert d["final_stop_frames"] == 2
    assert d["emergency_frames"] == 1
    # union: frame 0 (final_stop) + frame 2 (final_stop/emergency)
    # + frame 3 (minimal_risk level) = 3
    assert d["hard_stop_frames"] == 3


def test_end_zone_stop_is_split_out_explicitly():
    d = stop_digest(_run(speed=[0.0] * 4, rem_end=[50.0, 50.0, 2.0, 2.0]),
                    settle_s=0.0, end_zone_m=8.0)
    assert d["stop_frames"] == 4
    assert d["excl_end_zone"]["frames"] == 2
    assert d["excl_end_zone"]["end_zone_m"] == 8.0


def test_by_reason_reports_seconds_per_cause():
    d = stop_digest(
        _run(speed=[0.0] * 4,
             effective_rule=["no_drivable_path", "no_drivable_path",
                             "obstacle_risk", ""],
             reason=["no drivable path", "no drivable path",
                     "obstacle contact risk", "path hold (creep)"]),
        settle_s=0.0)
    assert d["by_reason"]["no_drivable_path"]["frames"] == 2
    assert d["by_reason"]["no_drivable_path"]["s"] == 2.0
    assert d["by_reason"]["obstacle_risk"]["s"] == 1.0
    # an empty effective_rule falls back to the reason text
    assert "path hold (creep)" in d["by_reason"]


def test_a_missing_speed_column_is_unknown_not_zero():
    d = stop_digest(_run(n=3), settle_s=0.0)      # no speed at all
    assert d["stop_frames"] is None
    assert d["stop_s"] is None
    assert d["speed_lt_0_3_frames"] is None
    assert "speed" in d["unknown"]


def test_a_missing_flag_column_is_unknown_not_zero():
    d = stop_digest(_run(n=3, speed=[0.0] * 3), settle_s=0.0)
    assert d["final_stop_frames"] is None
    assert d["emergency_frames"] is None
    assert "final_stop" in d["unknown"] and "emergency" in d["unknown"]


def test_a_missing_rem_end_makes_the_end_zone_split_unknown():
    d = stop_digest(_run(n=3, speed=[0.0] * 3), settle_s=0.0)
    assert d["excl_end_zone"] is None
    assert "rem_end" in d["unknown"]


def test_boolean_flags_are_counted_not_treated_as_missing():
    """A flag column read as numeric silently produced 0 violations."""
    d = stop_digest(_run(n=3, speed=[0.0] * 3,
                         final_stop=[True, True, False]), settle_s=0.0)
    assert d["final_stop_frames"] == 2
    assert "final_stop" not in d["unknown"]


def test_ref_stability_digest_counts_flips_jumps_and_authority():
    """P1-2 acceptance: flips, centre jumps, authority and provenance."""
    from scripts.m5_run_metrics import ref_stability_digest

    hist = [
        {"t": 0.0, "ref_lat_m": -1.5, "ref_authority": "limited",
         "ref_stable_ticks": 1, "ref_side_flips": 0, "lane_src_sel": "sensor"},
        {"t": 1.0, "ref_lat_m": -1.6, "ref_authority": "full",
         "ref_stable_ticks": 2, "ref_side_flips": 0, "lane_src_sel": "sensor",
         "lane_paired": 1, "pair_paired": 1},
        {"t": 2.0, "ref_lat_m": 1.4, "ref_authority": "limited",
         "ref_stable_ticks": 1, "ref_side_flips": 1, "ref_flip": 1,
         "lane_src_sel": "perception-unavailable"},
        {"t": 3.0, "lane_src_sel": "sensor", "lane_from": "divider_right_shift"},
    ]
    d = ref_stability_digest(hist)
    assert d["frames"] == 4
    assert d["side_flip_frames"] == 1
    assert d["ref_side_flips_total"] == 1
    assert d["centre_jump_m"]["n"] == 2          # the gap restarts the pairs
    assert d["centre_jump_m"]["max"] == pytest.approx(3.0)
    assert d["authority_frames"] == {"limited": 2, "full": 1, "absent": 1}
    assert d["lane_paired_frames"] == 1
    assert d["pair_paired_frames"] == 1
    assert d["provenance_frames"]["perception-unavailable"] == 1
    assert d["provenance_frames"]["sensor|divider_right_shift"] == 1


def test_ref_stability_digest_reports_unknown_not_zero_without_columns():
    from scripts.m5_run_metrics import ref_stability_digest

    d = ref_stability_digest([{"t": 0.0}, {"t": 1.0}])
    assert d["centre_jump_m"]["n"] == 0
    assert d["centre_jump_m"]["p50"] is None
    assert d["authority_frames"] == {"absent": 2}
