"""Offline tests for the FSD telemetry evaluator (no game needed)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.eval import assess_many, assess_run


def _hist(n: int = 10, lat_left: float = -2.0, lat_right: float = 2.0,
          speed: float = 5.0, rem_end=50.0, reversing: int = 0,
          level: str = "safe", source: str = "fsd", road_off: float = 0.0,
          throttle: float = 0.3, brake: float = 0.0,
          lane_sel: str = "sensor", lane_paired: int = 1):
    out = []
    for i in range(n):
        out.append({
            "t": float(i),
            "pos": [float(i), 0.0, 0.0],
            "heading": 0.0,
            "speed": speed,
            "source": source,
            "level": level,
            "reversing": reversing,
            "stuck": 0,
            "emergency": 0,
            "lat_left": lat_left,
            "lat_right": lat_right,
            "road_off": road_off,
            "rem_end": rem_end,
            "throttle": throttle,
            "brake": brake,
            "plan_speed": 6.0,
            "target_sm": 6.0,
            "lane_dev_m": 0.0,
            "lane_sel": lane_sel,
            "lane_paired": lane_paired,
        })
    return out


def test_clean_run() -> None:
    r = assess_run(_hist(), goal=(9.0, 0.0), cruise=6.0)
    assert r["frames"] == 10
    assert r["reversing_frames"] == 0
    assert r["cross_centre_frames"] == 0
    assert r["cross_right_frames"] == 0
    assert r["off_road_frames"] == 0
    assert r["stall_frames"] == 0
    assert r["goal_dist_m"] == pytest.approx(0.0, abs=1e-6)
    assert r["speed_max"] == 5.0


def test_crossing_and_off_road_detected() -> None:
    h = _hist(lat_left=0.4)          # + = inside the oncoming lane
    h[3]["lat_right"] = -0.3         # - = beyond the right road edge
    h[4]["road_off"] = 1.2           # off-road
    r = assess_run(h)
    assert r["cross_centre_frames"] == 10
    assert r["cross_right_frames"] == 1
    assert r["off_road_frames"] == 1
    assert r["max_cross_centre_m"] == pytest.approx(0.4)


def test_reversing_and_stalls() -> None:
    h = _hist(speed=0.3, rem_end=20.0)   # stalled away from the end zone
    h[1]["reversing"] = 1
    r = assess_run(h)
    assert r["stall_frames"] == 10
    assert r["reversing_frames"] == 1


def test_end_zone_stop_not_a_stall() -> None:
    # stopping inside the end zone (rem_end < 8) is arrival, not a stall
    r = assess_run(_hist(speed=0.2, rem_end=2.0))
    assert r["stall_frames"] == 0


def test_assess_many() -> None:
    rs = assess_many([_hist(), _hist(lat_left=0.15)], cruise=6.0)
    assert len(rs) == 2
    assert rs[1]["cross_centre_frames"] == 10


def test_off_road_reports_duration_magnitude_and_episodes() -> None:
    """Duration and magnitude, not only frames (plan P0-4).

    Frames at t = 0..9 s (1 s apart), so the sampled window is 9 s and a
    two-frame excursion at frames 4-5 lasts 2.0 s.
    """
    h = _hist()
    h[4]["road_off"] = 0.8
    h[5]["road_off"] = 1.2
    r = assess_run(h)
    assert r["off_road_frames"] == 2
    assert r["off_road_s"] == pytest.approx(2.0)
    assert r["off_road_longest_s"] == pytest.approx(2.0)
    assert r["off_road_episodes"] == 1
    assert r["off_road_max_m"] == pytest.approx(1.2)
    assert r["off_road_src"] == "road_off"
    assert r["off_road_measured"] is True
    assert r["settled_duration_s"] == pytest.approx(9.0)
    assert r["off_road_frac"] == pytest.approx(2.0 / 9.0, abs=1e-3)


def test_two_off_road_excursions_are_two_episodes() -> None:
    h = _hist()
    h[1]["road_off"] = 0.5
    h[2]["road_off"] = 0.5
    h[7]["road_off"] = 0.5
    r = assess_run(h)
    assert r["off_road_frames"] == 3
    assert r["off_road_episodes"] == 2
    assert r["off_road_longest_s"] == pytest.approx(2.0)
    assert r["off_road_s"] == pytest.approx(3.0)


def test_stall_reports_duration_and_episodes() -> None:
    h = _hist(speed=0.2, rem_end=50.0)
    for i in (3, 4, 5):
        h[i]["speed"] = 4.0              # one 3 s gap in the stall
    r = assess_run(h)
    assert r["stall_frames"] == 7
    assert r["stall_events"] == 2
    assert r["stall_s"] == pytest.approx(6.0)
    assert r["stall_longest_s"] == pytest.approx(3.0)
    assert r["stall_frac"] == pytest.approx(6.0 / 9.0, abs=1e-3)


def test_a_full_run_stall_never_exceeds_the_sampled_window() -> None:
    """Durations are bounded by the evidence: [t0, t_last] is 9 s here."""
    r = assess_run(_hist(speed=0.0, rem_end=50.0))
    assert r["stall_s"] == pytest.approx(r["settled_duration_s"])
    assert r["stall_s"] <= 9.0
    assert r["stall_frac"] == pytest.approx(1.0)


def test_missing_lateral_channel_is_none_not_zero() -> None:
    """No lateral samples means UNMEASURED, not "perfectly centred"."""
    h = _hist()
    for row in h:
        row.pop("lat_left", None)
        row.pop("lat_right", None)
    r = assess_run(h)
    assert r["lat_frames"] == 0
    assert r["max_cross_centre_m"] is None
    assert r["max_cross_right_m"] is None


def test_lane_continuity_counts_the_perception_lane() -> None:
    """The stall/creep metrics need the perception rate that causes them."""
    h = _hist(n=10)
    for i in range(6):
        h[i]["lane_sel"] = "perception-unavailable"
        h[i]["lane_paired"] = 0
    r = assess_run(h)
    assert r["lane_sensor_frames"] == 4
    assert r["lane_sensor_rate"] == pytest.approx(0.4)
    assert r["lane_paired_frames"] == 4
    assert r["lane_paired_rate"] == pytest.approx(0.4)
    assert r["lane_src_hist"] == {"perception-unavailable": 6, "sensor": 4}


def test_lane_continuity_falls_back_to_lane_src() -> None:
    """Older telemetry has no ``lane_sel``: read ``lane_src`` instead."""
    h = _hist(n=4)
    for f in h:
        del f["lane_sel"]
        f["lane_src"] = "bev/route"
    r = assess_run(h)
    assert r["lane_sensor_frames"] == 0
    assert r["lane_src_hist"] == {"bev/route": 4}


def test_lane_continuity_respects_the_settle_window() -> None:
    h = _hist(n=10)
    for i in range(5):
        h[i]["lane_sel"] = "perception-unavailable"
    r = assess_run(h, settle_s=5.0)
    assert r["settled_frames"] == 5
    assert r["lane_sensor_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# collision counting (plan §12 gate #1)
# ---------------------------------------------------------------------------

def _hist_with_damage(values):
    return [{"t": float(i), "speed": 3.0, "pos": [float(i), 0.0],
             "damage_total": v} for i, v in enumerate(values)]


def test_collision_events_counts_damage_increases() -> None:
    from beamng_autopilot.eval import collision_events
    rep = collision_events(_hist_with_damage([0.0, 0.0, 0.05, 0.05, 0.2]))
    assert rep["collision_count"] == 2
    assert rep["damage_frames"] == 5
    assert rep["damage_total"] == pytest.approx(0.2)
    assert rep["first_collision_t"] == pytest.approx(2.0)


def test_collision_events_ignores_subthreshold_noise() -> None:
    from beamng_autopilot.eval import collision_events
    rep = collision_events(_hist_with_damage([0.0, 0.001, 0.002]))
    assert rep["collision_count"] == 0
    rep2 = collision_events(_hist_with_damage([0.0, 0.001]),
                            min_delta=0.0005)
    assert rep2["collision_count"] == 1


def test_collision_count_is_none_when_not_measured() -> None:
    """"Not measured" must never read as "no collisions"."""
    from beamng_autopilot.eval import collision_events
    hist = [{"t": 0.0, "speed": 1.0}, {"t": 1.0, "speed": 1.0}]
    rep = collision_events(hist)
    assert rep["collision_count"] is None
    assert rep["damage_frames"] == 0
    assert collision_events([])["collision_count"] is None


def test_assess_run_reports_the_collision_fields() -> None:
    from beamng_autopilot.eval import assess_run
    out = assess_run(_hist_with_damage([0.0, 0.4]), settle_s=0.0)
    assert out["collision_count"] == 1
    out2 = assess_run([{"t": 0.0, "speed": 2.0, "pos": [0.0, 0.0]}],
                      settle_s=0.0)
    assert out2["collision_count"] is None


def test_worst_frames_ranks_measured_frames_and_skips_none():
    """判定要能"点进具体帧"：逐帧排序时，没有分母的帧不进榜（那是"没得比"）。"""
    from scripts.m5_seg_eval_matrix import worst_frames
    per = [{"frame": "a.npz", "iou": 0.9, "gt_px": 10},
           {"frame": "b.npz", "iou": 0.1, "gt_px": 20},
           {"frame": "c.npz", "iou": None, "gt_px": 0},
           {"frame": "d.npz", "iou": 0.5, "gt_px": 30}]
    r = worst_frames(per, k=2)
    assert r["n"] == 3 and r["n_no_denominator"] == 1
    assert [x["frame"] for x in r["worst"]] == ["b.npz", "d.npz"]
    assert [x["frame"] for x in r["best"]] == ["a.npz", "d.npz"]
    assert r["worst"][0]["gt_px"] == 20
