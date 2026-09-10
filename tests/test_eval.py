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
