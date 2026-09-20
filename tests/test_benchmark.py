"""Offline tests for the benchmark scoring layer (eval.score_run + script)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import math

import numpy as np
import pytest

from beamng_autopilot.eval import (
    assess_run,
    collision_events,
    score_many,
    score_run,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m5_fsd_benchmark.py"
_spec = importlib.util.spec_from_file_location("m5_fsd_benchmark", _SCRIPT)
m5_fsd_benchmark = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("m5_fsd_benchmark", m5_fsd_benchmark)
_spec.loader.exec_module(m5_fsd_benchmark)


def _hist(n=10, **over):
    rows = []
    for i in range(n):
        row = {
            "t": i * 0.5, "pos": [float(i), 0.0, 0.0], "speed": 3.0,
            "reversing": 0, "stuck": 0, "emergency": 0,
            "lat_left": -1.75, "lat_right": 1.75, "road_off": 0.0,
            "rem_end": 100.0, "throttle": 0.3, "brake": 0.0,
            # A CLEAN run must still have measured damage: the collision
            # gate is UNKNOWN without it (see the missing-channel tests),
            # so "clean" and "unmeasured" are different fixtures on
            # purpose.
            "damage_total": 0.0,
        }
        row.update(over)
        rows.append(row)
    return rows


def _hist_no_damage(n=10, **over):
    """Same run with the damage channel absent (an old log)."""
    rows = _hist(n, **over)
    for r in rows:
        r.pop("damage_total", None)
    return rows


def test_score_run_passes_clean_run() -> None:
    a = assess_run(_hist())
    v = score_run(a)
    assert v["status"] == "PASS"
    assert v["pass"] is True
    assert v["unknown"] == []
    assert all(v["checks"].values())


def test_unmeasured_collision_is_unknown_not_pass() -> None:
    """A run with no damage channel cannot clear the §12 collision gate.

    This is the hole the old verdict had: ``no_collision`` was not a check
    at all, so a run that never sampled damage passed on the other five.
    """
    a = assess_run(_hist_no_damage())
    assert a["collision_count"] is None
    v = score_run(a)
    assert v["checks"]["no_collision"] is False
    assert v["unknown"] == ["no_collision"]
    assert v["status"] == "UNKNOWN"
    assert v["pass"] is False


def test_confirmed_collision_fails() -> None:
    a = assess_run(_hist(damage_total=0.0))
    a["collision_count"] = 1          # a measured impact
    v = score_run(a)
    assert v["checks"]["no_collision"] is False
    assert "no_collision" not in v["unknown"]
    assert v["status"] == "FAIL"


def test_a_violation_outranks_an_unknown() -> None:
    """A confirmed failure is FAIL even when something else is unmeasured."""
    a = assess_run(_hist_no_damage(reversing=1))
    v = score_run(a)
    assert v["status"] == "FAIL"
    assert "no_collision" in v["unknown"]


def test_unmeasured_off_road_is_not_on_road() -> None:
    rows = _hist()
    for r in rows:
        r.pop("road_off", None)       # no off-road source at all
    a = assess_run(rows)
    assert a["off_road_frames"] is None
    assert a["off_road_measured"] is False
    assert a["off_road_s"] is None
    v = score_run(a)
    assert v["checks"]["on_road"] is False
    assert "on_road" in v["unknown"]


def test_score_many_reports_unknown_and_collision_ratio() -> None:
    good = assess_run(_hist())
    unknown = assess_run(_hist_no_damage())
    agg = score_many([good, unknown])
    assert agg["status"] == "UNKNOWN"
    assert agg["pass"] is False
    assert agg["n_pass"] == 1 and agg["n_unknown"] == 1
    assert agg["n_collided"] == 0
    assert agg["collision_run_ratio"] == 0.0
    # nothing measured at all -> the ratio is unknown, not zero
    agg2 = score_many([unknown, unknown])
    assert agg2["n_collided"] is None
    assert agg2["collision_run_ratio"] is None


def test_collision_episodes_dedupe_one_impact() -> None:
    """Damage rising on consecutive frames is ONE impact, not three."""
    rows = _hist(6)
    for i, d in enumerate([0.0, 0.0, 0.05, 0.09, 0.14, 0.14]):
        rows[i]["damage_total"] = d
    rep = collision_events(rows)
    assert rep["collision_count"] == 3        # three rising frames
    assert rep["collision_episodes"] == 1     # one physical impact
    assert rep["collided"] is True
    # two impacts two seconds apart stay two
    rows2 = _hist(10)
    for i, d in enumerate([0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05,
                           0.05, 0.30, 0.30]):
        rows2[i]["damage_total"] = d
    rep2 = collision_events(rows2)
    assert rep2["collision_count"] == 2
    assert rep2["collision_episodes"] == 2


def test_score_run_fails_each_hard_target() -> None:
    cases = [
        ({"reversing": 1}, "no_reversing"),
        ({"lat_left": 0.5}, "no_centre_crossing"),
        ({"lat_right": -0.5}, "no_edge_crossing"),
        ({"road_off": 0.5}, "on_road"),
        ({"speed": 0.1, "rem_end": 50.0}, "no_stall"),
    ]
    for over, key in cases:
        a = assess_run(_hist(**over))
        v = score_run(a)
        assert v["pass"] is False, over
        assert v["checks"][key] is False, over


def test_score_run_empty_hist_never_passes() -> None:
    v = score_run(assess_run([]))
    assert v["pass"] is False
    assert v["checks"]["has_frames"] is False


def test_score_run_goal_tolerance() -> None:
    # final pos is (4.5, 0): goal 5.5 m away passes, 100 m away fails
    a = assess_run(_hist(), goal=(10.0, 0.0))
    assert score_run(a, require_goal=True)["checks"]["reached_goal"] is True
    a2 = assess_run(_hist(), goal=(200.0, 0.0))
    v = score_run(a2, require_goal=True)
    assert v["pass"] is False
    assert v["checks"]["reached_goal"] is False


def test_score_many_requires_every_run() -> None:
    good = assess_run(_hist())
    bad = assess_run(_hist(reversing=1))
    agg = score_many([good, bad])
    assert agg["pass"] is False and agg["n_pass"] == 1
    agg2 = score_many([good, good])
    assert agg2["pass"] is True and agg2["n_pass"] == 2
    assert agg2["runs"][0]["pass"] is True


def test_scenario_args_override_and_namespace_complete() -> None:
    ns = m5_fsd_benchmark.scenario_args(
        "mountain", {"strict": True, "lane_mode": "sensor",
                     "goal": [1.0, 2.0]},
        Path("out.json"))
    # scenario layer wins
    assert ns.seconds == 90.0 and ns.speed == 6.0
    assert ns.teleport == [729.6, 763.9, 45.0]
    assert ns.out == "out.json"
    # CLI layer kept
    assert ns.strict is True and ns.lane_mode == "sensor"
    assert ns.goal == [1.0, 2.0]
    # every fsd_drive.run argument is present
    for key in m5_fsd_benchmark._DRIVE_ARG_DEFAULTS:
        assert hasattr(ns, key), key


def test_score_telemetry_roundtrip(tmp_path) -> None:
    p = tmp_path / "run.json"
    rows = _hist(24, lat_left=0.5)   # t = 0 .. 11.5 s
    p.write_text(__import__("json").dumps(rows), encoding="utf-8")
    r = m5_fsd_benchmark.score_telemetry(p, require_goal=False)
    assert r["pass"] is False
    # settle_s=8.0 mirrors the drive's WARMUP_S: the first 16 frames
    # (t < 8) are excluded, the remaining 8 are real violations
    assert r["assessed"]["settled_frames"] == 8
    assert r["assessed"]["cross_centre_frames"] == 8


def test_settle_window_excludes_spawn_transient() -> None:
    # a violation in the first 3 s counts with settle_s=0 but is
    # excluded with the benchmark's settle_s=3.0
    rows = _hist(12)
    rows[2]["lat_right"] = -0.5          # t = 1.0 s: spawn transient
    a0 = assess_run(rows)
    a3 = assess_run(rows, settle_s=3.0)
    assert a0["cross_right_frames"] == 1
    assert a3["cross_right_frames"] == 0
    assert a3["settled_frames"] == len(rows) - sum(
        1 for r in rows if r["t"] < 3.0)
    # score flips on the settle window alone
    assert score_run(a0)["pass"] is False
    assert score_run(a3)["pass"] is True


def test_score_run_requires_settled_frames() -> None:
    a = assess_run(_hist(2), settle_s=3.0)   # both frames inside window
    assert a["frames"] == 2 and a["settled_frames"] == 0
    assert score_run(a)["pass"] is False


def test_body_aware_crossing_catches_yawed_car() -> None:
    """A yawed car crosses the line with its BODY while the ego centre
    point still reads in-lane - the centre-only metric reported 0
    crossings while the user photographed left wheels ON the line
    (fsd_benchmark town 2026-09-06)."""
    rows = _hist(20)
    # straight and centred except one yawed segment 1 m right of the
    # lane centre with 12 deg of yaw: centre passes, body crosses
    for r in rows:
        r["lat_left"] = -1.0
        r["route_bear"] = 0.0
        r["heading"] = 0.0
    rows[10]["lat_left"] = -1.0
    rows[10]["heading"] = 12.0     # hist stores degrees
    rows[10]["route_bear"] = 0.0
    a = assess_run(rows)
    assert a["cross_centre_frames"] == 0
    assert a["body_cross_centre_frames"] >= 1
    assert a["max_body_left_m"] is not None and a["max_body_left_m"] > 0.1
    v = score_run(a)
    assert v["checks"]["no_centre_crossing"] is False


def test_body_aware_no_false_positive_when_straight() -> None:
    rows = _hist(20)
    for r in rows:
        r["lat_left"] = -1.0
        r["route_bear"] = 0.0
        r["heading"] = 0.0
    a = assess_run(rows)
    assert a["body_cross_centre_frames"] == 0
    assert score_run(a)["checks"]["no_centre_crossing"] is True
