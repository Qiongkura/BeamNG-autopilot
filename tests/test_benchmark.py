"""Offline tests for the benchmark scoring layer (eval.score_run + script)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from beamng_autopilot.eval import assess_run, score_many, score_run

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
        }
        row.update(over)
        rows.append(row)
    return rows


def test_score_run_passes_clean_run() -> None:
    a = assess_run(_hist())
    v = score_run(a)
    assert v["pass"] is True
    assert all(v["checks"].values())


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
