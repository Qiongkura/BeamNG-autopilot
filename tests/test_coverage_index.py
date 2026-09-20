"""Tests for the telemetry coverage index (plan P0-4/P0-5).

The point of this script is that "this frame does not say confirmed on_road"
has several different causes, and collapsing them hides the useful reading.
These tests pin each cause apart.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _mod():
    return _load("m5_coverage_index", ROOT / "scripts" / "m5_coverage_index.py")


# ---------------------------------------------------------------- load_run

def test_load_run_reads_a_frame_list(tmp_path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps([{"t": 0.0}, {"t": 0.1}]), encoding="utf-8")
    frames = _mod().load_run(str(p))
    assert frames is not None and len(frames) == 2


def test_load_run_reads_a_wrapped_frame_list(tmp_path):
    """Some writers wrap the frames under a key; both shapes must load."""
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"frames": [{"t": 0.0}]}), encoding="utf-8")
    frames = _mod().load_run(str(p))
    assert frames is not None and len(frames) == 1


def test_load_run_skips_non_frames(tmp_path):
    """A scorecard or report is not a run - it must not be counted as one."""
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"mean_travel_m": 55.6}), encoding="utf-8")
    assert _mod().load_run(str(p)) is None


def test_load_run_skips_broken_json(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json", encoding="utf-8")
    assert _mod().load_run(str(p)) is None


def test_load_run_drops_non_dict_elements(tmp_path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps([{"t": 0.0}, 3, "x"]), encoding="utf-8")
    frames = _mod().load_run(str(p))
    assert frames is not None and len(frames) == 1


# ------------------------------------------------------- road_accounting

def test_absent_column_is_uninstrumented_not_unknown():
    """No column at all is a different statement from "the gate said unknown"."""
    acc = _mod().road_accounting([{"t": 0.0}, {"t": 0.1}])
    assert acc["no_state_col"] == 2
    assert acc["state_unknown"] == 0
    assert acc["state_on_road"] == 0


def test_states_are_counted_separately():
    frames = [
        {"road_surface": "on_road", "road_checked": 1},
        {"road_surface": "off_road", "road_checked": 1},
        {"road_surface": "unknown", "road_checked": 1},
        {"road_surface": None, "road_checked": 1},
        {"road_surface": "", "road_checked": 0},
    ]
    acc = _mod().road_accounting(frames)
    assert acc["state_on_road"] == 1
    assert acc["state_off_road"] == 1
    # 'unknown', None and '' all mean the gate has no answer
    assert acc["state_unknown"] == 3
    assert acc["checked_false"] == 1


def test_missing_checked_column_is_flagged_but_keeps_the_state():
    """The reader-ran flag landed later, so old runs carry the state only.

    That must be reported as "cannot tell whether the reader ran", NOT as a
    frame the gate skipped - the state itself is still a real reading.
    """
    acc = _mod().road_accounting([{"road_surface": "on_road"},
                                  {"road_surface": "on_road"}])
    assert acc["no_checked_col"] == 2
    assert acc["state_on_road"] == 2
    assert acc["checked_false"] == 0


def test_zero_checked_with_a_state_is_a_skipped_reader():
    acc = _mod().road_accounting([{"road_surface": "unknown", "road_checked": 0}])
    assert acc["checked_false"] == 1
    assert acc["no_checked_col"] == 0


def test_empty_input_is_all_zero():
    acc = _mod().road_accounting([])
    assert acc["total"] == 0
    assert all(v == 0 for k, v in acc.items() if k != "total")


# ------------------------------------------------------------------- main

def test_main_reports_no_frames_instead_of_crashing(tmp_path, capsys):
    """A selection with no run JSON must exit non-zero, not traceback."""
    mod = _mod()
    import sys
    old = sys.argv
    sys.argv = ["m5_coverage_index.py", str(tmp_path / "nope.json")]
    try:
        rc = mod.main()
    finally:
        sys.argv = old
    assert rc == 1
    assert "no run JSON" in capsys.readouterr().out
