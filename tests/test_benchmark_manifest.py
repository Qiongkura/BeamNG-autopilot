"""Tests for the benchmark entry's run manifest and exclusivity gate (P0-1/P0-2).

No game, no git mutation: the manifest builder takes env and the process list
as arguments, so both the record and the contamination gate are testable here.
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
    return _load("m5_fsd_benchmark_manifest",
                 ROOT / "scripts" / "m5_fsd_benchmark.py")


def _fake_proc(pid: int, cmdline: list[str]):
    return {"pid": pid, "name": "python.exe", "cmdline": cmdline,
            "create_time": 1.0}


# --------------------------------------------------------------- manifest

def test_manifest_is_written_and_records_the_resolved_switches(tmp_path):
    mod = _mod()
    man = mod.write_manifest(
        tmp_path, 12345, ["town"],
        {"strict": True, "goal": [1.0, 2.0], "lane_mode": "sensor"},
        env={"BEAMNG_SCHED_KEEPALIVE": "1"}, procs=[])
    path = tmp_path / "manifest_12345.json"
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["manifest_version"] == 1
    assert saved["switches"]["BEAMNG_SCHED_KEEPALIVE"] == "1"
    # An unset switch is NOT the same reading as an explicit "0".
    assert saved["switches"]["BEAMNG_ROAD_SURFACE_GATE"] is None
    assert saved["run"]["scenarios"] == ["town"]
    assert saved["run"]["strict"] is True
    assert saved["run"]["lane_mode"] == "sensor"
    assert man["run"]["goal"] == [1.0, 2.0]


def test_manifest_carries_the_scenario_start_state(tmp_path):
    """A run is only reproducible if its spawn pose and duration are recorded."""
    mod = _mod()
    man = mod.write_manifest(tmp_path, 7, ["town"], {}, env={}, procs=[])
    assert man["run"]["teleport"]["town"]        # town has a fixed spawn pose
    assert man["run"]["seconds"]["town"] == 120.0
    assert man["run"]["speed_mps"]["town"] == 6.0


def test_manifest_records_the_commit_and_dirty_state(tmp_path):
    mod = _mod()
    man = mod.write_manifest(tmp_path, 8, ["town"], {}, env={}, procs=[])
    git = man["git"]
    assert git is not None and git.get("commit")
    # The record must not silently drop the dirty list - a run from a dirty
    # tree is not reproducible from the commit alone.
    assert "dirty" in git and "dirty_count" in git


def test_manifest_reports_a_missing_artifact_instead_of_skipping_it(tmp_path):
    mod = _mod()
    man = mod.write_manifest(tmp_path, 9, ["town"], {}, env={}, procs=[])
    # artifacts is keyed by path, so a missing file still has a record.
    assert any("tech_smallgrid.pt" in key for key in man["artifacts"])
    for rec in man["artifacts"].values():
        assert "present" in rec


# ------------------------------------------------------------ exclusivity

def test_exclusivity_passes_when_no_manifest():
    """A missing record must not block a run."""
    assert _mod().check_exclusivity(None) is True


def test_exclusivity_passes_when_alone():
    assert _mod().check_exclusivity(
        {"exclusivity": {"ok": True, "others": []}}) is True


def test_exclusivity_flags_another_controller():
    man = {"exclusivity": {"ok": False, "others": [
        {"pid": 42, "cmdline": ["python", "m5_fsd_drive.py"]}]}}
    assert _mod().check_exclusivity(man) is False


def test_manifest_records_a_second_controller_from_the_process_list(tmp_path):
    """The contamination that cost four runs must land in the record."""
    mod = _mod()
    man = mod.write_manifest(
        tmp_path, 11, ["town"], {}, env={},
        procs=[_fake_proc(4242, ["python", "scripts/m5_fsd_drive.py"])])
    assert man["exclusivity"]["ok"] is False
    assert man["exclusivity"]["others"][0]["pid"] == 4242
    assert not mod.check_exclusivity(man)


def test_an_unrelated_process_is_not_a_controller(tmp_path):
    mod = _mod()
    man = mod.write_manifest(
        tmp_path, 12, ["town"], {}, env={},
        procs=[_fake_proc(4243, ["python", "-m", "pytest", "tests/"])])
    assert man["exclusivity"]["ok"] is True
    assert mod.check_exclusivity(man) is True


def test_manifest_records_the_warmup_phase(tmp_path):
    """P0-1 asks for the warmup; the drive's own constant is the source."""
    mod = _mod()
    man = mod.write_manifest(tmp_path, 13, ["town"], {}, env={}, procs=[])
    assert man["run"]["warmup_s"] == 8.0


def test_manifest_says_the_seed_is_not_controlled(tmp_path):
    """BeamNG.tech owns the randomness; the stack exposes no seed for it.

    Recording None with an explicit flag is the honest answer - and it is
    part of why town mileage varies 30.9-92.9 m inside a single arm.
    """
    mod = _mod()
    man = mod.write_manifest(tmp_path, 14, ["town"], {}, env={}, procs=[])
    assert man["run"]["seed"] is None
    assert man["run"]["seed_controlled"] is False


def test_manifest_leaves_the_vehicle_id_unknown_before_attach(tmp_path):
    """The vid only exists once the connector is attached, which happens
    inside fsd_drive - after this manifest is written."""
    mod = _mod()
    man = mod.write_manifest(tmp_path, 15, ["town"], {}, env={}, procs=[])
    assert man["run"]["vehicle_id"] is None


def test_manifest_records_the_tech_build(tmp_path):
    mod = _mod()
    man = mod.write_manifest(
        tmp_path, 16, ["town"], {},
        env={"BEAMNG_TECH_HOME": r"G:/BeamNG.tech.v0.38.5.0"}, procs=[])
    assert man["tech"]["version"] == "v0.38.5.0"


def test_manifest_records_the_warmup_phase(tmp_path):
    """P0-1 asks for the warmup; the drive's own constant is the source."""
    mod = _mod()
    man = mod.write_manifest(tmp_path, 13, ["town"], {}, env={}, procs=[])
    assert man["run"]["warmup_s"] == 8.0


def test_manifest_says_the_seed_is_not_controlled(tmp_path):
    """BeamNG.tech owns the randomness; the stack exposes no seed for it.

    Recording None with an explicit flag is the honest answer - and it is
    part of why town mileage varies 30.9-92.9 m inside a single arm.
    """
    mod = _mod()
    man = mod.write_manifest(tmp_path, 14, ["town"], {}, env={}, procs=[])
    assert man["run"]["seed"] is None
    assert man["run"]["seed_controlled"] is False


def test_manifest_leaves_the_vehicle_id_unknown_before_attach(tmp_path):
    """The vid only exists once the connector is attached, which happens
    inside fsd_drive - after this manifest is written."""
    mod = _mod()
    man = mod.write_manifest(tmp_path, 15, ["town"], {}, env={}, procs=[])
    assert man["run"]["vehicle_id"] is None


def test_manifest_records_the_tech_build(tmp_path):
    mod = _mod()
    man = mod.write_manifest(
        tmp_path, 16, ["town"], {},
        env={"BEAMNG_TECH_HOME": r"G:\BeamNG.tech.v0.38.5.0"}, procs=[])
    assert man["tech"]["version"] == "v0.38.5.0"
