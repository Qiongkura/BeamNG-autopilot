"""Offline tests for the per-run provenance manifest (plan P0-1/P0-2).

Pure logic: a temp git repo, an injected environment and a fake process
list, so nothing here needs BeamNG or the real repo state.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from beamng_autopilot.run_manifest import (
    BEHAVIOUR_SWITCHES,
    DECLARED_SWITCHES,
    artifact_identity,
    build_manifest,
    exclusivity,
    git_state,
    other_controller_processes,
    own_process_lineage,
    own_process_lineage,
    switch_state,
    tech_version,
)


def _git(cwd, *args):
    subprocess.run(("git",) + args, cwd=str(cwd), check=True,
                   capture_output=True, text=True)


def _repo(tmp_path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("hello\n", encoding="utf-8")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-q", "-m", "first commit")
    return root


class TestSwitchState:
    def test_unset_declared_switch_is_none_not_missing(self):
        st = switch_state(env={})
        for name in DECLARED_SWITCHES:
            assert name in st, name
            assert st[name] is None

    def test_none_and_explicit_zero_are_different(self):
        """The module default being in force != an explicit "0"."""
        st = switch_state(env={"BEAMNG_SCHED_KEEPALIVE": "0"})
        assert st["BEAMNG_SCHED_KEEPALIVE"] == "0"
        assert st["BEAMNG_ROAD_SURFACE_GATE"] is None

    def test_set_value_is_recorded_verbatim(self):
        st = switch_state(env={"BEAMNG_ASYNC_HEADS": "1"})
        assert st["BEAMNG_ASYNC_HEADS"] == "1"

    def test_an_undeclared_beamng_var_is_still_captured(self):
        """Drift guard: a new switch must not vanish from the record."""
        st = switch_state(env={"BEAMNG_BRAND_NEW": "7"})
        assert st["BEAMNG_BRAND_NEW"] == "7"

    def test_non_beamng_env_is_ignored(self):
        st = switch_state(env={"PATH": "/usr/bin", "BEAMNG_PORT": "64257"})
        assert "PATH" not in st
        assert st["BEAMNG_PORT"] == "64257"

    def test_every_behaviour_switch_is_declared(self):
        assert len(BEHAVIOUR_SWITCHES) >= 15
        assert set(BEHAVIOUR_SWITCHES) <= set(DECLARED_SWITCHES)


class TestGitState:
    def test_clean_repo_records_commit_and_no_dirty(self, tmp_path):
        root = _repo(tmp_path)
        g = git_state(root)
        assert g["commit"] and len(g["commit"]) == 40
        assert g["commit_short"] == g["commit"][:7]
        assert g["head_subject"] == "first commit"
        assert g["dirty"] == []
        assert g["dirty_count"] == 0

    def test_dirty_paths_are_listed(self, tmp_path):
        """A run from a dirty tree is not reproducible from the commit."""
        root = _repo(tmp_path)
        (root / "a.txt").write_text("changed\n", encoding="utf-8")
        (root / "new.txt").write_text("x\n", encoding="utf-8")
        g = git_state(root)
        assert g["dirty_count"] == 2
        assert "a.txt" in g["dirty"]
        assert "new.txt" in g["dirty"]

    def test_non_repo_reports_none_instead_of_raising(self, tmp_path):
        g = git_state(tmp_path)
        assert g["commit"] is None
        assert g["dirty"] == []


class TestArtifactIdentity:
    def test_present_file_is_hashed(self, tmp_path):
        p = tmp_path / "w.pt"
        p.write_bytes(b"weights")
        want = hashlib.sha256(b"weights").hexdigest()[:16]
        rec = artifact_identity(("w.pt",), root=tmp_path)["w.pt"]
        assert rec["present"] is True
        assert rec["sha256"] == want
        assert rec["size"] == 7

    def test_missing_file_is_reported_absent_not_skipped(self, tmp_path):
        rec = artifact_identity(("nope.pt",), root=tmp_path)["nope.pt"]
        assert rec["present"] is False
        assert "sha256" not in rec

    def test_a_changed_file_changes_the_digest(self, tmp_path):
        p = tmp_path / "w.pt"
        p.write_bytes(b"one")
        a = artifact_identity(("w.pt",), root=tmp_path)["w.pt"]["sha256"]
        p.write_bytes(b"two")
        b = artifact_identity(("w.pt",), root=tmp_path)["w.pt"]["sha256"]
        assert a != b


class _FakeProc:
    def __init__(self, pid, cmdline):
        self.info = {"pid": pid, "name": "python.exe",
                     "cmdline": cmdline, "create_time": 1000.0}


class TestExclusivity:
    def test_another_controller_is_detected(self):
        procs = [
            _FakeProc(11, ["python", "scripts/m5_fsd_benchmark.py"]),
            _FakeProc(12, ["python", "scripts/something_else.py"]),
        ]
        found = other_controller_processes(pid=99, procs=procs)
        assert [r["pid"] for r in found] == [11]
        assert found[0]["create_time"] == 1000.0

    def test_self_is_never_counted(self):
        procs = [_FakeProc(99, ["python", "scripts/m5_fsd_drive.py"])]
        assert other_controller_processes(pid=99, procs=procs) == []

    def test_no_other_controller_is_ok(self):
        procs = [_FakeProc(12, ["python", "scripts/m5_lane_metrics.py"])]
        assert exclusivity(pid=99, procs=procs)["ok"] is True

    def test_two_controllers_is_not_ok(self):
        procs = [
            _FakeProc(11, ["python", "scripts/m5_fsd_benchmark.py"]),
            _FakeProc(13, ["python", "scripts/m5_e2e_test.py"]),
        ]
        e = exclusivity(pid=99, procs=procs)
        assert e["ok"] is False
        assert [r["pid"] for r in e["others"]] == [11, 13]


class TestBuildManifest:
    def test_manifest_has_every_required_section(self, tmp_path):
        root = _repo(tmp_path)
        m = build_manifest(root, run={"scenario": "town"}, env={},
                           procs=[])
        for key in ("manifest_version", "created_wall", "git", "switches",
                    "artifacts", "process", "exclusivity", "run"):
            assert key in m, key
        assert m["run"]["scenario"] == "town"
        assert m["exclusivity"]["ok"] is True
        assert m["process"]["pid"] > 0

    def test_run_section_is_copied_not_aliased(self, tmp_path):
        root = _repo(tmp_path)
        run = {"scenario": "town"}
        m = build_manifest(root, run=run, env={}, procs=[])
        run["scenario"] = "mutated"
        assert m["run"]["scenario"] == "town"


class TestTechVersion:
    """The Tech build is part of the experiment; nothing recorded it before."""

    def test_reads_the_version_off_the_install_path(self):
        assert tech_version(r"G:/BeamNG.tech.v0.38.5.0") == "v0.38.5.0"

    def test_reads_a_posix_path(self):
        assert tech_version("/opt/BeamNG.tech.v0.37.4.0") == "v0.37.4.0"

    def test_a_two_part_version_is_accepted(self):
        assert tech_version("BeamNG.tech.v1.2") == "v1.2"

    def test_no_version_in_the_path_is_none_not_a_guess(self):
        assert tech_version("/opt/beamng") is None

    def test_none_input_is_none(self):
        assert tech_version(None) is None

    def test_a_path_object_is_accepted(self):
        assert tech_version(Path(r"G:/BeamNG.tech.v0.38.5.0")) == "v0.38.5.0"

    def test_the_manifest_carries_home_and_version(self, tmp_path):
        root = _repo(tmp_path)
        m = build_manifest(root, env={}, procs=[],
                           tech_home=r"G:/BeamNG.tech.v0.38.5.0")
        assert m["tech"]["version"] == "v0.38.5.0"
        assert "v0.38.5.0" in m["tech"]["home"]

    def test_the_version_falls_back_to_the_env_home(self, tmp_path):
        root = _repo(tmp_path)
        m = build_manifest(
            root, env={"BEAMNG_TECH_HOME": r"G:/BeamNG.tech.v0.38.5.0"},
            procs=[])
        assert m["tech"]["version"] == "v0.38.5.0"

    def test_an_unversioned_home_records_none(self, tmp_path):
        root = _repo(tmp_path)
        m = build_manifest(root, env={"BEAMNG_TECH_HOME": "/opt/beamng"},
                           procs=[])
        assert m["tech"]["version"] is None


class TestTechVersion:
    """The Tech build is part of the experiment; nothing recorded it before."""

    def test_reads_the_version_off_the_install_path(self):
        assert tech_version(r"G:\BeamNG.tech.v0.38.5.0") == "v0.38.5.0"

    def test_reads_a_posix_path(self):
        assert tech_version("/opt/BeamNG.tech.v0.37.4.0") == "v0.37.4.0"

    def test_a_two_part_version_is_accepted(self):
        assert tech_version("BeamNG.tech.v1.2") == "v1.2"

    def test_no_version_in_the_path_is_none_not_a_guess(self):
        assert tech_version("/opt/beamng") is None

    def test_none_input_is_none(self):
        assert tech_version(None) is None

    def test_a_path_object_is_accepted(self):
        assert tech_version(Path(r"G:\BeamNG.tech.v0.38.5.0")) == "v0.38.5.0"

    def test_the_manifest_carries_home_and_version(self, tmp_path):
        root = _repo(tmp_path)
        m = build_manifest(root, env={}, procs=[],
                           tech_home=r"G:\BeamNG.tech.v0.38.5.0")
        assert m["tech"]["version"] == "v0.38.5.0"
        assert "v0.38.5.0" in m["tech"]["home"]

    def test_the_version_falls_back_to_the_env_home(self, tmp_path):
        root = _repo(tmp_path)
        m = build_manifest(
            root, env={"BEAMNG_TECH_HOME": r"G:\BeamNG.tech.v0.38.5.0"},
            procs=[])
        assert m["tech"]["version"] == "v0.38.5.0"

    def test_an_unversioned_home_records_none(self, tmp_path):
        root = _repo(tmp_path)
        m = build_manifest(root, env={"BEAMNG_TECH_HOME": "/opt/beamng"},
                           procs=[])
        assert m["tech"]["version"] is None


class TestOwnLineageIsNotAnotherController:
    """On Windows the venv python is a shim that spawns the real
    interpreter as a CHILD, so the shim - carrying the same command line -
    is this process's parent.  Excluding only os.getpid() made every live
    run refuse to start by reporting its own launcher as a rival."""

    def _proc(self, pid, cmd):
        return {"pid": pid, "cmdline": cmd, "create_time": 1.0}

    def test_self_is_excluded(self):
        procs = [self._proc(10, ["py", "m5_fsd_benchmark.py"])]
        assert other_controller_processes(pid=10, procs=procs,
                                          kin={10}) == []

    def test_the_parent_shim_is_excluded(self):
        procs = [self._proc(9, ["py", "m5_fsd_benchmark.py"]),
                 self._proc(10, ["py", "m5_fsd_benchmark.py"])]
        got = other_controller_processes(pid=10, procs=procs, kin={9, 10})
        assert got == []

    def test_a_grandparent_is_excluded(self):
        procs = [self._proc(8, ["py", "fsd_drive"])]
        assert other_controller_processes(pid=10, procs=procs,
                                          kin={8, 9, 10}) == []

    def test_a_sibling_from_the_same_shell_is_still_reported(self):
        """Two controllers started from one shell are siblings, not
        ancestors - they really do fight over the vehicle."""
        procs = [self._proc(11, ["py", "m5_fsd_benchmark.py"]),
                 self._proc(10, ["py", "m5_fsd_benchmark.py"])]
        got = other_controller_processes(pid=10, procs=procs, kin={9, 10})
        assert [g["pid"] for g in got] == [11]

    def test_the_real_game_process_is_not_a_controller(self):
        procs = [self._proc(99, ["G:////BeamNG.tech.exe", "-tcom"])]
        assert other_controller_processes(pid=10, procs=procs,
                                          kin={10}) == []

    def test_lineage_includes_self_even_without_psutil(self):
        # An injected pid may not exist; the set must still contain it.
        assert 4242 in own_process_lineage(4242)


class TestOwnLineageIsNotAnotherController:
    """On Windows the venv python is a shim that spawns the real
    interpreter as a CHILD, so the shim - carrying the same command line -
    is this process's parent.  Excluding only os.getpid() made every live
    run refuse to start by reporting its own launcher as a rival."""

    def _proc(self, pid, cmd):
        return {"pid": pid, "cmdline": cmd, "create_time": 1.0}

    def test_self_is_excluded(self):
        procs = [self._proc(10, ["py", "m5_fsd_benchmark.py"])]
        assert other_controller_processes(pid=10, procs=procs,
                                          kin={10}) == []

    def test_the_parent_shim_is_excluded(self):
        procs = [self._proc(9, ["py", "m5_fsd_benchmark.py"]),
                 self._proc(10, ["py", "m5_fsd_benchmark.py"])]
        got = other_controller_processes(pid=10, procs=procs, kin={9, 10})
        assert got == []

    def test_a_grandparent_is_excluded(self):
        procs = [self._proc(8, ["py", "fsd_drive"])]
        assert other_controller_processes(pid=10, procs=procs,
                                          kin={8, 9, 10}) == []

    def test_a_sibling_from_the_same_shell_is_still_reported(self):
        """Two controllers started from one shell are siblings, not
        ancestors - they really do fight over the vehicle."""
        procs = [self._proc(11, ["py", "m5_fsd_benchmark.py"]),
                 self._proc(10, ["py", "m5_fsd_benchmark.py"])]
        got = other_controller_processes(pid=10, procs=procs, kin={9, 10})
        assert [g["pid"] for g in got] == [11]

    def test_the_real_game_process_is_not_a_controller(self):
        procs = [self._proc(99, [r"G:\BeamNG.tech.exe", "-tcom"])]
        assert other_controller_processes(pid=10, procs=procs,
                                          kin={10}) == []

    def test_lineage_includes_self_even_without_psutil(self):
        # An injected pid may not exist; the set must still contain it.
        assert 4242 in own_process_lineage(4242)
