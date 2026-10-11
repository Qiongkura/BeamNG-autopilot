"""Automatic truth admission and interruption recovery without a game."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from beamng_autopilot.experiments import autonomous_data as auto
from beamng_autopilot.experiments.auto_truth import make_synthetic_batch
from beamng_autopilot.experiments.controller import LoopConfig, plan_once


def exporter():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m5_auto_truth_export.py"
    spec = importlib.util.spec_from_file_location("test_autonomous_export", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def package(root, *, positive=False):
    batch = make_synthetic_batch(with_line=positive, n_frames=2)
    frames = [dict(fr, rgb=fr["rgb"], annotation=fr["annotation"],
                   path=Path(f"frame_{i:05d}.npz")) for i, fr in enumerate(batch["frames"])]
    module = exporter()
    kwargs = dict(out_root=root, scene="known_a0s0", anchor=0,
                  map_name="italy", generator={"version": "test", "script_sha16": "abcdef"})
    if positive:
        rec = {"camera_alignment": {"per_frame": [{"after_px": 1.0}] * 2}}
        res = module.export_scene(rec, batch["frames"][0]["truth_points"], frames, **kwargs)
    else:
        res = module.export_line_free_package(frames, evidence={"line_px": 0}, **kwargs)
    assert res["verified"], res
    (root / "export_report.json").write_text(json.dumps({"exported": [res], "isolated": []}),
                                            encoding="utf-8")
    return Path(res["dir"])


@pytest.mark.parametrize("positive", [False, True])
def test_existing_exporter_output_is_content_bound(tmp_path, positive):
    view = package(tmp_path, positive=positive)
    result = auto.qualify_export(tmp_path, expected_map="italy")
    assert result["qualified"] and result["frames"] == 2, result
    assert result["line_free_frames"] == (0 if positive else 2)
    assert result["admitted"][0]["channels"], "per-channel unknowns must survive export"
    # Change one actual RGB pixel without changing its proof.
    path = view / "frame_00000.npz"
    with np.load(path) as data:
        rgb, lab = data["colour"].copy(), data["label"].copy()
    rgb[0, 0, 0] ^= 1
    np.savez(path, colour=rgb, label=lab)
    result = auto.qualify_export(tmp_path, expected_map="italy")
    assert not result["qualified"]
    assert "matches verified proof" in " ".join(result["isolated"][0]["reasons"])


@pytest.mark.parametrize("change", ["unverified", "negative_evidence", "aggregate", "metadata", "map"])
def test_incomplete_or_contradictory_proof_is_isolated(tmp_path, change):
    view = package(tmp_path)
    path = view.parent / "annotation.json"
    cred = json.loads(path.read_text(encoding="utf-8"))
    if change == "unverified":
        cred["truth_provenance"]["report"]["verified"] = False
    elif change == "negative_evidence":
        cred["truth_provenance"]["labels"]["line_free_evidence"] = None
    elif change == "aggregate":
        cred["truth_provenance"]["labels"]["label_sha"] = "wrong"
    elif change == "metadata":
        meta_path = view.parent / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["frames"][0]["path"] = "../elsewhere.npz"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    path.write_text(json.dumps(cred), encoding="utf-8")
    result = auto.qualify_package(view, expected_map="other" if change == "map" else "italy")
    assert not result["qualified"] and result["reasons"]


def test_export_cannot_admit_an_external_package(tmp_path):
    view = package(tmp_path / "outside")
    out = tmp_path / "attempt"
    out.mkdir()
    (out / "export_report.json").write_text(json.dumps(
        {"exported": [{"dir": str(view), "verified": True, "n_frames": 2}]}), encoding="utf-8")
    assert not auto.qualify_export(out, expected_map="italy")["qualified"]


def test_bounded_retry_advances_anchor_and_reuses_verified_result(tmp_path):
    calls = []

    def runner(cmd, log, timeout, game):
        calls.append(cmd)
        if game and len(calls) == 1:
            return {"rc": 1, "cleanup": {"closed": True}}
        if not game:
            package(Path(cmd[cmd.index("--out") + 1]))
        return {"rc": 0, "cleanup": {"closed": True}}

    spec = auto.AutoTruthSpec(anchor_offset=7)
    kwargs = dict(python="python", scripts=Path("scripts"), runner=runner)
    first = auto.run_collection(spec, tmp_path, **kwargs)
    assert first["ok"] and first["frames"] == 2
    assert [c[c.index("--anchor-offset") + 1] for c in calls if "--anchor-offset" in c] == ["7", "8"]
    again = auto.run_collection(spec, tmp_path, **kwargs)
    assert again["status"] == "reused_verified" and len(calls) == 3
    with pytest.raises(ValueError, match="changed"):
        auto.run_collection(auto.AutoTruthSpec(anchor_offset=8), tmp_path, **kwargs)


def test_interrupted_attempt_consumes_budget_without_overwriting(tmp_path):
    spec = auto.AutoTruthSpec(attempts=1)

    def interrupted(*args):
        raise KeyboardInterrupt()

    kwargs = dict(python="python", scripts=Path("scripts"), runner=interrupted)
    with pytest.raises(KeyboardInterrupt):
        auto.run_collection(spec, tmp_path, **kwargs)
    state = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert auto.run_collection(spec, tmp_path, **kwargs)["status"] == "attempt_budget_exhausted"
    assert (tmp_path / "state.json").read_text(encoding="utf-8") == state


def test_unproven_cleanup_stops_before_export_or_retry(tmp_path):
    calls = []

    def runner(cmd, *args):
        calls.append(cmd)
        return {"rc": 0, "cleanup": {"closed": False}}

    res = auto.run_collection(auto.AutoTruthSpec(), tmp_path, python="python",
                              scripts=Path("scripts"), runner=runner)
    assert not res["ok"] and res["status"] == "cleanup_unproven" and len(calls) == 1


def test_removed_admitted_evidence_stops_instead_of_recollecting(tmp_path):
    def runner(cmd, log, timeout, game):
        if not game:
            package(Path(cmd[cmd.index("--out") + 1]))
        return {"rc": 0, "cleanup": {"closed": True}}

    kwargs = dict(python="python", scripts=Path("scripts"), runner=runner)
    spec = auto.AutoTruthSpec()
    assert auto.run_collection(spec, tmp_path, **kwargs)["ok"]
    (tmp_path / "attempt_00" / "qualified" / "export_report.json").unlink()
    result = auto.run_collection(spec, tmp_path, **kwargs)
    assert not result["ok"] and result["status"] == "evidence_changed"


def test_resource_block_before_launch_does_not_spend_collection_budget(tmp_path):
    kwargs = dict(python="python", scripts=Path("scripts"))
    spec = auto.AutoTruthSpec(attempts=1, anchor_offset=5)
    result = auto.run_collection(spec, tmp_path, runner=lambda *a: {"blocked": True, "rc": 6}, **kwargs)
    assert result["status"] == "resource_blocked"
    commands = []

    def runner(cmd, log, timeout, game):
        commands.append(cmd)
        if not game:
            package(Path(cmd[cmd.index("--out") + 1]))
        return {"rc": 0, "cleanup": {"closed": True}}

    assert auto.run_collection(spec, tmp_path, runner=runner, **kwargs)["ok"]
    assert commands[0][commands[0].index("--anchor-offset") + 1] == "5"


def test_resource_pause_resumes_export_without_launching_another_game(tmp_path):
    calls = []

    def runner(cmd, log, timeout, game):
        calls.append(game)
        if len(calls) == 2:
            return {"blocked": True, "rc": 6}
        if not game:
            package(Path(cmd[cmd.index("--out") + 1]))
        return {"rc": 0, "cleanup": {"closed": True}}

    kwargs = dict(python="python", scripts=Path("scripts"), runner=runner)
    spec = auto.AutoTruthSpec(attempts=1)
    assert auto.run_collection(spec, tmp_path, **kwargs)["status"] == "resource_blocked_before_export"
    assert auto.run_collection(spec, tmp_path, **kwargs)["ok"]
    assert calls == [True, False, False]


def test_controlled_recipe_hash_and_plan_match_real_commands():
    cfg = LoopConfig(collect="tech", collect_source="controlled", collect_truth_anchor_offset=8)
    plan = plan_once(cfg, run_id="fixture", candidate_id="c", dataset_id="d",
                     python="python", script_dir=Path("scripts"), seed=42)
    assert [a.name for a in plan.actions[:2]] == ["generate_auto_truth", "certify_auto_truth"]
    assert cfg.hash() != LoopConfig(collect="tech", collect_source="controlled").hash()
    assert "--anchor-offset" in plan.actions[0].cmd
    with pytest.raises(ValueError, match="Tech"):
        LoopConfig(collect_source="controlled", collect_runtime="steam")


def test_lost_lease_terminates_only_the_owned_child(tmp_path):
    class Child:
        returncode = None
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            return 0

    child = Child()
    res = auto.run_stage(["fixture"], tmp_path / "stage.log", timeout_s=10,
                         heartbeat=lambda: False, cwd=tmp_path,
                         process_factory=lambda *a, **k: child)
    assert child.stopped and res["rc"] == 125 and "lease lost" in res["error"]
