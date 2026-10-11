"""Execute two rounds and failed-task restart with fixed synthetic measurements."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("interrupt", [False, True])
def test_later_round_and_restart_keep_baseline_task_metrics(tmp_path, monkeypatch, interrupt):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("baseline_reuse_loop", root / "scripts/m5_seg_autoloop.py")
    loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loop)
    monkeypatch.setattr(loop, "exp_dir", lambda rid: tmp_path / "exp" / rid)
    monkeypatch.setattr(loop, "machine_gpu_minutes_today", lambda: 0.0)
    monkeypatch.setattr(loop, "machine_lease", lambda: SimpleNamespace(
        acquire=lambda **_: {"acquired": True}, release=lambda: None))
    monkeypatch.setattr(loop, "_rounds_audit", lambda *a: ({"dev_dataset_id": "fixed-dev"}, 0))
    monkeypatch.setattr(loop, "_ckpt_train_args", lambda p: {"n_train": 4})
    monkeypatch.setattr(loop, "ckpt_steps_done", lambda *a, **k: 3)
    monkeypatch.setattr(loop, "_train_meta", lambda p: {"steps_done": 3, "total_steps": 3})
    monkeypatch.setattr(loop, "plateau_from_hist", lambda p: {"at_plateau": True})
    monkeypatch.setattr(loop, "plateau_check", lambda h: {"at_plateau": True})
    measurements = {"line_iou": 0.4, "line_recall": 0.8, "line_precision": 0.7,
                    "offroad_false_line_px": 2.0, "inference_ms_p95": 10.0,
                    "inference_ms_p50": 9.0, "offroad_false_frac_of_pred": 0.0,
                    "negative_line": {"n_frames": 4}}
    monkeypatch.setattr(loop, "_pixel_eval", lambda *a, **k: dict(measurements))
    monkeypatch.setattr(loop, "identity_metrics", lambda *a, **k: {
        "candidate_identity_rate": 0.6, "candidate_reference_coverage": 0.7,
        "left_right_role_agreement": 0.8})
    trained = []
    pending_interrupt = [interrupt]

    def train(cmd, timeout):
        out = Path(cmd[cmd.index("--out") + 1])
        if "round1" in out.parts and pending_interrupt[0]:
            pending_interrupt[0] = False
            return subprocess.CompletedProcess(cmd, 9, "", "interrupted fixture")
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoint_last.pt").write_bytes(str(out).encode())
        (out / "train_hist.json").write_text("{}", encoding="utf-8")
        trained.append(out)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(loop, "_train_once", train)
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps({"proposals": [{"candidate_id": "change",
        "factor": {"line_tversky_weight": 0.1}}]}), encoding="utf-8")
    argv = ["rounds", "--run-id", "reuse", "--rounds", "2", "--runs", "base",
            "--eval-runs", "dev", "--proposals", str(proposals), "--seeds", "42", "43",
            "--total-steps", "3", "--epochs", "1", "--device", "cpu"]
    first = loop.main(argv)
    if interrupt:
        assert first == 1
        assert loop.main(argv + ["--resume"]) == 0
    else:
        assert first == 0
    directory = loop.exp_dir("reuse")
    first_decision = json.loads((directory / "decision_change-r0.json").read_text(encoding="utf-8"))
    later = json.loads((directory / "decision_change-r1.json").read_text(encoding="utf-8"))
    for name in loop.TASK_METRICS:
        assert first_decision["pairings"][name]["n"] == later["pairings"][name]["n"] == 2
        assert first_decision["pairings"][name]["champion"] == later["pairings"][name]["champion"]
    assert first_decision["plateau_baseline_by_seed"] == later["plateau_baseline_by_seed"]
    assert first_decision["negative_line_by_seed"]["baseline"] == later["negative_line_by_seed"]["baseline"]
    assert len([p for p in trained if "baseline" in p.parts]) == 2
    assert len([p for p in trained if "round0" in p.parts]) == 2
