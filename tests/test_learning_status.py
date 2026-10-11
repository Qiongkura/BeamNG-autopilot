from __future__ import annotations

import json

from beamng_autopilot.experiments.learning_status import read_learning_status
from beamng_autopilot.experiments.metrics import MetricsStore
from beamng_autopilot.experiments.monitor_server import metrics_query


def write(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob), encoding="utf-8")


def test_available_frames_do_not_become_learned_frames(tmp_path):
    write(tmp_path / "auto_truth" / "admission.json",
          {"status": "admitted", "frames_total": 60, "isolated": []})
    result = read_learning_status(tmp_path)
    assert result["qualified_frames"] == 60
    assert result["unique_seen"] is None and result["steps_done"] is None
    assert result["missing"]


def test_read_actual_sampler_recipe_decision_and_admission(tmp_path):
    write(tmp_path / "round0" / "seed42" / "train_hist.json", {
        "steps_done": 120, "total_steps": 120, "stopped_by": "step_budget",
        "sampler_report": {"unique_available": 60, "unique_seen": 47, "exposures_total": 480},
        "init": {"init_sha16": "shared", "init_source": "checkpoint"}})
    write(tmp_path / "decision_candidate.json", {
        "candidate_id": "c", "decision": {"decision": "rejected", "reasons": ["recall below gate"]}})
    result = metrics_query(MetricsStore(tmp_path), 0, "fixture")["pipeline"]
    assert result["unique_available"] == 60 and result["unique_seen"] == 47
    assert result["steps_done"] == 120 and result["init_sha16"] == "shared"
    assert result["decisions"][0]["reasons"] == ["recall below gate"]


def test_partial_artifact_is_visible_and_not_zero(tmp_path):
    path = tmp_path / "auto_truth" / "admission.json"
    path.parent.mkdir()
    path.write_text('{"frames_total":', encoding="utf-8")
    result = read_learning_status(tmp_path)
    assert result["qualified_frames"] is None and result["problems"]
