from __future__ import annotations

import json
from pathlib import Path

import pytest

from beamng_autopilot.experiments.checkpoint import file_sha16
from beamng_autopilot.experiments.reproducibility import (
    read_baseline_evidence, write_baseline_evidence)


def _record(tmp_path):
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"frozen weights")
    identity = {"dev_dataset_id": "dev", "source_sha256": "source"}
    record = {"schema": "baseline-evidence-v1", "evaluation_identity": identity,
              "by_seed": {"42": 0.4}, "steps_by_seed": {"42": 60},
              "checkpoint_sha16": {"42": file_sha16(checkpoint)},
              "task_by_seed": {"42": {"line_recall": None}},
              "negative_by_seed": {"42": {"false_positive_px": 2}},
              "plateau_by_seed": {"42": {"at_plateau": False}},
              "train_meta_by_seed": {"42": {"steps_done": 60}}}
    path = tmp_path / "champion.json"
    write_baseline_evidence(path, record)
    return path, checkpoint, identity, record


def test_pinned_baseline_preserves_unknown_and_all_measured_evidence(tmp_path):
    path, checkpoint, identity, record = _record(tmp_path)
    assert read_baseline_evidence(path, identity=identity,
                                 checkpoints={"42": checkpoint}) == record
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("change", ["weights", "missing", "identity", "legacy", "seed"])
def test_baseline_drift_or_incomplete_record_cannot_be_reused(tmp_path, change):
    path, checkpoint, identity, record = _record(tmp_path)
    if change == "weights":
        checkpoint.write_bytes(b"other weights")
    elif change == "missing":
        checkpoint.unlink()
    elif change == "identity":
        identity = {**identity, "source_sha256": "different"}
    elif change == "legacy":
        record.pop("schema")
        write_baseline_evidence(path, record)
    else:
        record["task_by_seed"] = {"43": {"line_recall": 0.8}}
        write_baseline_evidence(path, record)
    with pytest.raises(ValueError):
        read_baseline_evidence(path, identity=identity, checkpoints={"42": checkpoint})
