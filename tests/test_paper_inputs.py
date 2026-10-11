from __future__ import annotations

import hashlib
import json

import pytest

from beamng_autopilot.experiments.paper_inputs import read_figure_inputs


def fixture_manifest(tmp_path):
    source = tmp_path / "hist.json"
    source.write_text('{"epochs": [0, 1]}', encoding="utf-8")
    record = {"schema": "paper-inputs-v1", "figures": {"fig21": {
        "scope": "two measured seeds", "inputs": [{"path": source.name,
        "role": "history", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}}}
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    return source, record, manifest


def test_new_logs_do_not_enter_a_frozen_figure(tmp_path):
    source, _, manifest = fixture_manifest(tmp_path)
    before = read_figure_inputs(tmp_path, manifest, "fig21")
    (tmp_path / "new_train_hist.json").write_text('{"epochs": [99]}', encoding="utf-8")
    source.touch()
    assert read_figure_inputs(tmp_path, manifest, "fig21") == before
    assert len(before) == 1 and before[0]["scope"] == "two measured seeds"


@pytest.mark.parametrize("change", ["content", "missing", "duplicate", "escape", "scope", "schema"])
def test_incomplete_or_changed_inputs_never_silently_change_a_figure(tmp_path, change):
    source, record, manifest = fixture_manifest(tmp_path)
    spec = record["figures"]["fig21"]
    if change == "content":
        source.write_text('{"epochs": [99]}', encoding="utf-8")
    elif change == "missing":
        source.unlink()
    elif change == "duplicate":
        spec["inputs"] *= 2
    elif change == "escape":
        spec["inputs"][0]["path"] = "../outside.json"
    elif change == "scope":
        del spec["scope"]
    else:
        record["schema"] = "legacy"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        read_figure_inputs(tmp_path, manifest, "fig21")
