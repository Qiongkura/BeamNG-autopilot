from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from beamng_autopilot.experiments.evidence_archive import build_archive


def test_archive_hashes_match_actual_exported_bytes(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"evidence": true}', encoding="utf-8")
    output = tmp_path / "evidence.zip"
    manifest = build_archive(output, [(source, "run/source.json")])
    assert manifest["complete"]
    with zipfile.ZipFile(output) as archive:
        assert manifest == json.loads(archive.read("MANIFEST.json"))
        assert manifest["entries"][0]["sha256"] == hashlib.sha256(archive.read("run/source.json")).hexdigest()
    with pytest.raises(FileExistsError):
        build_archive(output, [(source, "run/source.json")])


def test_missing_evidence_never_reports_a_complete_archive(tmp_path):
    manifest = build_archive(tmp_path / "incomplete.zip", [(tmp_path / "missing", "missing.json")])
    assert not manifest["complete"] and manifest["missing"][0]["arc"] == "missing.json"


def test_frozen_digest_checks_the_actual_archived_bytes(tmp_path):
    source = tmp_path / "hist.json"
    source.write_bytes(b"original measured history")
    pinned = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"changed after selection")
    output = tmp_path / "changed.zip"
    manifest = build_archive(output, [(source, "hist.json")], expected_sha256={"hist.json": pinned})
    assert not manifest["complete"] and len(manifest["validation_errors"]) == 1
    with zipfile.ZipFile(output) as archive:
        assert manifest == json.loads(archive.read("MANIFEST.json"))
        assert manifest["entries"][0]["sha256"] == hashlib.sha256(archive.read("hist.json")).hexdigest()
