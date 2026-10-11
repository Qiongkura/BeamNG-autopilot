from __future__ import annotations

from beamng_autopilot.experiments.reproducibility import SOURCE_DIRS, source_snapshot


def test_source_identity_includes_uncommitted_code_and_excludes_outputs(tmp_path):
    for name in SOURCE_DIRS:
        (tmp_path / name).mkdir()
    code = tmp_path / "beamng_autopilot" / "module.py"
    code.write_text("value = 1\n", encoding="utf-8")
    output = tmp_path / "logs"
    output.mkdir()
    (output / "new.py").write_text("generated = True", encoding="utf-8")
    first = source_snapshot(tmp_path)
    assert first["complete"] and len(first["files"]) == 1
    (output / "new.py").write_text("generated = False", encoding="utf-8")
    assert source_snapshot(tmp_path)["sha256"] == first["sha256"]
    code.write_text("value = 2\n", encoding="utf-8")
    assert source_snapshot(tmp_path)["sha256"] != first["sha256"]


def test_missing_source_scope_is_visible(tmp_path):
    result = source_snapshot(tmp_path)
    assert not result["complete"] and len(result["errors"]) == len(SOURCE_DIRS)
