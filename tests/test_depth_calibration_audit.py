from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_depth_calibration_audit", ROOT / "scripts" / "m5_depth_calibration_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pava_calibration_is_monotone_and_reports_supported_bins():
    mod = _load()
    # noisy monotone sensor readings: the fit must not invent a decreasing map
    raw = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    truth = np.asarray([1.0, 2.2, 2.0, 4.0, 5.0])
    model = mod._pava(raw, truth)
    assert np.all(np.diff(model[:, 1]) >= 0)
    pred = mod._predict(model, raw)
    assert np.all(np.diff(pred) >= 0)


def test_scene_report_is_not_treated_as_a_depthcal_scene(tmp_path):
    mod = _load()
    (tmp_path / "scene_report.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "report.json"
    assert mod.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    import json
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["scenes"] == []
    assert blob["missing_depthcal_scenes"] == []
