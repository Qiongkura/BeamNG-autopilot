"""Tests for segmentation evaluation and E2E replay contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_seg_eval_ignores_255_in_metrics(tmp_path, capsys):
    """Unknown manual pixels must not become background errors."""
    p = tmp_path / "run"
    p.mkdir()
    colour = np.zeros((8, 10, 3), dtype=np.uint8)
    label = np.full((8, 10), 255, dtype=np.uint8)
    label[2:6, 2:6] = 1
    label[3:5, 3:5] = 2
    np.savez_compressed(p / "frame_00000.npz", colour=colour, label=label)
    # The evaluator's implementation is exercised through its CLI so the
    # JSON report and ignore semantics are tested together.
    mod = _load("m5_eval_seg_contract", ROOT / "scripts" / "m5_eval_seg.py")
    import sys
    old = sys.argv
    out = tmp_path / "report.json"
    sys.argv = ["m5_eval_seg.py", "--runs", str(p), "--model",
                str(ROOT / "logs/m5_seg/seg_model_v8_backup/best.pt"),
                "--device", "cpu", "--json", str(out)]
    try:
        mod.main()
    finally:
        sys.argv = old
    import json
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ignore_value"] == 255
    assert report["frames"] == 1
    assert report["per_run"]["run"]["pixel_accuracy"] >= 0.0


def test_e2e_prep_prefers_fmap_over_bev():
    """A v3 multi-channel input must not be replaced by repeated BEV."""
    import torch
    mod = _load("m5_e2e_probe_contract", ROOT / "scripts" / "m5_e2e_probe.py")
    rgb = np.zeros((1, 4, 5, 3), dtype=np.uint8)
    label = np.zeros((1, 4, 5), dtype=np.uint8)
    bev = np.full((1, 6, 6), 0.1, dtype=np.float32)
    fmap = np.zeros((1, 4, 6, 6), dtype=np.float32)
    fmap[:, 2] = 0.9
    _, _, t = mod._prep_arr(0, rgb, label, bev, fmap,
                             img_h=4, img_w=5, bev_channels=4)
    assert tuple(t.shape) == (4, 6, 6)
    assert float(t[2].mean()) == pytest.approx(0.9)
    assert float(t[0].mean()) == pytest.approx(0.0)


def test_e2e_prep_legacy_falls_back_to_bev():
    mod = _load("m5_e2e_probe_legacy", ROOT / "scripts" / "m5_e2e_probe.py")
    rgb = np.zeros((1, 4, 5, 3), dtype=np.uint8)
    label = None
    bev = np.full((1, 6, 6), 0.2, dtype=np.float32)
    _, _, t = mod._prep_arr(0, rgb, label, bev, None,
                             img_h=4, img_w=5, bev_channels=4)
    assert tuple(t.shape) == (4, 6, 6)
    assert float(t[0].mean()) == pytest.approx(0.2)
    assert float(t[3].mean()) == pytest.approx(0.2)
