"""Offline tests for the data-tool helpers added around replay eval.

Covers: bad-episode filtering from the replay report, worst-frame PNG
export, and dataset frame dedup.  All pure offline (no game).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from beamng_autopilot.neural.dataset import GRID_N
from beamng_autopilot.recording import ShadowFrame, ShadowRecorder
from beamng_autopilot.neural import ShadowMultimodalDataset


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _make_episode(tmp_path: Path, name: str, n: int = 4,
                  speed: float = 3.0, steer: float = 0.0,
                  throttle: float = 0.35) -> Path:
    rec = ShadowRecorder(tmp_path, name)
    rgb = np.zeros((16, 24, 3), dtype=np.uint8)
    lab = np.zeros((16, 24), dtype=np.uint8)
    for i in range(n):
        rec.add(ShadowFrame(
            x=float(i), y=0.0, heading=0.0, speed=speed,
            throttle=throttle, brake=0.0, steer=steer,
            bev_raster=np.zeros((GRID_N, GRID_N), dtype=np.float32),
            drivable=np.ones((GRID_N, GRID_N), dtype=np.uint8),
            trajectory=np.array([[i, 0.0], [i + 1, 0.0]], dtype=float),
            target_speed=5.0, lane_src="semantic", cost=0.0,
            kind="arc", rgb=rgb, label=lab, quality=1.0))
    out = rec.save()
    assert out is not None
    return out


def test_dataset_dedup_skips_identical_frames(tmp_path) -> None:
    ep = _make_episode(tmp_path, "dedup")
    ds0 = ShadowMultimodalDataset([ep], min_quality=0.0, min_speed=0.0,
                                  history=0, dedup=False)
    ds1 = ShadowMultimodalDataset([ep], min_quality=0.0, min_speed=0.0,
                                  history=0, dedup=True)
    assert len(ds0) == 4
    assert len(ds1) == 1


def test_dataset_dedup_keeps_changing_frames(tmp_path) -> None:
    # steer changes every frame -> nothing is near-duplicate
    rec = ShadowRecorder(tmp_path, "changing")
    for i in range(4):
        rec.add(ShadowFrame(
            x=float(i), y=0.0, speed=3.0, throttle=0.35, steer=0.05 * i,
            bev_raster=np.zeros((GRID_N, GRID_N), dtype=np.float32),
            quality=1.0))
    ep = rec.save()
    assert ep is not None
    ds1 = ShadowMultimodalDataset([ep], min_quality=0.0, min_speed=0.0,
                                  history=0, dedup=True)
    assert len(ds1) == 4


def test_filter_high_takeover_drops_bad_episodes(tmp_path) -> None:
    train = _load_script("m5_train_e2e.py")
    good = tmp_path / "good.npz"
    bad = tmp_path / "bad.npz"
    good.touch()
    bad.touch()
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"episodes": [
        {"episode": "good.npz", "takeover_rate": 0.2},
        {"episode": "bad.npz", "takeover_rate": 0.8},
    ]}), encoding="utf-8")
    keep, dropped, rates = train.filter_high_takeover(
        [good, bad], report, 0.5)
    assert keep == [good]
    assert dropped == [bad]
    assert rates["bad.npz"] == pytest.approx(0.8)


def test_filter_high_takeover_missing_report_keeps_all(tmp_path) -> None:
    train = _load_script("m5_train_e2e.py")
    ep = tmp_path / "ep.npz"
    ep.touch()
    keep, dropped, rates = train.filter_high_takeover(
        [ep], tmp_path / "nope.json", 0.5)
    assert keep == [ep] and dropped == [] and rates == {}


def test_save_worst_writes_png_and_meta(tmp_path) -> None:
    probe = _load_script("m5_e2e_probe.py")
    ep = _make_episode(tmp_path, "worst_ep", n=2)
    out = tmp_path / "worst"
    probe._save_worst([(0.9, ep, 0, 0.2, 0.3, 0.05, 0.1)], 1, out)
    pngs = list(out.glob("*.png"))
    assert len(pngs) == 1
    meta = json.loads((out / "worst.json").read_text(encoding="utf-8"))
    assert meta[0]["episode"] == ep.name
    assert meta[0]["frame"] == 0
