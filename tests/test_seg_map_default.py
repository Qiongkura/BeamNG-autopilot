"""Per-map specialist checkpoint routing for the default seg model."""

from __future__ import annotations

from beamng_autopilot.vision import segmentation


def test_default_model_routes_by_map(tmp_path, monkeypatch):
    from beamng_autopilot import config

    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    segmentation.set_active_map(None)

    base = tmp_path / "m5_seg" / "seg_model"
    assert segmentation.default_model_path() is None     # 什么都没有

    base.mkdir(parents=True)
    (base / "best.pt").write_bytes(b"base")
    assert segmentation.default_model_path() == base / "best.pt"

    by_map = base / "by_map" / "east_coast_usa"
    by_map.mkdir(parents=True)
    (by_map / "best.pt").write_bytes(b"us")

    segmentation.set_active_map("east_coast_usa")
    assert segmentation.default_model_path() == by_map / "best.pt"

    segmentation.set_active_map("italy")                 # 无专家 -> 回退
    assert segmentation.default_model_path() == base / "best.pt"

    segmentation.set_active_map(None)                    # 复位
    assert segmentation.default_model_path() == base / "best.pt"
