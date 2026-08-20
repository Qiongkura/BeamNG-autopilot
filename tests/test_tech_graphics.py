"""Tech graphics-quality preflight regression (pure logic, no game).

The Camera/LiDAR providers must refuse to start when the Tech user's
graphics preset is 'Lowest': that preset never allocates the GPU prepass
buffer, so sensor creation makes BeamNG spam "Failed to get prepass
buffer" thousands of times per second and the game window hangs.
"""
from __future__ import annotations

import json

import pytest

from beamng_autopilot_tech.providers import check_graphics_quality


def _write_settings(tmp_path, quality: str) -> None:
    d = tmp_path / "current" / "settings"
    d.mkdir(parents=True)
    (d / "settings.json").write_text(json.dumps({
        "GraphicOverallQuality": quality,
        "GraphicLightingQuality": quality,
        "GraphicShaderQuality": quality,
        "GraphicMeshQuality": quality,
        "GraphicTextureQuality": quality,
        "GraphicShadowsQuality": quality,
    }), encoding="utf-8")


class TestCheckGraphicsQuality:
    def test_lowest_raises(self, tmp_path):
        _write_settings(tmp_path, "Lowest")
        with pytest.raises(RuntimeError, match="Lowest"):
            check_graphics_quality(tmp_path)

    def test_mixed_lowest_raises(self, tmp_path):
        _write_settings(tmp_path, "Low")
        d = tmp_path / "current" / "settings"
        data = json.loads((d / "settings.json").read_text(encoding="utf-8"))
        data["GraphicLightingQuality"] = "Lowest"
        (d / "settings.json").write_text(
            json.dumps(data), encoding="utf-8")
        with pytest.raises(RuntimeError, match="Lighting Quality"):
            check_graphics_quality(tmp_path)

    def test_low_passes(self, tmp_path):
        _write_settings(tmp_path, "Low")
        assert check_graphics_quality(tmp_path) is None

    def test_missing_settings_passes(self, tmp_path):
        # Missing/unreadable settings must not block sensor creation; the
        # helper only warns.
        assert check_graphics_quality(tmp_path) is None
