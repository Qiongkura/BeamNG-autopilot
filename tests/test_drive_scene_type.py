# -*- coding: utf-8 -*-
"""--scenario-json 的 scene_type 从 JSON 读取（默认 known_line）。

试车场需要 proving_lane（先铺深色铺装）；旧实现把 scene_type 硬编码成
known_line，导致场景 JSON 声明无效——这里钉住入口与默认值。
（scene JSON 在 logs/ 下被 .gitignore 忽略，因此测试只查代码契约。）
"""
from pathlib import Path


def test_drive_scene_type_from_json():
    root = Path(__file__).resolve().parents[1]
    src = (root / "beamng_autopilot" / "fsd_drive.py").read_text("utf-8")
    assert '_doc.get("scene_type") or "known_line"' in src
    assert "scene_types=[_scene_type]" in src
