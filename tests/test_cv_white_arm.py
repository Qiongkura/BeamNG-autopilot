# -*- coding: utf-8 -*-
"""BEAMNG_CV_WHITE 研究开关（默认 1=现行为）。

GridMap 试车场的浅色地面会被经典亮度白漆臂当成白漆：学习掩码里两条
30m solid 线与 cv_white 并成无效大块（0 marking）。显式
BEAMNG_CV_WHITE=0 让候选只来自学习掩码（+黄色臂），用于受控场地诊断；
生产默认必须不变。
"""
from pathlib import Path


def test_cv_white_switch_default_on_and_scoped():
    src = (Path(__file__).resolve().parents[1]
           / "beamng_autopilot" / "vision" / "segmentation.py").read_text("utf-8")
    assert 'BEAMNG_CV_WHITE", "1"' in src
    assert "if _cv_white_off:" in src
    assert '"cv_white_off": bool(_cv_white_off)' in src
