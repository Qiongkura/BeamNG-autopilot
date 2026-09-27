"""漆线外观证据的测试（T16 §4.2：贴花不被标注时的替代证据）。

三个层次：
1. `line_appearance`：白/黄/暗路面的判定与保守阈值；
2. `appearance_line_check`：逐点"可见 + 外观像线"，跳过规则与
   `verify_projection` 同一套；
3. `verify_projection(line_evidence="appearance")`：与 annotation 模式的**语义
   差别**——label 里有线类但 RGB 里没有漆线时，annotation 模式过、外观模式必须
   报 mismatch（否则这个替代证据等于没加）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.auto_truth import (  # noqa: E402
    appearance_line_check, line_appearance, make_synthetic_batch,
    verify_batch, verify_projection,
)


def test_line_appearance_white_yellow_and_asphalt():
    white = np.full((9, 9, 3), 60, np.uint8)
    white[4, 4] = (235, 235, 235)
    ap = line_appearance(white, 4, 4)
    assert ap["line_like"] and ap["white"] and not ap["yellow"], ap

    yellow = np.full((9, 9, 3), 60, np.uint8)
    yellow[4, 4] = (210, 200, 60)
    ap = line_appearance(yellow, 4, 4)
    assert ap["line_like"] and ap["yellow"], ap

    # 亮灰路面（italy 沥青实测特征：max≈170、chroma≈10）不该判成线
    asphalt = np.full((9, 9, 3), 160, np.uint8)
    asphalt[:, :, 0] = 170
    asphalt[:, :, 1] = 165
    ap = line_appearance(asphalt, 4, 4)
    assert not ap["line_like"], ap
    assert ap["chroma"] is not None


def test_line_appearance_out_of_frame_is_not_line_like():
    img = np.full((9, 9, 3), 255, np.uint8)
    ap = line_appearance(img, -3, 4)
    assert not ap["line_like"] and ap["max"] is None


def test_appearance_check_counts_line_like_points_on_a_known_line_batch():
    batch = make_synthetic_batch(with_line=True)
    frame = batch["frames"][0]
    rep = appearance_line_check(frame)
    assert rep["status"] == "measured" and rep["checked"] > 0, rep
    assert rep["line_like"] > 0, rep
    assert rep["line_like"] + rep["not_line_like"] == rep["checked"]


def test_appearance_check_flags_paint_removed_from_rgb():
    """把 RGB 里的漆线抹掉（label 保留线类）：外观检查必须报 not_line_like。"""
    batch = make_synthetic_batch(with_line=True)
    frame = dict(batch["frames"][0])
    rgb = np.full(np.asarray(frame["rgb"]).shape, 140, np.uint8)  # 整帧换成灰底
    frame["rgb"] = rgb
    rep = appearance_line_check(frame)
    assert rep["checked"] > 0, rep
    assert rep["line_like"] == 0 and rep["not_line_like"] == rep["checked"], rep


def test_projection_modes_differ_when_label_has_line_but_rgb_does_not():
    """关键语义差别：annotation 模式看 label，appearance 模式看 RGB。"""
    batch = make_synthetic_batch(with_line=True)
    frame = dict(batch["frames"][0])
    rgb = np.full(np.asarray(frame["rgb"]).shape, 140, np.uint8)  # 保留 label、抹掉外观
    frame["rgb"] = rgb
    ann = verify_projection(frame, line_evidence="annotation")
    app = verify_projection(frame, line_evidence="appearance")
    assert ann["checked"] > 0 and ann["mismatches"] == 0, ann
    assert app["checked"] > 0 and app["mismatches"] > 0, app
    assert any(d.get("evidence") == "appearance" for d in app["details"]), app


def test_verify_batch_reports_the_evidence_source():
    batch = make_synthetic_batch(with_line=True)
    rep = verify_batch(batch, line_evidence="appearance")
    assert rep["stats"]["line_evidence"] == "appearance"
    ap = rep["stats"]["appearance"]
    assert ap["checked"] > 0 and ap["line_like"] > 0, ap
    assert ap["status"] == "measured"
    # 默认（annotation）不受影响
    rep2 = verify_batch(batch)
    assert rep2["stats"]["line_evidence"] == "annotation"
