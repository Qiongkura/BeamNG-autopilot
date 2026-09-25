"""agent 核对式标注工具（`scripts/m5_line_truth_agent.py`）的规则与记账。

为什么有这个文件：要求是把那 32 帧标注出来。工具必须把三件事钉死，否则"机器画的"
很容易被当成"人工真值"用出去：

1. **未知区写 255(ignore)**——指标 (`known = lab != 255`) 与损失都尊重它，
   未知既不算假阳也不算漏检；写成 0 就等于替人断言"那里没有漆线"。
2. **规则里"看图判断"的部分要能在产物里看见**：宽亮带（路缘石）判背景这一条
   不是算法推出来的，是逐帧看出来的，所以每帧都记 `rejected_wide_px` 等计数；
   逐视角还能关掉 RGB 补线（`views_add_rgb`，复核抓到 pillar_right 一处假阳后用的）。
3. **身份必须跟着走**：npz 带 map/source/pos/heading，视角目录带 meta.json。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_line_truth_agent", ROOT / "scripts" / "m5_line_truth_agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_line_truth_agent"] = mod
    spec.loader.exec_module(mod)
    return mod


def _frame(h=60, w=80):
    """合成一帧：中灰路面 + 引擎标的左边缘线 + 一条引擎没标的细长亮条。"""
    colour = np.full((h, w, 3), 90, np.uint8)
    colour[20:, :] = (150, 150, 152)                   # 路面
    road = np.zeros((h, w), bool)
    road[20:, :] = True
    engine = np.zeros((h, w), bool)
    engine[25:55, 8:11] = True                         # 引擎标线（3 px 宽、30 px 长）
    colour[25:55, 8:11] = (215, 215, 216)              # 白漆
    colour[26:54, 60:63] = (214, 214, 215)             # 另一条细长亮条（引擎没标）
    colour[30:58, 40:52] = (205, 205, 206)             # 混凝土带（宽 12、高 28）
    return colour, road, engine


def test_engine_line_is_kept_whole_and_thin_strokes_are_added():
    t = _load()
    colour, road, engine = _frame()
    line, _unknown, st = t.propose(colour, engine, road)
    assert st["engine_px"] == int(engine.sum())
    assert int((line & engine).sum()) == int(engine.sum()), "引擎标线必须整块保留"
    assert st["added_px"] > 0, "引擎没标的细长亮条要补进来（虚线/边缘线）"
    assert int(line.sum()) == st["line_px"]


def test_add_rgb_off_keeps_engine_only():
    """逐视角复核用：某视角发现假阳时，可以只保留引擎标线。"""
    t = _load()
    colour, road, engine = _frame()
    _with, _u1, st_a = t.propose(colour, engine, road, add_rgb=True)
    without, _u2, st_b = t.propose(colour, engine, road, add_rgb=False)
    assert st_a["added_px"] > 0 and st_b["added_px"] == 0
    assert int(without.sum()) == int(engine.sum())


def test_build_writes_identity_provenance_and_only_legal_labels(tmp_path):
    t = _load()
    src = tmp_path / "coll" / "front_main"
    src.mkdir(parents=True)
    colour, road, engine = _frame()
    lab = np.zeros(colour.shape[:2], np.uint8)
    lab[road] = 1
    lab[engine] = 2
    for i in range(2):
        np.savez_compressed(src / f"frame_{i:05d}.npz",
                            colour=(colour + np.uint8(i * 2)), label=lab)
    (tmp_path / "coll" / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": "ring_test",
        "frames": [{"path": f"front_main/frame_{i:05d}.npz",
                    "pos": [1.0, 2.0, 3.0], "heading": 0.5}
                   for i in range(2)]}), encoding="utf-8")
    out = tmp_path / "anno"
    side = t.build({"sources": [[str(src), "front_main"]],
                    "views_add_rgb": ["front_main"],
                    "overrides": {"front_main/frame_00001.npz": {
                        "drop_rect": [[0, 0, 5, 5]]}}}, out=out)
    assert side["label_source"] == "agent_revision"
    assert len(side["frames"]) == 2
    assert side["frames"][1]["overrides"] is True
    assert (out / "front_main" / "meta.json").exists(), "身份要跟着走"
    # 凭证也要跟着帧走（方案 §6.1）：只写在根目录时，读取方按 self->parent 回退
    # 会先读到采集 meta（引擎来源），把 agent 数据误判成 engine_annotation
    cred = json.loads((out / "front_main" / "annotation.json").read_text(
        encoding="utf-8"))
    assert cred["label_source"] == "agent_revision", cred
    assert cred["view"] == "front_main" and len(cred["frames"]) == 2, cred
    with np.load(out / "front_main" / "frame_00000.npz") as z:
        assert str(z["map_name"].item()) == "italy"
        assert str(z["source_id"].item()) == "ring_test"
        assert z["pos"].size == 3 and z["heading"].size == 1
        lab2 = np.asarray(z["label"], np.uint8)
        assert set(np.unique(lab2).tolist()) <= {0, 1, 2, 255}, np.unique(lab2)
        unknown_px = int((lab2 == 255).sum())
        if unknown_px:
            assert np.asarray(z["unknown_kind"], np.uint8).size == lab2.size, \
                "写了 255 就要有 unknown_kind 说明为什么"
    from beamng_autopilot.experiments.manifest import DatasetManifest
    mf = DatasetManifest.build([out / "front_main"], root=tmp_path)
    assert not [r for r in mf.records if r.reject_reason], \
        [r.reject_reason for r in mf.records]


def test_the_agent_source_is_usable_but_never_a_gate():
    """`agent_revision`：可训练、可测，但晋级仍需人确认。"""
    from beamng_autopilot.experiments.labels import (
        PAINT_SOURCE_RANK, audit_label,
    )
    assert PAINT_SOURCE_RANK["agent_revision"] == "agent"
    lab = np.zeros((20, 30), np.uint8)
    lab[5:15, :] = 1
    lab[10, :8] = 2
    q = audit_label(lab, paint_source="agent_revision").paint
    assert q.valid is False, "机器画的不能当门槛真值"
    assert q.usable is True, "但可以学、可以测"
    assert "agent" in q.reason





def test_haze_is_not_taken_for_paint():

    """天际线雾不能被当成漆线：判别特征是**饱和度**。



    实测（2026-09-25）：真漆线（白/灰）mx-mn = 1.2–12，远处雾/云边 = 30–32；

    亮度（188–190 vs 199–213）与形状（同样细长）都分不开。阈值收到 20 后

    雾消失、真漆线保留（逐帧目视复核过）。

    """

    t = _load()

    colour, road, engine = _frame()

    # 在路面上画一条"雾色"细长条：亮 189、饱和 31

    colour[30:54, 20:23] = (189, 175, 158)

    _line, _unk, st = t.propose(colour, engine, road)

    assert t.SAT_MAX <= 20, "阈值被改回去了？"

    # 雾条不能进 "added"

    assert st["added_px"] < 200, st

