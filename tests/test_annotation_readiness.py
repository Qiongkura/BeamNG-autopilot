"""标注就绪核对的行为（防"标注落地但审计拒收"）。

实测背景：项目里发生过"48 帧人工修订因为**丢了身份**被审计拒收且不可恢复"。
标注是人的时间，所以必须在跑训练之前先核对：凭据（human_revision）、标注真的
落盘（npz 有 label）、身份完整（位姿）、以及标的是"无线"还是"有线"。
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
        "m5_annotation_readiness", ROOT / "scripts" / "m5_annotation_readiness.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_annotation_readiness"] = mod
    spec.loader.exec_module(mod)
    return mod


def _annotated(tmp_path, name: str, *, with_label=True, identity_missing=(),
               line_px=0, n=3, label_source="human_revision",
               with_pose=True) -> Path:
    d = tmp_path / name / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        kw = {"colour": np.full((12, 16, 3), 30 + i, np.uint8)}
        if with_label:
            lab = np.zeros((12, 16), np.uint8)
            if line_px:
                lab[5, :line_px] = 2
            kw["label"] = lab
        np.savez(d / f"frame_{i:05d}.npz", **kw)
        fr = {"path": f"front_main/frame_{i:05d}.npz", "view": "front_main",
              "exposure": i,
              "classes_painted": {"line": int(line_px), "road": 100,
                                  "background": 92 - line_px, "unknown": 0}}
        if with_pose:
            fr["pos"] = [1.0, 2.0, 3.0]
            fr["heading"] = 0.1
        frames.append(fr)
    (d / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": f"ring_{name}",
        "label_source": label_source,
        "annotation": {"tool": "m5_annotate_manual.py", "reviewer": "owner",
                       "annotated_at": "2026-09-27T10:00:00",
                       "identity_missing": list(identity_missing)},
        "frames": frames}), encoding="utf-8")
    return d


def test_a_fully_annotated_negative_pack_is_ready(tmp_path):
    mod = _load()
    d = _annotated(tmp_path, "neg", line_px=0)
    rep = mod.check_dir(d)
    assert rep["ok"] is True, rep["reasons"]
    assert rep["kind"] == "negative_pack"
    assert rep["line_stats"]["n_with_line"] == 0


def test_a_positive_pack_is_recognised_by_line_pixels(tmp_path):
    mod = _load()
    d = _annotated(tmp_path, "pos", line_px=4)
    rep = mod.check_dir(d)
    assert rep["ok"] is True, rep["reasons"]
    assert rep["kind"] == "positive_pack"


def test_unannotated_copy_is_not_ready(tmp_path):
    """只有 colour 的包副本（还没标）必须判未就绪，且原因写清楚。"""
    mod = _load()
    d = _annotated(tmp_path, "raw", with_label=False,
                   label_source="beamng_annotation (road dense; ...)")
    rep = mod.check_dir(d)
    assert rep["ok"] is False
    joined = " | ".join(rep["reasons"])
    assert "human_revision" in joined and "没有 label" in joined


def test_missing_identity_is_refused(tmp_path):
    """缺位姿 = 空间隔离核对不了 -> 未就绪（历史事故的同一类）。"""
    mod = _load()
    d = _annotated(tmp_path, "nopose", identity_missing=["pos", "heading"],
                   with_pose=False)
    rep = mod.check_dir(d)
    assert rep["ok"] is False
    assert any("身份缺字段" in r for r in rep["reasons"])
    assert rep["frames_missing_pose"] == 3


def test_a_mixed_pack_is_flagged_not_hidden(tmp_path):
    mod = _load()
    d = tmp_path / "mix" / "front_main"
    d.mkdir(parents=True)
    frames = []
    for i in range(4):
        lab = np.zeros((12, 16), np.uint8)
        if i < 2:
            lab[5, :4] = 2
        np.savez(d / f"frame_{i:05d}.npz",
                 colour=np.full((12, 16, 3), 20 + i, np.uint8), label=lab)
        frames.append({"path": f"front_main/frame_{i:05d}.npz", "view": "front_main",
                       "pos": [1.0, 2.0, 3.0], "heading": 0.0,
                       "classes_painted": {"line": 4 if i < 2 else 0}})
    (d / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": "ring_mix",
        "label_source": "human_revision",
        "annotation": {"reviewer": "owner", "identity_missing": []},
        "frames": frames}), encoding="utf-8")
    rep = mod.check_dir(d)
    assert rep["ok"] is True
    assert rep["kind"] == "mixed"
    assert any("按用途分别使用" in n for n in rep["notes"]), rep["notes"]
