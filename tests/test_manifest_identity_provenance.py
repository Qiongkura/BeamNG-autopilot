"""地图身份的来源证据：值存在不等于可信。

实测缺陷：旧采集器把构造参数当作地图名写入 meta，`east_coast_usa` /
`gridmap_v2` 会话的采集因此带着 `map_name="italy"`，且没有
`map_name_source` 键；修复后的采集（`t13_*`）才有该键。审计必须把"身份值
没有来源记录"变成一条可见的证据，而不是当作可信身份直接用于划分/决策。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from beamng_autopilot.experiments.manifest import (  # noqa: E402
    DatasetManifest, dir_group,
)


def _collection(tmp_path: Path, *, with_provenance: bool) -> Path:
    view = tmp_path / "coll" / "front_main"
    view.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(view / "frame_00000.npz"),
        colour=np.full((16, 20, 3), 128, np.uint8),
        label=np.full((16, 20), 1, np.uint8))
    meta = {"map_name": "italy", "source_id": "ring_20260923_183022",
            "frames": [{"path": "front_main/frame_00000.npz",
                        "view": "front_main", "exposure": 3,
                        "pos": [1.0, 2.0, 3.0], "heading": 0.1}]}
    if with_provenance:
        meta["map_name_source"] = "session.get_current().level"
    (tmp_path / "coll" / "meta.json").write_text(json.dumps(meta),
                                                 encoding="utf-8")
    return view


def test_map_name_without_provenance_is_flagged(tmp_path):
    view = _collection(tmp_path, with_provenance=False)
    mf = DatasetManifest.build([view], root=ROOT)
    notes = [n for n in mf.notes if "no provenance" in n]
    assert notes, mf.notes
    assert "map_name_source" in notes[0] and "italy" in notes[0]
    # 只是提示：分组与身份值本身不变，格式也不因此改判
    assert mf.records[0].group == "italy/ring_20260923_183022"
    assert mf.records[0].reject_reason == ""


def test_provenance_present_is_not_flagged(tmp_path):
    view = _collection(tmp_path, with_provenance=True)
    mf = DatasetManifest.build([view], root=ROOT)
    assert not [n for n in mf.notes if "no provenance" in n]


def test_dir_group_reads_identity_and_never_guesses_from_the_name(tmp_path):
    """场景/组键与 manifest 同一定义：`map/source_id`，缺身份不猜地图（W2）。

    为什么要有独立测试：分场景硬门、空间隔离和数据清单必须指同一个"场景"，
    否则"这个场景过没过门"会随入口而变；而按目录名猜地图会把
    `coll_ring1` 当成地图名，污染分组。
    """
    d = tmp_path / "coll_a" / "front_main"
    d.mkdir(parents=True)
    (d.parent / "meta.json").write_text(json.dumps(
        {"map_name": "italy", "source_id": "ring_a"}), encoding="utf-8")
    assert dir_group(d) == "italy/ring_a", "身份在父目录也要读到"
    # 没有 meta：不猜地图，也**不按目录名兜底**（两个采集的 front_main 同名，
    # 按名字当键会把它们合成一个场景）——用完整路径保证不合并
    e = tmp_path / "no_meta" / "front_main"
    e.mkdir(parents=True)
    g = dir_group(e)
    assert g.startswith("dir/") and "front_main" in g and "no_meta" in g, g
    f = tmp_path / "other" / "front_main"
    f.mkdir(parents=True)
    assert dir_group(f) != g, "同名视角目录绝不合并成一个场景"


def test_the_manifest_reads_the_directorys_own_credential(tmp_path):
    """资格以**目录自己的凭证**为准（方案 §6.1/A1），不看调用方声明。

    实测踩到：人工复核过的帧 sidecar 写着 ``label_source=human_revision``，
    但 ``DatasetManifest.build`` 用 ``paint_sources.get(...) or
    "engine_annotation"`` —— 从不读凭证，于是 14 帧全被判 rank=unreliable /
    valid=False，人工复核的成果直接被误判。
    """
    import numpy as np
    from beamng_autopilot.experiments.manifest import DatasetManifest

    def _dir(name: str, source: str) -> Path:
        d = tmp_path / name / "front_main"
        d.mkdir(parents=True)
        label = np.zeros((20, 30), np.uint8)
        label[5:15, :] = 1                    # road
        label[10, :6] = 2                     # line
        np.savez_compressed(d / "frame_00000.npz",
                            colour=np.full((20, 30, 3), 50, np.uint8),
                            label=label)
        (d / "meta.json").write_text(json.dumps({
            "map_name": "italy", "source_id": f"ring_{name}",
            "label_source": source, "frames": [
                {"path": "front_main/frame_00000.npz", "view": "front_main",
                 "exposure": 0, "pos": [0.0, 0.0, 0.0], "heading": 0.0}]}),
            encoding="utf-8")
        return d

    human = _dir("human", "human_revision")
    agent = _dir("agent", "agent_revision")
    mf = DatasetManifest.build([human, agent], root=tmp_path)
    by_src = {}
    for r in mf.records:
        q = (r.quality or {}).get("paint") or {}
        by_src[r.source_id] = q.get("rank")   # run 是视角目录名，会重名
    assert by_src.get("ring_human") == "verified", by_src
    assert by_src.get("ring_agent") == "agent", by_src
    # 调用方声明**不能**抬高质量：给 agent 目录声明 human 仍然是 agent
    mf2 = DatasetManifest.build([agent], root=tmp_path,
                                paint_sources={agent.name: "human_revision",
                                               str(agent): "human_revision"})
    q2 = (mf2.records[0].quality or {}).get("paint") or {}
    assert q2.get("rank") == "agent", q2
    # 没有凭证的目录：退到默认（unreliable）并在 notes 里写清"没有凭证"
    d3 = tmp_path / "naked" / "front_main"
    d3.mkdir(parents=True)
    label = np.zeros((20, 30), np.uint8)
    label[5:15, :] = 1
    np.savez_compressed(d3 / "frame_00000.npz",
                        colour=np.full((20, 30, 3), 60, np.uint8), label=label)
    (d3 / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": "ring_naked", "frames": [
            {"path": "front_main/frame_00000.npz", "view": "front_main",
             "exposure": 0, "pos": [0.0, 0.0, 0.0], "heading": 0.0}]}),
        encoding="utf-8")
    mf3 = DatasetManifest.build([d3], root=tmp_path)
    q3 = (mf3.records[0].quality or {}).get("paint") or {}
    assert q3.get("rank") == "unreliable", q3
    # 有 sidecar 但没声明来源：明确写出来（不是静默按默认值处理）
    assert any("declares no label_source" in n for n in mf3.notes), mf3.notes
