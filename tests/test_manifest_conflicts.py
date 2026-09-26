"""方案 v2 §3.1 / 验收 T01–T02：重复**别名**与内容**冲突**的显式归属。

独立复核发现的实测缺口（本文件逐条钉住）：

1. 旧 ``_reject_content_duplicates`` 只按图像内容哈希去重：同 RGB、不同标签
   的第二份被当成 "byte-identical" 静默拒绝，**谁活下来取决于目录顺序**，
   ``dataset_id`` 也跟着变——正是方案禁止的"因路径排序静默覆盖权威标签"；
2. 审计输出里没有 conflict 字段，也没有几何身份（尺寸/相机/位姿）比较；
3. 来源别名丢失（调用方只能看到"被拒绝的副本"，拿不到"同一张图的全部来源"）。

规则（manifest 里写死，与目录传入顺序无关）：

* (a) 同内容 + 同标签 + 同几何 ⇒ 一个评价样本 + 全部来源路径记为别名；
* (b) 同内容、标签不同 ⇒ 冲突；只有**恰好一侧**有 verified 凭证继承时才保留
  该侧并记继承，否则两份都隔离；
* (c) 同内容、几何身份不同 ⇒ 身份冲突，两份都隔离（相同像素不是独立样本）。

全部用合成 npz，不碰 GPU/游戏/真实 logs。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.manifest import DatasetManifest  # noqa: E402

#: 两个只差 fov 的相机：同图不同内参 = 身份冲突
_CAM_A = {"offset": [0.0, 1.5, 1.4], "fwd": [0.0, 1.0, 0.0],
          "up": [0.0, 0.0, 1.0], "fov_deg": 65.0, "width": 16, "height": 12}
_CAM_B = dict(_CAM_A, fov_deg=35.0)


def _label(kind: int, h: int = 12, w: int = 16) -> np.ndarray:
    lab = np.zeros((h, w), np.uint8)
    lab[4:8, :] = 1                            # road
    if kind:
        lab[6, 2:6] = 2                        # 冲突标签：多一条线
    return lab


def _write_collection(root: Path, name: str, *, frames: int = 2,
                      map_name: str = "italy", source_id: str | None = None,
                      label_kind: int = 0, base_gray: int = 60,
                      label_source: str | None = None,
                      dup_of: Path | None = None,
                      camera: dict | None = None,
                      pose: bool = True) -> Path:
    """写一个 ``<root>/<name>/front_main`` 采集，返回视角目录。

    ``dup_of``：逐帧复制该目录的 colour（内容哈希相同）；标签仍按
    ``label_kind`` 生成，所以能造出"同图不同标签"。
    """
    d = root / name / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    meta_frames = []
    for i in range(frames):
        if dup_of is not None:
            src = sorted(dup_of.glob("frame_*.npz"))[i]
            colour = np.asarray(np.load(src)["colour"], np.uint8)
        else:
            colour = np.full((12, 16, 3), base_gray + i, np.uint8)
        np.savez(d / f"frame_{i:05d}.npz", colour=colour,
                 label=_label(label_kind))
        frame = {"i": i, "view": "front_main", "exposure": i,
                 "t_wall": 1000.0 + i,
                 "path": f"front_main/frame_{i:05d}.npz"}
        if pose:
            frame["pos"] = [float(i), 0.0, 0.0]
            frame["heading"] = 0.0
        meta_frames.append(frame)
    meta = {"stamp": "20260926_000000", "roles": {"front_main": frames},
            "width": 16, "height": 12, "map_name": map_name,
            "map_name_source": "test",
            "source_id": source_id or f"ring_{name}", "frames": meta_frames}
    if camera is not None:
        meta["cameras"] = {"front_main": camera}
    if label_source is not None:
        meta["label_source"] = label_source
    (root / name / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# T01：相同图像/标签复制到两个目录，交换目录顺序
def test_t01_identical_copies_are_one_sample_with_aliases(tmp_path) -> None:
    a = _write_collection(tmp_path, "coll_a", frames=4)
    b = _write_collection(tmp_path, "coll_b", frames=4, dup_of=a)
    fwd = DatasetManifest.build([a, b], root=tmp_path)
    rev = DatasetManifest.build([b, a], root=tmp_path)

    for tag, mf in (("fwd", fwd), ("rev", rev)):
        accepted = [r for r in mf.records if not r.reject_reason]
        assert len(accepted) == 4, tag            # 一次评价，不是 8 次
        assert len(mf.by_split("train")) == 4, tag
        assert len(mf.rejected()) == 4, tag
        assert all(r.reject_code == "alias_duplicate"
                   for r in mf.rejected()), tag
        assert mf.conflicts == [], tag
        assert len(mf.aliases) == 4, tag
        for info in mf.aliases.values():
            assert info["n"] == 2 and len(info["paths"]) == 2
            assert info["n_accepted"] == 1
            assert Path(info["kept"]) in [Path(p) for p in info["paths"]]
        assert all(len(r.aliases) == 2 for r in accepted)
    # 全部 8 条来源路径都在别名里（两个目录都留下）
    alias_paths = {p for info in fwd.aliases.values() for p in info["paths"]}
    assert len(alias_paths) == 8
    assert {Path(p).parent.parent.name for p in alias_paths} == {"coll_a",
                                                                 "coll_b"}
    # 目录顺序不改变评价样本：同一个 dataset_id、同一批路径、同一份分数
    assert fwd.dataset_id == rev.dataset_id
    assert (sorted(r.path for r in fwd.by_split("train"))
            == sorted(r.path for r in rev.by_split("train")))
    assert fwd.coverage() == rev.coverage()
    assert fwd.audit()["n_records"] == rev.audit()["n_records"]
    assert fwd.audit()["n_rejected"] == rev.audit()["n_rejected"] == 4
    assert fwd.audit()["alias_groups"]["n"] == 4
    assert fwd.audit()["alias_groups"]["n_extra_paths"] == 4


# ---------------------------------------------------------------------------
# T02：同图像的两个冲突标签
def test_t02_label_conflict_isolates_every_copy_in_both_orders(tmp_path) -> None:
    a = _write_collection(tmp_path, "coll_a", frames=2, label_kind=0)
    b = _write_collection(tmp_path, "coll_b", frames=2, dup_of=a,
                          label_kind=1)
    fwd = DatasetManifest.build([a, b], root=tmp_path)
    rev = DatasetManifest.build([b, a], root=tmp_path)

    for tag, mf in (("fwd", fwd), ("rev", rev)):
        assert mf.by_split("train") == [], tag     # 一个都不许静默通过
        assert len(mf.rejected()) == 4, tag
        assert all(r.reject_code == "label_conflict"
                   for r in mf.rejected()), tag
        assert not mf.aliases, tag
        assert len(mf.conflicts) == 2, tag         # 每张图一条冲突
        record_paths = {r.path for r in mf.records}
        for c in mf.conflicts:
            assert c["kind"] == "label", c
            assert c["resolution"] == "isolated" and c["kept"] is None, c
            assert c["path_a"] != c["path_b"]
            assert c["label_sha16_a"] != c["label_sha16_b"]
            assert "label" in c["why"] and c["why"]
            # 两个来源路径都能定位到真实记录上
            assert c["path_a"] in record_paths
            assert c["path_b"] in record_paths
        assert mf.audit()["conflicts"]["n"] == 2
        assert mf.audit()["conflicts"]["n_isolated_copies"] == 4
    # 两个方向的选择必须一致：不能"最后读到的一份赢"
    assert fwd.dataset_id == rev.dataset_id
    assert (sorted(r.path for r in fwd.rejected())
            == sorted(r.path for r in rev.rejected()))
    sides = {(Path(c["path_a"]).parent.parent.name,
              Path(c["path_b"]).parent.parent.name) for c in fwd.conflicts}
    assert sides == {("coll_a", "coll_b")}


def test_a_verified_revision_wins_the_label_conflict_in_both_orders(tmp_path) -> None:
    """有明确修订继承（目录凭证 = human_revision）时保留已确认版本并记继承。"""
    engine = _write_collection(tmp_path, "coll_engine", frames=3,
                               label_kind=0)
    human = _write_collection(tmp_path, "coll_human", frames=3, dup_of=engine,
                              label_kind=1, label_source="human_revision")
    for order in ([engine, human], [human, engine]):
        mf = DatasetManifest.build(order, root=tmp_path)
        accepted = [r for r in mf.records if not r.reject_reason]
        assert len(accepted) == 3, order
        assert all("coll_human" in r.path for r in accepted), \
            "verified 凭证的一侧必须胜出，与目录顺序无关"
        assert all(r.quality["paint"]["rank"] == "verified"
                   for r in accepted)
        superseded = [r for r in mf.records if r.reject_reason]
        assert len(superseded) == 3
        assert all(r.reject_code == "label_conflict_superseded"
                   for r in superseded)
        assert len(mf.conflicts) == 3
        for c in mf.conflicts:
            assert c["resolution"] == "kept_verified"
            assert "coll_human" in c["kept"] and c["superseded"]
            assert "verified" in (c["source_rank_a"], c["source_rank_b"])
            assert {c["source_rank_a"], c["source_rank_b"]} == \
                {"verified", "unreliable"}
    assert (DatasetManifest.build([engine, human], root=tmp_path).dataset_id
            == DatasetManifest.build([human, engine], root=tmp_path).dataset_id)


def test_aliases_are_kept_within_the_verified_winner_side(tmp_path) -> None:
    """胜出侧自己有多份来源时，保留一个样本但把来源路径全记下来。"""
    engine = _write_collection(tmp_path, "coll_engine", frames=2, label_kind=0)
    human1 = _write_collection(tmp_path, "coll_human1", frames=2, dup_of=engine,
                               label_kind=1, label_source="human_revision")
    human2 = _write_collection(tmp_path, "coll_human2", frames=2, dup_of=engine,
                               label_kind=1, label_source="human_revision")
    mf = DatasetManifest.build([engine, human1, human2], root=tmp_path)
    accepted = [r for r in mf.records if not r.reject_reason]
    assert len(accepted) == 2
    assert all("coll_human1" in r.path for r in accepted), \
        "胜出侧内部也按排序取最小路径"
    assert len(mf.aliases) == 2
    assert all(a["n"] == 2 and a["n_accepted"] == 1
               for a in mf.aliases.values())
    assert all(len(r.aliases) == 2 for r in accepted)
    assert sorted(r.reject_code for r in mf.rejected()) == \
        ["alias_duplicate", "alias_duplicate",
         "label_conflict_superseded", "label_conflict_superseded"]


# ---------------------------------------------------------------------------
# 几何身份冲突：同图字节、不同相机/位姿
def test_same_bytes_different_camera_are_an_identity_conflict(tmp_path) -> None:
    a = _write_collection(tmp_path, "coll_a", frames=2, camera=_CAM_A)
    b = _write_collection(tmp_path, "coll_b", frames=2, dup_of=a,
                          camera=_CAM_B)
    fwd = DatasetManifest.build([a, b], root=tmp_path)
    rev = DatasetManifest.build([b, a], root=tmp_path)
    for tag, mf in (("fwd", fwd), ("rev", rev)):
        assert mf.by_split("train") == [], tag
        assert len(mf.rejected()) == 4, tag
        assert all(r.reject_code == "identity_conflict"
                   for r in mf.rejected()), tag
        assert len(mf.conflicts) == 2, tag
        for c in mf.conflicts:
            assert c["kind"] == "identity"
            assert c["camera_sha16_a"] != c["camera_sha16_b"]
            assert c["resolution"] == "isolated"
            assert "identity" in c["why"] or "geometry" in c["why"]
    assert fwd.dataset_id == rev.dataset_id


def test_same_bytes_with_and_without_pose_are_isolated(tmp_path) -> None:
    a = _write_collection(tmp_path, "coll_a", frames=2, pose=True)
    b = _write_collection(tmp_path, "coll_b", frames=2, dup_of=a, pose=False)
    mf = DatasetManifest.build([a, b], root=tmp_path)
    assert not [r for r in mf.records if not r.reject_reason]
    assert all(r.reject_code == "identity_conflict" for r in mf.rejected())
    assert len(mf.conflicts) == 2
    for c in mf.conflicts:
        assert c["kind"] == "identity"
        assert c["pose_state_a"] != c["pose_state_b"], c
        assert {c["pose_state_a"], c["pose_state_b"]} == {"pos+heading",
                                                          "unknown"}


def test_label_and_geometry_conflict_together_still_isolates(tmp_path) -> None:
    a = _write_collection(tmp_path, "coll_a", frames=1, label_kind=0,
                          pose=True)
    b = _write_collection(tmp_path, "coll_b", frames=1, dup_of=a,
                          label_kind=1, pose=False)
    mf = DatasetManifest.build([a, b], root=tmp_path)
    assert len(mf.rejected()) == 2
    assert len(mf.conflicts) == 1
    assert mf.conflicts[0]["kind"] == "label+identity"
    assert all(r.reject_code == "identity_conflict" for r in mf.rejected())


# ---------------------------------------------------------------------------
# 回归：正常的多目录、互不相同的帧不许被过度隔离
def test_normal_distinct_frames_are_not_over_rejected(tmp_path) -> None:
    a = _write_collection(tmp_path, "coll_a", frames=4, base_gray=40)
    b = _write_collection(tmp_path, "coll_b", frames=4, base_gray=120)
    mf = DatasetManifest.build([a, b], root=tmp_path)
    assert len(mf.rejected()) == 0
    assert len(mf.by_split("train")) == 8
    assert mf.conflicts == [] and mf.aliases == {}
    assert mf.audit()["n_rejected"] == 0
    assert mf.audit()["conflicts"]["n"] == 0
    assert mf.audit()["alias_groups"]["n"] == 0


def test_survivor_is_the_canonical_min_not_the_first_directory(tmp_path) -> None:
    """生存者是排序最小 ``(run, path)``，不是 ``runs`` 里第一个目录。"""
    a = _write_collection(tmp_path, "coll_a", frames=2)
    b = _write_collection(tmp_path, "coll_b", frames=2, dup_of=a)
    mf = DatasetManifest.build([b, a], root=tmp_path)   # 故意把 b 放前面
    accepted = [r for r in mf.records if not r.reject_reason]
    assert len(accepted) == 2
    assert all("coll_a" in r.path for r in accepted), \
        "把 b 放前面不能改变生存者"


# ---------------------------------------------------------------------------
# 报告结构：冲突/别名要能被保存/加载后继续报告
def test_manifest_json_round_trip_keeps_conflicts_and_aliases(tmp_path) -> None:
    a = _write_collection(tmp_path, "coll_a", frames=2)
    b = _write_collection(tmp_path, "coll_b", frames=2, dup_of=a)
    bad = _write_collection(tmp_path, "coll_bad", frames=1, label_kind=1)
    # 让 bad 的第一帧变成"与 a 的第一帧同图、不同标签"的冲突
    np.savez(bad / "frame_00000.npz",
             colour=np.asarray(np.load(sorted(a.glob("frame_*.npz"))[0])
                               ["colour"]),
             label=_label(1))
    mf = DatasetManifest.build([a, b, bad], root=tmp_path)
    assert mf.aliases and mf.conflicts
    p = mf.save(tmp_path / "ds.json")
    back = DatasetManifest.load(p)
    assert back.dataset_id == mf.dataset_id
    assert back.conflicts == mf.conflicts
    assert back.aliases == mf.aliases
    assert [r.aliases for r in back.records] == [r.aliases for r in mf.records]
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert "conflicts" in raw and "aliases" in raw


def test_dataset_id_differs_when_the_conflict_outcome_differs(tmp_path) -> None:
    """接受集真的不同（全隔离 vs 保留 verified 侧）⇒ dataset_id 必须可见地不同。"""
    engine = _write_collection(tmp_path, "coll_engine", frames=1, label_kind=0)
    other = _write_collection(tmp_path, "coll_other", frames=1, label_kind=1,
                              dup_of=engine)
    isolated = DatasetManifest.build([engine, other], root=tmp_path)
    human = _write_collection(tmp_path, "coll_human", frames=1, label_kind=1,
                              dup_of=engine, label_source="human_revision")
    kept = DatasetManifest.build([engine, human], root=tmp_path)
    assert not [r for r in isolated.records if not r.reject_reason]
    assert [r for r in kept.records if not r.reject_reason]
    assert isolated.dataset_id != kept.dataset_id