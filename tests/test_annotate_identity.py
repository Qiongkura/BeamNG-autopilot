"""标注导出的身份记录：搬运真实身份，缺身份就如实报缺。

背景（实测缺陷）：`t14_3h_20260924_1906` 审计里 48 帧人工修订全部被判
"no map identity in the recording"——npz 只有 `colour`/`label`，标注器
明明握着来源文件却没写下来。这组测试钉住两件事：

1. 来源 recording 里有 `map_name`/`source_id`/`pos`/`heading` 时，导出必须
   携带，并且**组名与来源一致**（否则注释副本会变成"新组"，把泄漏隔离
   打开一个口子）；
2. 来源没有身份时，字段保持空、进 `identity_missing`，审计据此拒绝——绝不
   默认填一个地图名。参数也不能声明身份。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "m5_annotate_manual", ROOT / "scripts" / "m5_annotate_manual.py")
ann = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ann)

from beamng_autopilot.experiments.manifest import DatasetManifest  # noqa: E402


def _rgb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(24, 32, 3), dtype=np.uint8)


def _label(seed: int) -> np.ndarray:
    lab = np.zeros((24, 32), np.uint8)
    lab[10:14, :] = ann.CLS_ROAD
    lab[12, :] = ann.CLS_LINE
    return lab


def _write_source_npz(path: Path, seed: int) -> None:
    np.savez_compressed(str(path), colour=_rgb(seed), label=_label(seed))


def _source_collection(tmp_path: Path, *, frames: int = 2):
    """`<tmp>/coll/front_main/*.npz` + `<tmp>/coll/meta.json`（采集器契约）。"""
    view = tmp_path / "coll" / "front_main"
    view.mkdir(parents=True, exist_ok=True)
    recs = []
    for i in range(frames):
        name = f"frame_{i:05d}.npz"
        _write_source_npz(view / name, i)
        recs.append({"i": i, "view": "front_main", "exposure": 100 + i,
                     "t_wall": 1.5 + i, "pos": [100.0 + i, 200.0, 3.5],
                     "heading": 0.25 + i, "path": f"front_main/{name}"})
    (tmp_path / "coll" / "meta.json").write_text(json.dumps({
        "map_name": "italy", "map_name_source": "session.get_current().level",
        "source_id": "ring_20260923_135342", "frames": recs,
    }, ensure_ascii=False), encoding="utf-8")
    return view


def _export_all(out_dir: Path, view: Path, *, in_place: bool = False):
    """走标注器的加载/导出/边车三步，返回 (idents, sidecar_path)。"""
    frames, idents, _resume, _paths = ann.load_frame_dir(
        view, out_dir=out_dir if in_place else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seed = ann._sidecar_seed(out_dir)
    for i, (rgb, _idx) in enumerate(frames):
        fp = out_dir / f"frame_{i + 1:05d}.npz"
        ann.export_frame(fp, rgb, _label(i), idents[i])
        records.append({"path": fp.name,
                        "source_path": idents[i].get("source_path") or "",
                        "map_name": idents[i].get("map_name") or "",
                        "source_id": idents[i].get("source_id") or "",
                        "view": idents[i].get("view"),
                        "exposure": idents[i].get("exposure"),
                        "pos": idents[i].get("pos"),
                        "heading": idents[i].get("heading"),
                        "identity_source": idents[i]["identity_source"],
                        "identity_missing": list(idents[i]["identity_missing"])})
    side = ann.write_sidecar(out_dir, records, identity=idents[0], seed=seed)
    return idents, side


# ---------------------------------------------------------------- 有身份时

def test_identity_carried_from_collection_meta(tmp_path):
    view = _source_collection(tmp_path)
    idents, side = _export_all(tmp_path / "labeled", view)

    assert idents[0]["map_name"] == "italy"
    assert idents[0]["source_id"] == "ring_20260923_135342"
    assert idents[0]["pos"] == [100.0, 200.0, 3.5]
    assert idents[0]["heading"] == 0.25
    assert idents[0]["view"] == "front_main"
    assert idents[0]["exposure"] == 100
    assert idents[0]["identity_missing"] == []
    assert idents[0]["identity_provenance"]["map_name"] == "meta:run"
    assert idents[0]["identity_provenance"]["pos"] == "meta:frame"

    # npz: 每帧都带身份，且没有 object dtype（严格加载器会拒绝）
    with np.load(tmp_path / "labeled" / "frame_00001.npz") as z:
        assert str(z["map_name"]) == "italy"
        assert str(z["source_id"]) == "ring_20260923_135342"
        assert z["pos"].tolist() == [100.0, 200.0, 3.5]
        assert float(z["heading"]) == 0.25
        for k in z.files:
            assert z[k].dtype.kind in "USfiu", (k, z[k].dtype)

    # sidecar: 审计按同样的键读取
    side_json = json.loads(side.read_text(encoding="utf-8"))
    assert side_json["map_name"] == "italy"
    assert side_json["source_id"] == "ring_20260923_135342"
    assert side_json["label_source"] == "human_revision"
    assert side_json["map_name_source"] == "session.get_current().level"
    by_name = {Path(f["path"]).name: f for f in side_json["frames"]}
    assert by_name["frame_00001.npz"]["pos"] == [100.0, 200.0, 3.5]
    assert by_name["frame_00001.npz"]["exposure"] == 100


def test_annotated_group_equals_source_group(tmp_path):
    """注释副本不得变成新组：那会让泄漏隔离看不见同一次采集。"""
    view = _source_collection(tmp_path)
    _export_all(tmp_path / "labeled", view)
    src = DatasetManifest.build([view], root=ROOT)
    out = DatasetManifest.build([tmp_path / "labeled"], root=ROOT,
                                paint_sources={"labeled": "human_revision"})
    assert src.records[0].group == "italy/ring_20260923_135342"
    assert out.records[0].group == src.records[0].group
    assert out.records[0].map_name == "italy"
    assert out.records[0].source_id == "ring_20260923_135342"
    assert out.records[0].reject_reason == ""
    assert out.records[0].split == "train"
    assert out.records[0].exposure == 100          # 泄漏检查要用的曝光号


def test_npz_identity_beats_run_meta(tmp_path):
    """npz 是逐帧证据，优先于 run 级 meta。"""
    view = _source_collection(tmp_path)
    with np.load(view / "frame_00000.npz") as z:
        rgb, lab = z["colour"], z["label"]
    np.savez_compressed(str(view / "frame_00000.npz"), colour=rgb, label=lab,
                        map_name=np.array("east_coast_usa"),
                        source_id=np.array("ring_ec_1"),
                        pos=np.array([1.0, 2.0, 0.0]), heading=np.array(0.5))
    _frames, idents, _r, _p = ann.load_frame_dir(view)
    assert idents[0]["map_name"] == "east_coast_usa"
    assert idents[0]["source_id"] == "ring_ec_1"
    assert idents[0]["identity_provenance"]["map_name"].startswith("npz:")


# ---------------------------------------------------------------- 缺身份时

def test_missing_identity_reported_not_invented(tmp_path):
    view = tmp_path / "raw"
    view.mkdir()
    _write_source_npz(view / "frame_00000.npz", 7)     # 没有 meta.json
    idents, side = _export_all(tmp_path / "labeled_raw", view)

    assert idents[0]["map_name"] is None
    assert idents[0]["source_id"] is None
    assert sorted(idents[0]["identity_missing"]) == \
        ["heading", "map_name", "pos", "source_id"]
    assert idents[0]["identity_source"] == "unavailable"

    with np.load(tmp_path / "labeled_raw" / "frame_00001.npz") as z:
        assert str(z["map_name"]) == ""
        assert str(z["source_id"]) == ""
        assert "pos" not in z.files and "heading" not in z.files
        blob = json.loads(str(z["identity_json"]))
        assert blob["identity_missing"]

    text = side.read_text(encoding="utf-8")
    assert "italy" not in text                        # 没有默认地图名
    side_json = json.loads(text)
    assert side_json["map_name"] == ""
    assert side_json["identity_note"]
    assert sorted(side_json["annotation"]["identity_missing"]) == \
        ["heading", "map_name", "pos", "source_id"]

    # 审计必须拒绝，而不是当成可训练帧
    mf = DatasetManifest.build([tmp_path / "labeled_raw"], root=ROOT)
    assert mf.records[0].reject_reason.startswith("no map identity")
    assert mf.records[0].split == "none"
    assert len(mf.rejected()) == 1


def test_identity_cannot_be_declared_by_flag():
    opts = {o for a in ann.build_parser()._actions
            for o in a.option_strings}
    assert not {o for o in opts if "map" in o or "source" in o}, opts


# ---------------------------------------------------------------- 合并与冲突

def test_sidecar_merge_keeps_existing_frames(tmp_path):
    """原地标注：已有 meta 的其它帧记录不能被覆盖掉。"""
    view = _source_collection(tmp_path, frames=2)
    extra = {"path": "front_main/frame_00099.npz", "exposure": 999,
             "pos": [9.0, 9.0, 0.0], "view": "front_main"}
    meta = json.loads((tmp_path / "coll" / "meta.json").read_text("utf-8"))
    meta["frames"] = list(meta["frames"]) + [extra]
    (tmp_path / "coll" / "meta.json").write_text(json.dumps(meta),
                                                 encoding="utf-8")

    _export_all(view, view, in_place=True)            # out_dir == 源目录
    side = json.loads((view / "meta.json").read_text(encoding="utf-8"))
    names = {Path(f["path"]).name for f in side["frames"]}
    assert "frame_00099.npz" in names                 # 外来帧保留
    assert "frame_00000.npz" in names                 # 未标注帧也保留
    assert side["map_name"] == "italy"


def test_conflicting_identity_kept_and_reported(tmp_path):
    """目录已有身份与来源冲突：保留原值并写明冲突，不静默改写。"""
    out = tmp_path / "labeled"
    out.mkdir()
    (out / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": "ring_A", "frames": []}),
        encoding="utf-8")
    ident = ann.frame_identity([("npz:frame", {
        "map_name": "east_coast_usa", "source_id": "ring_B",
        "pos": [0.0, 0.0, 0.0], "heading": 0.0})])
    side = ann.write_sidecar(out, [{"path": "frame_00001.npz"}],
                             identity=ident)
    got = json.loads(side.read_text(encoding="utf-8"))
    assert got["map_name"] == "italy" and got["source_id"] == "ring_A"
    assert got["identity_conflicts"]


def test_inplace_resume_paths_kept(tmp_path):
    """重构不得破坏原地续标：回访帧必须覆盖自己的输出。"""
    view = _source_collection(tmp_path, frames=2)
    frames, idents, resume_labels, resume_paths = ann.load_frame_dir(
        view, out_dir=view)
    assert len(frames) == len(idents) == 2
    assert set(resume_labels) == {0, 1}
    assert set(resume_paths) == {0, 1}
    assert resume_paths[0][0].name == "frame_00000.npz"
    assert resume_paths[0][1].name == "preview_00000.png"
