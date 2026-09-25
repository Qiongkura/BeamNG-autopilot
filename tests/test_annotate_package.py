"""标注任务包（`m5_annotate_package.py`）的接线与不变量。

为什么单独测：本项目唯一挡住晋级的硬门是标线/身份指标，而它现在是 UNKNOWN——
引擎不给 line 类，标线真值只能人工画。把"该画哪几帧"变成两条命令的这一步如果
悄悄出错（身份没带上、包里的 label 把预填顶掉、空包也产出），人工修订就会白做。
实测教训：3h 轮有 48 帧人工修订因为丢了身份被审计拒收且**不可恢复**。

钉住四件事：

1. 包里 npz **不带 label**（带 label 会命中标注器的续标分支，`--prefill-model` 被跳过）；
2. 身份**带着走**：`<out>/<view>/meta.json` 是采集 meta 的复制，`load_frame_dir`
   读回来四个身份字段都齐全（读不齐就返回 4，不让人去画）；
3. 每视角取前 N 帧（队列本身按漆线像素降序，不重排）；
4. 空队列/源文件不在 → 明确失败码，不产出空包。
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
        "m5_annotate_package", ROOT / "scripts" / "m5_annotate_package.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_annotate_package"] = mod
    spec.loader.exec_module(mod)
    return mod


def _collection(tmp_path: Path, *, views=("front_main", "pillar_left"),
                n=3) -> Path:
    """假采集目录：meta.json 带身份 + 每视角 n 帧 npz（colour/label/annotation_raw）。"""
    root = tmp_path / "collect_x"
    root.mkdir(parents=True, exist_ok=True)
    recs = []
    for v in views:
        d = root / v
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            colour = np.full((12, 16, 3), 40 + i * 10, np.uint8)
            label = np.zeros((12, 16), np.uint8)
            label[4:9, :] = 1
            np.savez_compressed(d / f"frame_{i:05d}.npz", colour=colour,
                                label=label,
                                annotation_raw=np.zeros_like(colour))
            recs.append({"i": i, "view": v, "exposure": i,
                         "path": f"{v}/frame_{i:05d}.npz",
                         "line_pixels": 1000 - i * 100,
                         "pos": [10.0 + i, 20.0, 0.0], "heading": 0.5})
    (root / "meta.json").write_text(json.dumps(
        {"map_name": "italy", "map_name_source": "session.get_current().level",
         "source_id": "ring_test", "roles": {v: n for v in views},
         "frames": recs}, ensure_ascii=False), encoding="utf-8")
    return root


def _queue(tmp_path: Path, coll: Path, views=("front_main", "pillar_left"),
           n=3) -> Path:
    recs = [f for f in json.loads((coll / "meta.json").read_text(
        encoding="utf-8"))["frames"]]
    recs.sort(key=lambda r: -int(r["line_pixels"]))
    p = tmp_path / "review_queue_collect_x.json"
    p.write_text(json.dumps({"why": "test", "out_dir": str(coll),
                             "frames": recs}, ensure_ascii=False),
                 encoding="utf-8")
    return p


def test_package_carries_identity_and_drops_the_label(tmp_path):
    pkg = _load()
    coll = _collection(tmp_path)
    q = _queue(tmp_path, coll)
    out = tmp_path / "pkg"
    assert pkg.main(["--review-queue", str(q), "--out", str(out),
                     "--per-view", "2"]) == 0
    from scripts.m5_annotate_manual import load_frame_dir
    for view in ("front_main", "pillar_left"):
        d = out / view
        assert (d / "meta.json").exists(), "身份要跟着走（审计按 <dir>/meta.json 读组）"
        npzs = sorted(d.glob("frame_*.npz"))
        assert len(npzs) == 2, npzs          # 每视角前 2 帧
        with np.load(npzs[0]) as z:
            assert "colour" in z.files
            assert "label" not in z.files, \
                "带 label 会走续标分支，--prefill-model 就不生效了"
            assert str(z["map_name"].item()) == "italy"
            assert str(z["source_id"].item()) == "ring_test"
        frames, idents, _r, _p = load_frame_dir(d)
        assert len(frames) == 2
        for it in idents:
            assert it["map_name"] and it["source_id"], it
            assert it["pos"] and it["heading"], it
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "m5_annotate_manual.py" in readme and "--prefill-model" in readme
    assert "m5_seg_dataset_audit.py" in readme, "要给出身份自查命令"
    assert "遮挡" in readme and "油漆桶" in readme, "键位要写清楚"


def test_per_view_cap_picks_the_most_paint_first(tmp_path):
    pkg = _load()
    coll = _collection(tmp_path, views=("front_main",), n=3)
    q = _queue(tmp_path, coll, views=("front_main",))
    picked = pkg.pick_per_view(pkg.load_review_queue(q)[1], 2)
    assert [f["line_pixels"] for f in picked] == [1000, 900]


def test_a_missing_source_frame_is_reported_not_swallowed(tmp_path):
    pkg = _load()
    coll = _collection(tmp_path, views=("front_main",), n=2)
    q = _queue(tmp_path, coll, views=("front_main",))
    blob = json.loads(q.read_text(encoding="utf-8"))
    blob["frames"].append({"view": "front_main", "path": "front_main/gone.npz",
                           "line_pixels": 5, "pos": [1, 2, 3], "heading": 0.0})
    q.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "pkg2"
    assert pkg.main(["--review-queue", str(q), "--out", str(out),
                     "--per-view", "5"]) == 0
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "缺 1 个源文件" in readme, readme[:400]


def test_an_empty_queue_does_not_produce_an_empty_package(tmp_path):
    pkg = _load()
    coll = _collection(tmp_path)
    q = tmp_path / "empty.json"
    q.write_text(json.dumps({"out_dir": str(coll), "frames": []}),
                 encoding="utf-8")
    out = tmp_path / "pkg3"
    assert pkg.main(["--review-queue", str(q), "--out", str(out)]) == 3
    assert not (out / "README.md").exists()


def test_a_queue_pointing_at_a_missing_collection_fails_loudly(tmp_path):
    pkg = _load()
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"out_dir": str(tmp_path / "nope"),
                             "frames": [{"view": "front_main",
                                         "path": "front_main/x.npz",
                                         "line_pixels": 1}]}),
                 encoding="utf-8")
    assert pkg.main(["--review-queue", str(q),
                     "--out", str(tmp_path / "p")]) == 2


def test_a_view_level_meta_is_used_and_filtered_to_the_package(tmp_path):
    """视角级 meta（agent 标注池那种）也要能打包，且只留包内帧。

    实测踩到：agent 标注池每个**视角目录**各有一份 meta（里面列着全部视角的帧），
    采集目录下没有 meta.json -> 打包直接 FileNotFoundError。回退到视角级 meta
    后还必须**按视角+文件名过滤**：不同视角的同名帧文件名相同，只按文件名过滤
    会把别的视角的帧一起留下（实测 4 帧的包留下 32 条记录）。
    """
    pkg = _load()
    coll = _collection(tmp_path, views=("front_main", "pillar_left"), n=3)
    # 把 meta 挪到视角目录（每个视角一份，列出全部视角的帧）
    blob = json.loads((coll / "meta.json").read_text(encoding="utf-8"))
    for v in ("front_main", "pillar_left"):
        (coll / v / "meta.json").write_text(
            json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    (coll / "meta.json").unlink()
    frames = [f for f in blob["frames"] if f["view"] == "front_main"][:2]
    q = tmp_path / "q_view_meta.json"
    q.write_text(json.dumps({"why": "t", "frames": frames}),
                 encoding="utf-8")
    out = tmp_path / "pkg_view_meta"
    rc = pkg.main(["--review-queue", str(q), "--collection", str(coll),
                   "--out", str(out)])
    assert rc == 0, rc
    meta = json.loads((out / "front_main" / "meta.json").read_text(
        encoding="utf-8"))
    assert meta["map_name"] == "italy" and meta["source_id"] == "ring_test", meta
    # 只留包内帧（2 帧），且不含别的视角
    assert len(meta["frames"]) == 2, meta["frames"]
    assert {r["view"] for r in meta["frames"]} == {"front_main"}, meta["frames"]
    assert len(list((out / "front_main").glob("frame_*.npz"))) == 2
    # 采集级 meta 仍然优先（老路径不变）
    coll2 = _collection(tmp_path / "second", views=("front_main",), n=2)
    q2 = tmp_path / "q2.json"
    q2.write_text(json.dumps({"why": "t", "frames": json.loads(
        (coll2 / "meta.json").read_text(encoding="utf-8"))["frames"]}),
        encoding="utf-8")
    out2 = tmp_path / "pkg_coll_meta"
    assert pkg.main(["--review-queue", str(q2), "--collection", str(coll2),
                     "--out", str(out2)]) == 0
    assert (out2 / "front_main" / "meta.json").is_file()


def test_the_annotation_sidecar_records_who_reviewed_and_which_classes(tmp_path):
    """验收要求「能查询任意帧是**谁**、何时、对哪些类别和区域做了复核」。

    实测缺口：sidecar 有 tool/annotated_at/unknown_px，但没有复核人，也没有
    "这一帧标了哪些类别"——于是"谁复核的"只能靠记忆。现在写
    ``annotation.reviewer`` 与逐帧 ``classes_painted``。
    """
    import importlib.util
    import sys
    from pathlib import Path as _P

    import numpy as np

    root = _P(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_annotate_manual_t", root / "scripts" / "m5_annotate_manual.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_annotate_manual_t"] = mod
    spec.loader.exec_module(mod)

    out = tmp_path / "annotated" / "front_main"
    out.mkdir(parents=True)
    label = np.zeros((6, 8), np.uint8)
    label[1, :] = 2                     # line
    label[2:5, :] = 1                   # road
    label[5, :3] = 255                  # unknown
    np.savez_compressed(out / "frame_00000.npz",
                        colour=np.zeros((6, 8, 3), np.uint8), label=label)
    rec = {"path": "frame_00000.npz", "view": "front_main", "exposure": 0,
           "pos": [1.0, 2.0, 0.0], "heading": 0.0,
           "classes_painted": {"line": int((label == 2).sum()),
                               "road": int((label == 1).sum()),
                               "background": int((label == 0).sum()),
                               "unknown": int((label == 255).sum())}}
    fp = mod.write_sidecar(out, [rec],
                           identity={"map_name": "italy", "source_id": "ring_x"},
                           annotation_reviewer="tester")
    import json
    blob = json.loads(fp.read_text(encoding="utf-8"))
    assert blob["label_source"] == "human_revision", blob["label_source"]
    assert blob["annotation"]["reviewer"] == "tester", blob["annotation"]
    assert blob["annotation"]["annotated_at"], blob["annotation"]
    got = blob["frames"][0]["classes_painted"]
    assert got == {"line": 8, "road": 24, "background": 13, "unknown": 3}, got
    # 复核人缺省不编名字（拿不到就空）
    assert mod._default_reviewer() is not None
