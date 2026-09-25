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
