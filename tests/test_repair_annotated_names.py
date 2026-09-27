"""错位标注的机械修复（检测在 `m5_annotation_readiness`，修复在这里）。

实测背景：保存名先自增导致整批错位 +1（frame_00001 里装的是 frame_00000），
导出帧自带 identity_provenance 可以自证来源，因此能**按源名改正**而不必重标；
但映射必须一一对应、目标必须在目录里，否则拒绝（交回重标）。
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
        "m5_repair_annotated_names",
        ROOT / "scripts" / "m5_repair_annotated_names.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_repair_annotated_names"] = mod
    spec.loader.exec_module(mod)
    return mod


def _misaligned_view(tmp_path, *, shift=1, n=4, stray=False) -> Path:
    d = tmp_path / "view"
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for i in range(n):
        name = f"frame_{i + shift:05d}.npz"          # 错位：名字比源大 shift
        src = f"npz:frame_{i:05d}.npz"
        np.savez(d / name, colour=np.full((8, 8, 3), 10 + i, np.uint8),
                 label=np.zeros((8, 8), np.uint8), exposure=i,
                 identity_json=json.dumps({"identity_provenance": {"pos": src}}))
        recs.append({"path": f"front_main/{name}", "exposure": i,
                     "classes_painted": {"line": 0, "road": 4}})
    # 未标注的原帧（错位 bug 的另一半：它会被标注版替换掉）
    np.savez(d / "frame_00000.npz", colour=np.full((8, 8, 3), 10, np.uint8),
             label=np.zeros((8, 8), np.uint8), exposure=0)
    recs.insert(0, {"path": "front_main/frame_00000.npz", "exposure": 0,
                    "classes_painted": {}})
    if stray:
        np.savez(d / f"frame_{n + shift:05d}.npz",
                 colour=np.full((8, 8, 3), 99, np.uint8),
                 label=np.zeros((8, 8), np.uint8))
    (d / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": "ring_x",
        "label_source": "human_revision",
        "annotation": {"reviewer": "owner", "identity_missing": []},
        "frames": recs}), encoding="utf-8")
    return d


def test_a_clean_shift_is_repaired_by_source_name(tmp_path):
    mod = _load()
    d = _misaligned_view(tmp_path)
    rep = mod.plan_view(d)
    assert not rep["refused"], rep["refused"]
    assert len(rep["mapping"]) == 4
    assert mod.main(["--dir", str(d), "--apply"]) in (0, 1)
    names = sorted(p.name for p in d.glob("frame_*.npz"))
    assert names == [f"frame_{i:05d}.npz" for i in range(4)], names
    assert not list(d.glob("*.tmp")), "改名后不留中间文件"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["frames"]) == 4, "未标注的种子条目要被丢掉"
    assert all(fr.get("classes_painted") for fr in meta["frames"])
    # 修完后内容与名字一致：再跑一遍应当"无需改名"
    assert not mod.plan_view(d)["mapping"]


def test_a_non_one_to_one_mapping_is_refused(tmp_path):
    mod = _load()
    d = _misaligned_view(tmp_path)
    # 把两个文件都指向同一个源名 -> 拒绝
    for name in ("frame_00001.npz", "frame_00002.npz"):
        np.savez(d / name, colour=np.full((8, 8, 3), 5, np.uint8),
                 label=np.zeros((8, 8), np.uint8),
                 identity_json=json.dumps(
                     {"identity_provenance": {"pos": "npz:frame_00000.npz"}}))
    rep = mod.plan_view(d)
    assert rep["refused"] and "一一对应" in rep["refused"][0]
