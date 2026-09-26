"""采集隔离核对脚本的行为（E1 解除规格里的"距 dev >50 m 自动核对"）。

实测背景：新训练负例不得取自开发/评价帧或其空间邻近段。不同地图天然隔离；
同地图必须比缓冲距离远；**缺位姿记 UNKNOWN，不能当成"离得远"**。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_collect_isolation_check",
        ROOT / "scripts" / "m5_collect_isolation_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_collect_isolation_check"] = mod
    spec.loader.exec_module(mod)
    return mod


def _collection(tmp_path, name: str, map_name: str, positions: list) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({
        "map_name": map_name, "source_id": f"ring_{name}",
        "frames": [{"path": f"front_main/frame_{i:05d}.npz", "view": "front_main",
                    **({"pos": list(p)} if p else {})}
                   for i, p in enumerate(positions)]}), encoding="utf-8")
    return d


def test_other_map_is_isolated_without_distance(tmp_path, monkeypatch, capsys):
    mod = _load()
    monkeypatch.setattr(mod, "dev_positions",
                        lambda: {"italy": [(0.0, 0.0, 0.0)]})
    d = _collection(tmp_path, "jv", "johnson_valley", [(1.0, 2.0, 3.0)])
    rc = mod.main(["--dir", str(d), "--out", str(tmp_path / "r.json")])
    rep = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert rc == 0 and rep["all_isolated"] is True
    entry = rep["candidates"][str(d)]
    assert entry["verdict"] == "isolated_other_map"
    assert "min_distance_m" not in entry, "异图不该报距离（没有可比对象）"


def test_same_map_far_and_near(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "dev_positions",
                        lambda: {"italy": [(0.0, 0.0, 0.0)]})
    far = _collection(tmp_path, "far", "italy", [(400.0, 0.0, 0.0)])
    near = _collection(tmp_path, "near", "italy", [(10.0, 0.0, 0.0)])
    rc_far = mod.main(["--dir", str(far)])
    rc_near = mod.main(["--dir", str(near)])
    assert rc_far == 0, "400 m > 50 m 应通过"
    assert rc_near == 1, "10 m <= 50 m 必须拒绝"


def test_missing_pose_is_unknown_not_isolated(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "dev_positions",
                        lambda: {"italy": [(0.0, 0.0, 0.0)]})
    d = _collection(tmp_path, "nopose", "italy", [None, None])
    rc = mod.main(["--dir", str(d)])
    assert rc == 1, "缺位姿不能算通过（缺位姿 ≠ 离得远）"


def test_buffer_boundary_is_inclusive_of_rejection(tmp_path, monkeypatch):
    """正好等于缓冲距离 -> 拒绝（与 `SPATIAL_BUFFER_M` 同口径：<= 缓冲算邻近）。"""
    mod = _load()
    monkeypatch.setattr(mod, "dev_positions",
                        lambda: {"italy": [(0.0, 0.0, 0.0)]})
    d = _collection(tmp_path, "edge", "italy", [(50.0, 0.0, 0.0)])
    assert mod.main(["--dir", str(d)]) == 1
