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

from beamng_autopilot.experiments.manifest import DatasetManifest  # noqa: E402


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
