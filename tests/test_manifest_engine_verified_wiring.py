"""清单/审计链对 `engine_verified` 的接线（T16 §4.3 反伪造 + 自动真值接入）。

实测缺口（2026-09-28）：`DatasetManifest.build` 调 `audit_label` 时**没传凭证**，
于是声明 `engine_verified` 的目录（带 truth_provenance v1）被判成"无凭证"降
`absent`，`paint_valid_frames=0`——自动真值导出的站点在清单里等于没接上。
本文件把两侧都钉住：

* 有凭证（v1 + report.verified）-> `paint_valid_frames > 0`（可当漆线真值）；
* 没凭证 / 伪造凭证 -> `paint_valid_frames == 0`（不得升格，方案 §4.3）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.manifest import DatasetManifest  # noqa: E402


def _pkg(tmp_path: Path, *, credential: dict | None) -> Path:
    """造一个最小采集包：front_main/frame_00000.npz + meta.json (+ annotation.json)。"""
    pkg = tmp_path / "pkg_auto"
    view = pkg / "front_main"
    view.mkdir(parents=True, exist_ok=True)
    colour = np.full((24, 32, 3), 120, np.uint8)
    label = np.zeros((24, 32), np.uint8)
    label[8:20, :] = 1                      # 路面
    label[14, :8] = 2                       # 漆线
    np.savez_compressed(view / "frame_00000.npz", colour=colour, label=label)
    (pkg / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": "m5auto_a5",
        "map_name_source": "m5_controlled_scenes.py (generated scenario)",
        "frames": [{"i": 0, "view": "front_main", "exposure": 0,
                    "t_wall": 1.0, "path": "front_main/frame_00000.npz"}]}),
        encoding="utf-8")
    if credential is not None:
        (pkg / "annotation.json").write_text(
            json.dumps(credential, ensure_ascii=False), encoding="utf-8")
    return view


def _cred(*, verified: bool = True, contract: str = "v1") -> dict:
    return {"label_source": "engine_verified", "truth_contract": contract,
            "truth_provenance": {
                "generator": {"name": "m5_controlled_scenes.py",
                              "version": "v1", "sha": "abc123"},
                "labels": {"label_sha": "deadbeef"},
                "report": {"verified": verified,
                           "verifier_version": "auto_truth/1"}},
            "frames": [{"frame_id": "f0", "label_sha": "deadbeef"}]}


def _paint_valid(view: Path) -> int:
    mf = DatasetManifest.build([view], root=ROOT,
                               paint_sources={str(view): "engine_verified"})
    cov = mf.coverage().get("train") or {}
    return int(cov.get("paint_valid_frames") or 0)


def test_engine_verified_with_provenance_counts_as_paint_truth(tmp_path):
    view = _pkg(tmp_path, credential=_cred())
    assert _paint_valid(view) > 0, "带 v1 凭证的 engine_verified 必须算可用漆线真值"


def test_engine_verified_without_credential_does_not_count(tmp_path):
    view = _pkg(tmp_path, credential=None)
    assert _paint_valid(view) == 0, "无凭证的 engine_verified 不得升格（反伪造）"


def test_engine_verified_with_unverified_report_does_not_count(tmp_path):
    view = _pkg(tmp_path, credential=_cred(verified=False))
    assert _paint_valid(view) == 0, "report.verified=False 时不得算真值"


def test_engine_verified_with_wrong_contract_does_not_count(tmp_path):
    view = _pkg(tmp_path, credential=_cred(contract="v0"))
    assert _paint_valid(view) == 0, "契约不是 v1 时不得算真值"
