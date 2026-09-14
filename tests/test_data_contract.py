"""Tests for the artifact/data provenance contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from beamng_autopilot.data_contract import (
    MANIFEST_VERSION,
    exclusion_reason,
    is_excluded_episode,
    make_seg_record,
    make_npz_record,
    stable_id,
    validate_record,
    write_manifest,
)


def _seg_dir(root: Path, name: str, value: int = 2) -> Path:
    p = root / name
    p.mkdir(parents=True)
    np.savez_compressed(
        p / "frame_00000.npz",
        colour=np.zeros((8, 10, 3), dtype=np.uint8),
        label=np.full((8, 10), value, dtype=np.uint8),
    )
    return p


def test_stable_id_uses_path_not_basename(tmp_path) -> None:
    a = _seg_dir(tmp_path / "a", "same")
    b = _seg_dir(tmp_path / "b", "same")
    assert stable_id(a, tmp_path) != stable_id(b, tmp_path)


def test_seg_record_carries_shape_and_manual_source(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    p = _seg_dir(root, "manual_current_full")
    rec = make_seg_record(p, root)
    assert rec["manifest_version"] == MANIFEST_VERSION
    assert rec["artifact_type"] == "seg_run"
    assert rec["domain"] == "manual_town"
    assert rec["label"]["source"] == "manual"
    assert rec["sensor"]["width"] == 10
    assert rec["sensor"]["height"] == 8
    assert rec["stats"]["frames"] == 1
    assert validate_record(rec) == []


def test_shadow_record_exposes_v2_v3_contract(tmp_path) -> None:
    p = tmp_path / "shadow_fsd_town.npz"
    np.savez_compressed(
        p, version=np.int64(3), t=np.zeros(2),
        rgb=np.zeros((2, 4, 5, 3), dtype=np.uint8),
        fmap=np.zeros((2, 4, 6, 6), dtype=np.float32),
        bev=np.zeros((2, 6, 6), dtype=np.float32),
    )
    rec = make_npz_record(p, tmp_path, "shadow_episode")
    assert rec["episode_version"] == 3
    assert rec["sensor"]["fmap_contract"] == "v3"
    assert rec["sensor"]["fmap_channels"] == 4
    assert rec["stats"]["frames"] == 2


def test_write_manifest_rejects_invalid_record(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_manifest([{"manifest_version": 1}], tmp_path / "m.json")


def test_bush_episode_is_excluded(tmp_path) -> None:
    name = "shadow_fsd_1789298914_20260913_193100.npz"
    assert is_excluded_episode(name)
    assert exclusion_reason(name) == "camera_in_vegetation_20260913"
    assert not is_excluded_episode("shadow_fsd_town.npz")
    assert exclusion_reason("shadow_fsd_town.npz") is None


def test_make_npz_record_marks_excluded_split(tmp_path) -> None:
    p = tmp_path / "shadow_fsd_1789298914_20260913_193100.npz"
    np.savez_compressed(
        p, version=np.int64(3), t=np.zeros(1),
        rgb=np.zeros((1, 4, 5, 3), dtype=np.uint8),
    )
    rec = make_npz_record(p, tmp_path, "shadow_episode")
    assert rec["split"] == "excluded"
    assert rec["label"]["source"] == "excluded"
    assert "vegetation" in rec["label"]["exclusion_reason"]


def test_episode_dataset_drops_excluded(tmp_path) -> None:
    from beamng_autopilot.recording import EpisodeDataset

    good = tmp_path / "shadow_fsd_good.npz"
    bad = tmp_path / "shadow_fsd_1789298914_20260913_193100.npz"
    for p in (good, bad):
        np.savez_compressed(
            p, version=np.int64(3), t=np.zeros(2),
            quality=np.ones(2, dtype=np.float32),
            bev=np.zeros((2, 4, 4), dtype=np.float32),
            rgb=np.zeros((2, 4, 4, 3), dtype=np.uint8),
            label=np.zeros((2, 4, 4), dtype=np.uint8),
            drivable=np.zeros((2, 4, 4), dtype=np.uint8),
        )
    ds = EpisodeDataset([good, bad], modalities=("bev",))
    assert [p.name for p in ds.files] == [good.name]
    assert len(ds) == 2
    ds2 = EpisodeDataset([good, bad], modalities=("bev",), drop_excluded=False)
    assert len(ds2.files) == 2


def test_write_manifest_roundtrip(tmp_path) -> None:
    p = _seg_dir(tmp_path, "run_a")
    rec = make_seg_record(p, tmp_path)
    out = write_manifest([rec], tmp_path / "manifest.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["manifest_version"] == 1
    assert data["records"][0]["run_id"] == rec["run_id"]
