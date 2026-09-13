"""S0 experiment-trust: spawn gate + true shadow map provenance."""

from __future__ import annotations

import json

import numpy as np
import pytest

from beamng_autopilot.data_contract import make_npz_record
from beamng_autopilot.fsd_drive import build_fsd_shadow_provenance
from beamng_autopilot.vision.spawn_gate import assess_spawn_frame


def _frame(h=120, w=160, rgb=(120, 120, 120)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = rgb
    return img


def test_spawn_gate_rejects_green_bush() -> None:
    # Strong green mid-bright foliage, almost no asphalt-like pixels.
    img = _frame(rgb=(20, 140, 30))
    a = assess_spawn_frame(img)
    assert not a.ok
    assert "too_green" in a.reasons or "no_road_like" in a.reasons
    assert a.green_frac > 0.3


def test_spawn_gate_rejects_dark() -> None:
    img = _frame(rgb=(8, 8, 8))
    a = assess_spawn_frame(img)
    assert not a.ok
    assert "too_dark" in a.reasons


def test_spawn_gate_accepts_roadlike() -> None:
    # Gray asphalt: low S, mid V, not green.
    img = _frame(rgb=(90, 90, 90))
    # Bottom band slightly brighter road texture
    img[60:] = (110, 110, 110)
    a = assess_spawn_frame(img)
    assert a.ok, a.reasons
    assert a.roadlike_frac > 0.05


def test_spawn_gate_bad_shape() -> None:
    a = assess_spawn_frame(np.zeros((10, 10), dtype=np.uint8))
    assert not a.ok


def test_fsd_provenance_uses_connector_map() -> None:
    p = build_fsd_shadow_provenance(
        runtime="tech",
        map_name="east_coast_usa",
        vehicle="etk800",
        speed_arg=6.0,
        strict=True,
    )
    assert p["map"] == "east_coast_usa"
    assert p["source"] == "fsd_drive"
    p2 = build_fsd_shadow_provenance(
        runtime="tech", map_name=None, vehicle=None,
        speed_arg=4.0, strict=False,
    )
    assert p2["map"] == "unknown"
    assert p2["vehicle"] == "unknown"


def test_make_npz_record_reads_episode_provenance(tmp_path) -> None:
    meta = {
        "schema": "fsd_shadow_episode",
        "sequence": "fsd_x",
        "frames": 1,
        "provenance": {
            "source": "fsd_drive",
            "runtime": "tech",
            "map": "east_coast_usa",
            "vehicle": "etk800",
        },
    }
    p = tmp_path / "shadow_fsd_east.npz"
    np.savez_compressed(
        p,
        version=np.int64(3),
        t=np.zeros(1),
        rgb=np.zeros((1, 8, 10, 3), dtype=np.uint8),
        meta=json.dumps(meta).encode("utf-8"),
    )
    rec = make_npz_record(p, tmp_path)
    assert rec["environment"]["map"] == "east_coast_usa"
    assert rec["environment"]["vehicle"] == "etk800"
    assert rec["environment"]["runtime"] == "tech"


def test_make_npz_record_unknown_when_meta_missing(tmp_path) -> None:
    p = tmp_path / "shadow_fsd_legacy.npz"
    np.savez_compressed(
        p, version=np.int64(3), t=np.zeros(1),
        rgb=np.zeros((1, 8, 10, 3), dtype=np.uint8),
    )
    rec = make_npz_record(p, tmp_path)
    assert rec["environment"]["map"] == "unknown"
    assert "italy" not in json.dumps(rec["environment"])
