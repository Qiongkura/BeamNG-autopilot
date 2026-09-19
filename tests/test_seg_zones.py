"""Near/mid/far zone thresholds for the probability gates (plan E2)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from beamng_autopilot.vision.seg_probs import SegProbabilityMaps
from beamng_autopilot.vision.seg_zones import (
    ZONE_FAR,
    ZONE_MID,
    ZONE_NEAR,
    ZoneSpec,
    default_zones,
    gate_masks_zoned,
    zone_of_row,
    zone_row_masks,
)

H, W = 100, 60


def _maps(line_value: float = 0.9, road_value: float = 0.9):
    return SegProbabilityMaps(
        line=np.full((H, W), float(line_value), dtype=np.float32),
        road=np.full((H, W), float(road_value), dtype=np.float32))


# ---------------------------------------------------------------------------
# zones
# ---------------------------------------------------------------------------

def test_default_zones_are_ordered_bottom_to_top() -> None:
    specs = default_zones()
    names = [s.name for s in specs]
    assert names == [ZONE_NEAR, ZONE_MID, ZONE_FAR]
    near, mid, far = specs
    assert near.row_from_frac > mid.row_from_frac > far.row_from_frac
    # the plan's intent: near strictest, far loosest
    assert near.line_min > mid.line_min > far.line_min
    assert near.hold_s < mid.hold_s < far.hold_s
    assert far.require_history is True
    assert near.require_history is False


def test_zone_row_masks_partition_the_image() -> None:
    bands = zone_row_masks((H, W))
    total = np.zeros((H, W), dtype=bool)
    for m in bands.values():
        assert not (total & m).any(), "bands must not overlap"
        total = total | m
    assert total.all(), "every row belongs to a zone"


def test_zone_of_row_matches_the_bands() -> None:
    assert zone_of_row(H - 1, H) == ZONE_NEAR     # bottom = closest
    assert zone_of_row(int(0.45 * H), H) == ZONE_MID
    assert zone_of_row(0, H) == ZONE_FAR          # top = farthest
    assert zone_of_row(10 ** 6, H) == ZONE_MID    # out of range -> mid


def test_custom_zones_are_honoured() -> None:
    zones = (ZoneSpec("only", 0.0, 1.0, line_min=0.99),)
    bands = zone_row_masks((H, W), zones)
    assert bands["only"].all()
    assert zone_of_row(0, H, zones) == "only"
    _, line, _ = gate_masks_zoned(_maps(0.5), zones)
    assert not line.any()


# ---------------------------------------------------------------------------
# per-zone thresholds
# ---------------------------------------------------------------------------

def test_a_mid_strength_line_passes_mid_and_fails_near() -> None:
    """The whole point of zones: one threshold cannot serve both bands."""
    m = _maps(line_value=0.55)
    _road, line, stats = gate_masks_zoned(m)
    bands = zone_row_masks((H, W))
    assert line[bands[ZONE_MID]].any()
    assert not line[bands[ZONE_NEAR]].any()
    # the far band also accepts it (loosest threshold + no history given)
    assert line[bands[ZONE_FAR]].any()
    assert stats.far_unconfirmed > 0      # ...but it is flagged as such
    assert set(stats.per_zone) == {ZONE_NEAR, ZONE_MID, ZONE_FAR}


def test_road_context_is_gated_per_zone_too() -> None:
    """Off-road white in the near band stays out; the far band is looser."""
    m = _maps(line_value=0.9, road_value=0.05)
    _road, line, stats = gate_masks_zoned(m)
    assert not line.any()
    assert stats.line_after == 0
    assert stats.per_zone[ZONE_NEAR]["line_after"] == 0


def test_far_candidates_require_history_support() -> None:
    m = _maps(line_value=0.9, road_value=0.9)
    support = np.zeros((H, W), dtype=bool)
    support[:20, :] = True                 # only some far rows are confirmed
    _road, line, stats = gate_masks_zoned(m, history_support=support)
    bands = zone_row_masks((H, W))
    far = bands[ZONE_FAR]
    assert line[far & support].any()
    assert not line[far & ~support].any(), "unconfirmed far paint is dropped"
    assert stats.far_unconfirmed > 0
    # near/mid are unaffected by the history requirement
    assert line[bands[ZONE_NEAR]].all()


def test_missing_history_keeps_far_candidates_but_flags_them() -> None:
    """Unknown history is not contradicted history."""
    m = _maps(line_value=0.9)
    _road, line, stats = gate_masks_zoned(m, history_support=None)
    bands = zone_row_masks((H, W))
    assert line[bands[ZONE_FAR]].any()
    assert stats.far_unconfirmed > 0


def test_extra_line_source_is_gated_per_zone() -> None:
    m = _maps(line_value=0.0, road_value=0.05)     # no line prob, no road
    extra = np.zeros((H, W), dtype=bool)
    extra[H - 5:H - 2, 10:12] = True               # a near-band blob
    _road, line, _stats = gate_masks_zoned(m, extra_line=extra)
    assert not line.any(), "off-road colour must not enter, per zone"
    m2 = _maps(line_value=0.0, road_value=0.95)
    _road2, line2, _stats2 = gate_masks_zoned(m2, extra_line=extra)
    assert line2[H - 5:H - 2, 10:12].all()


def test_gate_digest_is_json_safe() -> None:
    _road, _line, stats = gate_masks_zoned(_maps())
    text = json.dumps(stats.digest())
    assert "zones" in text and "nan" not in text.lower()


def test_mismatched_shapes_are_rejected() -> None:
    m = SegProbabilityMaps(line=np.zeros((4, 4), dtype=np.float32),
                           road=np.zeros((5, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        gate_masks_zoned(m)
