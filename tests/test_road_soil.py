"""铺装路面约束：土肩不得算作可行驶路面，但全土路仍算路。"""

from __future__ import annotations

import numpy as np

from beamng_autopilot.vision.segmentation import (
    snow_or_soil_mask, strip_soil_from_road,
)


def _frame(asphalt_frac: float, soil_frac: float, h: int = 40, w: int = 40):
    """Synthetic frame: left part asphalt-grey, right part soil-brown."""
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    n_soil = int(round(w * soil_frac))
    n_asph = int(round(w * asphalt_frac))
    rgb[:, :n_asph] = (128, 128, 128)          # grey asphalt
    rgb[:, n_asph:n_asph + n_soil] = (150, 105, 55)   # warm soil
    return rgb


def test_soil_mask_finds_warm_ground_only():
    rgb = _frame(0.5, 0.5)
    soil = snow_or_soil_mask(rgb)
    assert not soil[:, 0].any(), "grey asphalt is not soil"
    assert soil[:, -1].all(), "warm ground is soil"


def test_paved_road_drops_the_soil_shoulder():
    """Mask covering asphalt + shoulder: the shoulder must be removed.

    The fixture mirrors the measured worst paved frame (soil 24.8% of the
    mask), which must still be stripped - a floor that only bites on mild
    frames misses exactly the case this exists for.
    """
    rgb = _frame(0.752, 0.248)
    road = np.ones((40, 40), dtype=bool)       # model labelled everything road
    out = strip_soil_from_road(road, rgb)
    assert out[:, 0].all(), "asphalt must stay drivable"
    assert not out[:, -1].any(), "soil shoulder must go"


def test_dirt_road_keeps_the_soil_as_road_only_when_route_is_dirt():
    """Whether dirt counts as road is a ROUTE property, not a per-frame
    pixel share: the 0.70 escape hatch was measured live and removed
    (east_coast 2026-09-19 14:31 - paved share 0.40 skipped the strip,
    dirt entered the drivable layer and the car drove 40 frames off the
    road on lane=sensor).  The dirt-route case must be the explicit
    opt-in; the default (paved route) strips unconditionally, and a mask
    that is all soil returns EMPTY so strict mode fails closed instead
    of claiming the dirt is road."""
    rgb = _frame(0.0, 1.0)
    road = np.ones((40, 40), dtype=bool)
    # default: paved route -> soil stripped even when the share is low
    out = strip_soil_from_road(road, rgb)
    assert not out.any(), "paved route: all-soil mask must go empty"
    # explicit dirt route: soil stays road
    out_dirt = strip_soil_from_road(road, rgb, route_is_dirt=True)
    assert np.array_equal(out_dirt, road),         "an all-dirt route must stay drivable"


def test_mask_without_soil_is_untouched():
    rgb = _frame(1.0, 0.0)
    road = np.zeros((40, 40), dtype=bool)
    road[10:30, 10:30] = True
    out = strip_soil_from_road(road, rgb)
    assert np.array_equal(out, road)


def test_empty_and_all_soil_masks_are_safe():
    rgb = _frame(0.0, 1.0)
    assert not strip_soil_from_road(np.zeros((40, 40), bool), rgb).any()
    # a mask that is soil everywhere leaves no paved surface: EMPTY, so
    # the strict fail-closed path owns it (no road evidence -> no
    # candidate), never the dirt-as-road claim
    road = np.ones((40, 40), dtype=bool)
    assert not strip_soil_from_road(road, rgb).any()
