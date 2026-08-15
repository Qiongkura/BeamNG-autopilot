"""Tech annotation pixel-truth helpers (pure logic, no game)."""
from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot_tech.annotations import ANN_ASPHALT, road_share, to_label


def _ann(shape=(40, 60, 3), color=(0, 0, 0)):
    img = np.zeros(shape, dtype=np.uint8)
    img[...] = np.asarray(color, dtype=np.uint8)
    return img


class TestToLabel:
    def test_asphalt_maps_to_road_class(self):
        img = _ann(color=ANN_ASPHALT)
        lab = to_label(img)
        assert lab.shape == img.shape[:2]
        assert lab.dtype == np.uint8
        assert np.all(lab == 1)

    def test_unknown_color_is_background(self):
        img = _ann(color=(10, 20, 30))
        assert np.all(to_label(img) == 0)

    def test_mixed_frame(self):
        img = _ann(color=(10, 20, 30))
        img[:10, :20] = np.asarray(ANN_ASPHALT, dtype=np.uint8)
        img[10:20, 20:30] = np.asarray((255, 196, 128), dtype=np.uint8)
        lab = to_label(img)
        assert np.all(lab[:10, :20] == 1)      # asphalt
        assert np.all(lab[10:20, 20:30] == 2)  # solid line
        assert np.all(lab[30:, 40:] == 0)      # background


class TestRoadShare:
    def test_all_road_is_one(self):
        assert road_share(_ann(color=ANN_ASPHALT)) == pytest.approx(1.0)

    def test_alt_road_color_counts(self):
        # smallgrid's road material annotates as (128,196,255)
        assert road_share(_ann(color=(128, 196, 255))) == pytest.approx(1.0)

    def test_no_road_is_zero(self):
        assert road_share(_ann(color=(10, 20, 30))) == pytest.approx(0.0)

    def test_roi_only_counts_lower_part(self):
        # asphalt fills only the top 30% (outside the lower 66% ROI)
        img = _ann(color=(10, 20, 30))
        img[:12, :, :] = np.asarray(ANN_ASPHALT, dtype=np.uint8)
        assert road_share(img) == pytest.approx(0.0)

    def test_empty_frame_is_zero(self):
        assert road_share(np.empty((0, 0, 3), dtype=np.uint8)) == 0.0
        assert road_share(None) == 0.0