"""路面约束不得把标线芯挖空（predict 后处理回归）。"""

from __future__ import annotations

import numpy as np

from beamng_autopilot.vision.segmentation import (
    constrain_line_to_road, fill_interior_holes,
)


def _stripe(shape, x0: int, width: int, y0: int = 10, y1: int = 30):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x0 + width] = True
    return m


def test_fill_interior_holes_fills_enclosed_only():
    road = np.zeros((20, 20), dtype=bool)
    road[2:18, 2:18] = True
    road[8:12, 8:12] = False          # enClosed hole
    filled = fill_interior_holes(road)
    assert filled[9, 9]              # hole filled
    assert not filled[0, 0]          # outside border stays outside
    assert not filled[19, 19]


def test_paint_enclosed_by_road_survives_the_constraint():
    """The model calls bright paint NOT road, so the road mask carries a
    hole exactly along the marking.  The marking must still come through
    whole, not as a hollow outline."""
    road = np.ones((40, 60), dtype=bool)
    road[10:30, 26:34] = False       # the paint is a hole in the road
    line = _stripe((40, 60), 26, 8)

    kept = constrain_line_to_road(line, road)
    assert int(kept.sum()) == int(line.sum())
    assert np.array_equal(kept, line)


def test_wide_paint_is_not_eroded_to_its_border():
    """A marking wider than the 7 px dilation tolerance lost its core
    before (measured: 56.8% of every marking).  The centre column must
    survive."""
    road = np.ones((40, 60), dtype=bool)
    road[10:30, 20:40] = False
    line = _stripe((40, 60), 20, 20)

    kept = constrain_line_to_road(line, road)
    assert kept[20, 30]             # dead centre of the marking
    assert int(kept.sum()) == int(line.sum())


def test_line_outside_the_road_is_still_rejected():
    """Grass / wall / stone false lines sit OUTSIDE the road region - the
    case this constraint exists for."""
    road = np.zeros((40, 60), dtype=bool)
    road[:, :20] = True
    line = _stripe((40, 60), 45, 8)

    kept = constrain_line_to_road(line, road)
    assert not kept.any()


def test_empty_line_mask_is_returned_untouched():
    road = np.ones((10, 10), dtype=bool)
    kept = constrain_line_to_road(np.zeros((10, 10), dtype=bool), road)
    assert not kept.any()


# --- T01: the off-switch and the legacy arm -------------------------------
def test_elongated_frac_none_disables_the_second_tier():
    """``None`` is the off-switch; ``0.0`` is NOT (it lets everything pass).

    Pinned because the four-arm report used ``elongated_frac=0.0`` while
    describing the arm as "constraint without the elongated fallback" - a
    stroke lying almost entirely off the road survived the tier it was
    supposed to be excluded from (plan T01).
    """
    road = np.zeros((40, 60), dtype=bool)
    road[:, :20] = True
    line = _stripe((40, 60), 40, 8)      # 40 px right of the road region

    assert not constrain_line_to_road(
        line, road, elongated_frac=None).any()      # tier off -> dropped
    assert constrain_line_to_road(
        line, road, elongated_frac=0.0).any()       # 0.0 -> tier keeps it


def test_the_legacy_pixel_constraint_is_not_a_bare_intersection():
    """The two differ on exactly the road-mask holes paint sits in."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m5_seg_stage_eval import legacy_pixel_road_constraint

    road = np.zeros((40, 60), dtype=bool)
    road[5:35, 5:55] = True
    road[18:22, 25:35] = False           # a hole (the paint the model calls
    line = np.zeros((40, 60), dtype=bool)  # "not asphalt")
    line[18:22, 25:35] = True
    bare = line & road
    legacy = legacy_pixel_road_constraint(line, road)
    assert not bare.any()
    assert legacy.any(), "the legacy version fills the hole first"
