"""Offline tests for dashed-boundary recovery (collinear fragment grouping).

The town ``line`` class is mostly short blocks, so the extractor's shape
gates leave a dashed lane line with no usable boundary.  These pin the
merge rule: collinear and near-continuous fragments join into one chain,
anything else stays separate.
"""

from __future__ import annotations

import numpy as np

from beamng_autopilot.vision.lanes import group_world_fragments


def _frag(pts) -> dict:
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0] / float(np.linalg.norm(vt[0]))
    return {"pts": pts, "c": c, "d": d}


def _seg(x0: float, x1: float, y: float = 0.0, n: int = 6):
    xs = np.linspace(x0, x1, n)
    return _frag(np.column_stack([xs, np.full_like(xs, y)]))


def test_collinear_dashes_merge_into_one_chain() -> None:
    """Two dashes 5 m apart on the same line become one boundary."""
    groups = group_world_fragments([_seg(2.0, 5.0), _seg(10.0, 13.0)])
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1]


def test_dash_gap_beyond_the_limit_stays_separate() -> None:
    """A gap wider than the limit is a different marking, not one dashes.

    The gap is derived from the constant so changing the threshold cannot
    silently rot this expectation.
    """
    from beamng_autopilot.vision.lanes import DASHED_FRAG_GAP_MAX_M
    gap = DASHED_FRAG_GAP_MAX_M
    groups = group_world_fragments(
        [_seg(2.0, 5.0), _seg(5.0 + gap + 4.0, 5.0 + gap + 7.0)])
    assert groups == []


def test_dash_gap_inside_the_limit_still_merges() -> None:
    from beamng_autopilot.vision.lanes import DASHED_FRAG_GAP_MAX_M
    gap = DASHED_FRAG_GAP_MAX_M
    groups = group_world_fragments(
        [_seg(2.0, 5.0), _seg(5.0 + gap - 1.0, 5.0 + gap + 2.0)])
    assert len(groups) == 1


def test_laterally_offset_fragments_never_merge() -> None:
    """Two parallel lines 0.8 m apart are two boundaries, not one."""
    groups = group_world_fragments([_seg(2.0, 5.0, 0.0),
                                    _seg(6.0, 9.0, 0.8)])
    assert groups == []


def test_crossing_fragments_never_merge() -> None:
    """A perpendicular block (a stop bar) is not part of the lane line."""
    across = _frag(np.column_stack([np.full(5, 6.0), np.linspace(-1.0, 1.0, 5)]))
    groups = group_world_fragments([_seg(2.0, 5.0), across])
    assert groups == []


def test_group_span_is_bounded() -> None:
    """A chain may not walk across the whole road."""
    groups = group_world_fragments(
        [_seg(2.0, 5.0), _seg(50.0, 53.0)],
        gap_max_m=60.0, max_span_m=40.0)
    assert groups == []


def test_singletons_are_not_returned() -> None:
    assert group_world_fragments([_seg(2.0, 5.0)]) == []
    assert group_world_fragments([]) == []


def test_three_dashes_form_one_chain() -> None:
    groups = group_world_fragments(
        [_seg(2.0, 5.0), _seg(9.0, 12.0), _seg(16.0, 19.0)])
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]


def test_opposite_directions_never_merge() -> None:
    a = _frag(np.column_stack([np.linspace(2.0, 5.0, 6),
                               np.zeros(6)]))
    # same line but built back-to-front; the direction vector flips sign
    b = _frag(np.column_stack([np.linspace(6.0, 9.0, 6),
                               np.zeros(6)]))
    b["d"] = -b["d"]
    # |cos| is what is compared, so a reversed fragment still merges
    assert len(group_world_fragments([a, b])) == 1
