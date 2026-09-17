"""Offline tests for dashed-boundary recovery (collinear fragment grouping).

The town ``line`` class is mostly short blocks, so the extractor's shape
gates leave a dashed lane line with no usable boundary.  These pin the
merge rule: collinear and near-continuous fragments join into one chain,
anything else stays separate.
"""

from __future__ import annotations

import numpy as np
import pytest

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


# --- chain fitting / smoothing -------------------------------------------


def _chain(frag_pts):
    from beamng_autopilot.vision.lanes import order_chain_points
    pts = np.vstack([np.asarray(p, dtype=float) for p in frag_pts])
    pix = np.column_stack([pts[:, 0] * 10.0, np.zeros(len(pts))])
    return order_chain_points(pts, pix, np.zeros(2))


def test_chain_points_are_ordered_along_the_line() -> None:
    """Fragments arriving out of order still publish one ordered polyline."""
    far = np.column_stack([np.linspace(20.0, 24.0, 5), np.zeros(5)])
    near = np.column_stack([np.linspace(2.0, 6.0, 5), np.zeros(5)])
    pts, pix = _chain([far, near])          # deliberately far-first
    assert np.all(np.diff(pts[:, 0]) > 0.0), "chain must run one way"
    assert len(pix) == len(pts)
    assert pix[0, 0] == pytest.approx(pts[0, 0] * 10.0), \
        "the pixel trace must follow the same permutation"


def test_curved_chain_is_ordered_along_the_arc() -> None:
    """On a bend the walk follows the arc instead of jumping across it."""
    ang = np.linspace(0.0, 0.5, 24)
    arc = np.column_stack([10.0 * np.sin(ang), 10.0 * (1 - np.cos(ang))])
    far = arc[16:]
    mid = arc[8:16]
    near = arc[:8]
    pts, _pix = _chain([far, mid, near])
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert float(np.max(step)) < 2.0, \
        "no consecutive jump may cross the arc"


def test_smoothing_removes_the_zigzag_without_shortening_the_chain():
    from beamng_autopilot.vision.lanes import (
        _resample_pair, _smooth_polyline, DASHED_FRAG_CHAIN_STATIONS,
    )
    x = np.linspace(0.0, 20.0, 24)
    y = np.array([0.6 if i % 2 else -0.6 for i in range(24)])
    noisy = np.column_stack([x, y])
    raw_len = float(np.sum(np.linalg.norm(np.diff(noisy, axis=0), axis=1)))
    raw_span = float(np.linalg.norm(noisy[-1] - noisy[0]))
    w, _pix = _resample_pair(noisy, noisy, DASHED_FRAG_CHAIN_STATIONS)
    sm = _smooth_polyline(w)
    sm_len = float(np.sum(np.linalg.norm(np.diff(sm, axis=0), axis=1)))
    sm_span = float(np.linalg.norm(sm[-1] - sm[0]))
    assert raw_len > 1.3 * raw_span, "the fixture must actually zig-zag"
    assert sm_len < 1.1 * sm_span, "the fitted chain must read as a line"
    assert sm_span == pytest.approx(raw_span, rel=1e-6), \
        "pinning the endpoints keeps the span (span gates) intact"


def test_recovered_boundary_is_ordered_and_smooth() -> None:
    """End to end on a synthetic mask: dashes become one clean polyline."""
    from beamng_autopilot.vision.lanes import recover_dashed_boundaries
    from beamng_autopilot.vision.projection import default_camera

    cam = default_camera(536, 403)
    mask = np.zeros((403, 536), dtype=np.uint8)
    # three short dashes down the image: nearer rows, further down
    for v0, v1 in ((250, 262), (272, 284), (294, 306)):
        mask[v0:v1, 300:306] = 255
    pos = np.array([0.0, 0.0, 1.4])
    markings = recover_dashed_boundaries(mask, cam, pos, 0.0, ground_z=0.0)
    assert markings, "a dashed chain must be recovered"
    mk = markings[0]
    w = np.asarray(mk.world, dtype=float)
    assert len(w) >= 4
    assert len(np.asarray(mk.pixels, dtype=float)) == len(w)
    length = float(np.sum(np.linalg.norm(np.diff(w, axis=0), axis=1)))
    span = float(np.linalg.norm(w[-1] - w[0]))
    assert length < 1.15 * span, "the published chain must be a clean line"

