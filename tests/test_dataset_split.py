"""Dataset split and sample selection (plan E7)."""

from __future__ import annotations

import json

import pytest

from beamng_autopilot.vision.dataset_split import (
    WEAK_LINE_HIGH,
    WEAK_LINE_LOW,
    FrameRef,
    coverage_digest,
    leak_check,
    select_hard_negatives,
    select_weak_lines,
    split_by_group,
)


def _frames(run: str, n: int, *, map_name: str = "", line_frac: float = 0.01,
            hard_neg: float = 0.0, start_index: int = 0):
    return [FrameRef(index=start_index + i, run=run, map_name=map_name,
                     t=float(i), line_frac=line_frac, hard_neg=hard_neg)
            for i in range(n)]


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

def test_group_key_is_map_qualified_when_the_map_is_known() -> None:
    assert FrameRef(run="run_1", map_name="italy").group == "italy/run_1"
    assert FrameRef(run="run_1").group == "run_1"
    assert FrameRef(map_name="italy").group == "italy"
    assert FrameRef().group == "unknown"


# ---------------------------------------------------------------------------
# split by group
# ---------------------------------------------------------------------------

def test_no_group_straddles_the_split() -> None:
    """The plan's core requirement: adjacent frames cannot leak."""
    refs = (
        _frames("run_a", 10, map_name="italy", start_index=0)
        + _frames("run_b", 10, map_name="italy", start_index=10)
        + _frames("run_c", 10, map_name="town", start_index=20))
    plan = split_by_group(refs, val_frac=0.3)
    tg = {r.group for r in plan.train}
    vg = {r.group for r in plan.val}
    # a group may appear on both sides ONLY through the time tail, which
    # is the point (chronological holdout), so the leak test is about
    # FRAMES: no frame may be on both sides, and the tail must be val
    assert not leak_check(plan)["leak"]
    assert len(plan.train) + len(plan.val) == len(refs)
    assert len({r.index for r in plan.train} & {r.index for r in plan.val}) == 0
    assert tg and vg


def test_validation_is_the_time_tail_of_each_group() -> None:
    refs = _frames("run_a", 10, map_name="italy")
    plan = split_by_group(refs, val_frac=0.3)
    val_t = [r.t for r in plan.val]
    train_t = [r.t for r in plan.train]
    assert val_t == [7.0, 8.0, 9.0]
    assert train_t == [float(i) for i in range(7)]


def test_holdout_groups_go_entirely_to_validation() -> None:
    refs = (_frames("run_a", 8, map_name="italy", start_index=0)
            + _frames("run_b", 8, map_name="town", start_index=8))
    plan = split_by_group(refs, val_frac=0.25, holdout_groups=["town/run_b"])
    assert all(r.map_name == "town" for r in plan.val if r.group == "town/run_b")
    assert len([r for r in plan.val if r.group == "town/run_b"]) == 8
    assert not [r for r in plan.train if r.group == "town/run_b"]


def test_a_single_frame_group_stays_in_training_and_says_so() -> None:
    refs = _frames("run_solo", 1) + _frames("run_a", 5)
    plan = split_by_group(refs, val_frac=0.5)
    assert [r.group for r in plan.train if r.group == "run_solo"] == \
        ["run_solo"]
    assert any("single frame" in n for n in plan.notes)


def test_split_never_empties_a_group_s_training_side() -> None:
    refs = _frames("run_a", 2)
    plan = split_by_group(refs, val_frac=1.0)
    assert len(plan.train) == 1 and len(plan.val) == 1


def test_empty_input_is_safe() -> None:
    plan = split_by_group([], val_frac=0.2)
    assert plan.train == [] and plan.val == []
    assert not leak_check(plan)["leak"]
    assert coverage_digest(plan)["n_maps"] == 0


def test_leak_check_catches_a_hand_made_leak() -> None:
    """A frame appearing on both sides must be reported, not tolerated."""
    refs = _frames("run_a", 4)
    plan = split_by_group(refs, val_frac=0.5)
    plan.train.append(plan.val[0])          # inject the leak
    rep = leak_check(plan)
    assert rep["leak"] is True
    assert rep["leaked_frames"] == [plan.val[0].index]


def test_coverage_digest_flags_maps_without_validation() -> None:
    refs = (_frames("run_a", 6, map_name="italy", start_index=0)
            + _frames("run_b", 1, map_name="west_coast", start_index=6))
    plan = split_by_group(refs, val_frac=0.5)
    cov = coverage_digest(plan)
    assert cov["n_maps"] == 2
    assert cov["maps"]["italy"]["val"] >= 1
    assert "west_coast" in cov["maps_without_val"]


def test_split_is_deterministic() -> None:
    refs = (_frames("run_a", 7) + _frames("run_b", 7, start_index=7))
    a = split_by_group(refs, val_frac=0.3)
    b = split_by_group(list(refs), val_frac=0.3)
    assert [r.index for r in a.train] == [r.index for r in b.train]
    assert [r.index for r in a.val] == [r.index for r in b.val]
    assert json.dumps(a.digest()) == json.dumps(b.digest())


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def test_hard_negatives_are_taken_worst_first_and_capped() -> None:
    refs = [FrameRef(index=i, run="r", hard_neg=s)
            for i, s in enumerate([0.0, 0.4, 0.9, 0.2, 0.6])]
    picked = select_hard_negatives(refs, want=2)
    assert [r.index for r in picked] == [2, 4]
    assert [r.hard_neg for r in picked] == [0.9, 0.6]


def test_hard_negatives_respect_a_score_floor_and_zero_want() -> None:
    refs = [FrameRef(index=i, run="r", hard_neg=s)
            for i, s in enumerate([0.05, 0.3])]
    # want > pool: everything comes back, worst first (the ranking is
    # the point of the function, so the order is asserted, not the set)
    assert [r.index for r in select_hard_negatives(refs, want=5)] == [1, 0]
    assert [r.index for r in select_hard_negatives(refs, want=5,
                                                   min_score=0.1)] == [1]
    assert select_hard_negatives(refs, want=0) == []
    assert select_hard_negatives([], want=3) == []


def test_hard_negative_ties_break_by_index() -> None:
    refs = [FrameRef(index=i, run="r", hard_neg=0.5) for i in (3, 1, 2)]
    assert [r.index for r in select_hard_negatives(refs, want=2)] == [1, 2]


def test_weak_line_band_bounds_are_inclusive() -> None:
    refs = [FrameRef(index=i, run="r", line_frac=f)
            for i, f in enumerate([0.0, WEAK_LINE_LOW, 0.002,
                                   WEAK_LINE_HIGH, 0.05])]
    picked = [r.index for r in select_weak_lines(refs)]
    assert picked == [1, 2, 3]
    assert select_weak_lines(refs, low=0.003, high=0.004) == []


def test_weak_line_selection_survives_reversed_bounds_and_nan() -> None:
    refs = [FrameRef(index=0, run="r", line_frac=0.002),
            FrameRef(index=1, run="r", line_frac=float("nan"))]
    picked = select_weak_lines(refs, low=0.01, high=0.0005)
    assert [r.index for r in picked] == [0]     # bounds swapped, NaN dropped


def test_frame_defaults_are_conservative() -> None:
    r = FrameRef()
    assert r.line_frac == 0.0 and r.hard_neg == 0.0
    assert select_weak_lines([r]) == []          # an empty frame is not "weak"
    assert select_hard_negatives([r], want=1) == []
