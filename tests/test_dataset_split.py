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
    """Frames are never shared; the group sharing is REPORTED, not hidden.

    The default split is a chronological tail holdout, so by construction a
    group appears on both sides.  That is a different protocol from
    group/episode isolation (map holdout) and must not be reported as "no
    leak": ``leak_check`` returns the two numbers separately (plan
    T10/§1.5-20).  This test pins both semantics.
    """
    refs = (
        _frames("run_a", 10, map_name="italy", start_index=0)
        + _frames("run_b", 10, map_name="italy", start_index=10)
        + _frames("run_c", 10, map_name="town", start_index=20))
    plan = split_by_group(refs, val_frac=0.3)
    tg = {r.group for r in plan.train}
    vg = {r.group for r in plan.val}
    rep = leak_check(plan)
    # no FRAME may be on both sides
    assert rep["leaked_frames"] == []
    assert len({r.index for r in plan.train} & {r.index for r in plan.val}) == 0
    # ...while the group overlap of a temporal-tail split is REAL and said
    # out loud, with the per-side counts
    assert rep["leaked_groups"] == sorted(tg & vg) and rep["leaked_groups"]
    assert rep["leak"] is True
    assert all(c["train"] and c["val"]
               for c in rep["shared_groups"].values())
    assert len(plan.train) + len(plan.val) == len(refs)
    assert tg and vg


def test_a_map_holdout_is_group_pure_for_the_held_out_map() -> None:
    """T10: the protocols are different and the report must say which ran.

    A map holdout keeps the held-out group entirely in val; the REMAINING
    groups are still time-tail split, so they are still shared - which is
    exactly why "temporal-tail validation" and "group isolation" cannot be
    the same claim.  Full group purity needs every group held out.
    """
    refs = (_frames("run_a", 10, map_name="italy", start_index=0)
            + _frames("run_c", 10, map_name="town", start_index=20))
    plan = split_by_group(refs, val_frac=0.3,
                          holdout_groups=["town/run_c"])
    rep = leak_check(plan)
    held = [r for r in refs if r.group == "town/run_c"]
    assert held and all(r in plan.val for r in held)
    assert not any(r in plan.train for r in held)
    assert "town/run_c" not in rep["shared_groups"]
    assert rep["shared_groups"].keys() == {"italy/run_a"}

    # the fully held-out protocol IS group-pure
    plan2 = split_by_group(refs, val_frac=0.3,
                           holdout_groups=["town/run_c", "italy/run_a"])
    rep2 = leak_check(plan2)
    assert rep2["leaked_groups"] == []
    assert rep2["leak"] is False


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


class TestFrozenTestset:
    """T10: the final test set must be frozen and re-checkable."""

    def test_freezing_writes_identities_and_a_digest(self, tmp_path):
        from beamng_autopilot.vision.dataset_split import (freeze_testset,
                                                           split_by_group)
        refs = _frames("run_a", 10, map_name="italy", start_index=0)
        plan = split_by_group(refs, val_frac=0.3)
        rep = freeze_testset(plan, tmp_path / "frozen.json",
                             note="round-7 stage-C")
        assert rep["n_frames"] == len(plan.val) and rep["digest"]
        data = json.loads((tmp_path / "frozen.json").read_text("utf-8"))
        assert data["frames"] and all("group" in f for f in data["frames"])

    def test_a_changed_split_is_not_the_frozen_set(self, tmp_path):
        from beamng_autopilot.vision.dataset_split import (check_frozen_testset,
                                                           freeze_testset,
                                                           split_by_group)
        refs = _frames("run_a", 10, map_name="italy", start_index=0)
        plan = split_by_group(refs, val_frac=0.3)
        p = tmp_path / "frozen.json"
        freeze_testset(plan, p)
        assert check_frozen_testset(p, plan)["ok"] is True
        # a different split over the same data must NOT pass
        other = split_by_group(refs, val_frac=0.5)
        rep = check_frozen_testset(p, other)
        assert rep["ok"] is False and "changed" in rep["reason"]

    def test_a_missing_frozen_set_is_not_a_pass(self, tmp_path):
        from beamng_autopilot.vision.dataset_split import (check_frozen_testset,
                                                           split_by_group)
        plan = split_by_group(_frames("run_a", 6, map_name="italy"),
                              val_frac=0.3)
        rep = check_frozen_testset(tmp_path / "nope.json", plan)
        assert rep["ok"] is False and "no frozen set" in rep["reason"]


def _ring_meta(frames_per_view: int = 3, views=("front_main", "front_fisheye"),
               with_ids: bool = True):
    """A ring collector's meta: one grab = one exposure, all views share it."""
    frames = []
    for e in range(frames_per_view):
        for v in views:
            rec = {"i": e, "view": v, "exposure": e,
                   "t_wall": 1000.0 + e, "line_pixels": 10, "pixels": 100}
            frames.append(rec)
    meta = {"frames": frames}
    if with_ids:
        meta.update({"map_name": "italy", "source_id": "ring_20260922"})
    return meta


class TestRealIdentities:
    """T10: real map/episode/time/exposure ids, and the fallback is reported."""

    def test_identity_comes_from_the_recording_not_the_directory(self):
        from beamng_autopilot.vision.dataset_split import frame_refs_from_meta
        refs, notes = frame_refs_from_meta(_ring_meta(), run="ring_dir_name")
        assert notes == []
        assert {r.group for r in refs} == {"italy/ring_20260922"}
        assert refs[0].t_wall == 1000.0 and refs[0].t == 1000.0
        assert refs[0].t_is_index is False
        assert refs[0].line_frac == pytest.approx(0.1)

    def test_a_recording_without_ids_reports_every_missing_identity(self):
        from beamng_autopilot.vision.dataset_split import frame_refs_from_meta
        refs, notes = frame_refs_from_meta(_ring_meta(with_ids=False),
                                           run="run_12")
        joined = " | ".join(notes)
        assert "no map_name" in joined and "no source_id" in joined
        assert refs[0].group == "run_12", "the directory is only a fallback"
        # this meta still carries the exposure counter, so that note must NOT
        # appear - the notes name exactly what is missing
        assert "no exposure counter" not in joined

    def test_a_missing_wall_clock_is_flagged_not_guessed(self):
        from beamng_autopilot.vision.dataset_split import frame_refs_from_meta
        meta = {"map_name": "italy", "source_id": "ep1",
                "frames": [{"i": 0}, {"i": 1}]}
        refs, notes = frame_refs_from_meta(meta, run="ep1")
        assert all(r.t_is_index for r in refs)
        assert any("t is a frame INDEX" in n for n in notes)
        assert all(r.t == float(k) for k, r in enumerate(refs))


class TestCrossViewGrouping:
    """The same exposure seen by two mounts is ONE instant."""

    def test_views_of_one_exposure_group_together(self):
        from beamng_autopilot.vision.dataset_split import (
            cross_view_groups, frame_refs_from_meta)
        refs, _ = frame_refs_from_meta(_ring_meta(2), run="r")
        groups = cross_view_groups(refs)
        assert len(groups) == 2
        assert all(len(v) == 2 for v in groups.values())

    def test_no_exposure_counter_means_no_grouping_and_no_check(self):
        from beamng_autopilot.vision.dataset_split import (
            cross_view_groups, frame_refs_from_meta)
        refs, _ = frame_refs_from_meta({"frames": [{"i": 0}, {"i": 1}]},
                                       run="r")
        assert cross_view_groups(refs) == {}

    def test_splitting_one_exposure_across_sides_is_a_leak(self):
        from beamng_autopilot.vision.dataset_split import (
            SplitPlan, cross_view_leak, frame_refs_from_meta)
        refs, _ = frame_refs_from_meta(_ring_meta(2), run="r")
        both_sides = SplitPlan(train=[refs[0], refs[2]],
                              val=[refs[1], refs[3]])
        rep = cross_view_leak(both_sides, refs)
        assert rep["checked"] is True and len(rep["leaked_groups"]) == 2
        kept_together = SplitPlan(train=[refs[0], refs[1]],
                                  val=[refs[2], refs[3]])
        assert cross_view_leak(kept_together, refs)["leaked_groups"] == []

    def test_an_uncheckable_plan_is_not_reported_as_clean(self):
        from beamng_autopilot.vision.dataset_split import (
            SplitPlan, cross_view_leak, frame_refs_from_meta)
        refs, _ = frame_refs_from_meta({"frames": [{"i": 0}, {"i": 1}]},
                                       run="r")
        plan = SplitPlan(train=[refs[0]], val=[refs[1]])
        rep = cross_view_leak(plan, refs)
        assert rep["checked"] is False and rep["leaked_groups"] == []
        assert "no exposure counters" in rep["reason"]
