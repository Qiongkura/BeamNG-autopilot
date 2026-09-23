"""Dataset splitting and sample selection (plan phase E7).

The plan's warning is specific: a dataset must be split by MAP and SCENE,
never by shuffling frames - adjacent frames of one run leak into both
sides and the validation score becomes fiction, which is exactly how a
model that is nearly blind on a map still "wins" an evaluation.

This module is that policy as pure logic, so the rules are testable
without a training run:

* :func:`split_by_group` partitions by group key (map + run, or run when
  the map is not recorded) and takes each validation group's TIME TAIL,
  which keeps neighbouring frames on the same side and keeps the
  validation set chronological - the repo's ``--split per-run`` semantics,
  now with the group boundary made explicit;
* :func:`leak_check` reports any group that ended up on both sides (it
  must be empty) plus the maps that are missing from validation entirely;
* :func:`select_hard_negatives` ranks frames by their false-positive line
  evidence (line-like pixels the label says are not paint - kerbs,
  reflections, wall edges, red-white posts) and takes the worst ``want``;
* :func:`select_weak_lines` takes the low-but-nonzero line fraction band
  (faded / far-field paint), which is the data class a mIoU-driven
  pipeline silently drops;
* :func:`coverage_digest` reports per-map frame counts on both sides so a
  split can be audited instead of trusted.

Every threshold is a parameter with a documented default; nothing here
reads files or a map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Weak / faded line band: below ``low`` there is effectively no paint to
# learn from, above ``high`` the frame is a dense-marking sample the
# pipeline already trains on well.  The repo's own dense-run filter uses
# 0.003 as "dense", so 0.0005-0.01 brackets the faded middle.
WEAK_LINE_LOW = 0.0005
WEAK_LINE_HIGH = 0.01


@dataclass
class FrameRef:
    """One dataset frame and the evidence used to place it."""

    index: int = 0
    run: str = ""
    map_name: str = ""
    t: float = 0.0
    line_frac: float = 0.0      # labelled line pixels / frame pixels
    hard_neg: float = 0.0       # false-positive line evidence (see module doc)
    # --- real identities (T10) -------------------------------------------
    #: Real map/episode/capture id when the recording carries one; the run
    #: DIRECTORY name is not an identity (a re-export renames it).
    source_id: str = ""
    #: Wall-clock capture time (epoch seconds) when recorded.
    t_wall: float | None = None
    #: Exposure counter: all views of one grab share it, so the same scene
    #: seen from several cameras can be kept on one side of the split.
    exposure: int | None = None
    #: Camera role for ring captures ("front_main", ...), "" for single-view.
    view: str = ""
    #: True when ``t`` is a frame index because no wall clock was recorded.
    t_is_index: bool = False
    #: The recording's own file path, when the collector wrote one: the
    #: strongest duplicate evidence there is (a copied "golden frame" has
    #: two paths or two identical bytes, not two positions in time).
    path: str = ""

    @property
    def group(self) -> str:
        """The split group: map-qualified when the map is known."""
        m = (self.map_name or "").strip()
        r = (self.source_id or self.run or "").strip()
        if m and r:
            return f"{m}/{r}"
        return r or m or "unknown"



@dataclass
class SplitPlan:
    """Which frames go where, and the audit trail for that decision."""

    train: list = field(default_factory=list)      # list[FrameRef]
    val: list = field(default_factory=list)
    groups_train: list = field(default_factory=list)
    groups_val: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def digest(self) -> dict:
        return {"train": len(self.train), "val": len(self.val),
                "groups_train": list(self.groups_train),
                "groups_val": list(self.groups_val),
                "notes": list(self.notes)}


def _group_frames(refs):
    groups: dict[str, list] = {}
    for r in refs:
        groups.setdefault(r.group_key if hasattr(r, "group_key")
                          else r.group, []).append(r)
    return groups


def frame_refs_from_meta(meta, *, run: str, start_index: int = 0):
    """Build FrameRefs from a collector's meta record.

    Returns ``(refs, notes)``.  Real identities are used when the
    recording carries them (``map_name``, ``source_id``/``episode``,
    ``t_wall``, ``exposure``, ``view``); when time was not recorded the
    frame index is used and the fallback is WRITTEN INTO THE NOTES - the
    plan's rule is that a name/identity mismatch must be reported, not
    silently patched by the training entry.
    """
    frames = list((meta or {}).get("frames") or [])
    map_name = str((meta or {}).get("map_name") or
                   (meta or {}).get("map") or "")
    source_id = str((meta or {}).get("source_id") or
                    (meta or {}).get("episode") or "")
    refs: list[FrameRef] = []
    no_wall = 0
    no_exposure = 0
    for k, rec in enumerate(frames):
        rec = dict(rec or {})
        t_wall = rec.get("t_wall")
        t_wall = None if t_wall is None else float(t_wall)
        exposure = rec.get("exposure")
        exposure = None if exposure is None else int(exposure)
        if t_wall is None:
            no_wall += 1
        if exposure is None:
            no_exposure += 1
        fraq = float(rec.get("line_frac") or 0.0)
        if not fraq:
            px = rec.get("line_pixels")
            tot = rec.get("pixels") or rec.get("n_pixels")
            if px and tot:
                fraq = float(px) / float(tot)
        refs.append(FrameRef(
            index=int(start_index) + int(rec.get("i", k)),
            run=str(run),
            map_name=map_name,
            source_id=source_id,
            t=float(t_wall if t_wall is not None else k),
            t_wall=t_wall,
            t_is_index=t_wall is None,
            exposure=exposure,
            view=str(rec.get("view") or rec.get("role") or ""),
            line_frac=fraq,
            hard_neg=float(rec.get("hard_neg") or 0.0),
            path=str(rec.get("path") or rec.get("file") or "")))
    notes: list[str] = []
    if not map_name:
        notes.append(f"{run}: no map_name in the recording; the group is "
                     f"the directory name only")
    if not source_id:
        notes.append(f"{run}: no source_id/episode in the recording")
    if no_wall:
        notes.append(f"{run}: {no_wall}/{len(frames)} frames have no wall "
                     f"clock; t is a frame INDEX (t_is_index=True)")
    if no_exposure:
        notes.append(f"{run}: {no_exposure}/{len(frames)} frames have no "
                     f"exposure counter; cross-view grouping is unavailable")
    return refs, notes


def cross_view_groups(refs) -> dict:
    """Frames that are the SAME EXPOSURE seen by several views.

    Returns ``{key: [refs]}`` for exposures with more than one view; a
    recording without exposure counters yields an empty mapping (and the
    caller must report that it could not check, not that it passed).
    """
    by_key: dict[str, list] = {}
    for r in refs:
        if r.exposure is None:
            continue
        by_key.setdefault(f"{r.group}#e{int(r.exposure)}", []).append(r)
    return {k: v for k, v in by_key.items()
            if len({(r.view or "") for r in v}) > 1}


def cross_view_leak(plan, refs=None) -> dict:
    """Are two VIEWS of one exposure on opposite sides of the split?

    Frames of the same exposure are the same instant of the same scene, so
    they must never be split - and a plan that has no exposure information
    reports ``checked=False`` rather than a clean result.  When the list
    holds only ONE view the cross-view case cannot arise at all, and the
    reason says so: it is NOT that the recording lacks exposure counters
    (measured: a single-view training list was told "no exposure counters"
    while all 173 refs carried one).
    """
    all_refs = list(refs if refs is not None else
                    (list(plan.train) + list(plan.val)))
    groups = cross_view_groups(all_refs)
    in_train = {id(r) for r in plan.train}
    leaked: list[str] = []
    for key, members in sorted(groups.items()):
        sides = {(id(r) in in_train) for r in members}
        if len(sides) > 1:
            leaked.append(key)
    with_exposure = sum(1 for r in all_refs if r.exposure is not None)
    if groups:
        reason = ""
    elif with_exposure:
        reason = ("no exposure carries more than one view in this list, so "
                  "the cross-view case cannot arise here")
    else:
        reason = "no exposure counters in this recording"
    return {"checked": bool(groups), "n_cross_view_groups": len(groups),
            "leaked_groups": leaked,
            "n_refs_with_exposure": with_exposure,
            "reason": reason}


def duplicate_groups(refs) -> dict:
    """Groups of frames that are the SAME SAMPLE recorded more than once.

    The plan's T10 wants duplicate samples reported SEPARATELY from frame
    and group overlap, because a copied "golden frame" inflates whichever
    side it lands on while looking like two independent samples.  Evidence
    is used in order of strength and each group says which one fired:

    * ``same_path`` - the recording names the same file twice.  The path is
      matched WITHIN one run: collectors write collection-relative paths
      (``front_main/frame_00000.npz``), so keying on the path alone groups
      identically-named frames of DIFFERENT collections - measured on a
      six-collection training list that produced 39 false groups while
      every frame was unique;
    * ``same_exposure_view`` - the same grab, same camera, twice;
    * ``same_wall_clock`` - the same instant in the same view twice.

    Two VIEWS of one exposure are NOT duplicates (that is the cross-view
    case, reported separately), and a recording without paths, exposures
    or wall clocks yields an empty mapping - the caller must then report
    that it could not check, not that it passed.
    """
    out: dict[str, dict] = {}
    seen_path: dict[str, list] = {}
    seen_exp: dict[tuple, list] = {}
    seen_t: dict[tuple, list] = {}
    for r in refs:
        if r.path:
            seen_path.setdefault(f"{r.run}|{r.path}", []).append(r)
        if r.exposure is not None:
            seen_exp.setdefault((r.group, int(r.exposure), r.view),
                                []).append(r)
        if r.t_wall is not None:
            seen_t.setdefault((r.group, r.view, round(float(r.t_wall), 3)),
                              []).append(r)
    # One finding per set of frames, carrying the STRONGEST evidence that
    # fired for it: a recording that writes paths AND exposures would
    # otherwise report the same duplicate twice and inflate the count.
    best: dict[frozenset, tuple] = {}

    def _add(key: str, why: str, rank: int, members: list) -> None:
        if len(members) <= 1:
            return
        ids = frozenset(int(m.index) for m in members)
        if ids in best and best[ids][0] <= rank:
            return
        best[ids] = (rank, key, why, sorted(ids))

    for path, members in sorted(seen_path.items()):
        _add(f"path:{path}", "same_path", 0, members)
    for (group, exp, view), members in sorted(seen_exp.items()):
        _add(f"{group}#e{exp}:{view}", "same_exposure_view", 1, members)
    for (group, view, t), members in sorted(seen_t.items()):
        _add(f"{group}@{t}:{view}", "same_wall_clock", 2, members)
    for _, key, why, ids in best.values():
        out[key] = {"why": why, "n": len(ids), "indexes": ids}
    return out


def _same_sample_pairs(refs) -> set:
    """Index pairs that are the same sample (duplicate or same exposure)."""
    pairs = set()
    for info in duplicate_groups(refs).values():
        idx = info["indexes"]
        for i, a in enumerate(idx):
            for b in idx[i + 1:]:
                pairs.add(frozenset((a, b)))
    for members in cross_view_groups(refs).values():
        idx = [int(m.index) for m in members]
        for i, a in enumerate(idx):
            for b in idx[i + 1:]:
                pairs.add(frozenset((a, b)))
    return pairs


def temporal_neighbours(refs, *, gap_s: float = 0.5,
                        exclude_pairs=None) -> list:
    """Frame PAIRS closer in wall-clock time than ``gap_s``.

    Adjacent segments that a split must not separate: two frames recorded
    a fraction of a second apart are the same moment of the same drive,
    so putting one in training and the other in validation leaks content
    even when their indexes, groups and exposures all differ.  Pairs that
    are already reported as duplicates or as one exposure seen by several
    views are EXCLUDED (``exclude_pairs``), so the four statements stay
    separate.  Only frames with a real wall clock participate.
    """
    skip = exclude_pairs or set()
    timed = [r for r in refs if r.t_wall is not None]
    timed.sort(key=lambda r: float(r.t_wall))
    out = []
    for a, b in zip(timed, timed[1:]):
        if frozenset((int(a.index), int(b.index))) in skip:
            continue
        dt = abs(float(b.t_wall) - float(a.t_wall))
        if dt <= float(gap_s):
            out.append({"dt_s": round(dt, 4), "a": int(a.index),
                        "b": int(b.index),
                        "groups": sorted({a.group, b.group})})
    return out


def split_audit(plan, refs=None, *, gap_s: float = 0.5) -> dict:
    """Report every leak kind SEPARATELY, each with whether it could run.

    The plan's T10 lists statements that must not be merged into one "no
    leak": frame overlap, group overlap, temporal adjacency, duplicated
    samples, and the cross-view exposure case.  Each section carries
    ``checked`` plus a reason when it could not run, so a recording
    without clocks reports "unchecked" rather than "clean".
    """
    all_refs = list(refs if refs is not None else
                    (list(plan.train) + list(plan.val)))
    train_idx = {int(r.index) for r in plan.train}
    base = leak_check(plan, all_refs)
    dup = duplicate_groups(all_refs)
    dup_cross = [k for k, v in dup.items()
                 if len({i in train_idx for i in v["indexes"]}) > 1]
    near = temporal_neighbours(all_refs, gap_s=gap_s,
                               exclude_pairs=_same_sample_pairs(all_refs))
    near_cross = [p for p in near
                  if len({p["a"] in train_idx, p["b"] in train_idx}) > 1]
    xview = cross_view_leak(plan, all_refs)
    timed = [r for r in all_refs if r.t_wall is not None]
    # "checked" must mean "the evidence existed to run this check", NOT
    # "something was found": collapsing the two made a clean list report
    # checked=False with the reason "no duplicate evidence available" even
    # though all 173 refs carried a path, an exposure and a wall clock
    # (measured on the T13 training list).
    has_dup_evidence = any(r.path or r.exposure is not None
                           or r.t_wall is not None for r in all_refs)
    return {
        "frame_overlap": {"checked": True,
                          "n": len(base["leaked_frames"]),
                          "detail": base["leaked_frames"][:8]},
        "group_overlap": {"checked": True,
                          "n": len(base["leaked_groups"]),
                          "detail": base["leaked_groups"][:8]},
        "temporal_adjacency": {
            "checked": bool(timed),
            "n_pairs": len(near),
            "n_crossing": len(near_cross),
            "gap_s": float(gap_s),
            "detail": near_cross[:6],
            "reason": ("" if timed else
                       "no wall clock in this recording: adjacency "
                       "unchecked")},
        "duplicates": {"checked": has_dup_evidence,
                       "n_groups": len(dup),
                       "n_crossing": len(dup_cross),
                       "n_evidence": {
                           "path": sum(1 for r in all_refs if r.path),
                           "exposure": sum(1 for r in all_refs
                                           if r.exposure is not None),
                           "wall_clock": len(timed)},
                       "detail": dup_cross[:6],
                       "reason": ("" if has_dup_evidence else
                                  "no duplicate evidence available "
                                  "(no path/exposure/wall clock)")},
        "cross_view_exposure": {"checked": bool(xview["checked"]),
                                "n_groups": int(xview["n_cross_view_groups"]),
                                "n_crossing": len(xview["leaked_groups"]),
                                "detail": xview["leaked_groups"][:6],
                                "reason": xview["reason"]},
    }


def split_by_group(refs, val_frac: float = 0.2, *,
                   holdout_groups=None, min_val_frames: int = 1
                   ) -> SplitPlan:
    """Split by group; each validation group contributes its time tail.

    ``holdout_groups`` forces the named groups entirely into validation
    (a map/scene held out on purpose, which is how a generalisation claim
    should be built).  A group with a single frame cannot be split, so it
    stays in training - and says so in the plan's notes rather than
    silently vanishing from both sides.
    """
    plan = SplitPlan()
    recs = [r for r in (refs or ()) if r is not None]
    if not recs:
        return plan
    holdout = {str(g) for g in (holdout_groups or ())}
    groups = _group_frames(recs)
    for name in sorted(groups):
        items = sorted(groups[name], key=lambda r: (float(r.t), int(r.index)))
        if name in holdout:
            plan.groups_val.append(name)
            plan.val.extend(items)
            continue
        k = len(items)
        if k <= 1:
            plan.notes.append(f"{name}: single frame stays in train")
            plan.groups_train.append(name)
            plan.train.extend(items)
            continue
        n_val = max(int(min_val_frames), int(k * float(val_frac)))
        n_val = min(n_val, k - 1)          # never empty the training side
        if n_val <= 0:
            plan.notes.append(f"{name}: val_frac too small, all train")
            plan.groups_train.append(name)
            plan.train.extend(items)
            continue
        plan.groups_train.append(name)
        plan.groups_val.append(name)
        plan.train.extend(items[:k - n_val])
        plan.val.extend(items[k - n_val:])
    return plan


def leak_check(plan: SplitPlan, refs=None) -> dict:
    """Report groups leaking across the split, and maps without val frames.

    The split must be group-pure: a group on both sides is a leak (the
    adjacent frames of one run share scenery and time), and a map with no
    validation frames means the score says nothing about that map.

    Two independent checks are reported, and NEITHER is conditional on the
    other: frame-index overlap (the same frame on both sides) and group
    purity (the same group on both sides).  The group check used to be
    skipped whenever ``plan.groups_val`` was non-empty - which is exactly
    the default time-tail split, where a group lands in BOTH lists by
    construction - so a genuinely shared group was reported as
    ``leak=False`` (plan §1.5-20, T10).
    """
    train_ids = {int(r.index) for r in plan.train}
    val_ids = {int(r.index) for r in plan.val}
    overlap = sorted(train_ids & val_ids)
    seen: dict[str, dict[str, int]] = {}
    tg = {r.group for r in plan.train}
    vg = {r.group for r in plan.val}
    for g in sorted(tg & vg):
        seen[g] = {
            "train": sum(1 for r in plan.train if r.group == g),
            "val": sum(1 for r in plan.val if r.group == g),
        }
    leaked = sorted(seen)
    return {"leaked_frames": overlap, "leaked_groups": leaked,
            "shared_groups": {g: dict(c) for g, c in seen.items()},
            "leak": bool(overlap or leaked)}


def coverage_digest(plan: SplitPlan, refs=None) -> dict:
    """Per-map frame counts on both sides (for auditing a split)."""
    maps: dict[str, dict] = {}

    def _add(items, side: str):
        for r in items:
            key = (r.map_name or r.run or "unknown") or "unknown"
            row = maps.setdefault(key, {"train": 0, "val": 0})
            row[side] += 1

    _add(plan.train, "train")
    _add(plan.val, "val")
    missing = sorted(m for m, c in maps.items() if c["val"] == 0)
    return {"maps": maps, "maps_without_val": missing,
            "n_maps": len(maps)}


def select_hard_negatives(refs, want: int, *,
                          min_score: float = 0.0) -> list:
    """The ``want`` frames with the strongest false-positive line evidence.

    ``hard_neg`` is the per-frame false-positive score (line-like pixels
    where the label says "not paint") - kerbs, wet reflections, wall
    edges, red-white posts.  Ties break by index so a selection is
    reproducible.
    """
    if not want or int(want) <= 0:
        return []
    pool = [r for r in (refs or ())
            if r is not None and float(getattr(r, "hard_neg", 0.0) or 0.0)
            > float(min_score)]
    pool.sort(key=lambda r: (-float(r.hard_neg), int(r.index)))
    return pool[:int(want)]


def select_weak_lines(refs, *, low: float = WEAK_LINE_LOW,
                      high: float = WEAK_LINE_HIGH) -> list:
    """Frames in the faded / far-field line band (low but non-zero).

    A pipeline that keeps only dense-marking frames never sees the paint
    that is hardest to detect; this band is that data, bounded on both
    sides so empty frames (nothing to learn) and dense frames (already
    represented) stay out.
    """
    lo, hi = float(low), float(high)
    if hi < lo:
        lo, hi = hi, lo
    out = []
    for r in refs or ():
        if r is None:
            continue
        f = float(getattr(r, "line_frac", 0.0) or 0.0)
        if not math.isfinite(f):
            continue
        if lo <= f <= hi:
            out.append(r)
    return out


def freeze_testset(plan: SplitPlan, path, *, note: str = "") -> dict:
    """Write the validation side as an immutable test-set manifest (T10).

    The plan's rule: once a set has been used to tune thresholds it is
    development data, and the final test set must be frozen separately -
    "任何规则/权重/标签调整后，原测试集转为开发资料，最终测试重新冻结".
    This writes what a later comparison must reuse: the frame identities,
    their group keys, and a digest of the set, so a re-run can prove it used
    the same frames instead of asserting it.
    """
    import hashlib
    import json
    from pathlib import Path as _P

    frames = [{"index": int(r.index), "run": str(r.run),
               "group": str(r.group),
               "map": str(getattr(r, "map_name", "") or ""),
               "t": float(r.t)} for r in plan.val]
    digest = hashlib.blake2b(
        json.dumps([[f["index"], f["run"], round(f["t"], 3)]
                    for f in frames], sort_keys=True).encode(),
        digest_size=8).hexdigest()
    payload = {"protocol": "frozen_testset", "n_frames": len(frames),
               "digest": digest, "note": note, "frames": frames}
    out = _P(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return {"path": str(out), "n_frames": len(frames), "digest": digest}


def check_frozen_testset(path, plan: SplitPlan) -> dict:
    """Does this split still match the frozen set? (T10)

    Returns ``{"ok": bool, "reason": str}``: a set whose frames or digests
    moved is NOT the frozen set, and a comparison run on it is a
    development result, not a test result.
    """
    import json
    from pathlib import Path as _P

    p = _P(path)
    if not p.is_file():
        return {"ok": False, "reason": "no frozen set at this path"}
    try:
        frozen = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:                       # unreadable = not frozen
        return {"ok": False, "reason": f"unreadable: {exc}"}
    want = {(int(f["index"]), str(f["run"])) for f in frozen.get("frames", [])}
    have = {(int(r.index), str(r.run)) for r in plan.val}
    if want != have:
        return {"ok": False, "reason": (
            f"validation set changed: {len(have - want)} new, "
            f"{len(want - have)} missing"),
            "frozen_digest": frozen.get("digest")}
    return {"ok": True, "reason": "matches the frozen set",
            "frozen_digest": frozen.get("digest"),
            "n_frames": len(have)}
