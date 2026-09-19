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

    @property
    def group(self) -> str:
        """The split group: map-qualified when the map is known."""
        m = (self.map_name or "").strip()
        r = (self.run or "").strip()
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
    """
    train_ids = {int(r.index) for r in plan.train}
    val_ids = {int(r.index) for r in plan.val}
    overlap = sorted(train_ids & val_ids)
    leaked = []
    if overlap:
        seen = {}
        for r in list(plan.train) + list(plan.val):
            if int(r.index) in overlap:
                seen.setdefault(r.group, {"train": 0, "val": 0})
        for r in plan.train:
            if int(r.index) in overlap:
                seen[r.group]["train"] += 1
        for r in plan.val:
            if int(r.index) in overlap:
                seen[r.group]["val"] += 1
        leaked = sorted(g for g, c in seen.items()
                        if c["train"] and c["val"])
    if not leaked:
        # group-purity check independent of the index overlap
        tg = {r.group for r in plan.train}
        vg = {r.group for r in plan.val}
        leaked = sorted(tg & vg) if not plan.groups_val else []
    return {"leaked_frames": overlap, "leaked_groups": leaked,
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
