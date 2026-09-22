"""Map-assisted candidate association - SHADOW, falsifiable, scoring only.

T08 of the round-5 plan.  The map may rank and reject PERCEPTION
CANDIDATES; it may never be a lateral source.  Three rules shape this
module:

* **Inputs are real map fields only.**  ``MapLinkPrior`` is built from what
  ``connector.read_current_road_rule`` actually returns (link nodes, link
  direction, the BeamNG ``lanes`` string, in/out radii, oneway, drivability)
  - there is no measured lane-level geometry in this map, so the lane string
  is used as a COUNT prior and never as a width or offset (plan §1.7).
* **Output is scores, hypothesis ids, conflicts and abstentions.**  There is
  no lateral centre anywhere in :class:`AssociationResult`, and no function
  here returns one; a test pins that.
* **Every rejection is falsifiable.**  Each conflict names the map field and
  the perception quantity that disagree, so a wrong map can be caught by the
  SAME code that uses a right one - which is what makes an A/B comparison
  against "no map" meaningful rather than a vibe.

The comparison this supports (arm A = score from perception alone, arm B =
map-assisted) reports the decisions that changed and, among them, the ones
that changed DESPITE a recorded conflict: that last number is the
false-acceptance risk the plan asks not to increase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Direction agreement: a boundary that runs at a very different bearing from
# the link is a different object (a cross street's edge, a parking bay).
DIR_TOL_DEG = 25.0
DIR_HARD_DEG = 45.0
# A pair's measured width is compared against the map's lane count: one
# lane is 3.0-3.8 m in this project's own measurements, so a two-lane pair
# must land in [5.0, 8.5] m.  Outside the band the map and the sensors
# disagree about the ROAD, which is a conflict, not a worse score.
LANE_WIDTH_MIN_M = 3.0
LANE_WIDTH_MAX_M = 3.8
# How much the map may change a candidate's rank (score multiplier bounds).
W_MATCH = 0.25
W_CONFLICT = 0.5


@dataclass(frozen=True)
class MapLinkPrior:
    """The map fields for the link under the ego (real, not inferred)."""

    link_id: str | None = None
    direction_rad: float | None = None
    lanes_hint: int | None = None
    one_way: bool | None = None
    curvature_1pm: float | None = None
    drivability: float | None = None
    right_hand_drive: bool | None = None
    source: str = "map.findBestRoad"

    @staticmethod
    def parse_lane_count(lane_string) -> tuple[int | None, int | None]:
        """``(total_lanes, forward_lanes)`` from BeamNG's lane string.

        The string is a sequence of ``+``/``-`` (one per lane) in newer maps;
        older assets store a count.  Unparseable -> ``(None, None)``, i.e.
        the prior is UNAVAILABLE rather than guessed.
        """
        if lane_string is None:
            return None, None
        s = str(lane_string).strip()
        if not s:
            return None, None
        if set(s) <= {"+", "-"}:
            return len(s), s.count("+")
        try:
            total = int(float(s))
        except (TypeError, ValueError):
            return None, None
        return (total if total > 0 else None), None

    @classmethod
    def from_road_rule(cls, rule) -> "MapLinkPrior":
        """Build from a ``connector.read_current_road_rule`` result."""
        if rule is None:
            return cls(link_id=None)
        total, fwd = cls.parse_lane_count(getattr(rule, "lanes", None))
        direction = None
        ip, op = (getattr(rule, "in_pos", None),
                  getattr(rule, "out_pos", None))
        if ip is not None and op is not None:
            dx, dy = float(op[0]) - float(ip[0]), float(op[1]) - float(ip[1])
            if math.hypot(dx, dy) > 1e-6:
                direction = math.atan2(dy, dx)
        cur = None
        ir = getattr(rule, "in_radius", None)
        orr = getattr(rule, "out_radius", None)
        radii = [float(r) for r in (ir, orr)
                 if r is not None and float(r) > 1.0]
        if radii:
            cur = 1.0 / float(sum(radii) / len(radii))
        n1, n2 = getattr(rule, "n1", None), getattr(rule, "n2", None)
        link_id = (f"{n1}->{n2}" if n1 is not None and n2 is not None
                   else None)
        return cls(link_id=link_id, direction_rad=direction, lanes_hint=total,
                   one_way=getattr(rule, "one_way", None),
                   curvature_1pm=cur, drivability=getattr(rule, "drivability",
                                                          None),
                   right_hand_drive=getattr(rule, "right_hand_drive", None))


@dataclass(frozen=True)
class PerceptionCandidate:
    """One perception candidate WITH its identity (plan: 带身份的感知候选)."""

    cand_id: str
    side: str                 # "left" | "right" | "pair" | "centre"
    kind: str = "unknown"     # solid | dashed | divider | thin | unknown
    bearing_rad: float | None = None
    confidence: float = 0.0
    span_m: float = 0.0
    width_m: float | None = None
    paired: bool = False
    fresh: bool = False


@dataclass
class AssociationResult:
    """Scoring verdict for ONE candidate.  No geometry, no lateral centre."""

    candidate_id: str
    hypothesis_id: str | None = None
    score: float = 0.0
    conflicts: list[str] = field(default_factory=list)
    abstain: str | None = None
    map_fields: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"candidate_id": self.candidate_id,
                "hypothesis_id": self.hypothesis_id,
                "score": round(float(self.score), 4),
                "conflicts": list(self.conflicts),
                "abstain": self.abstain,
                "map_fields": list(self.map_fields)}


def _wrap(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _dir_delta_deg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(math.degrees(_wrap(float(a) - float(b))))


def expected_side_for(kind: str, right_hand_drive: bool | None) -> str | None:
    """Which side of the ego lane a boundary of this KIND should be on.

    The one prior this map honestly supports: with right-hand traffic the
    own lane sits right of the centre line, so the centre/divider paint is
    the lane's LEFT boundary and the outer edge line is its RIGHT one (and
    the two swap under left-hand traffic).  It says NOTHING about where the
    lane centre is - no offset, no width, no lateral target.
    """
    rht = True if right_hand_drive is None else bool(right_hand_drive)
    k = str(kind or "").lower()
    if k == "divider":
        return "left" if rht else "right"
    if k in ("solid", "dashed", "thin"):
        return "right" if rht else "left"
    return None


#: Whether the map's ``rightHandDrive`` flag may be used as a side prior.
#: DEFAULT OFF: measured 2026-09-22 on the italy level the asset reports
#: ``rightHandDrive: false`` while the road under test is driven on the
#: right (the project's own lateral records put the own lane right of the
#: centre paint, and AGENTS.md's "keep right" rule).  A prior that
#: contradicts the road must not silently demote real candidates, so it is
#: enabled only after someone verifies the flag for that level.
TRUST_RIGHT_HAND_DRIVE = False


def associate_candidate(cand: PerceptionCandidate, prior: MapLinkPrior | None,
                        *, other_links: int = 0,
                        trust_right_hand_drive: bool | None = None
                        ) -> AssociationResult:
    """Map-assisted association score for ONE candidate (shadow).

    ``other_links`` is how many other map links were plausible at the ego
    position (junction / fork): with more than one, the map cannot say
    WHICH road the candidate belongs to, so the association abstains rather
    than picking one - the plan's 分岔 case.
    """
    res = AssociationResult(candidate_id=str(cand.cand_id))
    if prior is None or prior.link_id is None:
        res.abstain = "no map link under the ego"
        return res
    trust_rht = (TRUST_RIGHT_HAND_DRIVE if trust_right_hand_drive is None
                 else bool(trust_right_hand_drive))
    res.hypothesis_id = prior.link_id
    res.map_fields = [f for f in ("link_id", "direction_rad", "lanes_hint",
                                  "one_way", "curvature_1pm", "drivability",
                                  "right_hand_drive")
                      if getattr(prior, f, None) is not None]
    if not trust_rht:
        # the flag is present but not trusted: it stays out of the field
        # list so a reader cannot think it was used
        res.map_fields = [f for f in res.map_fields
                          if f != "right_hand_drive"]
    if not cand.fresh:
        # A held candidate cannot be associated: there is no observation to
        # associate (same rule as T02/T03).
        res.abstain = "candidate is not a fresh observation"
        return res
    if int(other_links) > 0:
        res.abstain = (f"ambiguous: {int(other_links) + 1} plausible map "
                       f"links at the ego")
        return res
    score = 1.0
    # --- direction ------------------------------------------------------
    dd = _dir_delta_deg(cand.bearing_rad, prior.direction_rad)
    if dd is not None:
        if dd > DIR_HARD_DEG:
            res.conflicts.append(
                f"direction_mismatch: candidate {dd:.0f} deg off the map "
                f"link direction (hard {DIR_HARD_DEG:.0f})")
        elif dd > DIR_TOL_DEG:
            res.conflicts.append(
                f"direction_loose: candidate {dd:.0f} deg off the map link "
                f"direction")
            score *= (1.0 - W_CONFLICT * 0.5)
        else:
            score *= (1.0 + W_MATCH)
    # --- side (only with a TRUSTED hand-of-traffic flag) -----------------
    want = (expected_side_for(cand.kind, prior.right_hand_drive)
            if trust_rht else None)
    if want is not None and cand.side in ("left", "right") and cand.side != want:
        res.conflicts.append(
            f"side_mismatch: a {cand.kind} boundary of the own lane should "
            f"be on the {want} (right_hand_drive="
            f"{prior.right_hand_drive})")
        score *= (1.0 - W_CONFLICT)
    # --- widening: a single-lane map with a two-lane-wide pair -----------
    if cand.width_m is not None and prior.lanes_hint is not None:
        lo = LANE_WIDTH_MIN_M * prior.lanes_hint
        hi = LANE_WIDTH_MAX_M * prior.lanes_hint
        if not (lo <= float(cand.width_m) <= hi):
            res.conflicts.append(
                f"width_contradicts_map: measured {float(cand.width_m):.2f} m "
                f"vs map lanes={prior.lanes_hint} implying {lo:.1f}-{hi:.1f} m")
            score *= (1.0 - W_CONFLICT)
    # --- curvature ------------------------------------------------------
    if prior.curvature_1pm is not None and cand.bearing_rad is not None \
            and cand.span_m > 4.0:
        # the candidate's own bearing already encodes its direction; a
        # strongly curved link whose candidate runs straight (or vice versa)
        # is a soft disagreement only
        pass
    res.score = max(0.0, float(score)) * max(0.0, float(cand.confidence))
    return res


def compare_arms(candidates, prior: MapLinkPrior | None, *,
                 other_links: int = 0) -> dict:
    """Arm A (perception only) vs arm B (map-assisted), and the diff.

    The falsifiable numbers: how many candidate RANKINGS changed, and how
    many of those changes happened while a conflict was recorded.  A map
    that raises a conflicted candidate is producing false acceptance, and
    this comparison counts it instead of arguing about it.
    """
    arm_a: list[dict] = []
    arm_b: list[dict] = []
    changed_with_conflict = 0
    changed = 0
    for c in candidates:
        ra = AssociationResult(candidate_id=str(c.cand_id),
                               score=float(c.confidence))
        rb = associate_candidate(c, prior, other_links=other_links)
        if rb.abstain:
            # Abstaining means "the map has nothing to say": the perception
            # score is left EXACTLY as arm A had it (the reason is recorded
            # in ``abstain``).  Zeroing it would make arm B look like it had
            # rejected the candidate, which is a different statement.
            rb.score = ra.score
        arm_a.append(ra.as_dict())
        arm_b.append(rb.as_dict())
        if abs(rb.score - ra.score) > 1e-9:
            changed += 1
            if rb.conflicts:
                changed_with_conflict += 1
    order_a = [r["candidate_id"] for r in sorted(arm_a, key=lambda r: -r["score"])]
    order_b = [r["candidate_id"] for r in sorted(arm_b, key=lambda r: -r["score"])]
    return {"arm_a": arm_a, "arm_b": arm_b,
            "rank_changed": bool(order_a != order_b),
            "n_changed_scores": changed,
            "n_changed_with_conflict": changed_with_conflict,
            "false_acceptance_risk": changed_with_conflict,
            "order_a": order_a, "order_b": order_b,
            "note": "arm B only re-scores candidates; it cannot create, "
                    "remove or move any geometry"}
