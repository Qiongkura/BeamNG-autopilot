"""Fine-grained marking classes for lane candidates (plan phase E5).

The extractor labels a marking with a coarse ``kind`` ("solid" / "dashed" /
"thin") and a colour, and the pairing policy uses those to decide what may
bound a lane.  Plan phase E5 asks for the taxonomy the downstream rules
actually talk about:

``solid_white`` / ``dashed_white`` / ``yellow_center`` / ``white_edge`` /
``curb`` / ``roadside_artifact``

The distinctions matter because the four "real" classes have different
jobs - a centre line and a lane edge are both paint but only one of them
may bound the ego lane on a given side, a dashed line is a lane divider
that may be crossed while a solid one may not, and the last two are NOT
paint at all: a curb is a physical road edge (allowed as a boundary only
under the AGENTS.md trust order, never as a painted line) and a roadside
artifact is a red-white reflector post, a wall edge or a graffiti-like
blob that must never become lane geometry.

Pure logic: every input is evidence the caller already has (colour, the
coarse kind, span/width in metres, the painted/gap split of the span, the
marking's lateral offset from the ego lane centre, the own-lane half
width, how far the marking sits from the observed pavement edge, whether
it lands on drivable evidence, and an optional raised-height hint).  The
classifier reads no image, no map and no route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# The taxonomy (plan E5).  ``MARK_UNKNOWN`` exists so an
# under-evidenced marking is labelled honestly instead of being forced
# into a class by a default.
MARK_SOLID_WHITE = "solid_white"
MARK_DASHED_WHITE = "dashed_white"
MARK_YELLOW_CENTER = "yellow_center"
MARK_WHITE_EDGE = "white_edge"
MARK_CURB = "curb"
MARK_ROADSIDE_ARTIFACT = "roadside_artifact"
MARK_UNKNOWN = "unknown"

MARK_PAINT = (MARK_SOLID_WHITE, MARK_DASHED_WHITE, MARK_YELLOW_CENTER,
              MARK_WHITE_EDGE)
MARK_NOT_PAINT = (MARK_CURB, MARK_ROADSIDE_ARTIFACT)

# A marking whose painted fraction is below this is dashed (the plan's
# dashed recovery already uses gaps of up to ~12 m, so the threshold is
# deliberately low: a long solid line with one dropout must stay solid).
MARK_DASHED_GAP_MIN = 0.25
# Painted fraction above this plus a short span is a fragment, not a line.
MARK_FRAGMENT_SPAN_M = 1.5
# A marking this far outside the own-lane half width is not the lane's
# paint any more.
MARK_EDGE_MARGIN_M = 0.35
# Off-pavement markings closer than this to the pavement edge are curbs /
# kerb stones; farther out they are roadside clutter.
MARK_CURB_MAX_OFF_PAVEMENT_M = 0.8
# A raised edge (kerb) is at least this tall when the caller can say.
MARK_CURB_MIN_HEIGHT_M = 0.04


@dataclass
class MarkingClassResult:
    """The class plus the evidence that produced it."""

    mark_class: str = MARK_UNKNOWN
    confidence: float = 0.0
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def is_paint(self) -> bool:
        return self.mark_class in MARK_PAINT

    @property
    def usable_as_lane_boundary(self) -> bool:
        """Paint may bound a lane; a curb only under the trust order; an
        artifact and an unknown never do."""
        return self.mark_class in MARK_PAINT + (MARK_CURB,)

    def digest(self) -> dict:
        def _r(v):
            if v is None:
                return None
            if isinstance(v, float):
                return None if not math.isfinite(v) else round(v, 4)
            return v
        return {"class": self.mark_class,
                "conf": round(float(self.confidence), 3),
                "why": ",".join(self.reasons) if self.reasons else "",
                "m": {k: _r(v) for k, v in self.metrics.items()}}


def classify_marking(*, color: str | None = None,
                     kind: str | None = None,
                     span_m: float | None = None,
                     width_m: float | None = None,
                     gap_ratio: float | None = None,
                     lateral_m: float | None = None,
                     lane_half_m: float | None = None,
                     off_pavement_m: float | None = None,
                     on_drivable: bool | None = None,
                     raised_height_m: float | None = None,
                     ) -> MarkingClassResult:
    """Classify one marking candidate.

    ``gap_ratio`` is the fraction of the span with no paint (0 = a solid
    stroke), ``lateral_m`` the signed offset of the marking from the ego
    lane centre, ``off_pavement_m`` how far it sits outside the observed
    pavement edge, and ``raised_height_m`` the physical height when the
    caller measured one (a kerb is raised, paint is not).

    Order of decisions is deliberate: "is this paint at all?" comes before
    "which paint?", because a red-white reflector post with a dashed
    appearance must not be rescued into ``dashed_white`` by its gaps.
    """
    res = MarkingClassResult()
    res.metrics = {"span_m": span_m, "width_m": width_m,
                   "gap_ratio": gap_ratio, "lateral_m": lateral_m,
                   "off_pavement_m": off_pavement_m,
                   "height_m": raised_height_m}

    col = (None if color is None else str(color).lower())
    coarse = (None if kind is None else str(kind).lower())
    short = (span_m is not None
             and math.isfinite(float(span_m))
             and float(span_m) < MARK_FRAGMENT_SPAN_M)
    off_pavement = (off_pavement_m is not None
                    and math.isfinite(float(off_pavement_m))
                    and float(off_pavement_m) > 0.0)
    raised = (raised_height_m is not None
              and math.isfinite(float(raised_height_m))
              and float(raised_height_m) >= MARK_CURB_MIN_HEIGHT_M)
    drivable = on_drivable

    # --- 1) is it paint at all? ------------------------------------------
    if drivable is False:
        # "its own points are not on the drivable surface" is already
        # enough to refuse lane geometry - the DISTANCE off the pavement
        # (when known) only separates a plausible kerb from clutter, and
        # a kerb additionally needs the raised / adjacency evidence.
        far_out = (off_pavement_m is not None
                   and float(off_pavement_m) > MARK_CURB_MAX_OFF_PAVEMENT_M)
        if raised:
            res.mark_class = MARK_CURB
            res.reasons.append("raised_edge")
            res.confidence = 0.75
        elif far_out:
            res.mark_class = MARK_ROADSIDE_ARTIFACT
            res.reasons.append("off_pavement_far")
            res.confidence = 0.7
        elif short:
            res.mark_class = MARK_ROADSIDE_ARTIFACT
            res.reasons.append("fragment_off_road")
            res.confidence = 0.6
        elif off_pavement:
            res.mark_class = MARK_CURB
            res.reasons.append("off_pavement")
            res.confidence = 0.5
        else:
            res.mark_class = MARK_ROADSIDE_ARTIFACT
            res.reasons.append("off_drivable")
            res.confidence = 0.55
        return res

    # --- 2) which paint? --------------------------------------------------
    if col == "yellow":
        res.mark_class = MARK_YELLOW_CENTER
        res.reasons.append("yellow_paint")
        res.confidence = 0.85
        return res
    if col == "white":
        # the ego lane's outer paint: only when a lane width was supplied
        # and the marking sits beyond it
        if (lateral_m is not None and lane_half_m is not None
                and math.isfinite(float(lateral_m))
                and math.isfinite(float(lane_half_m))
                and abs(float(lateral_m))
                >= float(lane_half_m) + MARK_EDGE_MARGIN_M):
            res.mark_class = MARK_WHITE_EDGE
            res.reasons.append("outside_own_lane_half")
            res.confidence = 0.7
            return res
        if gap_ratio is not None and math.isfinite(float(gap_ratio)) \
                and float(gap_ratio) >= MARK_DASHED_GAP_MIN:
            res.mark_class = MARK_DASHED_WHITE
            res.reasons.append("gappy_stroke")
            res.confidence = 0.8
            return res
        if coarse == "dashed":
            res.mark_class = MARK_DASHED_WHITE
            res.reasons.append("extractor_dashed")
            res.confidence = 0.7
            return res
        if coarse in ("solid", "thin") or short:
            res.mark_class = MARK_SOLID_WHITE
            res.reasons.append("continuous_stroke")
            res.confidence = 0.8 if coarse == "solid" else 0.6
            return res
        res.reasons.append("white_but_unmeasured")
        res.confidence = 0.3
        return res

    # --- 3) no colour evidence: only a solid white reading is safe -------
    if coarse == "solid":
        res.mark_class = MARK_SOLID_WHITE
        res.reasons.append("solid_without_colour")
        res.confidence = 0.4
        return res
    res.reasons.append("no_evidence")
    return res


def classify_marking_object(mk, **evidence) -> MarkingClassResult:
    """Convenience wrapper for a ``vision.lanes.LaneMarking``-shaped item.

    Reads only what the object carries (colour, coarse kind, span) and
    passes the caller's extra geometry through, so the classifier stays
    pure while the call sites stay short.
    """
    world = getattr(mk, "world", None)
    span = evidence.pop("span_m", None)
    if span is None and world is not None:
        try:
            import numpy as np
            pts = np.asarray(world, dtype=float)[:, :2]
            if len(pts) >= 2:
                span = float(np.sum(np.linalg.norm(np.diff(pts, axis=0),
                                                   axis=1)))
        except Exception:
            span = None
    return classify_marking(color=getattr(mk, "color", None),
                            kind=getattr(mk, "kind", None),
                            span_m=span, **evidence)
