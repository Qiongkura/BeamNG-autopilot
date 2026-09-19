"""Semantic (road / lane-line) task head.

Wraps the learned ``Segmenter`` (road + line masks) and the classic
``LaneDetector`` lane pipeline into one HydraNet head: given one frame
it returns the road / line / offroad masks plus the world-space lane
markings the planner consumes.
"""

from __future__ import annotations

import numpy as np

from ..hydra import FrameContext, TaskOutput


# Probability gate (plan phase E1), default OFF like every other
# behaviour-changing perception switch here: when enabled, the semantic
# head asks the segmenter for the class probabilities of the SAME
# forward pass and removes line pixels that fail the dual gate (line
# probability + road context).  The gate can only REMOVE pixels: the
# existing pipeline, the yellow prior and the evidence accumulator keep
# their roles, so enabling it is a precision/recall trade decided on
# live frames, not offline.
import os as _os
SEG_PROB_GATE_ENABLED = _os.environ.get("BEAMNG_SEG_PROB_GATE", "0") == "1"

# Near/mid/far thresholds (plan phase E2), also opt-in.  Enabling this
# implies the probability gate above and switches it to per-zone
# thresholds, with the far band additionally requiring history support
# from the evidence accumulator; the zone band fractions are starting
# points, not a calibration (see vision.seg_zones).
SEG_ZONES_ENABLED = _os.environ.get("BEAMNG_SEG_ZONES", "0") == "1"

# Fine-grained marking classes (plan phase E5), opt-in: label every
# extracted marking with the taxonomy the downstream rules speak
# (solid/dashed white, yellow centre, white edge, curb, roadside
# artifact) and keep the NON-paint classes out of the lane geometry.
# Off by default like the other perception switches: the thresholds are
# evidence-shaped starting points, not a calibration.
MARK_CLASS_ENABLED = _os.environ.get("BEAMNG_MARK_CLASS", "0") == "1"


class SemanticHead:
    """Road / painted-line segmentation + lane markings task head."""

    name = "semantic"

    def __init__(self, segmenter=None, lane_detector=None,
                 enable_evidence: bool = True, line_segmenter=None):
        # Either may be None: lazily import/construct only when actually
        # used so the head stays usable in offline unit tests without a
        # trained model or a game.  ``line_segmenter`` is an optional
        # deliberate split-model path: a paved-shoulder fine-tune can own
        # the ROAD mask while the proven deployed checkpoint owns the
        # PAINTED-LINE mask.  The two labels have different failure modes;
        # sharing one checkpoint made a road-edge read like a lane line and
        # sent the pairing gate looking for a line at the pavement edge.
        self.segmenter = segmenter
        self.line_segmenter = line_segmenter
        self.lane_detector = lane_detector
        self._lane_fallback = None
        # Multi-frame line evidence (world-space accumulation); front
        # camera only.  Fills single-frame line dropouts from recent
        # sightings without propagating unconfirmed single-frame noise.
        self.enable_evidence = bool(enable_evidence)
        self._evidence = None

    def reset(self) -> None:
        """Drop accumulated line evidence (location-bound; teleport)."""
        if self._evidence is not None:
            self._evidence.reset()
        if self.segmenter is not None and hasattr(self.segmenter, "reset"):
            self.segmenter.reset()
        if (self.line_segmenter is not None
                and self.line_segmenter is not self.segmenter
                and hasattr(self.line_segmenter, "reset")):
            self.line_segmenter.reset()

    def _get_segmenter(self):
        if self.segmenter is None:
            from ..segmentation import Segmenter
            self.segmenter = Segmenter()
        return self.segmenter

    def _get_lanes(self):
        if self.lane_detector is None:
            from ..lanes import LaneDetector
            self.lane_detector = LaneDetector()
        return self.lane_detector

    def run(self, ctx: FrameContext) -> TaskOutput:
        out = TaskOutput()
        h, w = ctx.frame_rgb.shape[:2]
        road = np.ones((h, w), dtype=bool)
        line = np.zeros((h, w), dtype=bool)
        markings: list = []
        prediction_ok = False
        _prob_maps = None
        try:
            seg = self._get_segmenter()
            if ((SEG_PROB_GATE_ENABLED or SEG_ZONES_ENABLED)
                    and self.line_segmenter is None
                    and hasattr(seg, "predict_with_probs")):
                # one inference, both the pipeline masks and the class
                # probabilities (plan E1)
                road, line, _prob_maps = seg.predict_with_probs(ctx.frame_rgb)
            else:
                road, line = seg.predict(ctx.frame_rgb)
            if self.line_segmenter is not None and self.line_segmenter is not seg:
                _road_unused, line = self.line_segmenter.predict(ctx.frame_rgb)
                # those probabilities would belong to the OTHER model
                _prob_maps = None
            prediction_ok = True
        except Exception as exc:
            self.reset()
            out.meta.setdefault("line_errors", {})["predict"] = str(exc)
            # No trained model / inference error: the planner simply
            # has no sensor lane this frame (existing fallback).
            road = np.ones((h, w), dtype=bool)
            line = np.zeros((h, w), dtype=bool)
        raw_line = line
        out.meta["line_pixels_raw"] = int(np.count_nonzero(raw_line))
        # US yellow centre paint: UNet is white-line biased; union the HSV
        # prior so pairing / evidence see yellow as LINE (east_coast).
        # BEAMNG_YELLOW_FUSION=0 disables for A/B.
        import os
        _yellow_prior = None
        if prediction_ok and os.environ.get("BEAMNG_YELLOW_FUSION", "1") != "0":
            try:
                from ..yellow_line_mask import yellow_line_mask
                ym = yellow_line_mask(ctx.frame_rgb)
                if ym.shape == line.shape:
                    _yellow_prior = ym.astype(bool)
                    line = line | _yellow_prior
            except Exception as exc:
                out.meta.setdefault("line_errors", {})["yellow"] = str(exc)
        out.meta["line_pixels_yellow"] = int(
            np.count_nonzero(line) - np.count_nonzero(raw_line))
        if _prob_maps is not None:
            # Dual probability gate (plan E1) - or its per-zone form (plan
            # E2) when that switch is on.  Either way a line pixel must
            # clear the line threshold AND sit in road context; the zoned
            # form applies each band's own thresholds and additionally
            # requires HISTORY for the loose far band.  Applied to the
            # CURRENT frame's mask, before the temporal evidence fuse, so
            # the dropout survival that fuse provides is untouched, and the
            # gate can only REMOVE pixels.
            try:
                if SEG_ZONES_ENABLED:
                    _support = None
                    _ev = getattr(self, "_evidence", None)
                    if _ev is not None and ctx.cam is not None:
                        try:
                            _support = _ev.support_mask(
                                ctx.frame_rgb.shape[:2], ctx.cam, ctx.pos,
                                ctx.heading, ctx.ground_z)
                        except Exception:
                            _support = None
                    from ..seg_zones import gate_masks_zoned
                    _road_g, _line_g, _zstats = gate_masks_zoned(
                        _prob_maps, extra_line=_yellow_prior,
                        history_support=_support)
                    line = line & _line_g
                    out.meta["seg_gate_zones"] = _zstats.digest()
                    out.meta["seg_gate_mode"] = "zoned"
                else:
                    from ..seg_probs import gate_masks
                    _road_g, _line_g, _gate_stats = gate_masks(
                        _prob_maps, extra_line=_yellow_prior)
                    line = line & _line_g
                    out.meta["seg_gate"] = _gate_stats.digest()
                    out.meta["seg_gate_mode"] = "flat"
            except Exception as exc:
                out.meta.setdefault("line_errors", {})["seg_gate"] = str(exc)
        if prediction_ok and self.enable_evidence and ctx.role == "front_main":
            try:
                if self._evidence is None:
                    from ..line_evidence import LineEvidenceAccumulator
                    self._evidence = LineEvidenceAccumulator()
                # Provenance-aware fuse (plan phase E3): the mask keeps
                # the historical support, and the meta says how much of
                # it is a FRESH observation vs. held evidence, plus the
                # continuous-loss expiry state - a consumer must be able
                # to tell "seen now" from "remembered".  Duck-typed: an
                # accumulator that only implements ``fuse`` (test stubs,
                # alternate evidence sources) must keep working - losing
                # the whole evidence channel over a missing accessor
                # would be worse than not reporting the confidence.
                _fuse_conf = getattr(self._evidence, "fuse_with_confidence",
                                     None)
                if callable(_fuse_conf):
                    line, _ev_info = _fuse_conf(
                        line, ctx.cam, ctx.pos, ctx.heading, ctx.ground_z,
                        now=ctx.timestamp)
                    out.meta["line_evidence"] = dict(_ev_info)
                else:
                    line = self._evidence.fuse(
                        line, ctx.cam, ctx.pos, ctx.heading, ctx.ground_z,
                        now=ctx.timestamp)
            except Exception as exc:
                if self._evidence is not None:
                    self._evidence.reset()
                out.meta.setdefault("line_errors", {})["evidence"] = str(exc)
        out.meta["line_pixels_added"] = int(np.count_nonzero(line & ~raw_line))
        out.masks["road"] = road
        out.masks["line"] = line
        out.masks["offroad"] = ~road
        # Lane markings in world space (only meaningful on the front
        # camera; other roles leave this empty).
        if ctx.role == "front_main":
            if prediction_ok:
                try:
                    _mark_seg = (self.line_segmenter
                                 if self.line_segmenter is not None else seg)
                    markings = _mark_seg.detect_lines(
                        ctx.frame_rgb, ctx.cam, ctx.pos, ctx.heading,
                        ground_z=ctx.ground_z, line_mask=line,
                        road_mask=road)
                    # Preserve the yellow prior's COLOR.  The bool union
                    # above is correct for segmentation, but passing that
                    # union as one mask to ``detect_lines`` labels the
                    # yellow centre paint as WHITE.  The pairing policy
                    # uses color to distinguish a centre line from a white
                    # edge; losing it was why the live base run exposed a
                    # centre candidate at med_lat~0.0 as white and could
                    # not build an own-lane frame.  Re-project the yellow
                    # prior separately and append it only when the extractor
                    # did not already return a yellow marking.
                    # The learned line mask can contain white edge-like
                    # fragments while the classic color detector still sees
                    # the US yellow centre paint.  Always add yellow classic
                    # candidates when the learned extractor returned none;
                    # waiting for ``not markings`` discarded the centreline
                    # precisely when a spurious white edge candidate was
                    # present, leaving pairing with only the wrong side.
                    if not any(getattr(m, "color", None) == "yellow"
                               for m in markings or ()):
                        try:
                            _classic_y = self._get_lanes().detect(
                                ctx.frame_rgb, ctx.cam, ctx.pos, ctx.heading,
                                ground_z=ctx.ground_z)
                            markings.extend([
                                m for m in (_classic_y or ())
                                if getattr(m, "color", None) == "yellow"
                            ])
                            if not any(getattr(m, "color", None) == "yellow"
                                       for m in markings or ()):
                                from ..yellow_line_mask import yellow_line_mask
                                from ..lanes import _mask_to_markings
                                _ym = yellow_line_mask(ctx.frame_rgb)
                                if _ym.any():
                                    markings.extend(_mask_to_markings(
                                        _ym.astype(np.uint8) * 255, "yellow",
                                        ctx.cam, ctx.pos, ctx.heading,
                                        ground_z=ctx.ground_z))
                        except Exception as _cye:
                            out.meta.setdefault("line_errors", {})[
                                "yellow_classic"] = str(_cye)
                    if line.any():
                        for marking in markings:
                            if getattr(marking, "kind", None) == "unknown":
                                marking.kind = "thin"
                except Exception as exc:
                    out.meta.setdefault("line_errors", {})["markings"] = str(exc)
                    markings = []
            if not markings:
                # The learned mask can contain real paint while its
                # mask-to-polyline extractor returns no usable component
                # (short/faded US paint).  Classic CV is an independent
                # geometric fallback and must still get a chance; it does
                # not re-run segmentation or revive historical evidence.
                try:
                    markings = self._get_lanes().detect(
                        ctx.frame_rgb, ctx.cam, ctx.pos, ctx.heading,
                        ground_z=ctx.ground_z)
                    # The classic detector may call a small but bright paint
                    # fragment "unknown" because its geometry is short.  If
                    # the learned LINE mask also has evidence on this frame,
                    # promote only those fallback fragments to a conservative
                    # thin marking; pairing still applies all spatial gates.
                    if line.any():
                        for marking in markings:
                            if getattr(marking, "kind", None) == "unknown":
                                marking.kind = "thin"
                except Exception:
                    markings = []
        if MARK_CLASS_ENABLED and markings:
            # Plan E5: refine every candidate's class and keep the classes
            # that are not lane paint out of the geometry a lane may be
            # paired from.  A roadside artifact (reflector post, wall edge)
            # is dropped here; a CURB is kept but labelled, because the
            # AGENTS.md trust order allows a physical road edge as a
            # boundary while forbidding it as a painted line - the label
            # is what lets the pairing policy tell them apart.
            try:
                from ...lane.marking_class import (
                    MARK_ROADSIDE_ARTIFACT, classify_marking_object,
                )
                _kept = []
                _counts: dict = {}
                for _mk in markings:
                    # the drivable evidence the head already holds: project
                    # the marking's world points into the road mask.  When
                    # there is no camera or no mask the answer stays None,
                    # and the classifier treats unknown as "not
                    # contradicted" rather than as off-road.
                    _on_drivable = None
                    if ctx.cam is not None and road is not None:
                        try:
                            _w = np.asarray(
                                getattr(_mk, "world", None),
                                dtype=float)[:, :2]
                            if len(_w) >= 1:
                                _w3 = np.column_stack([
                                    _w, np.full(len(_w),
                                                float(ctx.ground_z))])
                                _u, _v, _ok = ctx.cam.project(
                                    _w3, np.asarray(ctx.pos, dtype=float),
                                    float(ctx.heading))
                                _ok = np.asarray(_ok, dtype=bool)
                                if _ok.any():
                                    _uu = np.asarray(_u, dtype=float)[_ok]                                         .astype(int)
                                    _vv = np.asarray(_v, dtype=float)[_ok]                                         .astype(int)
                                    _hh, _ww = road.shape
                                    _inb = ((_uu >= 0) & (_uu < _ww)
                                            & (_vv >= 0) & (_vv < _hh))
                                    if _inb.any():
                                        _on_drivable = bool(np.mean(
                                            road[_vv[_inb], _uu[_inb]]) >= 0.3)
                        except Exception:
                            _on_drivable = None
                    _cls = classify_marking_object(_mk,
                                                   on_drivable=_on_drivable)
                    try:
                        _mk.mark_class = _cls.mark_class
                        _mk.mark_class_reasons = list(_cls.reasons)
                    except Exception:
                        pass
                    _counts[_cls.mark_class] = _counts.get(_cls.mark_class, 0) + 1
                    if _cls.mark_class == MARK_ROADSIDE_ARTIFACT:
                        continue
                    _kept.append(_mk)
                markings = _kept
                out.meta["mark_class"] = _counts
            except Exception as exc:
                out.meta.setdefault("line_errors", {})["mark_class"] = str(exc)
        out.meta["markings"] = markings
        return out