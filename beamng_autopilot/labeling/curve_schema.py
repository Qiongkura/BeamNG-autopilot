"""Annotation schema for lane curves (T10).

The plan's T10 asks for a schema that can carry what the pixel classes
alone cannot: which curve a stroke belongs to, whether it is the left or
right boundary or a divider, which parts are VISIBLE, where occlusion
starts and ends, its attributes, and what is simply UNKNOWN.  It also
requires that **centre lines, pixel paint and inferred extensions are
stored separately** - an interpolated curve is a different kind of
statement from an observed one, and merging them is how an inference
becomes "truth".

The schema is deliberately small and strict:

* a curve's segments carry a ``source``; ``inferred_extension`` segments
  must name the observations they were derived from (``derived_from``)
  and are never accepted as measured paint;
* ``unknown`` is an explicit list of field names, so "not annotated" is
  distinguishable from "annotated as absent";
* pixel classes are pinned here (``0=background, 1=road, 2=line,
  255=ignore``) and any other value in a mask is REPORTED, not coerced.

Nothing in this module drives anything: it reads and writes label files
and validates them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1

# Pixel classes.  255 is IGNORE (unlabelled), never background: an
# unlabelled pixel must not be counted as a true negative by a metric.
CLS_BACKGROUND = 0
CLS_ROAD = 1
CLS_LINE = 2
CLS_IGNORE = 255
PIXEL_CLASS_NAMES = {
    CLS_BACKGROUND: "background",
    CLS_ROAD: "road",
    CLS_LINE: "line",
    CLS_IGNORE: "ignore",
}

#: Boundary roles.  ``divider`` is a line that separates two same-direction
#: lanes; ``centre`` is the centre line (which may be a divider or an
#: opposite-direction separator - the annotation says which, the role here
#: only records what the annotator marked).
ROLES = ("left", "right", "divider", "centre", "unknown")

#: Where a segment's geometry comes from.  The first two are MEASURED
#: (pixels a human or the Tech annotation actually marked at that frame);
#: the third is an interpolation and must carry provenance.
SRC_PIXEL_PAINT = "pixel_paint"
SRC_CENTRE_LINE = "centre_line"
SRC_INFERRED = "inferred_extension"
SOURCES = (SRC_PIXEL_PAINT, SRC_CENTRE_LINE, SRC_INFERRED)
MEASURED_SOURCES = (SRC_PIXEL_PAINT, SRC_CENTRE_LINE)

ATTR_KEYS = ("colour", "style", "width_m")


def pixel_value_report(mask) -> dict:
    """Counts per known class plus any value the schema does not define.

    A mask with an undefined value is reported, not silently relabelled:
    the plan's rule is that unknown is not a class.
    """
    import numpy as np

    arr = np.asarray(mask)
    if arr.size == 0:
        return {"n": 0, "counts": {}, "unknown_values": []}
    values, counts = np.unique(arr, return_counts=True)
    known: dict[str, int] = {}
    unknown: list[int] = []
    for v, c in zip(values.tolist(), counts.tolist()):
        name = PIXEL_CLASS_NAMES.get(int(v))
        if name is None:
            unknown.append(int(v))
        else:
            known[name] = int(c)
    return {"n": int(arr.size), "counts": known,
            "unknown_values": sorted(unknown)}


@dataclass
class CurvePoint:
    """One annotated point of a curve on one frame."""

    u: float
    v: float
    frame: int | None = None
    t: float | None = None
    world: list | None = None      # measured 3D point, when one exists


@dataclass
class Occlusion:
    """An occluded span of a curve: 'these frames are not visible'."""

    by: str                       # e.g. "vehicle", "wall", "unknown"
    frame_start: int
    frame_end: int


@dataclass
class CurveSegment:
    """A run of frames over which a curve is one continuous statement."""

    frame_start: int
    frame_end: int
    source: str = SRC_PIXEL_PAINT
    visible: bool = True
    occlusion: Occlusion | None = None
    derived_from: list = field(default_factory=list)
    points: list = field(default_factory=list)      # list[CurvePoint]


@dataclass
class Curve:
    """One curve identity across frames (id is stable within a record)."""

    curve_id: str
    role: str = "unknown"
    attributes: dict = field(default_factory=dict)
    unknown: list = field(default_factory=list)     # field NAMES, explicit
    segments: list = field(default_factory=list)    # list[CurveSegment]


@dataclass
class FrameAnnotation:
    """One annotation batch, anchored at ``frame_index``.

    A record is anchored at the first frame its segments cover: a curve
    that spans frames 4-9 appears ONCE, as a segment with that span and its
    per-frame points - not as six near-identical records.
    """

    frame_index: int
    t: float | None = None
    image_w: int | None = None
    image_h: int | None = None
    curves: list = field(default_factory=list)      # list[Curve]
    run: str | None = None
    map_name: str | None = None
    episode: str | None = None
    source_id: str | None = None
    version: int = SCHEMA_VERSION
    notes: list = field(default_factory=list)


def to_dict(ann: FrameAnnotation) -> dict:
    return asdict(ann)


def _occlusion_from(value) -> Occlusion | None:
    if value is None:
        return None
    if isinstance(value, Occlusion):
        return value
    return Occlusion(**value)


def _segment_from(value) -> CurveSegment:
    if isinstance(value, CurveSegment):
        return value
    d = dict(value)
    d["occlusion"] = _occlusion_from(d.get("occlusion"))
    d["points"] = [p if isinstance(p, CurvePoint) else CurvePoint(**p)
                   for p in d.get("points", [])]
    return CurveSegment(**d)


def _curve_from(value) -> Curve:
    if isinstance(value, Curve):
        return value
    d = dict(value)
    d["segments"] = [_segment_from(s) for s in d.get("segments", [])]
    return Curve(**d)


def from_dict(payload: dict) -> FrameAnnotation:
    d = dict(payload)
    d["curves"] = [_curve_from(c) for c in d.get("curves", [])]
    return FrameAnnotation(**d)


def validate(ann: FrameAnnotation) -> list[str]:
    """Return the schema violations of one frame annotation (empty = ok)."""
    errs: list[str] = []
    if int(getattr(ann, "version", -1)) != SCHEMA_VERSION:
        errs.append(f"version {getattr(ann, 'version', None)!r} != "
                    f"{SCHEMA_VERSION}")
    if ann.frame_index is None or int(ann.frame_index) < 0:
        errs.append("frame_index missing or negative")
    if ann.image_w is not None and int(ann.image_w) <= 0:
        errs.append("image_w must be positive")
    if ann.image_h is not None and int(ann.image_h) <= 0:
        errs.append("image_h must be positive")
    ids: set[str] = set()
    for curve in ann.curves:
        if not curve.curve_id:
            errs.append("a curve has no curve_id")
        elif curve.curve_id in ids:
            errs.append(f"duplicate curve_id {curve.curve_id!r}")
        ids.add(curve.curve_id)
        if curve.role not in ROLES:
            errs.append(f"{curve.curve_id}: role {curve.role!r} not in "
                        f"{ROLES}")
        for name in curve.unknown:
            if name not in ATTR_KEYS and name != "role":
                errs.append(f"{curve.curve_id}: unknown field {name!r} is "
                            f"not an annotatable field")
        for name in curve.attributes:
            if name not in ATTR_KEYS:
                errs.append(f"{curve.curve_id}: attribute {name!r} not in "
                            f"{ATTR_KEYS}")
        if not curve.segments:
            errs.append(f"{curve.curve_id}: no segments")
        for i, seg in enumerate(curve.segments):
            where = f"{curve.curve_id}[{i}]"
            if seg.source not in SOURCES:
                errs.append(f"{where}: source {seg.source!r} not in {SOURCES}")
            if seg.frame_start is None or seg.frame_end is None:
                errs.append(f"{where}: segment needs frame_start/end")
            elif int(seg.frame_end) < int(seg.frame_start):
                errs.append(f"{where}: frame_end < frame_start")
            if seg.source == SRC_INFERRED and not seg.derived_from:
                errs.append(f"{where}: an inferred segment must name the "
                            f"observations it was derived from")
            if seg.source in MEASURED_SOURCES and seg.derived_from:
                errs.append(f"{where}: a measured segment must not claim "
                            f"derived_from")
            if seg.occlusion is not None:
                occ = seg.occlusion
                if int(occ.frame_end) < int(occ.frame_start):
                    errs.append(f"{where}: occlusion frame_end < frame_start")
                if not str(occ.by).strip():
                    errs.append(f"{where}: occlusion without a cause")
                if seg.visible:
                    errs.append(f"{where}: an occluded span must be stored as "
                                f"visible=False (it is not a visible segment)")
            for p in seg.points:
                if p.u is None or p.v is None:
                    errs.append(f"{where}: a point has no pixel")
                if p.frame is not None and not (
                        int(seg.frame_start) <= int(p.frame)
                        <= int(seg.frame_end)):
                    errs.append(f"{where}: point frame {p.frame} outside the "
                                f"segment span")
    return errs


def dump_jsonl(path, anns) -> int:
    """Write annotations as JSON Lines; each line is validated first."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for ann in anns:
            errs = validate(ann)
            if errs:
                raise ValueError(f"refusing to write an invalid annotation: "
                                 f"{errs[:3]}")
            fh.write(json.dumps(to_dict(ann), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path) -> list:
    """Read JSON Lines, rejecting records the schema does not accept."""
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            ann = from_dict(json.loads(line))
        except TypeError as exc:
            raise ValueError(f"line {i + 1}: {exc}") from exc
        errs = validate(ann)
        if errs:
            raise ValueError(f"line {i + 1}: {errs[:3]}")
        out.append(ann)
    return out
