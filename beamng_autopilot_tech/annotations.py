"""BeamNG.tech annotation pixel-truth helpers shared by the collectors.

The Tech camera can render per-object colour annotations alongside the RGB
frame.  The running simulator exposes the active palette through
``BeamNGpy.get_annotations()``; callers should resolve that palette once per
session instead of assuming that a colour means the same thing on every map.
Collectors use these helpers to build segmentation labels and to reject
low-quality driving frames at the source instead of poisoning the BC dataset.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

ANN_ASPHALT = (128, 128, 128)
ANN_ALT_ROAD = (128, 196, 255)
ANN_SOLID_LINE = (255, 196, 128)
ANN_DASHED_LINE = (196, 196, 255)
ANN_ZEBRA = (255, 128, 128)
_LINE_COLORS = (ANN_SOLID_LINE, ANN_DASHED_LINE, ANN_ZEBRA)

# Static compatibility palette used by offline callers and old tests.  A
# live Tech collector should use ``annotation_palette(bng.get_annotations())``
# because (128,196,255) is SKY in the current italy session, not asphalt.
ANN_ROAD_COLORS = (ANN_ASPHALT, ANN_ALT_ROAD)
PALETTE_VERSION = "known_road_line_v1"
TECH_PALETTE_VERSION = "beamng_get_annotations_v1"

# These are class names returned by BeamNG.tech's annotation API.  The first
# group contains drivable street surfaces; the second contains painted lane
# markings.  Missing optional classes are simply omitted from a map palette.
_ROAD_CLASSES = ("ASPHALT", "STREET", "RESTRICTED_STREET", "COBBLESTONE")
_LINE_CLASSES = ("SOLID_LINE", "DASHED_LINE", "ZEBRA_CROSSING")


def _color(value) -> tuple[int, int, int]:
    """Validate and normalize one Tech RGB triplet."""
    try:
        vals = tuple(int(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("annotation palette colors must be RGB triplets") from exc
    if len(vals) != 3 or any(v < 0 or v > 255 for v in vals):
        raise ValueError("annotation palette colors must be RGB triplets")
    return vals


def annotation_palette(annotations: Mapping | None = None) -> dict:
    """Resolve road/line colors from a Tech ``get_annotations`` response.

    When no response is supplied, return the historical static compatibility
    palette.  The explicit fallback is retained for offline tests and old
    tools; live Tech collectors pass the API response so SKY cannot be
    mistaken for ROAD merely because two maps used different colors.
    """
    if annotations is None:
        return {
            "road": ANN_ROAD_COLORS,
            "line": _LINE_COLORS,
            "version": PALETTE_VERSION,
            "source": "static_compatibility",
            "classes": {},
        }
    if not isinstance(annotations, Mapping):
        raise ValueError("Tech annotation palette must be a mapping")
    classes = {str(k): _color(v) for k, v in annotations.items()}
    road = tuple(classes[name] for name in _ROAD_CLASSES if name in classes)
    line = tuple(classes[name] for name in _LINE_CLASSES if name in classes)
    if not road:
        raise ValueError("Tech annotation palette has no road surface class")
    if not line:
        raise ValueError("Tech annotation palette has no lane-line class")
    return {
        "road": road,
        "line": line,
        "version": TECH_PALETTE_VERSION,
        "source": "beamng_get_annotations",
        "classes": classes,
    }


def _colors_or_default(colors, default):
    if colors is None:
        return default
    return tuple(_color(c) for c in colors)


def to_label(ann_rgb: np.ndarray, *, road_colors=None,
             line_colors=None) -> np.ndarray:
    """Annotation RGB frame -> 3-class label map (H, W) uint8.

    Classes: 0 background, 1 asphalt/road, 2 lane markings.  Pass colors
    from ``annotation_palette(bng.get_annotations())`` for live Tech data.
    """
    ann_rgb = np.asarray(ann_rgb)
    if (ann_rgb.ndim != 3 or ann_rgb.shape[2] not in (3, 4)
            or not ann_rgb.shape[0] or not ann_rgb.shape[1]):
        raise ValueError("annotation must be a nonempty HxWx3/4 image")
    if ann_rgb.dtype != np.uint8:
        raise ValueError("annotation must have uint8 palette values")
    ann_rgb = ann_rgb[:, :, :3]
    roads = _colors_or_default(road_colors, ANN_ROAD_COLORS)
    lines = _colors_or_default(line_colors, _LINE_COLORS)
    label = np.zeros(ann_rgb.shape[:2], dtype=np.uint8)
    for c in roads:
        label[(ann_rgb == np.asarray(c, dtype=np.uint8)).all(axis=2)] = 1
    for c in lines:
        label[(ann_rgb == np.asarray(c, dtype=np.uint8)).all(axis=2)] = 2
    return label


def road_share(ann_rgb: np.ndarray, roi_rows: float = 0.66, *,
               road_colors=None) -> float:
    """Fraction of road pixels in the lower part of the frame.

    ``roi_rows`` is the share of the frame height counted from the bottom.
    Pass the session-specific road colors for live Tech data; otherwise the
    historical static compatibility palette is used.
    """
    if ann_rgb is None or ann_rgb.size == 0:
        return 0.0
    ann_rgb = np.asarray(ann_rgb)
    if ann_rgb.ndim != 3 or ann_rgb.shape[2] not in (3, 4):
        return 0.0
    h = ann_rgb.shape[0]
    roi = ann_rgb[int(h * (1.0 - roi_rows)):, :, :3]
    road = np.zeros(roi.shape[:2], dtype=bool)
    for c in _colors_or_default(road_colors, ANN_ROAD_COLORS):
        road |= (roi == np.asarray(c, dtype=np.uint8)).all(axis=2)
    return float(np.mean(road))
