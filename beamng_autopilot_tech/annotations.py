"""BeamNG.tech annotation pixel-truth helpers shared by the collectors.

The Tech camera can render per-object colour annotations alongside the
RGB frame; the asphalt / lane-line colours below are BeamNG.tech's fixed
annotation palette.  Collectors use them to (a) build segmentation labels
and (b) reject low-quality driving frames (off-road, black view) at the
source instead of poisoning the BC dataset.
"""
from __future__ import annotations

import numpy as np

ANN_ASPHALT = (128, 128, 128)
ANN_SOLID_LINE = (255, 196, 128)
ANN_DASHED_LINE = (196, 196, 255)
ANN_ZEBRA = (255, 128, 128)
_LINE_COLORS = (ANN_SOLID_LINE, ANN_DASHED_LINE, ANN_ZEBRA)

# Road-surface annotation colors are map-dependent: italy's asphalt comes
# back as (128,128,128) while smallgrid's road material is (128,196,255).
# The quality gate must accept both.
ANN_ROAD_COLORS = (ANN_ASPHALT, (128, 196, 255))


def to_label(ann_rgb: np.ndarray) -> np.ndarray:
    """Annotation RGB frame -> 3-class label map (H, W) uint8.

    Classes: 0 background, 1 asphalt/road, 2 lane markings.
    """
    label = np.zeros(ann_rgb.shape[:2], dtype=np.uint8)
    label[(ann_rgb == np.asarray(ANN_ASPHALT, dtype=np.uint8)).all(axis=2)] = 1
    for c in _LINE_COLORS:
        label[(ann_rgb == np.asarray(c, dtype=np.uint8)).all(axis=2)] = 2
    return label


def road_share(ann_rgb: np.ndarray, roi_rows: float = 0.66) -> float:
    """Fraction of road pixels in the lower part of the frame.

    ``roi_rows`` is the share of the frame height counted from the bottom
    (the hood/road area in a front camera).  Any known road-surface
    annotation colour counts (see ``ANN_ROAD_COLORS``).  A frame where the
    expert is off-road or the camera is black has a low road share and
    should be dropped from BC training data.
    """
    if ann_rgb is None or ann_rgb.size == 0:
        return 0.0
    h = ann_rgb.shape[0]
    roi = ann_rgb[int(h * (1.0 - roi_rows)):, :, :]
    road = np.zeros(roi.shape[:2], dtype=bool)
    for c in ANN_ROAD_COLORS:
        road |= (roi == np.asarray(c, dtype=np.uint8)).all(axis=2)
    return float(np.mean(road))