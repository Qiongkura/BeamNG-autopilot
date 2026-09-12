"""Training/evaluation-only Tech annotation samples; never a driving input."""
from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from beamng_autopilot_tech.annotations import annotation_palette, to_label

LABEL_SOURCE = "beamng_tech_annotation"


def annotation_metadata(annotations: Mapping | None = None) -> dict:
    """Describe provenance, not a claim that the map's labels are correct."""
    palette = annotation_palette(annotations)
    result = {
        "label_source": LABEL_SOURCE,
        "label_usage": "training_evaluation_only",
        "palette_version": palette["version"],
        "palette_source": palette["source"],
        "annotation_review_required": True,
        "palette": {
            "road": [list(c) for c in palette["road"]],
            "line": [list(c) for c in palette["line"]],
        },
    }
    if palette["classes"]:
        result["annotation_classes"] = {
            name: list(color) for name, color in palette["classes"].items()
        }
    return result


def prepare_annotation_sample(colour, annotation, *, width: int,
                              height: int, palette: Mapping | None = None
                              ) -> tuple[dict, dict]:
    """Prepare one same-poll RGB/annotation pair, preserving original labels.

    Shape agreement is checked here; the caller must obtain both outputs
    from one camera poll. It is not proof of correct map/material labels.
    Existing trainers still read only ``colour`` and ``label``.
    """
    if annotation is None:
        raise ValueError("Tech annotation is missing; no pseudo-label fallback")
    resolved = annotation_palette() if palette is None else palette
    if not isinstance(resolved, Mapping):
        raise ValueError("resolved annotation palette must be a mapping")
    road_colors = resolved.get("road")
    line_colors = resolved.get("line")
    if not road_colors or not line_colors:
        raise ValueError("resolved annotation palette must contain road and line colors")
    ann = np.asarray(annotation)
    # Validate before resizing, so missing/float/corrupt palettes fail loud.
    to_label(ann, road_colors=road_colors, line_colors=line_colors)
    rgb = np.asarray(colour)
    if (rgb.ndim != 3 or rgb.shape[2] not in (3, 4)
            or rgb.shape[:2] != ann.shape[:2] or rgb.dtype != np.uint8):
        raise ValueError("RGB and annotation must be matching uint8 images")
    if width <= 0 or height <= 0:
        raise ValueError("output dimensions must be positive")
    small_ann = cv2.resize(ann[:, :, :3], (width, height),
                           interpolation=cv2.INTER_NEAREST)
    label = to_label(small_ann, road_colors=road_colors,
                     line_colors=line_colors)
    payload = {
        "colour": cv2.resize(rgb[:, :, :3], (width, height),
                              interpolation=cv2.INTER_AREA),
        "label": label,
        "annotation": small_ann,
        "annotation_raw": ann.copy(),
        "label_source": np.asarray(LABEL_SOURCE),
        "palette_version": np.asarray(resolved.get("version", "unknown")),
    }
    stats = {
        "road_pixels": int(np.count_nonzero(label == 1)),
        "line_pixels": int(np.count_nonzero(label == 2)),
        "background_or_unmapped_pixels": int(np.count_nonzero(label == 0)),
        # No-line frames can be legitimate negatives; flag, don't invent paint.
        "needs_line_review": not bool(np.any(label == 2)),
    }
    return payload, stats
