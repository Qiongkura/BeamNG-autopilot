"""Spawn / camera legality: reject vegetation pits and empty frames.

Roadnet nodes and teleports are not legal camera poses. The 2026-09-13
east_coast judgment drive sat in bushes; photo tour must mark such stops
instead of presenting them as usable spawn candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from beamng_autopilot.vision.veg_mask import green_vegetation_mask

# Lower band of the frame carries the road; sky/canopy tops are ignored.
ROI_TOP_FRAC = 0.30
GREEN_FRAC_MAX = 0.45
MEAN_V_MIN = 40.0
ROADLIKE_FRAC_MIN = 0.08
# Asphalt-ish: low saturation, mid brightness, not vegetation.
_ROAD_S_MAX = 70
_ROAD_V_MIN = 50
_ROAD_V_MAX = 220


@dataclass
class SpawnAssessment:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    green_frac: float = 0.0
    dark_frac: float = 0.0
    roadlike_frac: float = 0.0
    mean_v: float = 0.0

    def as_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "reasons": list(self.reasons),
            "green_frac": round(float(self.green_frac), 4),
            "dark_frac": round(float(self.dark_frac), 4),
            "roadlike_frac": round(float(self.roadlike_frac), 4),
            "mean_v": round(float(self.mean_v), 2),
        }


def assess_spawn_frame(rgb: np.ndarray) -> SpawnAssessment:
    """RGB uint8 frame -> whether this camera pose is worth teleporting to."""
    import cv2

    frame = np.ascontiguousarray(rgb)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return SpawnAssessment(ok=False, reasons=["bad_frame"])
    h = frame.shape[0]
    roi_top = int(h * ROI_TOP_FRAC)
    roi = frame[roi_top:]
    if roi.size == 0:
        return SpawnAssessment(ok=False, reasons=["bad_frame"])

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    veg = green_vegetation_mask(roi)
    # green_vegetation_mask already thresholds HSV; recompute on ROI size
    # via the shared helper so thresholds stay single-sourced.
    green_frac = float(np.mean(veg)) if veg.size else 1.0
    mean_v = float(np.mean(v))
    dark_frac = float(np.mean(v < 40))
    roadlike = (s <= _ROAD_S_MAX) & (v >= _ROAD_V_MIN) & (v <= _ROAD_V_MAX) \
        & (~veg)
    roadlike_frac = float(np.mean(roadlike))

    reasons: list[str] = []
    if green_frac > GREEN_FRAC_MAX:
        reasons.append("too_green")
    if mean_v < MEAN_V_MIN:
        reasons.append("too_dark")
    if roadlike_frac < ROADLIKE_FRAC_MIN:
        reasons.append("no_road_like")
    return SpawnAssessment(
        ok=not reasons,
        reasons=reasons,
        green_frac=green_frac,
        dark_frac=dark_frac,
        roadlike_frac=roadlike_frac,
        mean_v=mean_v,
    )
