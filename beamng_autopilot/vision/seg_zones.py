"""Near / mid / far zones for the probability gates (plan phase E2).

One threshold for the whole image forces a bad trade: a value strict
enough to keep a false line out of the near field also throws away the
weak, genuinely-there markings the planner needs early, and a value loose
enough to keep those lets roadside clutter in under the bumper.  The plan
asks for three bands with different jobs:

* ``near``  (image bottom) - strictest thresholds and a SHORT history
  hold: it directly decides what the car does right now, so precision
  matters more than recall and yesterday's evidence must not linger;
* ``mid``   - balanced thresholds, feeds the trajectory;
* ``far``   (image top) - loosest thresholds, because a distant line is
  weak and only used for anticipation - but every far candidate must be
  supported by HISTORY (the plan's "允许更宽松候选，但必须满足几何连续
  性" in the temporal form this layer can enforce; the geometric
  consistency itself belongs to E4).

Rows are the image's, since that is where the camera sees distance; the
fractions below are starting points, not a calibration, and a caller may
supply its own ``ZoneSpec`` list.  Nothing here reads a map or a route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from beamng_autopilot.vision.seg_probs import (
    SEG_LINE_PROB_MIN,
    SEG_ROAD_CONTEXT_PX,
    SEG_ROAD_PROB_MIN,
    gate_masks,
)

ZONE_NEAR = "near"
ZONE_MID = "mid"
ZONE_FAR = "far"
ZONE_ORDER = (ZONE_NEAR, ZONE_MID, ZONE_FAR)


@dataclass
class ZoneSpec:
    """One image band and the gate policy that applies inside it."""

    name: str
    row_from_frac: float        # 0.0 = top of the image
    row_to_frac: float          # 1.0 = bottom (closest to the car)
    line_min: float = SEG_LINE_PROB_MIN
    road_min: float = SEG_ROAD_PROB_MIN
    road_context_px: int = SEG_ROAD_CONTEXT_PX
    hold_s: float = 0.5         # history hold time reported for this zone
    require_history: bool = False


def default_zones() -> tuple[ZoneSpec, ...]:
    """Near (strict, short hold) / mid (balanced) / far (loose + history).

    The row fractions are image-space starting points for a forward
    camera pitched at the road: the bottom third is the near field, the
    next band the mid field, the top the far field.  They are NOT
    calibrated against a distance and should be tuned per mount.
    """
    return (
        ZoneSpec(ZONE_NEAR, 0.62, 1.0, line_min=0.60, road_min=0.40,
                 road_context_px=3, hold_s=0.15, require_history=False),
        ZoneSpec(ZONE_MID, 0.32, 0.62, line_min=0.50, road_min=0.35,
                 road_context_px=3, hold_s=0.5, require_history=False),
        ZoneSpec(ZONE_FAR, 0.0, 0.32, line_min=0.40, road_min=0.30,
                 road_context_px=4, hold_s=1.0, require_history=True),
    )


def _band_rows(spec: ZoneSpec, height: int) -> tuple[int, int]:
    """Half-open ``[from, to)`` row range of a band, clamped to the image."""
    h = max(1, int(height))
    a = int(round(float(spec.row_from_frac) * h))
    b = int(round(float(spec.row_to_frac) * h))
    a = max(0, min(h, a))
    b = max(0, min(h, b))
    if b < a:
        a, b = b, a
    return a, b


def zone_of_row(row: int, height: int, zones=None) -> str:
    """Which zone a row falls in (``ZONE_MID`` when it falls in none)."""
    specs = tuple(zones or default_zones())
    r = int(row)
    for spec in specs:
        a, b = _band_rows(spec, height)
        if a <= r < b:
            return spec.name
    return ZONE_MID


def zone_row_masks(shape, zones=None) -> dict[str, np.ndarray]:
    """Boolean ``(H, W)`` mask per zone; the bands partition the image."""
    specs = tuple(zones or default_zones())
    h, w = int(shape[0]), int(shape[1])
    out: dict[str, np.ndarray] = {}
    for spec in specs:
        a, b = _band_rows(spec, h)
        m = np.zeros((h, w), dtype=bool)
        if b > a:
            m[a:b, :] = True
        out[spec.name] = m
    return out


@dataclass
class ZoneGateStats:
    """Per-zone gate statistics (telemetry + tests)."""

    per_zone: dict = field(default_factory=dict)
    line_before: int = 0
    line_after: int = 0
    far_unconfirmed: int = 0

    def digest(self) -> dict:
        def _clean(v):
            if isinstance(v, float):
                return None if not math.isfinite(v) else round(v, 4)
            return v
        return {
            "line_before": int(self.line_before),
            "line_after": int(self.line_after),
            "far_unconfirmed": int(self.far_unconfirmed),
            "zones": {k: {kk: _clean(vv) for kk, vv in v.items()}
                      for k, v in self.per_zone.items()},
        }


def gate_masks_zoned(maps,
                     zones=None,
                     *,
                     extra_line: np.ndarray | None = None,
                     history_support: np.ndarray | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, ZoneGateStats]:
    """Apply the dual gate with PER-ZONE thresholds.

    Each band is gated with its own ``line_min`` / ``road_min`` /
    context radius; a band with ``require_history`` additionally needs the
    pixel in ``history_support`` (the temporally accumulated evidence)
    before it may be used.  When ``history_support`` is None the
    requirement cannot be evaluated: the candidates are kept - unknown
    evidence is not the same as contradicted evidence - but the count is
    reported in ``far_unconfirmed`` so telemetry shows that the far field
    is running unconfirmed.
    """
    if maps.line is None or maps.road is None:
        raise ValueError("line and road probability maps are required")
    if maps.line.shape != maps.road.shape:
        raise ValueError("line and road maps must share a shape")
    specs = tuple(zones or default_zones())
    h, w = maps.line.shape
    support = None
    if history_support is not None:
        support = np.asarray(history_support, dtype=bool)
        if support.shape != (h, w):
            support = None
    extra = None
    if extra_line is not None:
        extra = np.asarray(extra_line)
        if extra.shape != (h, w):
            extra = None

    road_total = np.zeros((h, w), dtype=bool)
    line_total = np.zeros((h, w), dtype=bool)
    stats = ZoneGateStats()
    for spec in specs:
        # slice by ROW RANGE, not by a boolean band mask: fancy-indexing
        # with a 2-D mask flattens the band to 1-D and the sliding max
        # filter (and every 2-D consumer) then breaks
        a, b = _band_rows(spec, h)
        if b <= a:
            continue
        sub = type(maps)(line=maps.line[a:b], road=maps.road[a:b])
        sub_extra = None if extra is None else extra[a:b]
        road_band, line_band, band_stats = gate_masks(
            sub, line_min=spec.line_min, road_min=spec.road_min,
            context_px=spec.road_context_px, extra_line=sub_extra)
        if spec.require_history:
            if support is None:
                stats.far_unconfirmed += int(np.count_nonzero(line_band))
            else:
                sup = support[a:b]
                stats.far_unconfirmed += int(np.count_nonzero(
                    line_band & ~sup))
                line_band = line_band & sup
        road_total[a:b] = road_band
        line_total[a:b] = line_band
        stats.line_before += int(band_stats.line_before)
        stats.line_after += int(np.count_nonzero(line_band))
        stats.per_zone[spec.name] = {
            "line_before": int(band_stats.line_before),
            "line_after": int(np.count_nonzero(line_band)),
            "line_min": float(spec.line_min),
            "road_min": float(spec.road_min),
            "hold_s": float(spec.hold_s),
            "history_required": int(bool(spec.require_history)),
            "mean_line_p": float(band_stats.mean_line_prob),
        }
    return road_total, line_total, stats
