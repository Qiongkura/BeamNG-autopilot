"""Canonical per-tick perception snapshot for the FSD pipeline.

This is the boundary between sensing and downstream consumers.  It keeps
all outputs from one tick together with the pose and freshness information
needed to decide whether a cached result is still safe to use.  Large arrays
remain references (the snapshot is a lightweight view); callers must treat
one returned snapshot as read-only for the rest of the tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PerceptionSnapshot:
    """One pose-consistent perception result and its provenance."""

    captured_at: float
    tick_id: int
    pos: np.ndarray
    heading: float
    frame: np.ndarray | None = None
    cam: Any = None
    head_outputs: dict[str, Any] = field(default_factory=dict)
    head_age_s: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    ray_hits: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    bev: np.ndarray | None = None
    drivable: np.ndarray | None = None
    observed: np.ndarray | None = None
    feature_map: Any = None
    lane_envelope: Any = None
    range_age_s: float | None = None
    bev_age_s: float | None = None

    @property
    def max_sensor_age_s(self) -> float:
        """Maximum age across all available perception modalities."""
        ages = []
        for value in self.head_age_s.values():
            if value is None:
                return float("inf")
            value = float(value)
            if not np.isfinite(value):
                return float("inf")
            ages.append(value)
        if self.range_age_s is not None:
            if not np.isfinite(float(self.range_age_s)):
                return float("inf")
            ages.append(float(self.range_age_s))
        if self.bev_age_s is not None:
            if not np.isfinite(float(self.bev_age_s)):
                return float("inf")
            ages.append(float(self.bev_age_s))
        if self.lane_envelope is not None:
            ages.append(float(self.lane_envelope.age_s))
        return max(ages, default=0.0)

    def freshness(self) -> dict[str, float | None]:
        """JSON-safe per-modality ages for SafetyMonitor/telemetry."""
        return {
            "head_max_s": (None if not self.head_age_s
                           or any(v is None for v in self.head_age_s.values())
                           else max(self.head_age_s.values())),
            "range_s": self.range_age_s,
            "bev_s": self.bev_age_s,
            "lane_s": (self.lane_envelope.age_s
                        if self.lane_envelope is not None else None),
            "max_s": self.max_sensor_age_s,
        }

    @property
    def valid(self) -> bool:
        """True when the snapshot has a finite pose and at least one sensor."""
        return (self.pos.size >= 2 and np.isfinite(self.pos[:2]).all()
                and (self.frame is not None or self.bev is not None
                     or bool(self.ray_hits)))

    def age_s(self, now: float) -> float:
        """Age of the complete snapshot at an explicit clock value."""
        return max(0.0, float(now) - float(self.captured_at))

    def meta(self) -> dict:
        """Small JSON-safe freshness/provenance summary for telemetry."""
        return {
            "tick_id": int(self.tick_id),
            "captured_at": float(self.captured_at),
            "head_age_s": {k: (None if v is None else round(float(v), 3))
                           for k, v in self.head_age_s.items()},
            "range_age_s": (None if self.range_age_s is None
                            else round(float(self.range_age_s), 3)),
            "bev_age_s": (None if self.bev_age_s is None
                          else round(float(self.bev_age_s), 3)),
            "freshness": {k: (None if v is None else round(float(v), 3))
                           for k, v in self.freshness().items()},
            "errors": dict(self.errors),
            "has_frame": int(self.frame is not None),
            "has_bev": int(self.bev is not None),
            "has_lane": int(self.lane_envelope is not None
                             and getattr(self.lane_envelope, "valid", False)),
            "n_tracks": len(self.tracks),
        }
