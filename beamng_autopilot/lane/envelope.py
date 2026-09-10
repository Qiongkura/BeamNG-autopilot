"""Canonical sensor-owned lane envelope shared by planning and safety.

A ``LaneFrame`` is a detector/fusion result.  ``SensorLaneEnvelope`` is
the live contract: one center/boundary geometry, one source/confidence,
and one age/uncertainty description consumed by planner, safety,
controller correction and telemetry.  It is deliberately immutable-ish
(copy-on-build arrays) so callers do not silently mutate a different
version of the lane during one tick.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class SensorLaneEnvelope:
    center: np.ndarray
    left: np.ndarray | None
    right: np.ndarray | None
    width_m: float
    source: str
    paired: bool
    confidence: float
    captured_at: float
    uncertainty_m: float = 0.0
    left_real: bool = False
    right_real: bool = False
    virtual_boundary: bool = False

    @classmethod
    def from_lane_frame(cls, frame, captured_at: float | None = None,
                        uncertainty_m: float = 0.0):
        """Build the canonical object from an existing LaneFrame."""
        if frame is None:
            return None
        arr = lambda x: (None if x is None else
                         np.asarray(x, dtype=float)[:, :2].copy())
        return cls(
            center=arr(frame.center),
            left=arr(getattr(frame, "left", None)),
            right=arr(getattr(frame, "right", None)),
            width_m=float(getattr(frame, "width", 0.0) or 0.0),
            source="+".join(getattr(frame, "sources", ()) or ()) or "unknown",
            paired=bool(getattr(frame, "paired", False)),
            confidence=float(getattr(frame, "confidence", 0.0) or 0.0),
            captured_at=float(time.time() if captured_at is None
                              else captured_at),
            uncertainty_m=float(uncertainty_m),
            left_real=getattr(frame, "left", None) is not None,
            right_real=getattr(frame, "right", None) is not None,
            virtual_boundary=not bool(getattr(frame, "paired", False)),
        )

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - float(self.captured_at))

    @property
    def valid(self) -> bool:
        return self.center is not None and len(self.center) >= 3 \
            and np.isfinite(self.center).all()

    def as_meta(self) -> dict:
        """Compact telemetry representation (no large polylines)."""
        return {
            "source": self.source,
            "paired": int(self.paired),
            "confidence": round(self.confidence, 3),
            "age_s": round(self.age_s, 3),
            "uncertainty_m": round(self.uncertainty_m, 3),
            "left_real": int(self.left_real),
            "right_real": int(self.right_real),
            "virtual_boundary": int(self.virtual_boundary),
            "width_m": round(self.width_m, 3),
        }
