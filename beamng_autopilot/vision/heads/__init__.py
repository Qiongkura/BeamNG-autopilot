"""Task heads wrapping the existing perception algorithms as HydraNets.

Every head implements ``HydraHead`` (``run(ctx) -> TaskOutput``) and
wraps an existing, already-working component instead of re-implementing
it, so the shared-backbone shape lands without touching the validated
perception code.
"""

from __future__ import annotations

from .semantic import SemanticHead
from .object import ObjectHead
from .traffic import TrafficSignalHead, merge_signal_vision, suggest_signal_state
from .topology import (
    LaneGraph,
    LaneTopologyHead,
    build_lane_graph,
)

__all__ = [
    "SemanticHead",
    "ObjectHead",
    "TrafficSignalHead",
    "merge_signal_vision",
    "suggest_signal_state",
    "LaneGraph",
    "LaneTopologyHead",
    "build_lane_graph",
]