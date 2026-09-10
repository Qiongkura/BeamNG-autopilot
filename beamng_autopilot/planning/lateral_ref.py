"""The single owner of the question "where is my lane?".

Real FSD splits *routing* ("where do I go") from *lane perception*
("where is the lane I am driving in").  The nav route is intent; it must
never place the car laterally.  This module is the one place that
answers the lateral question, so the policy cannot drift between
consumers - it used to be re-implemented in the safety monitor and in
the planner cost, and each copy had its own idea of when the map route
was allowed (that duplication is exactly how a route fallback survived
into the strict FSD path, docs/fsd_realism.md §2/§4).

Precedence, highest first:

* ``sensor``   - ``scene.lane_ref``, the own-lane centreline the stack
  derived from painted lines / LiDAR pairing.  The FSD reference.
* ``envelope`` - ``scene.lane_envelope.center``, the same geometry
  published as the canonical sensor contract.
* ``route``    - LEGACY ONLY.  ``scene.route`` is the nav polyline; it
  is allowed purely to keep the old rule stack (``lane_mode="map"``,
  ``strict_perception=False``) working.  Strict scenes never get it.
* ``none``     - nothing to steer on; the consumer must degrade
  (stop / hold heading), never fall back to map geometry.
"""

from __future__ import annotations

import numpy as np

REF_SENSOR = "sensor"
REF_ENVELOPE = "envelope"
REF_ROUTE = "route"
REF_NONE = "none"

# Perceptual sources - what a strict FSD run is allowed to align to.
REF_PERCEPTION = (REF_SENSOR, REF_ENVELOPE)


def _as_ref(value):
    """Normalise a candidate reference to an ``(N, 2)`` float array.

    Returns ``None`` for anything that cannot be used as a polyline
    (wrong shape, non-finite samples), so a malformed perception payload
    degrades instead of raising inside a control tick.
    """
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    arr = arr[:, :2]
    if not np.isfinite(arr).all():
        return None
    return arr


def lateral_reference(scene) -> tuple[np.ndarray | None, str]:
    """Answer "where is my lane" for one planning tick.

    Returns ``(reference, source)`` where ``source`` is one of
    ``REF_SENSOR`` / ``REF_ENVELOPE`` / ``REF_ROUTE`` / ``REF_NONE``.
    ``reference`` is an ``(N, 2)`` world polyline or ``None``.
    """
    ref = _as_ref(getattr(scene, "lane_ref", None))
    if ref is not None:
        return ref, REF_SENSOR
    envelope = getattr(scene, "lane_envelope", None)
    ref = _as_ref(getattr(envelope, "center", None)
                  if envelope is not None else None)
    if ref is not None:
        return ref, REF_ENVELOPE
    if getattr(scene, "strict_perception", False):
        # FSD mode: no map geometry may become the lane.
        return None, REF_NONE
    ref = _as_ref(getattr(scene, "route", None))
    if ref is not None:
        return ref, REF_ROUTE
    return None, REF_NONE


def perception_reference(scene) -> tuple[np.ndarray | None, str]:
    """Perception-only reference: the route is never returned."""
    ref, src = lateral_reference(scene)
    if src in REF_PERCEPTION:
        return ref, src
    return None, REF_NONE
