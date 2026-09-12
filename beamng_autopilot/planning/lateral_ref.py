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


# --- bounded tick-to-tick slew of the accepted reference --------------------
# The own-lane reference must not teleport sideways between ticks.  On the
# 2026-09-11 town runs (dashed recovery on) the accepted sensor reference's
# near-field lateral offset jumped up to 2.20 m between consecutive ticks
# (p50 0.23, p90 0.38, 5.3% of ticks > 1.0 m) while the envelope reference
# moved at most 0.41 m - the SELECTION was unstable, not the car, and the
# control loop followed the step into off-road frames.  A disagreement
# larger than ``LANE_REF_SLEW_MAX_M`` keeps the previous reference, but
# only for ``LANE_REF_SLEW_HOLD_MAX_S``: a persistent disagreement is a
# real change (junction, new lane) and must not be blocked forever.
LANE_REF_SLEW_MAX_M = 0.8
LANE_REF_SLEW_HOLD_MAX_S = 2.0


def near_lat(ref, pos, heading: float, within_m: float = 20.0
             ) -> float | None:
    """Mean lateral offset (left = +) of ``ref`` beside the ego, or None.

    Measured at the CURRENT pose, so normal forward motion does not read
    as a lateral step when two references are compared.
    """
    arr = _as_ref(ref)
    if arr is None:
        return None
    p = np.asarray(pos, dtype=float)[:2]
    if p.size < 2 or not np.isfinite(p).all():
        return None
    rel = arr - p
    near = arr[np.linalg.norm(rel, axis=1) <= float(within_m)]
    if len(near) == 0:
        return None
    fwd = np.array([np.cos(float(heading)), np.sin(float(heading))])
    left = np.array([-fwd[1], fwd[0]])
    return float(((near - p) @ left).mean())


def limit_reference_slew(prev, prev_hold_t: float, new, now: float,
                         pos, heading: float,
                         max_jump_m: float = LANE_REF_SLEW_MAX_M,
                         hold_max_s: float = LANE_REF_SLEW_HOLD_MAX_S
                         ) -> tuple[np.ndarray | None, float]:
    """Return ``(reference_to_publish, hold_started_at)``.

    ``prev`` is the previously published reference and ``prev_hold_t`` the
    time the current hold started (0.0 when not holding).  A new reference
    that disagrees laterally with the previous one by more than
    ``max_jump_m`` at the current pose is rejected in favour of ``prev``
    until ``hold_max_s`` has elapsed, after which the new one is accepted
    (a real change).  ``prev`` None (first tick / after a reset) accepts.
    """
    new_ref = _as_ref(new)
    if new_ref is None:
        return None, 0.0
    prev_ref = _as_ref(prev)
    if prev_ref is None:
        return new_ref, 0.0
    jump = near_lat(new_ref, pos, heading), near_lat(prev_ref, pos, heading)
    if jump[0] is None or jump[1] is None:
        return new_ref, 0.0
    if abs(jump[0] - jump[1]) <= float(max_jump_m):
        return new_ref, 0.0
    if prev_hold_t and (float(now) - float(prev_hold_t)) <= float(hold_max_s):
        return prev_ref, float(prev_hold_t)
    if not prev_hold_t:
        return prev_ref, float(now)
    return new_ref, 0.0

