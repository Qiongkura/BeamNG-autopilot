"""Vehicle damage readout - the honest source for ``collision_count``.

The plan's §12 hard gate #1 is ``collision_count = 0``, and until now the
benchmark had no way to measure it: crossings and off-road frames are
proxies for "the car went somewhere it should not", not for "the car hit
something".  BeamNG.tech's ``Damage`` sensor is the ground truth for that
(beamngpy's own docs call it a perfect-knowledge readout of how deformed
the body is), so this module turns its payload into ONE number per frame,
which the drive loop logs and :func:`beamng_autopilot.eval.collision_events`
turns into events.

The one rule that matters: a shape we do not recognise yields ``None`` -
"not measured" - never ``0.0``.  A collision metric that silently reports
zero when the sensor is missing is worse than no metric at all, because
it reads as a PASS.

Pure logic: dicts in, ``float | None`` out.
"""

from __future__ import annotations

import math

# Keys that, when present, ARE the total (the sensor's own summary), in
# order of preference.  Only read at the TOP level: a per-part dict may
# legitimately contain a key called "damage" for one part, and summing
# per-part damage must not be confused with a total.
TOTAL_KEYS = ("damage", "total", "total_damage", "damage_total")
# Keys holding per-part numbers to sum when no top-level total exists.
PART_KEYS = ("part_damage", "parts", "partDamage")


def _finite(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sum_numbers(node, depth: int = 0) -> float | None:
    """Sum every finite number in a nested dict/list (None when none)."""
    if depth > 6:
        return None
    if isinstance(node, dict):
        vals = [_sum_numbers(v, depth + 1) for v in node.values()]
    elif isinstance(node, (list, tuple)):
        vals = [_sum_numbers(v, depth + 1) for v in node]
    else:
        return _finite(node)
    known = [v for v in vals if v is not None]
    return float(sum(known)) if known else None


def damage_total(payload) -> float | None:
    """One damage number from a ``Damage`` sensor payload, or None.

    Accepts the sensor's dict shape (a top-level total, a per-part table,
    or any nesting of numbers), a bare number, or ``None``.  Returns None
    whenever nothing numeric can be read - never a default of zero.
    """
    if payload is None:
        return None
    direct = _finite(payload)
    if direct is not None:
        return direct
    if isinstance(payload, dict):
        for key in TOTAL_KEYS:
            if key in payload:
                v = _finite(payload[key])
                if v is not None:
                    return v
        for key in PART_KEYS:
            if key in payload:
                v = _sum_numbers(payload[key])
                if v is not None:
                    return v
        # fall back to summing whatever numeric leaves exist; an
        # unrecognised shape yields None, not 0.0
        return _sum_numbers(payload)
    if isinstance(payload, (list, tuple)):
        return _sum_numbers(payload)
    return None
