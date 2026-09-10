"""Offline tests for the single lateral lane-reference policy."""

from __future__ import annotations

import numpy as np

from beamng_autopilot.planning import (
    REF_ENVELOPE,
    REF_NONE,
    REF_ROUTE,
    REF_SENSOR,
    Scene,
    lateral_reference,
    perception_reference,
)


def _line(y: float = 0.0) -> np.ndarray:
    xs = np.linspace(0.0, 30.0, 31)
    return np.column_stack([xs, np.full_like(xs, y)])


def _scene(**kwargs) -> Scene:
    return Scene(pos=np.array([0.0, 0.0]), heading=0.0, **kwargs)


def test_sensor_reference_has_priority() -> None:
    scene = _scene(route=_line(0.0), lane_ref=_line(-1.8),
                   lane_envelope=type("Envelope", (), {
                       "center": _line(-2.0)})())
    ref, src = lateral_reference(scene)
    assert src == REF_SENSOR
    assert ref is not None
    assert float(np.median(ref[:, 1])) == -1.8


def test_envelope_is_perception_fallback() -> None:
    scene = _scene(route=_line(0.0),
                   lane_envelope=type("Envelope", (), {
                       "center": _line(-1.8)})())
    ref, src = lateral_reference(scene)
    assert src == REF_ENVELOPE
    assert ref is not None
    assert float(np.median(ref[:, 1])) == -1.8


def test_strict_scene_never_uses_route() -> None:
    scene = _scene(route=_line(0.0), strict_perception=True)
    ref, src = lateral_reference(scene)
    assert ref is None
    assert src == REF_NONE
    assert perception_reference(scene) == (None, REF_NONE)


def test_legacy_scene_can_use_route_fallback() -> None:
    scene = _scene(route=_line(0.0))
    ref, src = lateral_reference(scene)
    assert src == REF_ROUTE
    assert ref is not None


def test_malformed_reference_degrades_to_none() -> None:
    scene = _scene(
        route=_line(0.0),
        lane_ref=np.array([[0.0, np.nan], [1.0, 0.0]]),
        strict_perception=True,
    )
    ref, src = lateral_reference(scene)
    assert ref is None
    assert src == REF_NONE


def test_perception_reference_rejects_route() -> None:
    scene = _scene(route=_line(0.0))
    ref, src = perception_reference(scene)
    assert ref is None
    assert src == REF_NONE
