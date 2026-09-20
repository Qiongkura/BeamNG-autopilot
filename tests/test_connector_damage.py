"""Connector damage readout (the honest source for collision_count).

``read_damage_total`` is a thin wrapper over
:func:`beamng_autopilot.damage.damage_total`, but the WRAPPER is where the
honesty lives: it must return None - never 0.0 - when there is no vehicle,
no damage sensor, or a payload it cannot read, because a missing
measurement that reports zero reads as "no collisions" and turns the
plan's first hard gate into a false PASS.  These tests pin that, with a
fake vehicle, offline.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from beamng_autopilot.connector import BeamNGConnector


def _conn(sensors=None, vehicle=None):
    """A connector stub with the attributes read_damage_total touches."""
    c = BeamNGConnector.__new__(BeamNGConnector)
    c.io_lock = threading.RLock()
    if vehicle is not None:
        c.vehicle = vehicle
    else:
        c.vehicle = SimpleNamespace(sensors=sensors) if sensors is not None \
            else None
    return c


def test_reads_the_sensor_payload() -> None:
    c = _conn({"damage": SimpleNamespace(data={"damage": 0.25})})
    assert c.read_damage_total() == 0.25


def test_reads_a_part_table_payload() -> None:
    c = _conn({"damage": SimpleNamespace(
        data={"part_damage": {"hood": 0.1, "door": 0.2}})})
    assert c.read_damage_total() == 0.30000000000000004 or \
        abs(c.read_damage_total() - 0.3) < 1e-9


def test_missing_vehicle_is_none_not_zero() -> None:
    assert _conn().read_damage_total() is None


def test_missing_sensor_is_none_not_zero() -> None:
    """Steam, or a Tech session that never attached one: UNMEASURED."""
    assert _conn({}).read_damage_total() is None
    assert _conn({"camera": SimpleNamespace(data={})}).read_damage_total() \
        is None


def test_unreadable_payload_is_none_not_zero() -> None:
    assert _conn({"damage": SimpleNamespace(data=None)}).read_damage_total() \
        is None
    assert _conn({"damage": SimpleNamespace(data={})}).read_damage_total() \
        is None
    assert _conn({"damage": SimpleNamespace(data={"state": "ok"})}) \
        .read_damage_total() is None


def test_a_broken_sensor_object_does_not_raise() -> None:
    class _Boom:
        @property
        def data(self):
            raise RuntimeError("sensor died")

    assert _conn({"damage": _Boom()}).read_damage_total() is None
    assert _conn({"damage": "not a sensor"}).read_damage_total() is None


def test_zero_damage_is_reported_as_zero() -> None:
    """A measured zero IS a clean car - that must not become None."""
    c = _conn({"damage": SimpleNamespace(data={"damage": 0.0})})
    assert c.read_damage_total() == 0.0
