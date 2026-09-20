"""Vehicle damage readout (plan §12 gate #1: collision_count = 0)."""

from __future__ import annotations

import pytest

from beamng_autopilot.damage import damage_total


def test_bare_number_passes_through() -> None:
    assert damage_total(0.25) == pytest.approx(0.25)
    assert damage_total("0.5") == pytest.approx(0.5)


def test_top_level_total_wins_over_parts() -> None:
    """The sensor's own total must not be re-derived from parts."""
    payload = {"damage": 0.8, "part_damage": {"hood": 0.1, "door": 0.1}}
    assert damage_total(payload) == pytest.approx(0.8)
    for key in ("total", "total_damage", "damage_total"):
        assert damage_total({key: 0.4, "part_damage": {"a": 9.0}}) == \
            pytest.approx(0.4)


def test_part_table_is_summed_when_no_total_exists() -> None:
    payload = {"part_damage": {"hood": 0.1, "door_left": 0.2}}
    assert damage_total(payload) == pytest.approx(0.3)


def test_nested_shapes_are_summed() -> None:
    payload = {"parts": {"front": {"hood": 0.1, "bumper": 0.15},
                         "rear": {"trunk": 0.05}}}
    assert damage_total(payload) == pytest.approx(0.3)
    assert damage_total([0.1, [0.2, 0.3]]) == pytest.approx(0.6)


def test_unknown_shape_is_none_not_zero() -> None:
    """"Not measured" must never read as "no collisions"."""
    assert damage_total(None) is None
    assert damage_total({}) is None
    assert damage_total({"state": "pristine"}) is None
    assert damage_total({"part_damage": {}}) is None
    assert damage_total("not a number") is None
    assert damage_total(float("nan")) is None
    assert damage_total({"damage": None}) is None


def test_one_bad_leaf_does_not_hide_the_others() -> None:
    """A partly unreadable payload still reports what it does know."""
    payload = {"part_damage": {"hood": 0.4, "door": None, "roof": "junk"}}
    assert damage_total(payload) == pytest.approx(0.4)


def test_events_helper_and_this_agree_on_an_honest_run() -> None:
    """Round trip: sensor payload -> logged value -> counted events."""
    from beamng_autopilot.eval import collision_events
    payloads = [{"damage": 0.0}, {"damage": 0.0},
                {"damage": 0.12}, {"damage": 0.12}]
    hist = [{"t": float(i), "damage_total": damage_total(p)}
            for i, p in enumerate(payloads)]
    rep = collision_events(hist)
    assert rep["collision_count"] == 1
    assert rep["first_collision_t"] == pytest.approx(2.0)
    # and a run with no damage samples at all stays UNMEASURED
    assert collision_events([{"t": 0.0}])["collision_count"] is None
