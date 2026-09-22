"""P0-2: the lateral field contract is complete, documented and honest.

The review handoff asks for one thing above all: from a single frame of
telemetry plus the contract, a reader must be able to tell which side of
which line the car is on - and a missing measurement must never look like
a healthy zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from beamng_autopilot.telemetry_contract import (
    LATERAL_FIELD_SPECS,
    lateral_digest,
)

DOC = Path(__file__).resolve().parents[1] / "docs" / "LATERAL_FIELD_CONTRACT.md"
REQUIRED_KEYS = ("reference", "frame", "sign", "unit", "coverage", "zero",
                 "none")


def test_every_field_declares_the_full_contract():
    for name, spec in LATERAL_FIELD_SPECS.items():
        for key in REQUIRED_KEYS:
            assert spec.get(key), f"{name} is missing '{key}'"


def test_side_carrying_fields_state_a_sign_direction():
    """A field that can say which side must name the positive direction."""
    for name in ("line_lat", "ego_lat_route_m", "lat_route_m", "lat_car_m",
                 "lane_side_off_m", "lat_left", "lat_right",
                 "body_lat_left", "body_lat_right"):
        assert "+" in LATERAL_FIELD_SPECS[name]["sign"] or \
            "-" in LATERAL_FIELD_SPECS[name]["sign"], name


def test_the_unsigned_field_says_so():
    assert "UNSIGNED" in LATERAL_FIELD_SPECS["lane_dev_m"]["sign"]


def test_fields_that_are_unmeasurable_say_what_none_means():
    """lat_left/right and lane_dev_m are the ones that were read as 0."""
    for name in ("lat_left", "lat_right", "body_lat_left", "body_lat_right",
                 "lane_dev_m", "line_lat"):
        text = LATERAL_FIELD_SPECS[name]["none"] + \
            LATERAL_FIELD_SPECS[name]["zero"]
        assert ("UNKNOWN" in text or "NOT" in text or "not " in text), name


def test_the_doc_names_every_field():
    text = DOC.read_text(encoding="utf-8")
    for name in LATERAL_FIELD_SPECS:
        assert name in text, f"{name} missing from LATERAL_FIELD_CONTRACT.md"
    # and the doc must state the format it follows
    assert "reference + frame + sign + unit + coverage" in text


def test_the_doc_explains_the_two_similar_looking_numbers():
    """The +0.91 m / -0.368 m pair that looked contradictory."""
    text = DOC.read_text(encoding="utf-8")
    assert "+0.91" in text and "0.368" in text
    assert "不同帧" in text


def test_digest_reports_a_missing_column_as_unknown_not_zero():
    hist = [{"t": 0.0, "line_lat": 1.0}, {"t": 1.0}]
    d = lateral_digest(hist)
    assert d["fields"]["line_lat"]["status"] == "measured"
    assert d["fields"]["line_lat"]["measured_frames"] == 1
    assert d["fields"]["line_lat"]["missing_frames"] == 1
    lat_left = d["fields"]["lat_left"]
    assert lat_left["status"] == "UNKNOWN"
    assert lat_left["median"] is None and lat_left["min"] is None
    assert lat_left["measured_frames"] == 0
    assert lat_left["spec"] is LATERAL_FIELD_SPECS["lat_left"]


def test_digest_ignores_booleans_and_non_finite_values():
    hist = [{"lat_left": True}, {"lat_left": float("nan")},
            {"lat_left": float("inf")}, {"lat_left": 0.4}]
    d = lateral_digest(hist)["fields"]["lat_left"]
    assert d["measured_frames"] == 1
    assert d["median"] == pytest.approx(0.4)


class _Scene:
    """Minimal strict scene with no perception lane reference."""

    def __init__(self):
        from beamng_autopilot.occupancy import OccupancyGrid
        from beamng_autopilot.planning import Scene

        self.scene = Scene(pos=np.zeros(3), heading=0.0,
                           grid=OccupancyGrid(60, 60, 0.5),
                           route=None, lane_ref=None, strict_perception=True)


def test_lane_dev_is_none_when_there_is_no_reference():
    """It used to return 0.0, which read as "perfectly aligned"."""
    from beamng_autopilot.safety_monitor import SafetyMonitor

    path = np.column_stack([np.linspace(0.0, 20.0, 21), np.zeros(21)])
    mon = SafetyMonitor(max_speed=6.0)
    dev, src = mon._lane_deviation(_Scene().scene, path)
    assert dev is None
    assert src == "none"
    v = mon.evaluate(_Scene().scene, path)
    assert v.lane_dev_m is None


def test_lane_dev_none_does_not_trigger_the_lane_rules():
    """Behaviour is unchanged: no reference already fails closed earlier."""
    from beamng_autopilot.safety_monitor import SafetyMonitor

    path = np.column_stack([np.linspace(0.0, 20.0, 21), np.zeros(21)])
    v = SafetyMonitor(max_speed=6.0).evaluate(_Scene().scene, path)
    assert v.effective_rule != "path_off_lane"
    assert v.effective_rule != "path_near_lane_edge"
