"""Offline tests for the FSD Routing-intent layer (pure logic)."""

from __future__ import annotations

import numpy as np

from beamng_autopilot.planning.intent import (
    INTENT_LEFT,
    INTENT_RIGHT,
    INTENT_STRAIGHT,
    INTENT_U_TURN,
    INTENT_UNKNOWN,
    infer_route_intent,
)


def test_straight_route_is_straight() -> None:
    route = np.column_stack([np.linspace(0.0, 40.0, 41),
                             np.zeros(41)])
    intent = infer_route_intent(route, np.array([0.0, 0.0]), 0.0)
    assert intent.label == INTENT_STRAIGHT
    assert abs(intent.turn_deg) < 25.0
    # straight roads stay at the cruise speed
    assert intent.suggested_speed >= 12.0


def test_right_turn_classified_with_sign() -> None:
    # heading east, route bends south-east: negative turn -> right
    route = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0],
                      [10.0, 0.0], [14.0, 0.0], [20.0, 0.0],
                      [30.0, -10.0], [40.0, -26.0]])
    intent = infer_route_intent(route, np.array([0.0, 0.0]), 0.0)
    assert intent.label == INTENT_RIGHT
    assert intent.turn_deg < -25.0
    assert intent.is_turn


def test_left_turn_classified_with_sign() -> None:
    # heading east, route bends north-east: positive turn -> left
    route = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0],
                      [10.0, 0.0], [14.0, 0.0], [20.0, 0.0],
                      [30.0, 10.0], [40.0, 26.0]])
    intent = infer_route_intent(route, np.array([0.0, 0.0]), 0.0)
    assert intent.label == INTENT_LEFT
    assert intent.turn_deg > 25.0


def test_u_turn_classified() -> None:
    # the route doubles back behind the ego: net displacement over the
    # 35 m lookahead points backward -> |turn| >= 140 deg
    route = np.array([[0.0, 0.0], [5.0, 0.0], [3.0, -2.0],
                      [-2.0, -4.0], [-8.0, -3.0], [-14.0, 1.0],
                      [-20.0, 6.0], [-26.0, 12.0]])
    intent = infer_route_intent(route, np.array([0.0, 0.0]), 0.0)
    assert intent.label == INTENT_U_TURN
    assert intent.suggested_speed <= 5.0


def test_short_route_is_unknown() -> None:
    route = np.array([[0.0, 0.0], [4.0, 0.0]])
    intent = infer_route_intent(route, np.array([0.0, 0.0]), 0.0)
    assert intent.label == INTENT_UNKNOWN


def test_route_ahead_slowdown_for_turn() -> None:
    # a sharp 90-degree right bend ahead must lower the suggested speed
    route = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0],
                      [10.0, 0.0], [13.0, 0.0], [17.0, 0.0],
                      [20.0, 0.0], [20.0, -10.0], [20.0, -20.0]])
    intent = infer_route_intent(route, np.array([0.0, 0.0]), 0.0)
    assert intent.is_turn
    assert intent.suggested_speed < 12.0
