"""The last branch of the speed chain may not raise a hard constraint.

Plan P2.3.  The monitor's verdict, the emergency clearance branch and the
planned corner speed all feed one value; everything after them - the
longitudinal planner, the ramp, the corner governor - shapes HOW the car
gets to a speed.  Shaping is not choosing.
"""

from beamng_autopilot.fsd_drive import final_target_speed


def test_a_shaped_reference_above_the_cap_is_capped(self=None):
    """The bug this closes: the long planner clamped against the plan
    only, so a comfort-shaped 6.0 was published where the monitor had
    said 3.3."""
    assert final_target_speed(6.0, 6.0, 3.3) == 3.3


def test_the_plan_still_binds_when_it_is_lower():
    assert final_target_speed(3.0, 2.0, 5.0) == 2.0


def test_the_lowest_of_the_three_wins():
    assert final_target_speed(9.0, 4.0, 7.0) == 4.0


def test_when_all_agree_the_value_is_unchanged():
    assert final_target_speed(3.3, 3.3, 3.3) == 3.3


def test_a_force_stop_is_zero_not_shaped():
    # Comfort shaping must never delay a safety action.
    assert final_target_speed(6.0, 6.0, 3.3, force_stop=True) == 0.0


def test_an_unreadable_reference_does_not_relax_the_cap():
    # A None or non-numeric reference is not a higher ceiling.
    assert final_target_speed(None, 6.0, 3.3) == 3.3
    assert final_target_speed("nan", 6.0, 3.3) == 3.3


def test_zero_cap_stays_zero():
    assert final_target_speed(6.0, 6.0, 0.0) == 0.0


def test_a_negative_cap_is_not_corrected_upwards():
    # Clamping to 0 here would be a second decision; the caller owns it.
    assert final_target_speed(6.0, 6.0, -1.0) == -1.0
