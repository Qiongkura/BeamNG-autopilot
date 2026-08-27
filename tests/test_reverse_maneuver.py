"""Offline tests for the controlled reverse-escape state machine."""

from __future__ import annotations

import pytest

from beamng_autopilot.control.reverse_maneuver import (
    ReverseCommand,
    ReverseManeuver,
)


def _run(rm: ReverseManeuver, seq, dt: float = 0.5):
    """Feed a sequence of (has_fwd, rear_clear, signed) ticks; return cmds."""
    out = []
    pos = (0.0, 0.0)
    for has_fwd, rear, signed in seq:
        cmd = rm.decide(has_fwd, rear, signed, pos, dt=dt)
        out.append(cmd)
    return out


def test_forward_path_never_reverses() -> None:
    rm = ReverseManeuver()
    cmds = _run(rm, [(True, 10.0, 1.0)] * 10)
    assert all(not c.active for c in cmds)
    assert rm.state == "idle"


def test_arms_after_stall_with_clear_rear() -> None:
    rm = ReverseManeuver(stall_s=1.0)
    cmds = _run(rm, [(False, 6.0, 0.0), (False, 6.0, 0.0)])
    assert cmds[-1].active
    assert cmds[-1].gear == rm.rev_gear
    assert cmds[-1].target_speed_mps < 0.0
    assert rm.state == "reversing"


def test_does_not_arm_while_still_rolling() -> None:
    rm = ReverseManeuver(stall_s=0.2)
    cmds = _run(rm, [(False, 6.0, 1.2), (False, 6.0, 1.2)])
    assert not any(c.active for c in cmds)
    assert rm.state == "idle"


def test_does_not_arm_without_rear_space() -> None:
    rm = ReverseManeuver(stall_s=0.2, min_rear_clear_m=3.5)
    cmds = _run(rm, [(False, 2.0, 0.0), (False, 2.0, 0.0)])
    assert not any(c.active for c in cmds)


def test_does_not_arm_blind_without_rear_data() -> None:
    rm = ReverseManeuver(stall_s=0.2)
    cmds = _run(rm, [(False, None, 0.0), (False, None, 0.0)])
    assert not any(c.active for c in cmds)


def test_stops_at_distance_budget() -> None:
    rm = ReverseManeuver(stall_s=0.1, max_reverse_m=2.0, max_reverse_s=10.0)
    # arm (pos stays at start), then simulate moving back 1 m per tick
    c0 = rm.decide(False, 8.0, 0.0, (0.0, 0.0), dt=0.5)
    assert c0.active
    c1 = rm.decide(False, 8.0, -1.0, (0.0, -1.0), dt=0.5)
    assert c1.active
    c2 = rm.decide(False, 8.0, -1.0, (0.0, -2.1), dt=0.5)
    assert not c2.active
    assert rm.state == "paused"


def test_aborts_when_rear_space_shrinks() -> None:
    rm = ReverseManeuver(stall_s=0.1, min_rear_clear_m=3.5)
    c0 = rm.decide(False, 6.0, 0.0, (0.0, 0.0), dt=0.5)
    assert c0.active
    c1 = rm.decide(False, 2.0, -1.0, (0.0, -0.5), dt=0.5)
    assert not c1.active
    assert rm.state == "paused"


def test_returns_to_forward_when_path_reappears() -> None:
    rm = ReverseManeuver(stall_s=0.1, max_reverse_m=10.0, max_reverse_s=10.0)
    c0 = rm.decide(False, 8.0, 0.0, (0.0, 0.0), dt=0.5)
    assert c0.active
    c1 = rm.decide(True, 8.0, -0.05, (0.0, -0.3), dt=0.5)
    assert not c1.active
    assert rm.state == "paused"
    c2 = rm.decide(True, 8.0, 1.0, (0.0, -0.3), dt=0.5)
    assert not c2.active
    assert rm.state == "idle"


def test_retries_then_gives_up() -> None:
    rm = ReverseManeuver(stall_s=0.1, pause_s=0.2, max_reverse_m=1.0,
                         max_reverse_s=1.0, max_attempts=2)
    # attempt 1: arm -> budget done -> paused
    c0 = rm.decide(False, 8.0, 0.0, (0.0, 0.0), dt=0.5)
    assert c0.active
    c1 = rm.decide(False, 8.0, -1.0, (0.0, -1.5), dt=0.5)
    assert not c1.active and rm.state == "paused"
    # paused -> retry (attempt 2) -> budget done -> paused
    c2 = rm.decide(False, 8.0, 0.0, (0.0, -1.5), dt=0.5)
    assert c2.active
    c3 = rm.decide(False, 8.0, -1.0, (0.0, -3.0), dt=0.5)
    assert not c3.active and rm.state == "paused"
    # paused -> attempts exhausted -> give_up
    c4 = rm.decide(False, 8.0, 0.0, (0.0, -3.0), dt=0.5)
    assert not c4.active and rm.state == "give_up"
    c5 = rm.decide(False, 8.0, 0.0, (0.0, -3.0), dt=0.5)
    assert not c5.active and rm.state == "give_up"


def test_reset() -> None:
    rm = ReverseManeuver(stall_s=0.1)
    rm.decide(False, 8.0, 0.0, (0.0, 0.0), dt=0.5)
    assert rm.state == "reversing"
    rm.reset()
    assert rm.state == "idle"
    assert rm._attempts == 0


def test_runaway_backward_speed_gives_up() -> None:
    """A reverse attempt that runs away (steep downhill) must terminate."""
    rm = ReverseManeuver(stall_s=0.1, runaway_mps=-2.0)
    c0 = rm.decide(False, 8.0, 0.0, (0.0, 0.0), dt=0.5)
    assert c0.active and rm.state == "reversing"
    # Car accelerates far beyond the reverse target: give up for good.
    c1 = rm.decide(False, 8.0, -4.0, (0.0, -1.0), dt=0.5)
    assert not c1.active
    assert c1.reason == "runaway"
    assert rm.state == "give_up"
    # No retry even after a stall pause.
    c2 = rm.decide(False, 8.0, 0.0, (0.0, -1.0), dt=0.5)
    assert not c2.active and rm.state == "give_up"
def test_give_up_recovers_when_forward_path_reappears() -> None:
    """A terminal give-up must not lock the car forever: once a forward
    path exists again the state machine returns to normal driving."""
    rm = ReverseManeuver(stall_s=0.1, pause_s=0.2, max_reverse_m=1.0,
                         max_reverse_s=1.0, max_attempts=1)
    # stall -> arm -> budget done -> paused -> attempts exhausted -> give_up
    rm.decide(False, 8.0, 0.0, (0.0, 0.0), dt=0.5)
    rm.decide(False, 8.0, -1.0, (0.0, -1.5), dt=0.5)
    rm.decide(False, 8.0, 0.0, (0.0, -1.5), dt=0.5)
    c = rm.decide(False, 8.0, 0.0, (0.0, -1.5), dt=0.5)
    assert rm.state == "give_up"
    c2 = rm.decide(True, 8.0, 0.5, (0.0, -1.5), dt=0.5)
    assert not c2.active
    assert rm.state == "idle"


def test_runaway_recovers_when_forward_path_reappears() -> None:
    """Even after a runaway give-up, a fresh forward path restores normal
    driving (the caller re-verifies it and shifts back to D)."""
    rm = ReverseManeuver(stall_s=0.1, runaway_mps=-2.0)
    rm.decide(False, 8.0, 0.0, (0.0, 0.0), dt=0.5)
    c = rm.decide(False, 8.0, -4.0, (0.0, -1.0), dt=0.5)
    assert rm.state == "give_up" and c.reason == "runaway"
    c2 = rm.decide(True, 8.0, 1.0, (0.0, -1.0), dt=0.5)
    assert not c2.active and rm.state == "idle"


def test_non_runaway_give_up_retries_after_cooldown() -> None:
    """A plain (non-runaway) give-up re-arms after a cooldown at
    standstill with clear rear space - the parked-forever failure."""
    rm = ReverseManeuver(stall_s=0.1, pause_s=0.2, max_reverse_m=1.0,
                         max_reverse_s=1.0, max_attempts=1,
                         retry_cooldown_s=2.0)
    rm.decide(False, 8.0, 0.0, (0.0, 0.0), dt=0.5)
    rm.decide(False, 8.0, -1.0, (0.0, -1.5), dt=0.5)
    rm.decide(False, 8.0, 0.0, (0.0, -1.5), dt=0.5)
    assert rm.state == "give_up"
    # not enough cooldown yet
    c = rm.decide(False, 8.0, 0.0, (0.0, -1.5), dt=0.5)
    assert not c.active and rm.state == "give_up"
    # cooldown elapsed -> re-arm a new bounded attempt
    c2 = rm.decide(False, 8.0, 0.0, (0.0, -1.5), dt=2.0)
    assert c2.active and rm.state == "reversing"
    assert c2.reason == "cooldown_retry"


def test_runaway_give_up_never_retries_by_cooldown() -> None:
    """A runaway is terminal: the cooldown re-arm must not fire."""
    rm = ReverseManeuver(stall_s=0.1, runaway_mps=-2.0,
                         retry_cooldown_s=1.0)
    rm.decide(False, 8.0, 0.0, (0.0, 0.0), dt=0.5)
    c = rm.decide(False, 8.0, -4.0, (0.0, -1.0), dt=0.5)
    assert rm.state == "give_up" and c.reason == "runaway"
    for _ in range(5):
        c = rm.decide(False, 8.0, 0.0, (0.0, -1.0), dt=1.0)
        assert not c.active and rm.state == "give_up"
