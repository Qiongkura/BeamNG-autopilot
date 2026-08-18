"""Gearbox helpers for the attached vehicle.

BeamNG's numeric control ``gear`` value is NOT the same on every vehicle:
the project's default car (etk800, automatic) maps 1->P, 2->D, 3->S1 while
a manual gearbox maps 1->1st.  Hardcoding ``gear=1`` (the beamngpy doc
convention) therefore engaged P on the auto box, which is how the old
handover "pinned the gear to 1st" while the car was actually in park - and
could then end up reversing when the arcade box latched R.  These helpers
probe the real gearbox at a standstill and remember the numeric input that
selects a forward gear, so autopilot and handover can pin D/1st instead of
park.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_cache: dict[str, int] = {}
_PROBE_INPUTS = (1, 2, 3, 4, 5, 6)
_FORWARD_PREFIXES = ("1", "2", "3", "4", "5", "6", "7", "8")


def read_gearbox_mode(vehicle) -> str | None:
    """Return the current gearbox mode name or None when unavailable."""
    try:
        resp = vehicle.queue_lua_command(
            "return electrics.values.gearboxMode", response=True)
        if resp is None:
            return None
        if isinstance(resp, dict):
            resp = resp.get("value")
        if isinstance(resp, str):
            # Some BeamNG protocol versions double-encode the Lua return
            # value as a JSON string; unwrap it when that happens.
            try:
                parsed = json.loads(resp)
                if isinstance(parsed, str):
                    resp = parsed
            except (ValueError, TypeError):
                pass
        return str(resp) if resp else None
    except Exception as exc:
        # NOTE: bare except kept — Lua command can fail with any
        # transport error; we return None for graceful degradation.
        logger.warning("[gearbox] read mode failed: %s", exc)
        return None


def set_gearbox_mode(vehicle, mode: str) -> None:
    try:
        vehicle.queue_lua_command(
            f'controller.mainController.setGearboxMode("{mode}")')
    except Exception as exc:
        # NOTE: bare except kept — Lua command can fail with any
        # transport error; we silently ignore mode-set failures.
        logger.warning("[gearbox] set mode failed: %s", exc)


def read_gear(conn) -> str | None:
    """Current gear string ('P','R','N','D','1',...) or None."""
    try:
        resp = conn.vehicle.queue_lua_command(
            "return tostring(electrics.values.gear)", response=True)
        if resp is None:
            return None
        if isinstance(resp, dict):
            resp = resp.get("value")
        if isinstance(resp, str):
            try:
                parsed = json.loads(resp)
                if isinstance(parsed, str):
                    resp = parsed
            except (ValueError, TypeError):
                pass
            if resp:
                return resp.strip()
    except Exception as exc:
        # NOTE: bare except kept — Lua command can fail with any
        # transport error; we return None for graceful degradation.
        logger.warning("[gearbox] read gear failed: %s", exc)
    return None


def engage_neutral(conn) -> bool:
    """Shift the stopped vehicle into N with the parking brake on.

    Returns True when the gearbox reports N afterwards.  This is the safe
    state to hand the car back in: switching from realistic D into arcade
    leaves the arcade box in D and the car creeps forward on its own at
    ~2 m/s, while switching from N keeps it truly parked (verified live).
    Only call this at a standstill with the parking brake engaged - it sends
    ``gear=0``, which forces realistic mode if arcade was active.
    """
    try:
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=1.0, gear=0)
        conn.step(8)
        gear = read_gear(conn)
        if gear == "N":
            return True
        # A manual gearbox may need the clutch to engage neutral.
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=1.0, clutch=1.0, gear=0)
        conn.step(8)
        return read_gear(conn) == "N"
    except Exception as exc:
        # NOTE: bare except kept — any transport or Lua error means
        # we could not confirm neutral; return False.
        logger.warning("[gearbox] engage neutral failed: %s", exc)
        return False


def _signed_speed(conn) -> float:
    try:
        st = conn.get_state()
        return float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
    except Exception:
        # NOTE: bare except kept — get_state can fail with any transport
        # error; default to zero speed so the caller proceeds safely.
        return 0.0


def forward_gear_input(conn, force: bool = False) -> int:
    """Return the numeric control ``gear`` value that selects a forward gear
    for the attached vehicle (2 = D on the default auto, 1 = 1st on a
    manual).

    If the car is moving it is first braked to a standstill in realistic
    mode (where braking never latches reverse), then each candidate input
    is tried at a standstill and the first one that engages a forward gear
    is kept.  The result is cached per vehicle id.  On return the car is
    stopped, in realistic mode, in a forward gear, with the parking brake
    engaged.
    """
    vid = conn.vehicle.vid
    if not force and vid in _cache:
        return _cache[vid]

    if read_gearbox_mode(conn.vehicle) != "realistic":
        set_gearbox_mode(conn.vehicle, "realistic")
        conn.step(5)
    # Brake to a standstill: safe in realistic mode (no reverse latch).
    for _ in range(150):  # up to ~5 s of braking at 60 Hz
        conn.control(throttle=0.0, brake=1.0, steering=0.0)
        conn.step(2)
        if abs(_signed_speed(conn)) < 0.3:
            break
    conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=1.0)
    conn.step(5)

    for g in _PROBE_INPUTS:
        # Shift through neutral first so the previous gearbox state cannot
        # leak into the read-back and make input 1 look like a forward gear.
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=1.0, gear=0)
        conn.step(8)
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=1.0, gear=g)
        conn.step(8)
        gear = read_gear(conn)
        if gear and (gear == "D" or gear.startswith(_FORWARD_PREFIXES)):
            # Confirm the gearbox actually stays in the reported forward
            # gear before caching it (the first probe read is sometimes a
            # stale value from the previous gearbox mode).
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=1.0, gear=g)
            conn.step(8)
            gear2 = read_gear(conn)
            if gear2 == gear:
                _cache[vid] = g
                return g
    # Fall back to beamngpy's documented 1st-gear convention.
    _cache[vid] = 1
    return 1
