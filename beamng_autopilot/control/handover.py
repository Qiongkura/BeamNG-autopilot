"""Hand the vehicle back to the player without a stuck reverse gear.

Root causes, verified live on the default etk800:

1. BeamNG's arcade gearbox treats "brake while (nearly) stopped" as a
   reverse request: it latches R and then drives the car backward by
   itself, even after the brake is released.  Every braking manoeuvre in a
   handover must therefore happen in realistic mode, and arcade must never
   see a brake/gear input from us.

2. The numeric control ``gear`` value is vehicle-dependent.  gear=1
   engages P (park) on the default automatic, not 1st gear, so the old
   handover "pinned the gear to 1st" while the car was actually sitting in
   park - and the arcade box could then latch R when the car was released.
   All gear pinning now uses forward_gear_input(), which probes the real
   gearbox and returns the input that selects D/1st.

3. Handing a rolling car back into arcade leaves the arcade box in D and
   the car keeps driving itself: it first coasts down, then creeps forward
   on its own at ~2 m/s forever (engine idle in D).  A handover must
   therefore always end with the car fully stopped in N, never rolling.
   The old "fast path" (zero the pedals and hand back while the car is
   rolling faster than 2 m/s) triggered exactly this - the car never came
   to a stop, and once the player braked the arcade box latched R and the
   car reversed on its own.  This is the "the car drives / reverses by
   itself after autopilot" bug the user kept reporting.

Handover sequence (used for every exit, at any speed):
1. force realistic mode (if not already) - braking in realistic never
   latches R;
2. brake to a full standstill (explicit loop, regardless of how fast the
   car is approaching, so a cached forward gear never skips the stop);
3. probe and engage a forward gear (D/1st) with the parking brake on;
4. shift to N while stopped (realistic D handed into arcade would leave the
   arcade box in D and the car would creep forward on its own at ~2 m/s -
   verified live; N keeps it truly parked);
5. switch back to the player's gearbox mode while fully stopped with the
   parking brake engaged and zero pedals - verified live that arcade then
   sits in N and the car does not move;
6. hand over with all pedals zeroed and the parking brake holding.

The same code is used by the autopilot script and by the automated
obstacle / handover tests, so the tests exercise exactly what ships.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

from .gearbox import (
    engage_neutral,
    forward_gear_input,
    read_gearbox_mode,
    read_gear,
    set_gearbox_mode,
)


def signed_speed(st) -> float:
    """Longitudinal speed: positive = moving in the car's forward direction."""
    if st is None:
        return 0.0
    return float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])


def _parking_brake(conn) -> str | None:
    """Read the current parking-brake electric value as '0'/'1'/'nil'."""
    try:
        resp = conn.vehicle.queue_lua_command(
            "local pb = electrics.values.parkingbrake\n"
            "if pb == nil then return 'nil' end\n"
            "return tostring(pb)", response=True)
        if isinstance(resp, dict):
            resp = resp.get("value")
        if isinstance(resp, str):
            try:
                parsed = json.loads(resp)
                if isinstance(parsed, str):
                    resp = parsed
            except (ValueError, TypeError):
                pass
        return str(resp) if resp is not None else None
    except Exception as exc:
        # NOTE: bare except kept — Lua command can fail with any
        # transport error; we return None for graceful degradation.
        logger.warning("[handover] parking brake read failed: %s", exc)
        return None


def _handover_state(conn) -> dict:
    """Final gearbox snapshot used to verify a parked handover."""
    return {
        "gear": read_gear(conn),
        "mode": read_gearbox_mode(conn.vehicle),
        "parkingbrake": _parking_brake(conn),
    }


def _brake_to_standstill(conn, log=print) -> bool:
    """Brake the vehicle to a full standstill in realistic mode.

    Returns True when the car reached (nearly) zero signed speed.  The
    loop is independent of the forward-gear cache, so it also stops a car
    that is rolling fast on a slope or still coasting from autopilot.
    """
    for _ in range(300):  # up to ~10 s of braking at 60 Hz
        try:
            conn.control(throttle=0.0, brake=1.0, steering=0.0)
            conn.step(2)
            if abs(signed_speed(conn.get_state())) < 0.25:
                return True
        except Exception as exc:
            # NOTE: bare except kept — transport error during braking is
            # unrecoverable in this loop; return False immediately.
            logger.warning("[handover] brake loop error: %s", exc)
            return False
    return False


def handover_vehicle(conn, saved_gearbox: str | None,
                     gearbox_switched: bool, log=print) -> None:
    """Release control and restore the player's gearbox without a stuck R.

    ``conn`` is a BeamNGConnector, ``saved_gearbox`` the mode that was active
    before autopilot (``None`` when unknown) and ``gearbox_switched`` tells
    whether autopilot actually changed the mode (only then is it restored).
    The car is always braked to a full standstill in realistic mode, parked
    in N with the parking brake, and only then handed back - so it can
    never keep creeping forward (~2 m/s arcade D idle) or latch R and
    reverse on its own.
    """
    try:
        # Decide the mode to hand back in.  When autopilot switched the
        # box, restore what it was; otherwise restore whatever was active
        # (unless it is already realistic).
        current = read_gearbox_mode(conn.vehicle)
        if gearbox_switched and saved_gearbox:
            target = saved_gearbox
        elif current and current != "realistic":
            target = current
        else:
            target = None
        # Force realistic before any braking: in arcade, brake-at-stop is a
        # reverse request and drives the car backward by itself.
        if current != "realistic":
            set_gearbox_mode(conn.vehicle, "realistic")
            conn.step(5)
        # 1-3. Brake to a full standstill in realistic mode - where braking
        #      can never latch R - then probe and engage a forward gear
        #      (D/1st) with the parking brake on.
        if not _brake_to_standstill(conn, log=log):
            log("[handover] WARNING: could not confirm standstill - "
                "continuing with the forward-gear probe")
        fwd_gear = forward_gear_input(conn)
        log(f"[handover] forward gear input = {fwd_gear} (realistic, "
            f"stopped, parking brake on)")
        # 4. Shift to N before handing back: realistic D switched into
        #    arcade leaves the arcade box in D and the car creeps forward on
        #    its own (~2 m/s).  N + parking brake keeps it truly parked.
        if engage_neutral(conn):
            log("[handover] shifted to N at standstill (parking brake on)")
        else:
            log("[handover] WARNING: could not confirm N - handing back "
                "from the probed forward gear")
        # 5. Restore the player's gearbox while fully stopped with the
        #    parking brake engaged and zero pedals.  Verified live that the
        #    arcade box then sits in N and the car does not move; it only
        #    engages D when the player presses throttle, and never R.
        #
        #    IMPORTANT: never send a `gear` value after this point.  A gear
        #    input forces the gearbox back into realistic mode (beamngpy
        #    behavior observed live), which would silently undo the restore
        #    and re-open the stuck-R path.
        if target:
            set_gearbox_mode(conn.vehicle, target)
            conn.step(5)
            log(f"[handover] gearbox restored to {target} at standstill")
        # Verify the restored box did not latch R (or creep D in arcade)
        # while the parking brake was re-asserting.  If it did, re-park
        # through realistic neutral so the player never gets a stuck R.
        state = _handover_state(conn)
        bad_state = state["gear"] == "R"
        if (target and target.lower() == "arcade"
                and state["gear"] not in (None, "N", "P")):
            bad_state = True
        if bad_state:
            log(f"[handover] WARNING: final gearbox state {state}; "
                "re-parking through realistic neutral")
            set_gearbox_mode(conn.vehicle, "realistic")
            conn.step(5)
            if engage_neutral(conn):
                if target:
                    set_gearbox_mode(conn.vehicle, target)
                    conn.step(5)
                state = _handover_state(conn)
        # 6. Hand over with all pedals zeroed; the parking brake holds.
        conn.control(throttle=0.0, brake=0.0, steering=0.0)
        conn.step(5)
        log(f"[handover] car parked (gear={state['gear']}, "
            f"mode={state['mode']}, "
            f"parkingbrake={state['parkingbrake']}, pedals zeroed)")
    except Exception as exc:
        # NOTE: bare except kept — any error during the handover sequence
        # must not crash; we try to park the car and return.
        logger.warning("[handover] failed: %s", exc)
        try:
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=1.0)
            conn.step(3)
        except Exception:
            # NOTE: bare except kept — best-effort parking attempt
            # during error recovery; nothing more we can do.
            pass
