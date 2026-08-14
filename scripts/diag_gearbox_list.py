# Find the right Lua expression to read the gearbox gear list so we can
# compute the numeric control input for a forward gear generically.

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector


def lua(conn, chunk: str):
    resp = conn.vehicle.queue_lua_command(chunk, response=True)
    if resp is None:
        return None
    if isinstance(resp, str):
        try:
            return json.loads(resp)
        except (ValueError, TypeError):
            return resp
    return resp


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        vid = conn.vehicle.vid
        print(f"[diag] attached vid={vid}")

        candidates = [
            ("be:getVehicleByID", f"local v=be:getVehicleByID('{vid}') "
                                  "if not v then return jsonEncode({err='nil'}) end "
                                  "local gb=v:getGearbox() "
                                  "if not gb then return jsonEncode({err='no gb'}) end "
                                  "return jsonEncode({gears=gb.gears})"),
            ("be.getVehicleByID", f"local v=be.getVehicleByID('{vid}') "
                                  "if not v then return jsonEncode({err='nil'}) end "
                                  "local gb=v:getGearbox() "
                                  "if not gb then return jsonEncode({err='no gb'}) end "
                                  "return jsonEncode({gears=gb.gears})"),
            ("playerVehicle", "local v=be:getPlayerVehicle(0) "
                              "if not v then return jsonEncode({err='nil'}) end "
                              "local gb=v:getGearbox() "
                              "if not gb then return jsonEncode({err='no gb'}) end "
                              "return jsonEncode({gears=gb.gears})"),
            ("gears on controller", "local c=controller "
                                    "local g=c.gears "
                                    "if not g then return jsonEncode({err='no gears'}) end "
                                    "return jsonEncode({gears=g})"),
            ("electrics gears", "local g=electrics.values.gears "
                                "if not g then return jsonEncode({err='no gears'}) end "
                                "return jsonEncode({gears=g})"),
            ("gearboxGearsList", "local f=controller.gearboxGearsList "
                                 "if not f then return jsonEncode({err='no fn'}) end "
                                 "return jsonEncode({gears=f()})"),
        ]
        for label, chunk in candidates:
            try:
                print(f"  {label}: {lua(conn, chunk)}")
            except Exception as exc:
                print(f"  {label}: error {exc}")
    finally:
        try:
            lua(conn, 'controller.mainController.setGearboxMode("arcade")')
            conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=1.0)
            conn.step(20)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
