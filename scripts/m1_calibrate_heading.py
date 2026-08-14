"""M1 标定：实测 angle_to_quat 的 yaw 与车头朝向（dir）的映射关系。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.misc.quat import angle_to_quat

from beamng_autopilot import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto prefers BeamNG.tech when installed")
    args = ap.parse_args()

    bng = BeamNGpy(config.HOST, config.PORT,
                   home=str(config.runtime_home(args.runtime)),
                   user=str(config.runtime_user(args.runtime)))
    bng.open(launch=True)
    bng.set_steps_per_second(60)

    scenario = Scenario(config.DEFAULT_MAP, "calib_heading")
    yaws = [("c0", 0.0), ("c90", 90.0), ("c180", 180.0), ("cm90", -90.0)]
    for i, (name, yaw) in enumerate(yaws):
        v = Vehicle(name, model=config.DEFAULT_VEHICLE, color="Blue")
        scenario.add_vehicle(
            v,
            pos=(0.0, 0.0 + i * 6.0, 0.0),
            rot_quat=angle_to_quat((0.0, 0.0, yaw)),
            cling=True,
        )
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    bng.step(30)

    for name, yaw in yaws:
        v = scenario.vehicles[name]
        v.sensors.poll()
        st = v.state
        d = st["dir"]
        heading = math.degrees(math.atan2(d[1], d[0]))
        print(f"yaw={yaw:6.0f} deg -> dir=({d[0]:+.3f}, {d[1]:+.3f})  world_heading={heading:+7.1f} deg")

    bng.close()


if __name__ == "__main__":
    main()
