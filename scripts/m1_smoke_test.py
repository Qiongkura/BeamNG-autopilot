"""M0 冒烟测试：连接游戏、建场景、直行 3 秒并打印遥测。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto prefers BeamNG.tech when installed")
    args = ap.parse_args()

    with BeamNGConnector(
            args.map, args.vehicle,
            home=config.runtime_home(args.runtime)) as conn:
        conn.load_scenario()
        print("[smoke] 场景已开始，直行测试中...")

        start = time.time()
        while time.time() - start < 3.0:
            st = conn.get_state()
            conn.control(throttle=0.5, steering=0.0)
            conn.step(1)
            print(f"pos=({st.pos[0]:.2f},{st.pos[1]:.2f}) speed={st.speed:.2f} m/s heading={st.heading:.3f}")

        conn.control(throttle=0.0, brake=1.0)
        conn.step(30)
        print("[smoke] 完成")


if __name__ == "__main__":
    main()
