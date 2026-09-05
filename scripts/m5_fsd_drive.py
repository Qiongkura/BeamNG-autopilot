"""FSD-mode live driving: FSDStack planning -> safety monitor -> control.

This is the optional *real-driving* path of the FSD-style stack: instead
of only recording shadow data, it drives the car with the layered
planner's chosen trajectory, arbitrated every frame by the safety
monitor (which can degrade to a stop when the path is blocked, sensors
go stale, or the trajectory leaves the lane).  It is a separate entry
point from ``m5_autopilot.py`` so the proven rule autopilot (94.6%
route result) is never touched.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_fsd_drive.py --runtime tech \\
        --attach --seconds 30 --speed 8
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.fsd_drive import run


def main() -> int:
    ap = argparse.ArgumentParser(description="FSD-mode live driving")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="auto")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--cam-w", type=int, default=400)
    ap.add_argument("--cam-h", type=int, default=300)
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"),
                    help="teleport to an open stretch before driving")
    ap.add_argument("--out", type=str, default=None,
                    help="path for per-frame JSON telemetry export")
    ap.add_argument("--lane-mode", choices=("map", "auto", "sensor"),
                    default="map",
                    help="lane-keep reference policy: map (rule-stable "
                         "default), auto (sensor leads only when it agrees "
                         "with the map lane), sensor (perception-led; map "
                         "prior stays the hard guard-rail)")
    ap.add_argument("--strict", action="store_true",
                    help="FSD realism mode (docs/fsd_realism.md): with "
                         "--lane-mode sensor the map lane may NEVER lead; "
                         "no paired perception lane -> no-lane degradation")
    ap.add_argument("--e2e-model", type=str, default=None,
                    help="trained E2ENetTorch checkpoint to rank as the "
                         "neural planning candidate (default: "
                         "logs/m5_e2e/best_temporal.pt)")
    ap.add_argument("--no-e2e", action="store_true",
                    help="disable the E2E neural planning candidate")
    ap.add_argument("--bc-model", type=str, default=None,
                    help="trained DAVE-2 BC steering checkpoint to rank "
                         "as a neural candidate (default: "
                         "logs/m3_bc/bc_tech_smallgrid.pt)")
    ap.add_argument("--no-bc", action="store_true",
                    help="disable the DAVE-2 BC steering candidate")
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"),
                    help="set an in-game navigation route to this goal "
                         "before driving (else reuse the active nav route)")
    ap.add_argument("--no-signal", action="store_true",
                    help="disable the traffic-light stop (a confident "
                         "vision red or a game-side red signal normally "
                         "stops the car)")
    ap.add_argument("--ring", choices=("front", "all"), default="front",
                    help="camera ring to poll: front = front_main only "
                         "(the only frame the live tick consumes), "
                         "all = full 8-camera surround ring")
    ap.add_argument("--no-shadow", action="store_true",
                    help="disable shadow-episode recording during the drive")
    args = ap.parse_args()

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
