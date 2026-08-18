"""M5 autopilot assistant: manual in-game activation, Tesla-style vision
overlay, and post-drive throttle/brake/speed bar charts.

Workflow (recommended: you are already in a map with a vehicle in BeamNG):
    1. Run the script. It first probes for your running session and attaches
       to your current map/vehicle automatically (no --attach needed). If the
       game is not running at all, it launches one and loads a fresh scenario.
    2. Press F9 to toggle autopilot ON/OFF. Without a nav route the car
       keeps the sensor lane centre from camera lane lines / LiDAR walls.
       In the game open the big map (M), pick a destination, then press
       F10 to add the navigation route. F11 clears the route.
    3. F8 toggles the Tesla-style vision overlay (3D world route + front
       camera projection + bird view). When a session ends, a 3-panel
       throttle/brake/speed bar chart pops up.

Hotkeys (global, work while the game has focus):
    F8   vision overlay on/off
    F9   autopilot on/off
    F10  grab the in-game navigation route (set a destination on map M first)
    F11  clear route
    F12  quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.autopilot import AutopilotSession
from beamng_autopilot.planner import (
    RIGHT_OFFSET_M,
    SHARP_ANGLE_DEG,
    SHARP_CORNER_KPH,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 in-game autopilot assistant")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--attach", action="store_true",
                    help="attach to the vehicle in a running BeamNG session")
    ap.add_argument("--attach-vid", default=None,
                    help="vehicle id to attach to (default: first active)")
    ap.add_argument("--speed", type=float, default=20.0,
                    help="cruise speed in m/s")
    ap.add_argument("--port", type=int, default=None,
                    help="comms port (default: per-runtime - Steam 64256 / "
                         "Tech 64257)")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects Steam/tech after connect")
    ap.add_argument("--max-run", type=float, default=600.0,
                    help="max seconds for one autopilot session")
    ap.add_argument("--no-hud", action="store_true",
                    help="disable the live telemetry HUD window")
    ap.add_argument("--no-show", action="store_true",
                    help="save the telemetry chart to PNG without showing it")
    ap.add_argument("--front-camera", action="store_true",
                    help="switch the in-game camera to a fixed view ahead of "
                         "the car (default: leave the game camera and UI alone)")
    ap.add_argument("--no-vision-obstacles", action="store_true",
                    help="disable YOLO front-camera obstacle detection "
                         "(keeps raycast + vehicle sources)")
    ap.add_argument("--no-lanes", action="store_true",
                    help="disable front-camera lane-marking detection "
                         "(keeps raycast + vehicle sources)")
    ap.add_argument("--seg-model", default=None,
                    help="learned lane/road segmentation model path "
                         "(default: logs/m5_seg/seg_model/best.pt when "
                         "present; without it classic CV is used)")
    ap.add_argument("--no-seg", action="store_true",
                    help="do NOT auto-load the semantic segmentation model "
                         "(default: a found segmenter loads automatically). "
                         "Segmentation is the heaviest GPU load and the most "
                         "likely to destabilise BeamNG.tech; --no-seg falls "
                         "back to classic CV lane detection which is much "
                         "lighter, while YOLO + LiDAR remain active")
    ap.add_argument("--no-markers", action="store_true",
                    help="do not draw the yellow start / red goal spheres "
                         "in the game world")
    ap.add_argument("--nav-world", type=int, choices=(0, 1), default=0,
                    help="in-world nav line: 0 hides it (default) while the "
                         "route stays on the map; 1 shows arrows/ground "
                         "markers in the world")
    ap.add_argument("--no-overlay", action="store_true",
                    help="disable the 3D world overlay (route/obstacles)")
    ap.add_argument("--vision-conf", type=float, default=0.35,
                    help="YOLO confidence threshold for obstacle boxes")
    ap.add_argument("--vision-rate", type=float, default=3.0,
                    help="vision obstacle scan rate in Hz (default 3)")
    ap.add_argument("--right-offset", type=float, default=RIGHT_OFFSET_M,
                    help="optional right-hand offset from the nav route "
                         "(m, default 0 = follow the route centre)")
    ap.add_argument("--sharp-angle", type=float, default=SHARP_ANGLE_DEG,
                    help="corner angle threshold for the speed cap (deg)")
    ap.add_argument("--sharp-corner-kph", type=float,
                    default=SHARP_CORNER_KPH,
                    help="max speed through a sharp corner (km/h)")
    ap.add_argument("--bc-model", default=None,
                    help="DAVE-2 behavioural cloning model path "
                         "(e.g. logs/m3_bc/bc_tech_smallgrid.pt). "
                         "When loaded, press F7 to toggle BC steering mode")
    args = ap.parse_args()

    session = AutopilotSession(args)
    session.run()


if __name__ == "__main__":
    main()
