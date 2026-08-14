"""M4 realtime telemetry dashboard (standalone, second-terminal mode).

Renders the same HUD as the in-process dashboard (beamng_autopilot/hud.py)
but reads the JSON snapshot written by TelemetryBroadcaster, so it can run in
a second terminal while any driving / collect script is running:

    python scripts/m4_dashboard.py

Keys: q / ESC to quit.  Use --snapshot out.png to render one frame and exit
(useful for headless verification).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot.hud import LiveHUD, render_hud
from beamng_autopilot.telemetry import read_live


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None,
                    help="render one frame to a PNG and exit (headless check)")
    ap.add_argument("--fps", type=float, default=30.0, help="refresh rate")
    args = ap.parse_args()

    if args.snapshot:
        img = render_hud(read_live(), cam=None, fps=0.0)
        out = Path(args.snapshot)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        print(f"snapshot -> {out}")
        return

    hud = LiveHUD(name="BeamNG Telemetry (remote)", show_camera=False)
    while True:
        if not hud.update(read_live()):
            break
        time.sleep(1.0 / max(args.fps, 1.0))
    hud.close()


if __name__ == "__main__":
    main()
