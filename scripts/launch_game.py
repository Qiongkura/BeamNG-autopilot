"""Launch BeamNG.drive with the beamngpy comms port enabled.

Launching the game from Steam manually does NOT open the tcom port that
beamngpy uses to attach (64256 by default), so m5_autopilot.py cannot see
the running game. Run this script instead of launching from Steam:

    python scripts/launch_game.py
    python scripts/launch_game.py --runtime tech

It starts the game with the same flags beamngpy uses, so the autopilot
script can detect and attach to your map/vehicle afterwards.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamngpy.beamng.filesystem import determine_binary


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch BeamNG with tcom enabled")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto prefers BeamNG.tech when installed")
    args = ap.parse_args()
    resolved = config.resolve_launch_runtime(args.runtime)
    try:
        binary = determine_binary(config.runtime_home(resolved))
    except Exception as exc:
        print(f"[launch] binary not found: {exc}")
        sys.exit(1)

    cmd = [str(binary)]
    if resolved != "tech":
        cmd.append("-nosteam")
    cmd += ["-tcom", "-tport", str(config.PORT), "-console"]
    runtime_user = config.runtime_user(resolved)
    if runtime_user:
        cmd += ["-userpath", str(runtime_user)]

    print(f"[launch] runtime={resolved} "
          f"starting BeamNG with tcom port {config.PORT} ...")
    print(f"[launch] {' '.join(cmd)}")
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    # Wait for the port to come up (game needs ~10-40s to reach the menu).
    import socket

    deadline = time.time() + 90.0
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", config.PORT), timeout=1):
                print(f"[launch] port {config.PORT} is up - the autopilot "
                      "can now attach")
                return
        except OSError:
            time.sleep(1.0)
    print(f"[launch] game started but port {config.PORT} did not open within "
          "90s; check the game window and rerun m5_autopilot.py")
    sys.exit(2)


if __name__ == "__main__":
    main()
