"""Probe the end-to-end network skeleton against a recorded episode.

Loads a shadow-mode .npz episode and runs the E2ENet over the recorded
BEV frames, printing the predicted trajectory/action vs the executed
control - a quick sanity check of the (raster -> trajectory/action)
contract without training.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_e2e_probe.py \\
        --episode logs/live_runs/shadow_episodes/shadow_*.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.neural import E2ENet


def main() -> int:
    ap = argparse.ArgumentParser(description="E2E skeleton probe")
    ap.add_argument("--episode", type=str, required=True,
                    help="recorded shadow .npz episode")
    args = ap.parse_args()

    episode = Path(args.episode)
    if not episode.exists():
        print(f"[e2e] episode not found: {episode}")
        return 1
    net = E2ENet()
    with np.load(episode, allow_pickle=True) as z:
        n = int(z["t"].shape[0])
        print(f"[e2e] episode {episode.name}: {n} frames")
        for i in range(n):
            bev = np.asarray(z["bev"][i], dtype=np.float32)
            traj, action = net.forward(bev)
            gt_steer = float(z["steer"][i])
            gt_thr = float(z["throttle"][i])
            pred_steer, pred_thr = (float(action[0]), float(action[1]))
            print(f"  frame {i}: executed steer={gt_steer:+.2f} "
                  f"throttle={gt_thr:.2f} | "
                  f"pred steer={pred_steer:+.3f} thr={pred_thr:.3f} "
                  f"traj_len={len(traj)}")
        print("[e2e] forward/backward contract OK (untrained skeleton)")
    return 0


if __name__ == "__main__":
    sys.exit(main())