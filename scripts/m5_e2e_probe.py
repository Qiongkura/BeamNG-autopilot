"""Probe the end-to-end network against a recorded episode.

Loads a shadow-mode .npz episode and runs the E2E net over the recorded
BEV frames, printing the predicted trajectory/action vs the executed
control.  Without ``--weights`` the numpy linear skeleton is used (call
contract check); with ``--weights`` the real trained ``E2ENetTorch``
(multimodal CNN) is loaded and evaluated on the RGB/label/BEV frames.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_e2e_probe.py \\
        --episode logs/live_runs/shadow_episodes/shadow_*.npz
    .venv\\Scripts\\python.exe scripts\\m5_e2e_probe.py \\
        --episode logs/live_runs/shadow_episodes/shadow_*.npz \\
        --weights logs/m5_e2e/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.neural import E2ENet, E2ENetTorch


def main() -> int:
    ap = argparse.ArgumentParser(description="E2E probe")
    ap.add_argument("--episode", type=str, required=True,
                    help="recorded shadow .npz episode")
    ap.add_argument("--weights", type=str, default=None,
                    help="trained E2ENetTorch checkpoint (real CNN path)")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    episode = Path(args.episode)
    if not episode.exists():
        print(f"[e2e] episode not found: {episode}")
        return 1
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    net = E2ENet()
    net_torch = None
    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu")
        net_torch = E2ENetTorch(grid_n=ckpt["grid_n"],
                                n_waypoints=ckpt["n_waypoints"])
        net_torch.load_state_dict(ckpt["model"])
        net_torch.to(device)
        net_torch.eval()
        img_h, img_w = int(ckpt["img_h"]), int(ckpt["img_w"])
        print(f"[e2e] loaded {args.weights} (epoch {ckpt['epoch']}, "
              f"val {float(ckpt['val_loss']):.4f})")

    with np.load(episode, allow_pickle=True) as z:
        n = int(z["t"].shape[0])
        print(f"[e2e] episode {episode.name}: {n} frames")
        for i in range(n):
            bev = np.asarray(z["bev"][i], dtype=np.float32)
            gt_steer = float(z["steer"][i])
            gt_thr = float(z["throttle"][i])
            if net_torch is not None:
                rgb = np.asarray(z["rgb"][i], dtype=np.uint8)
                label = np.asarray(z["label"][i], dtype=np.uint8) \
                    if "label" in z else None
                import torch.nn.functional as F
                t_rgb = torch.from_numpy(
                    rgb.astype(np.float32)).permute(2, 0, 1)[None]
                t_rgb = F.interpolate(
                    t_rgb, size=(img_h, img_w), mode="bilinear",
                    align_corners=False)[0] / 255.0
                t_label = None
                if label is not None:
                    t_label = torch.from_numpy(
                        label.astype(np.float32))[None, None]
                    t_label = F.interpolate(
                        t_label, size=(img_h, img_w), mode="nearest")[0]
                t_bev = torch.from_numpy(bev)[None, None]
                with torch.no_grad():
                    traj, action = net_torch(
                        t_rgb[None].to(device),
                        t_label[None].to(device) if t_label is not None
                        else None,
                        t_bev.to(device))
                pred_steer = float(action[0, 0].cpu())
                pred_thr = float(action[0, 1].cpu())
                traj_len = int(traj.shape[1])
            else:
                traj, action = net.forward(bev)
                pred_steer, pred_thr = (float(action[0]), float(action[1]))
                traj_len = len(traj)
            print(f"  frame {i}: executed steer={gt_steer:+.2f} "
                  f"throttle={gt_thr:.2f} | "
                  f"pred steer={pred_steer:+.3f} thr={pred_thr:.3f} "
                  f"traj_len={traj_len}")
        if net_torch is None:
            print("[e2e] forward/backward contract OK (untrained skeleton)")
        else:
            print("[e2e] trained CNN inference OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
