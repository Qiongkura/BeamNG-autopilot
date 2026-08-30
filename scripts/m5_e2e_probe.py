"""Probe the end-to-end network against recorded episodes.

Single-episode mode (default): load one shadow-mode .npz episode, run the
E2E net over the recorded frames and print predicted vs executed control.

Batch replay mode (``--data``): run a trained ``E2ENetTorch`` over every
episode under a directory and write a JSON report with

  * action error  - steer / throttle MAE & RMS vs the executed control
  * takeover rate - fraction of frames whose predicted action deviates
    more than a threshold from what the driver actually did (the net
    would have needed human intervention)
  * trajectory error - recorded shadow trajectory (world) transformed to
    the ego frame vs the predicted ego-relative waypoints
  * trajectory sanity - NaN frames, path extent and curvature of the
    predicted paths

This is the offline closed-loop replay evaluation of the E2E stack: no
game needed, it answers "how much would the net fight the driver".

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_e2e_probe.py \\
        --episode logs/live_runs/shadow_episodes/shadow_*.npz
    .venv\\Scripts\\python.exe scripts\\m5_e2e_probe.py \\
        --data logs/live_runs/shadow_episodes --weights logs/m5_e2e/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.neural import E2ENet, E2ENetTorch


def _load_trained(weights: str, device: str):
    """Load an E2ENetTorch checkpoint; return (net, ckpt, img_h, img_w)."""
    ckpt = torch.load(weights, map_location="cpu")
    net_torch = E2ENetTorch(grid_n=ckpt["grid_n"],
                            n_waypoints=ckpt["n_waypoints"],
                            history=int(ckpt.get("history", 0)))
    net_torch.load_state_dict(ckpt["model"])
    net_torch.to(device)
    net_torch.eval()
    return net_torch, ckpt, int(ckpt["img_h"]), int(ckpt["img_w"])


def _prep(i, z, img_h: int, img_w: int):
    """Resize one recorded frame for the trained CNN."""
    import torch.nn.functional as F
    rgb = np.asarray(z["rgb"][i], dtype=np.uint8) \
        if "rgb" in z else np.zeros((img_h, img_w, 3), dtype=np.uint8)
    label = np.asarray(z["label"][i], dtype=np.uint8) \
        if "label" in z else None
    t_rgb = torch.from_numpy(
        rgb.astype(np.float32)).permute(2, 0, 1)[None]
    t_rgb = F.interpolate(t_rgb, size=(img_h, img_w),
                          mode="bilinear",
                          align_corners=False)[0] / 255.0
    t_label = None
    if label is not None:
        t_label = torch.from_numpy(
            label.astype(np.float32))[None, None]
        t_label = F.interpolate(t_label, size=(img_h, img_w),
                                mode="nearest")[0]
    t_bev = torch.from_numpy(
        np.asarray(z["bev"][i], dtype=np.float32))[None]
    return t_rgb, t_label, t_bev


def _predict(net_torch, net, z, i: int, device: str,
             img_h: int, img_w: int):
    """Run one frame; return (steer, throttle, traj_ego, valid_len)."""
    if net_torch is not None:
        h = net_torch.history
        need = h + 1
        i0 = max(0, i - h)
        frames = [_prep(j, z, img_h, img_w) for j in range(i0, i + 1)]
        pads = need - len(frames)  # missing at the episode start
        if pads:
            frames = [frames[0]] * pads + frames
        t_rgb = torch.stack([f[0] for f in frames])[None].to(device)
        t_label = torch.stack([(f[1] if f[1] is not None
                                else torch.zeros_like(f[0][:1]))
                               for f in frames])[None].to(device)
        t_bev = torch.stack([f[2] for f in frames])[None].to(device)
        t_speed = torch.tensor(
            [[float(z["speed"][i])]], dtype=torch.float32, device=device)
        with torch.no_grad():
            traj, action = net_torch(t_rgb, t_label, t_bev, t_speed)
        return (float(action[0, 0].cpu()), float(action[0, 1].cpu()),
                traj[0].cpu().numpy(), int(traj.shape[1]))
    traj, action = net.forward(np.asarray(z["bev"][i], dtype=np.float32))
    return (float(action[0]), float(action[1]),
            np.asarray(traj, dtype=float), len(traj))


def _traj_ego_gt(x: float, y: float, heading: float,
                 traj_world: np.ndarray):
    """World trajectory -> ego frame (same transform as the dataset)."""
    h = float(heading)
    c, s = np.cos(h), np.sin(h)
    rel = np.asarray(traj_world, dtype=np.float32) - \
        np.array([x, y], dtype=np.float32)
    rel = rel[np.isfinite(rel).all(axis=1)]
    if len(rel) == 0:
        return None
    return np.stack([rel[:, 0] * c + rel[:, 1] * s,
                     -rel[:, 0] * s + rel[:, 1] * c], axis=1)


def _path_metrics(traj_pred: np.ndarray) -> tuple[float, float]:
    """(extent, curvature) of a predicted ego path."""
    if traj_pred is None or len(traj_pred) < 2 or \
            not np.isfinite(traj_pred).all():
        return float("nan"), float("nan")
    extent = float(np.linalg.norm(traj_pred[-1]))
    d = np.linalg.norm(np.diff(traj_pred, axis=0), axis=1)
    L = float(d.sum())
    if L < 1e-3 or len(traj_pred) < 3:
        return extent, 0.0
    seg = np.diff(traj_pred, axis=0)
    angs = np.arctan2(seg[:, 1], seg[:, 0])
    turn = float(np.abs(np.diff((angs + np.pi) % (2.0 * np.pi)
                                - np.pi)).sum())
    return extent, turn / L


def _evaluate_episode(ep: Path, net_torch, net, device: str,
                      img_h: int, img_w: int,
                      th_steer: float, th_thr: float) -> dict:
    n_wp = net_torch.n_waypoints if net_torch is not None else 16
    steer_err: list[float] = []
    thr_err: list[float] = []
    takeovers = 0
    traj_mae: list[float] = []
    traj_frames = 0
    traj_nan = 0
    extents: list[float] = []
    curvs: list[float] = []
    with np.load(ep, allow_pickle=True) as z:
        n = int(z["t"].shape[0])
        has_traj = all(k in z for k in
                       ("trajectory", "trajectory_ok", "heading", "x", "y"))
        for i in range(n):
            ps, pt, traj_pred, _ = _predict(
                net_torch, net, z, i, device, img_h, img_w)
            gs = float(z["steer"][i])
            gt = float(z["throttle"][i])
            steer_err.append(ps - gs)
            thr_err.append(pt - gt)
            if abs(ps - gs) > th_steer or abs(pt - gt) > th_thr:
                takeovers += 1
            ext, curv = _path_metrics(traj_pred)
            if np.isnan(ext):
                traj_nan += 1
            else:
                extents.append(ext)
                curvs.append(curv)
            if has_traj and bool(z["trajectory_ok"][i]):
                gt_ego = _traj_ego_gt(
                    float(z["x"][i]), float(z["y"][i]),
                    float(z["heading"][i]),
                    np.asarray(z["trajectory"][i]))
                if gt_ego is not None and np.isfinite(traj_pred).all():
                    m = min(len(gt_ego), len(traj_pred), n_wp)
                    if m >= 2:
                        traj_mae.append(float(np.linalg.norm(
                            traj_pred[:m] - gt_ego[:m], axis=1).mean()))
                        traj_frames += 1
    n = max(1, n)
    se = np.asarray(steer_err, dtype=float)
    te = np.asarray(thr_err, dtype=float)
    return {
        "episode": ep.name,
        "frames": int(n),
        "steer_mae": float(np.abs(se).mean()),
        "steer_rms": float(np.sqrt(np.mean(se ** 2))),
        "throttle_mae": float(np.abs(te).mean()),
        "throttle_rms": float(np.sqrt(np.mean(te ** 2))),
        "takeover_rate": takeovers / n,
        "traj_mae": float(np.mean(traj_mae)) if traj_mae else None,
        "traj_frames": traj_frames,
        "traj_nan_frames": traj_nan,
        "traj_extent_mean": float(np.mean(extents)) if extents else None,
        "traj_curv_mean": float(np.mean(curvs)) if curvs else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="E2E probe")
    ap.add_argument("--episode", type=str, default=None,
                    help="single recorded shadow .npz episode")
    ap.add_argument("--data", type=str, default=None,
                    help="directory of episodes for batch replay eval")
    ap.add_argument("--weights", type=str, default=None,
                    help="trained E2ENetTorch checkpoint (real CNN path)")
    ap.add_argument("--report", type=str, default=None,
                    help="JSON report path (batch mode; default "
                         "logs/m5_e2e/report.json)")
    ap.add_argument("--takeover-steer", type=float, default=0.2,
                    help="|pred-gt| steer deviation counted as takeover")
    ap.add_argument("--takeover-throttle", type=float, default=0.3,
                    help="|pred-gt| throttle deviation counted as takeover")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    if args.episode is None and args.data is None:
        print("[e2e] give --episode or --data")
        return 2
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    net = E2ENet()
    net_torch = None
    ckpt = None
    img_h = img_w = 0
    if args.weights:
        net_torch, ckpt, img_h, img_w = _load_trained(args.weights, device)
        print(f"[e2e] loaded {args.weights} (epoch {ckpt['epoch']}, "
              f"val {float(ckpt['val_loss']):.4f}, "
              f"history {net_torch.history})")

    if args.data is not None:
        if net_torch is None:
            print("[e2e] batch replay needs --weights")
            return 2
        data = Path(args.data)
        eps = sorted(data.glob("*.npz"),
                     key=lambda p: p.stat().st_mtime)
        if not eps:
            print(f"[e2e] no episodes under {data}")
            return 1
        report_path = Path(args.report or (ROOT / "logs" / "m5_e2e"
                                           / "report.json"))
        per = []
        t0 = time.time()
        for ep in eps:
            r = _evaluate_episode(ep, net_torch, net, device,
                                  img_h, img_w,
                                  args.takeover_steer,
                                  args.takeover_throttle)
            per.append(r)
            traj_txt = "--" if r["traj_mae"] is None \
                else f"{r['traj_mae']:.2f}m"
            print(f"[e2e] {r['episode']}: n={r['frames']} "
                  f"steer_mae={r['steer_mae']:+.3f} "
                  f"thr_mae={r['throttle_mae']:.3f} "
                  f"takeover={r['takeover_rate'] * 100:.1f}% "
                  f"traj_mae={traj_txt} "
                  f"nan={r['traj_nan_frames']}")
        n_all = int(sum(r["frames"] for r in per))
        agg = {
            "episodes": len(per),
            "frames": n_all,
            "steer_mae": float(np.mean([r["steer_mae"] for r in per])),
            "throttle_mae": float(np.mean([r["throttle_mae"]
                                           for r in per])),
            "takeover_rate": float(np.mean(
                [r["takeover_rate"] for r in per])),
            "traj_mae": float(np.mean(
                [r["traj_mae"] for r in per if r["traj_mae"] is not None]))
            if any(r["traj_mae"] is not None for r in per) else None,
        }
        report = {
            "weights": args.weights,
            "history": int(net_torch.history),
            "model_val_loss": float(ckpt["val_loss"]),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "takeover_thresholds": {"steer": args.takeover_steer,
                                    "throttle": args.takeover_throttle},
            "aggregate": agg,
            "episodes": per,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=1),
            encoding="utf-8")
        traj_txt = "--" if agg["traj_mae"] is None \
            else f"{agg['traj_mae']:.2f}m"
        print(f"[e2e] {agg['episodes']} episodes / {agg['frames']} frames, "
              f"steer_mae={agg['steer_mae']:.3f} "
              f"thr_mae={agg['throttle_mae']:.3f} "
              f"takeover={agg['takeover_rate'] * 100:.1f}% "
              f"traj_mae={traj_txt} "
              f"({time.time() - t0:.0f}s) -> {report_path}")
        return 0

    with np.load(args.episode, allow_pickle=True) as z:
        n = int(z["t"].shape[0])
        print(f"[e2e] episode {Path(args.episode).name}: {n} frames")
        for i in range(n):
            ps, pt, traj_pred, traj_len = _predict(
                net_torch, net, z, i, device, img_h, img_w)
            gs = float(z["steer"][i])
            gt = float(z["throttle"][i])
            print(f"  frame {i}: executed steer={gs:+.2f} "
                  f"throttle={gt:.2f} | "
                  f"pred steer={ps:+.3f} thr={pt:.3f} "
                  f"traj_len={traj_len}")
        if net_torch is None:
            print("[e2e] forward/backward contract OK (untrained skeleton)")
        else:
            print("[e2e] trained CNN inference OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
