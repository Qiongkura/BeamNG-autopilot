"""Real end-to-end training on shadow episodes (multimodal CNN).

Trains ``E2ENetTorch`` on recorded shadow data - front RGB + seg label +
BEV raster in, trajectory (ego frame) + (steer, throttle) out - the
FSD-style "perception -> planning" supervision the recording harness
collects.  Pure offline: no game needed once the .npz episodes exist.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_train_e2e.py \\
        --data logs/live_runs/shadow_episodes --epochs 80 --batch 8
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

from beamng_autopilot.neural import E2ENetTorch, ShadowMultimodalDataset


def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """MSE over the valid waypoints only; 0 when a frame has none.

    Frames whose shadow planner found no trajectory are kept for
    action-only supervision, so the trajectory loss must not see their
    zero-filled padding.
    """
    err = (pred - target) ** 2 * mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    return err.sum() / denom


def main() -> int:
    ap = argparse.ArgumentParser(description="train the multimodal E2E net")
    ap.add_argument("--data", type=str, required=True,
                    help="dir of shadow .npz episodes, or a single file")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-h", type=int, default=120)
    ap.add_argument("--img-w", type=int, default=160)
    ap.add_argument("--min-quality", type=float, default=0.5)
    ap.add_argument("--min-speed", type=float, default=0.5,
                    help="drop near-static frames from training")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="logs/m5_e2e/best.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    files = sorted(data.glob("*.npz")) if data.is_dir() else [data]
    files = [p for p in files if p.exists()]
    if not files:
        print(f"[train-e2e] no episodes under {data}")
        return 1

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ShadowMultimodalDataset(
        files, min_quality=args.min_quality, min_speed=args.min_speed,
        img_h=args.img_h, img_w=args.img_w, augment=args.augment,
        seed=args.seed)
    n = len(ds)
    if n == 0:
        print("[train-e2e] no usable frames (quality/trajectory filter)")
        return 1
    n_val = max(1, int(round(n * args.val_split)))
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"[train-e2e] {len(files)} episodes, {n} frames "
          f"(train {len(train_idx)} / val {len(val_idx)}), device={device}")

    collate = ShadowMultimodalDataset.collate
    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, train_idx.tolist()),
        batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, val_idx.tolist()),
        batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = E2ENetTorch().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)
    act_w = torch.tensor([1.0, 0.3], dtype=torch.float32, device=device)

    def run_epoch(dl, train: bool) -> float:
        model.train(train)
        total = 0.0
        for obs, traj_t, mask_t, act_t in dl:
            rgb, label, bev = obs
            rgb = rgb.to(device)
            label = label.to(device) if label is not None else None
            bev = bev.to(device) if bev is not None else None
            traj_t = traj_t.to(device)
            mask_t = mask_t.to(device)
            act_t = act_t.to(device)
            traj_p, act_p = model(rgb, label, bev)
            # Trajectory error is normalised to metres/10 so the action
            # head is not starved by the much larger absolute waypoint
            # scale; frames without a feasible trajectory only pay the
            # action term (masked trajectory loss).
            loss = _masked_mse(traj_p / 10.0, traj_t / 10.0, mask_t) \
                + _mse(act_p * act_w, act_t * act_w)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += float(loss.detach().cpu()) * len(act_t)
        return total / max(1, len(dl.dataset))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch = 1e18, -1
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(train_dl, True)
        va = run_epoch(val_dl, False)
        sched.step()
        if va < best_val:
            best_val, best_epoch = va, ep
            torch.save({"model": model.state_dict(),
                        "epoch": ep, "val_loss": best_val,
                        "grid_n": model.grid_n,
                        "n_waypoints": model.n_waypoints,
                        "img_h": args.img_h, "img_w": args.img_w,
                        "min_quality": args.min_quality},
                       out)
        print(f"[train-e2e] epoch {ep:3d}  train={tr:.4f}  "
              f"val={va:.4f}  best={best_val:.4f}@{best_epoch}")
    stats = {"episodes": [str(p) for p in files], "frames": n,
             "best_val_loss": float(best_val), "best_epoch": int(best_epoch),
             "epochs": args.epochs, "device": device,
             "seconds": float(time.time() - t0)}
    (out.with_suffix(".json")).write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[train-e2e] best val {best_val:.4f} @ epoch {best_epoch} "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
