"""M3 behavioural cloning training (DAVE-2 style steering regressor).

Consumes one or more capture runs (m3_collect_bc layout, or the older
m2_capture layout with meta.json) and trains a small CNN to predict the
steering command from a single front camera frame.

Usage:
    python scripts/m3_train_bc.py --runs logs/m3_bc/20260811_* \
        --epochs 60 --out logs/m3_bc/bc_steer
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.bc import Dave2, conv_feature_size


def load_run(run_dir: Path, w: int, h: int):
    """Return list of (img_rgb_uint8, steer, t) for one capture run."""
    run_dir = Path(run_dir)
    samples = []

    jsonl = run_dir / "meta.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            fp = run_dir / "frames" / f"frame_{m['idx']:05d}.jpg"
            if fp.exists():
                samples.append((fp, float(m["steer"]), float(m["t"])))
        return samples

    meta_json = run_dir / "meta.json"
    if meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        for m in meta:
            fp = run_dir / f"frame_{m['idx']:05d}.jpg"
            if fp.exists():
                samples.append((fp, float(m["steer"]), float(m["t"])))
        return samples

    return samples


def load_dataset(runs, w: int, h: int, cache: dict):
    """Load all samples; cache decoded frames by path."""
    raw = []
    for r in runs:
        samples = load_run(r, w, h)
        if not samples:
            print(f"[train] WARNING: no samples in {r}")
        else:
            print(f"[train] {len(samples)} samples from {r}")
            raw.extend(samples)
    raw.sort(key=lambda s: s[2])  # time order
    out = []
    for fp, steer, t in raw:
        key = str(fp)
        img = cache.get(key)
        if img is None:
            img = cv2.cvtColor(cv2.imread(key), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            cache[key] = img
        out.append((img, float(steer), float(t)))
    return out


def augment(img: np.ndarray, steer: float, rng: random.Random):
    """Random horizontal flip + brightness/contrast jitter."""
    if rng.random() < 0.5:
        img = img[:, ::-1, :]
        steer = -steer
    alpha = 1.0 + rng.uniform(-0.15, 0.15)
    beta = rng.uniform(-15, 15)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img, steer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="capture run dirs (supports glob via shell)")
    ap.add_argument("--resize", default="200x66", help="model input WxH")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-aug", action="store_true",
                    help="disable flip/brightness augmentation")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default=None, help="output prefix (default logs/m3_bc/bc_steer)")
    args = ap.parse_args()

    w, h = (int(x) for x in args.resize.lower().split("x"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  input={w}x{h}")

    cache: dict = {}
    data = load_dataset(args.runs, w, h, cache)
    if len(data) < 20:
        print(f"[train] only {len(data)} samples; need >= 20. Abort.")
        sys.exit(1)

    n_val = max(1, int(len(data) * args.val_frac))
    train, val = data[: len(data) - n_val], data[len(data) - n_val:]
    print(f"[train] train={len(train)}  val={len(val)}  (time-ordered split)")

    feat_in = conv_feature_size(h, w)
    print(f"[train] conv feature size = {feat_in}")
    model = Dave2(feat_in=feat_in).to(device)
    mean_steer = float(np.mean([s[1] for s in train]))
    with torch.no_grad():
        model.fc[-1].bias.fill_(mean_steer)
    print(f"[train] head bias init = {mean_steer:+.3f} (train steer mean)")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    def make_batches(samples, batch, shuffle):
        idx = list(range(len(samples)))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), batch):
            yield [samples[j] for j in idx[i : i + batch]]

    def preprocess(imgs):
        arr = np.stack(imgs).astype(np.float32) / 255.0 * 2.0 - 1.0
        return torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(device)

    history = {"train_loss": [], "val_mae": [], "val_r2": []}
    best_mae, best_path = float("inf"), None
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        total, n_b = 0.0, 0
        for batch in make_batches(train, args.batch, shuffle=True):
            imgs, steer_labels = [], []
            for b in batch:
                img_a, steer_a = (b[0], b[1]) if args.no_aug else augment(b[0], b[1], rng)
                imgs.append(img_a)
                steer_labels.append(steer_a)
            steers = torch.tensor(steer_labels, dtype=torch.float32,
                                  device=device).view(-1, 1)
            opt.zero_grad()
            pred = model(preprocess(imgs))
            loss = loss_fn(pred, steers)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(batch)
            n_b += len(batch)
        sched.step()
        history["train_loss"].append(total / max(n_b, 1))

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in make_batches(val, args.batch, shuffle=False):
                imgs = [b[0] for b in batch]
                steers = torch.tensor([b[1] for b in batch], dtype=torch.float32,
                                      device=device).view(-1, 1)
                pred = model(preprocess(imgs))
                preds.extend(pred.cpu().numpy().ravel().tolist())
                truths.extend(steers.cpu().numpy().ravel().tolist())
        preds, truths = np.asarray(preds), np.asarray(truths)
        mae = float(np.mean(np.abs(preds - truths)))
        ss_res = float(np.sum((preds - truths) ** 2))
        ss_tot = float(np.sum((truths - truths.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        history["val_mae"].append(mae)
        history["val_r2"].append(r2)

        if best_path is None or mae < best_mae:
            best_mae = mae
            out_prefix = args.out or str(config.LOGS_DIR / "m3_bc" / "bc_steer")
            out_prefix = Path(out_prefix)
            out_prefix.parent.mkdir(parents=True, exist_ok=True)
            best_path = out_prefix.with_suffix(".pt")
            torch.save({"state_dict": model.state_dict(),
                        "resize": (w, h),
                        "val_mae": mae}, str(best_path))

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[train] ep {epoch + 1:3d}/{args.epochs}  "
                  f"loss={history['train_loss'][-1]:.4f}  "
                  f"val_mae={mae:.3f}  val_r2={r2:+.3f}  "
                  f"best_mae={best_mae:.3f}  ({time.time() - t0:.0f}s)")

    print(f"\n[train] best val MAE = {best_mae:.3f} -> {best_path}")

    # qualitative samples from the val split
    model.eval()
    with torch.no_grad():
        for b in make_batches(val[:8], 8, shuffle=False):
            imgs = [x[0] for x in b]
            pred = model(preprocess(imgs)).cpu().numpy().ravel()
            for i, (p, t) in enumerate(zip(pred, [x[1] for x in b])):
                print(f"    val#{i}: pred={p:+.3f}  truth={t:+.3f}")

    # save loss curves + val scatter
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_prefix = Path(args.out or str(config.LOGS_DIR / "m3_bc" / "bc_steer"))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history["train_loss"], label="train MSE")
        ax[0].set_title("train loss")
        ax[0].legend()
        ax[1].plot(history["val_mae"], label="val MAE", color="tab:red")
        ax[1].plot(history["val_r2"], label="val R2", color="tab:green")
        ax[1].set_title("validation")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(str(out_prefix.with_suffix(".png")))
        print(f"[train] curves -> {out_prefix.with_suffix('.png')}")
    except Exception as exc:
        print(f"[train] plot skipped: {exc}")


if __name__ == "__main__":
    main()
