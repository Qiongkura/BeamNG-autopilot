"""Run N short FSD drives at one spawn and aggregate run metrics.

Usage:
    .venv\\Scripts\\python.exe scripts/m5_fsd_multi_run.py --n 3 --seconds 60 \\
        --teleport 246.44 877.87 -107.62 --map east_coast_usa
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m5_run_metrics import metrics_from_text


def run_once(args, out_path: Path) -> dict:
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "m5_fsd_drive.py"),
        "--runtime", args.runtime, "--attach", "--map", args.map,
        "--teleport", str(args.teleport[0]), str(args.teleport[1]),
        str(args.teleport[2]),
        "--strict", "--lane-mode", "sensor",
        "--seconds", str(args.seconds), "--speed", str(args.speed),
        "--no-e2e", "--no-bc", "--no-dqn",
    ]
    if args.allow_unplaced:
        cmd.append("--allow-unplaced")
    print("[multi] run", out_path.name, flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, timeout=int(args.seconds) + 180)
    raw = p.stdout + b"\n" + p.stderr
    out_path.write_bytes(raw)
    text = raw.decode("utf-8", errors="replace")
    m = metrics_from_text(text)
    m["returncode"] = p.returncode
    m["wall_s"] = round(time.time() - t0, 1)
    print("[multi]", m, flush=True)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--runtime", default="tech")
    ap.add_argument("--map", default="east_coast_usa")
    ap.add_argument("--teleport", nargs=3, type=float,
                    default=[246.44, 877.87, -107.62])
    ap.add_argument("--allow-unplaced", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir) if args.out_dir else (
        ROOT / "logs" / "paper_notes" / f"multi_{int(time.time())}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n):
        rows.append(run_once(args, out / f"arm_{i}.log"))
    agg = {
        "n": len(rows),
        "sensor_rates": [r.get("sensor_rate") for r in rows],
        "placed": [r.get("placed") for r in rows],
        "no_path": [r.get("no_drivable_path") for r in rows],
        "median_sensor": sorted(r.get("sensor_rate") or 0 for r in rows)[
            len(rows) // 2],
        "placed_rate": sum(1 for r in rows if r.get("placed")) / max(len(rows), 1),
    }
    (out / "summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
