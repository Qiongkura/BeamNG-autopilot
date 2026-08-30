"""Automated FSD pipeline: collect -> train -> evaluate, loop.

Orchestrates the existing scripts in one process chain so data can be
grown overnight without babysitting:

  1. collect : ``m5_shadow_drive.py`` (road-graph route, perception
               guard, end-stop; records RGB+label+BEV+quality)
  2. train   : ``m5_train_e2e.py`` (temporal multimodal CNN)
  3. eval    : ``m5_e2e_probe.py`` against the newest episode

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_pipeline.py --cycles 3 \\
        --episodes 2 --collect-seconds 60 --speed 6 \\
        --teleport 729.6 763.9 45 --goal 572 533.5
    .venv\\Scripts\\python.exe scripts\\m5_pipeline.py --mode train  # offline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(args: list[str]) -> None:
    cmd = [PY, str(ROOT / args[0]), *args[1:]]
    print(f"[pipeline] $ {cmd[0]} {' '.join(str(a) for a in cmd[1:])}",
          flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="collect->train->eval pipeline")
    ap.add_argument("--mode", choices=("all", "collect", "train"),
                    default="all")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=2,
                    help="shadow episodes per cycle")
    ap.add_argument("--collect-seconds", type=float, default=60.0)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--teleport", nargs=3, type=float, default=None)
    ap.add_argument("--goal", nargs=2, type=float, default=None)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--history", type=int, default=2)
    ap.add_argument("--data", type=str,
                    default="logs/live_runs/shadow_episodes")
    ap.add_argument("--weights", type=str, default="logs/m5_e2e/best.pt")
    args = ap.parse_args()

    data = Path(args.data)
    weights = Path(args.weights)
    data.mkdir(parents=True, exist_ok=True)
    weights.parent.mkdir(parents=True, exist_ok=True)

    for cycle in range(1, args.cycles + 1):
        if args.mode in ("all", "collect"):
            for ep in range(1, args.episodes + 1):
                cmd = ["scripts/m5_shadow_drive.py", "--runtime", "tech",
                       "--attach", "--seconds",
                       str(args.collect_seconds), "--speed",
                       str(args.speed), "--drive",
                       "--min-quality", "0", "--lidar-every", "2",
                       "--out", str(data)]
                if args.teleport is not None:
                    cmd += ["--teleport"] + [str(v) for v in args.teleport]
                if args.goal is not None:
                    cmd += ["--goal"] + [str(v) for v in args.goal]
                _run(cmd)
                print(f"[pipeline] cycle {cycle}/{args.cycles} "
                      f"episode {ep}/{args.episodes} recorded", flush=True)
        if args.mode in ("all", "train"):
            _run(["scripts/m5_train_e2e.py", "--data", str(data),
                  "--epochs", str(args.epochs), "--history",
                  str(args.history), "--out", str(weights)])
            report = ROOT / "logs" / "m5_e2e" / "report.json"
            # 批量回放评测：对所有录到的 episodes 跑一次，报告落盘
            # logs/m5_e2e/report.json，方便跨轮次对比接管率/动作误差。
            _run(["scripts/m5_e2e_probe.py", "--data", str(data),
                  "--weights", str(weights), "--report", str(report)])
    print("[pipeline] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
