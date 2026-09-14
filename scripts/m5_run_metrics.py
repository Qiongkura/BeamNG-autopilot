"""Parse an FSD drive log or shadow npz into run quality metrics.

Usage:
    .venv\\Scripts\\python.exe scripts/m5_run_metrics.py --log path.txt
    .venv\\Scripts\\python.exe scripts/m5_run_metrics.py --shadow path.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def metrics_from_text(t: str) -> dict:
    sensor = t.count("lane=sensor")
    unav = t.count("lane=perception-unavailable")
    n = sensor + unav
    return {
        "lane_ticks": n,
        "sensor_ticks": sensor,
        "sensor_rate": round(sensor / n, 4) if n else 0.0,
        "unavailable_ticks": unav,
        "no_drivable_path": t.count("no drivable path"),
        "grazes_obstacle": t.count("path grazes obstacle"),
        "placed": "placed=True" in t,
        "unplaced": "UNPLACED" in t or "placed=False" in t,
    }


def metrics_from_shadow(path: Path) -> dict:
    import numpy as np
    d = np.load(path, allow_pickle=True)
    lane = [str(x) for x in d["lane_src"].tolist()]
    sensor = sum(1 for s in lane if s == "sensor")
    n = len(lane)
    return {
        "frames": n,
        "sensor_rate": round(sensor / n, 4) if n else 0.0,
        "trajectory_ok_rate": round(float(d["trajectory_ok"].mean()), 4)
        if "trajectory_ok" in d else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument("--shadow", type=str, default=None)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    out: dict = {}
    if args.log:
        raw = Path(args.log).read_bytes()
        text = (raw.decode("utf-16") if raw[:2] == b"\xff\xfe"
                else raw.decode("utf-8", errors="replace"))
        out.update(metrics_from_text(text))
    if args.shadow:
        out.update(metrics_from_shadow(Path(args.shadow)))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
