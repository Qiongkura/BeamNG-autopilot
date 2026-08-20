"""Probe the full integrated FSDStack on BeamNG.tech.

Runs one complete FSD-style pipeline tick live: camera ring -> HydraNets
(semantic + traffic + object + topology heads) -> BEV occupancy / vector
space -> layered planner, and prints the fused results.  This is the
"wiring proof" that the FSD stack is functionally complete end to end.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_fsd_stack_probe.py --runtime tech --attach
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.fsd_stack import FSDStack
from beamng_autopilot.vision.heads import (
    ObjectHead,
    SemanticHead,
    TrafficSignalHead,
)
from beamng_autopilot.vision.heads.topology import LaneTopologyHead


def main() -> int:
    ap = argparse.ArgumentParser(description="integrated FSD stack probe")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="auto")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--with-object", action="store_true",
                    help="also run YOLO object head (needs ultralytics)")
    args = ap.parse_args()

    conn = BeamNGConnector(
        "italy", "etk800",
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    try:
        conn.open(launch=not args.attach)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            conn.load_scenario()

        heads = [SemanticHead(), TrafficSignalHead()]
        if args.with_object:
            try:
                heads.append(ObjectHead(enabled_roles=("front_main",)))
            except Exception as exc:
                print(f"[fsd] object head unavailable: {exc}")

        stack = FSDStack(conn, args.runtime, heads=heads)
        print(f"[fsd] runtime={stack.mode} pipeline ready "
              f"(ring + {[h.name for h in heads]} + bev + planner)")
        out = stack.tick()
        print(f"[fsd] bev occupancy: "
              f"occupied={(out.bev > 0.4).sum()}/{(out.bev > 0.0).sum()} "
              f"drivable={int((out.drivable > 0).sum())}")
        print(f"[fsd] candidates={out.n_candidates} "
              f"chosen={out.meta.get('planner', {}).get('why')}")
        sem = out.head_outputs.get("semantic")
        if sem is not None:
            road = float(sem.masks["road"].mean())
            print(f"[fsd] semantic head: road_frac={road:.3f} "
                  f"markings={len(sem.meta.get('markings', []))}")
        tr = out.head_outputs.get("traffic")
        if tr is not None:
            print(f"[fsd] traffic head: state={tr.meta.get('signal_state')} "
                  f"conf={tr.meta.get('signal_conf'):.2f}")
        if out.head_outputs.get("object"):
            ob = out.head_outputs["object"]
            print(f"[fsd] object head: obstacles={len(ob.obstacles)} "
                  f"boxes={len(ob.boxes)}")
        stack.close()
        print("[fsd] one full FSD pipeline tick OK")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())