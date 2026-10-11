"""Serial comparison of a captured Tech range input and a pinned local reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamng_autopilot.experiments.collection import game_pids
from beamng_autopilot.experiments.range_replay import compare_replay


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--reference-source", type=Path, required=True)
    ap.add_argument("--reference-sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("use a fresh report path; preserve earlier evidence")
    active = game_pids()
    if active is None or active:
        ap.error("close the game before quiet CPU comparison; game state must be known")
    try:
        report = compare_replay(args.input, reference_source=args.reference_source,
                                reference_sha256=args.reference_sha256,
                                rounds=args.rounds, iterations=args.iterations,
                                warmup=args.warmup)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"range replay rejected: {exc}", flush=True)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"equal={report['outputs_equal']} calls={report['compared_calls']} "
          f"points={report['input_points']} best_p95_ms={report['best_p95_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
