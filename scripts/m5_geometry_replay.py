"""Audit exact production geometry and source-to-command joins offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamng_autopilot.geometry_probe import analyze_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("use a fresh report path; preserve earlier evidence")
    frames = json.loads(args.telemetry.read_text(encoding="utf-8"))
    if not isinstance(frames, list):
        parser.error("telemetry must be a list")
    result = analyze_frames(frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"frames={result['frames']} replayed={result['geometry_replayed']} "
          f"unknown={result['geometry_unknown']} failed_joins={result['command_joins_failed']} "
          f"safety_matches={result['safety_matches']} "
          f"safety_mismatches={result['safety_mismatches']} "
          f"safety_unknown={result['safety_assessment_unknown']}")
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
