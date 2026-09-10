"""Build the project artifact manifest used by data/evaluation gates.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_build_manifest.py
    .venv\\Scripts\\python.exe scripts\\m5_build_manifest.py --out logs\\manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.data_contract import scan_project, write_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="build artifact manifest")
    ap.add_argument("--root", type=Path, default=config.PROJECT_ROOT)
    ap.add_argument("--out", type=Path,
                    default=config.LOGS_DIR / "data_manifest.json")
    args = ap.parse_args()
    records = scan_project(args.root)
    out = write_manifest(records, args.out)
    by_type: dict[str, int] = {}
    for rec in records:
        by_type[rec["artifact_type"]] = by_type.get(rec["artifact_type"], 0) + 1
    print(f"[manifest] records={len(records)} by_type={by_type}")
    print(f"[manifest] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
