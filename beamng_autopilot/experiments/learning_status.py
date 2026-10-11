"""Read the existing learning artifacts for the monitor without training or control."""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import sanitize_numbers


def read_learning_status(run_dir: Path) -> dict:
    problems = []

    def read(path):
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected an object")
            return value
        except (OSError, ValueError) as exc:
            problems.append(f"{path.name}: {type(exc).__name__}")
            return None

    cfg = read(run_dir / "loop_config.json") or {}
    admission = read(run_dir / "auto_truth" / "admission.json")
    collection = read(run_dir / "auto_truth" / "state.json")
    resource = read(run_dir / "resource_state.json")
    # Fixed two-level layouts used by rounds. Never recurse through logs.
    paths = list(run_dir.glob("baseline/seed*/train_hist.json"))
    paths += list(run_dir.glob("round*/seed*/train_hist.json"))
    latest = max(paths, key=lambda p: p.stat().st_mtime) if paths else None
    hist = read(latest) if latest else None
    sampling = (hist or {}).get("sampler_report") or {}
    init = (hist or {}).get("init") or {}
    decisions = []
    for path in sorted(run_dir.glob("decision_*.json")):
        value = read(path)
        if value:
            decision = value.get("decision") or {}
            decisions.append({"candidate_id": value.get("candidate_id"),
                              "decision": decision.get("decision"),
                              "reasons": decision.get("reasons"),
                              "outcome": value.get("round_outcome")})
    attempts = (collection or {}).get("attempts") or []
    isolated = (admission or {}).get("isolated")
    if isolated is None and attempts:
        isolated = (attempts[-1].get("qualification") or {}).get("isolated")
    out = {
        "schema": "learning-status-v1",
        "collection_status": ((admission or {}).get("status")
                              or (attempts[-1].get("status") if attempts else None)),
        "collection_attempts": len(attempts) if collection is not None else None,
        "qualified_frames": (admission or {}).get("frames_total"),
        "line_free_frames": (admission or {}).get("line_free_frames"),
        "isolated_packages": len(isolated) if isolated is not None else None,
        "isolation_reasons": [r.get("reasons") for r in (isolated or [])],
        "channels": [r.get("channels") for r in (admission or {}).get("admitted", [])],
        "training_path": latest.relative_to(run_dir).as_posix() if latest else None,
        "steps_done": (hist or {}).get("steps_done"),
        "total_steps": (hist or {}).get("total_steps", cfg.get("total_steps")),
        "unique_available": sampling.get("unique_available"),
        "unique_seen": sampling.get("unique_seen"),
        "exposures_total": sampling.get("exposures_total"),
        "init_sha16": init.get("init_sha16"),
        "init_source": init.get("init_source"),
        "stopped_by": (hist or {}).get("stopped_by"),
        "decisions": decisions, "resource": resource, "problems": problems,
        "missing": ("No sampler report: unique_seen is UNKNOWN, not pool size"
                    if not sampling else None),
    }
    return sanitize_numbers(out)[0]
