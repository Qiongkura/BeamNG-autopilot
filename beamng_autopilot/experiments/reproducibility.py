"""Portable source identities for runs made from an uncommitted workspace."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .checkpoint import file_sha16


SOURCE_DIRS = ("beamng_autopilot", "beamng_autopilot_tech", "scripts")
SOURCE_SUFFIXES = {".py", ".ps1", ".lua", ".json", ".yaml", ".yml"}


def source_snapshot(root: Path) -> dict:
    """Hash only named source directories, never logs, weights or user files."""
    files = {}
    errors = []
    for name in SOURCE_DIRS:
        directory = root / name
        if not directory.is_dir():
            errors.append(f"missing source directory: {name}")
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            try:
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                errors.append(f"{path.relative_to(root)}: {type(exc).__name__}")
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, timeout=10)
        commit = proc.stdout.strip() if proc.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"schema": "source-content-v1", "commit": commit,
            "sha256": digest, "files": files, "errors": errors,
            "complete": bool(files) and not errors}


def write_baseline_evidence(path: Path, evidence: dict) -> None:
    """Publish the complete baseline record only after its bytes are written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=1, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_baseline_evidence(path: Path, *, identity: dict,
                           checkpoints: dict[str, Path]) -> dict:
    """Reuse measured values only with the same inputs and exact weights."""
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("baseline evidence must be an object")
    if evidence.get("schema") != "baseline-evidence-v1":
        raise ValueError("legacy baseline lacks complete evidence; use a new run id")
    if evidence.get("evaluation_identity") != identity:
        raise ValueError("baseline evaluation/config/source identity changed")
    seeds = set(checkpoints)
    for name in ("by_seed", "steps_by_seed", "task_by_seed", "negative_by_seed",
                 "plateau_by_seed", "train_meta_by_seed", "checkpoint_sha16"):
        values = evidence.get(name)
        if not isinstance(values, dict) or set(values) != seeds:
            raise ValueError(f"baseline {name} is missing or has different seeds")
        if name in ("task_by_seed", "negative_by_seed", "plateau_by_seed", "train_meta_by_seed"):
            if any(not isinstance(value, dict) for value in values.values()):
                raise ValueError(f"baseline {name} has malformed records")
    for seed, checkpoint in checkpoints.items():
        if (not checkpoint.is_file()
                or file_sha16(checkpoint) != evidence["checkpoint_sha16"][seed]):
            raise ValueError(f"baseline checkpoint changed or missing: seed {seed}")
    return evidence
