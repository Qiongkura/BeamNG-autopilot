"""Artifact/data provenance contract for the autonomous-driving pipeline.

The project has several independently-created artifacts (Tech segmentation
runs, manual labels, shadow episodes, telemetry and scorecards).  This module
provides one small, JSON-serialisable sidecar contract without changing the
legacy NPZ payloads.  Consumers can use the manifest to reproduce splits and
identify the exact runtime/model/data provenance used for a result.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from beamng_autopilot import config

MANIFEST_VERSION = 1
LABEL_CLASSES = ["background", "asphalt", "line"]


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return a stable content hash for provenance (streamed)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_id(path: Path, root: Path | None = None) -> str:
    """Stable artifact id; unlike ``Path.name`` it cannot collide."""
    root = root or config.PROJECT_ROOT
    text = _rel(path, root)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _frame_files(path: Path) -> list[Path]:
    return sorted(path.glob("frame_*.npz"))


def _first_frame_info(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    files = _frame_files(path)
    if not files:
        return {}, {"frames": 0, "valid_frames": 0}
    try:
        import numpy as np
        with np.load(files[0], allow_pickle=True) as z:
            colour = np.asarray(z["colour"])
            label = np.asarray(z["label"])
            line = int((label == 2).sum())
            return {
                "width": int(colour.shape[1]),
                "height": int(colour.shape[0]),
                "channels": int(colour.shape[2]) if colour.ndim == 3 else None,
                "annotations": True,
                "label_classes": LABEL_CLASSES,
                "ignore_value": 255,
            }, {
                "frames": len(files),
                "valid_frames": len(files),
                "line_pixels_first_frame": line,
            }
    except Exception as exc:
        return {"read_error": str(exc)}, {"frames": len(files), "valid_frames": 0}


def _json_meta(path: Path) -> dict[str, Any]:
    meta = path / "meta.json"
    if not meta.is_file():
        return {}
    try:
        value = json.loads(meta.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def make_seg_record(path: Path, root: Path | None = None,
                    split: str = "unassigned") -> dict[str, Any]:
    root = root or config.PROJECT_ROOT
    sensor, stats = _first_frame_info(path)
    meta = _json_meta(path)
    name = path.name
    domain = "manual" if name.startswith("manual_") else "tech_truth"
    label_source = "manual" if domain == "manual" else "tech_annotation"
    if name.startswith("manual_current") or name.startswith("manual_review"):
        domain = "manual_town"
    if "mountain" in name.lower():
        scenario = "mountain"
    elif "bridge" in name.lower():
        scenario = "bridge"
    elif "town" in name.lower() or "navroute" in name.lower():
        scenario = "town"
    else:
        scenario = "unknown"
    return {
        "manifest_version": MANIFEST_VERSION,
        "run_id": stable_id(path, root),
        "artifact_type": "seg_run",
        "path": _rel(path, root),
        "task": "segmentation",
        "domain": domain,
        "scenario_id": scenario,
        "group_id": f"seg:{stable_id(path, root)}",
        "split": split,
        "environment": {
            "runtime": "tech" if domain == "tech_truth" else "unknown",
            "map": meta.get("map", "italy"),
            "vehicle": meta.get("vehicle", "etk800"),
            "seed": meta.get("seed"),
        },
        "sensor": sensor,
        "label": {
            "classes": LABEL_CLASSES,
            "ignore_value": 255,
            "source": label_source,
            "line_only": domain.startswith("manual"),
        },
        "stats": stats,
        "provenance": {
            "meta_file": (_rel(path / "meta.json", root)
                           if (path / "meta.json").is_file() else None),
            "git_commit": None,
        },
    }


def make_npz_record(path: Path, root: Path | None = None,
                    artifact_type: str = "shadow_episode") -> dict[str, Any]:
    root = root or config.PROJECT_ROOT
    info: dict[str, Any] = {"frames": 0, "valid_frames": 0}
    sensor: dict[str, Any] = {}
    version = None
    fmap_channels = None
    try:
        import numpy as np
        with np.load(path, allow_pickle=True) as z:
            n = int(z["t"].shape[0]) if "t" in z else 0
            info["frames"] = n
            info["valid_frames"] = n
            version = int(np.asarray(z["version"]).item()) if "version" in z else None
            if "rgb" in z and z["rgb"].ndim == 4:
                sensor = {"width": int(z["rgb"].shape[2]),
                          "height": int(z["rgb"].shape[1]),
                          "channels": int(z["rgb"].shape[3])}
            if "fmap" in z and z["fmap"].ndim == 4:
                fmap_channels = int(z["fmap"].shape[1])
    except Exception as exc:
        info["read_error"] = str(exc)
    name = path.name.lower()
    scenario = "town" if "town" in name else "mountain" if "mountain" in name else "unknown"
    return {
        "manifest_version": MANIFEST_VERSION,
        "run_id": stable_id(path, root),
        "artifact_type": artifact_type,
        "path": _rel(path, root),
        "task": "closed_loop" if artifact_type == "telemetry" else "e2e",
        "domain": "shadow" if artifact_type == "shadow_episode" else "closed_loop",
        "scenario_id": scenario,
        "group_id": f"episode:{stable_id(path, root)}",
        "split": "unassigned",
        "environment": {"runtime": "tech", "map": "italy", "vehicle": "etk800"},
        "sensor": {**sensor, "fmap_channels": fmap_channels,
                   "fmap_contract": "v3" if fmap_channels else "legacy"},
        "label": {"source": "semantic_prediction" if artifact_type == "shadow_episode" else None},
        "stats": info,
        "provenance": {"git_commit": None,
                        "sha256": sha256_file(path) if path.is_file() else None},
        "episode_version": version,
    }


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return validation errors; an empty list means schema basics pass."""
    required = ("manifest_version", "run_id", "artifact_type", "path",
                "task", "domain", "scenario_id", "group_id", "split")
    errors = [f"missing:{k}" for k in required if k not in record]
    if record.get("manifest_version") != MANIFEST_VERSION:
        errors.append("manifest_version")
    if not isinstance(record.get("run_id"), str) or not record.get("run_id"):
        errors.append("run_id")
    return errors


def scan_project(root: Path | None = None) -> list[dict[str, Any]]:
    """Build records for current segmentation, shadow and telemetry artifacts."""
    root = root or config.PROJECT_ROOT
    logs = root / "logs"
    records: list[dict[str, Any]] = []
    seg_root = logs / "m5_seg"
    if seg_root.is_dir():
        for path in sorted(p for p in seg_root.iterdir() if p.is_dir()):
            if _frame_files(path):
                records.append(make_seg_record(path, root))
    for path in sorted((logs / "m5_e2e").glob("shadow_fsd_*.npz")):
        records.append(make_npz_record(path, root, "shadow_episode"))
    for path in sorted((logs / "fsd_benchmark").glob("scorecard_*.json")):
        records.append(make_npz_record(path, root, "scorecard"))
    return records


def write_manifest(records: Iterable[dict[str, Any]], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = list(records)
    errors = [(r.get("path"), validate_record(r)) for r in data]
    errors = [x for x in errors if x[1]]
    if errors:
        raise ValueError(f"invalid manifest records: {errors[:3]}")
    output.write_text(json.dumps({"manifest_version": MANIFEST_VERSION,
                                  "records": data},
                                 ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return output
