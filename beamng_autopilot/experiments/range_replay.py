"""Hashed Tech distance inputs and serial, same-input CPU comparisons.

No game connection or actuator calls in replay. These are algorithm witnesses,
not independent pavement/paint truth or proof that a path is safe to execute.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from beamng_autopilot.config import PROJECT_ROOT
from beamng_autopilot.perception import LidarClusterTracker, Obstacle
from .metrics import sanitize_numbers
from .reproducibility import source_snapshot

SCHEMA = "tech-range-replay-v1"
MAX_REPLAY_POINTS = 2_000_000


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _primitive(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        value = {key: _primitive(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [_primitive(item) for item in value]
    clean, bad = sanitize_numbers(value)
    if bad:
        raise ValueError(f"nonfinite replay fields: {bad}")
    return clean


def _validate_sample(sample):
    pos = np.asarray(sample["pos"], dtype=float)
    if (pos.shape != (3,) or not np.isfinite(pos).all()
            or not sample["ego_half_len"] > 0 or not sample["ego_half_width"] > 0
            or not sample["radius"] > 0):
        raise ValueError("invalid range pose/extents/radius")
    if sample["heading"] is not None and not np.isfinite(sample["heading"]):
        raise ValueError("invalid range heading")
    hits = np.asarray(sample["ray_hits"], dtype=float)
    if hits.size and (hits.ndim != 2 or hits.shape[1] != 2 or not np.isfinite(hits).all()):
        raise ValueError("invalid range ray hits")


def write_replay(directory, payload, *, pos, ego_half_len, ego_half_width) -> dict:
    """Exclusive capture; the complete manifest is the LAST published file."""
    directory = Path(directory)
    cloud = np.asarray(payload.cloud)
    if (cloud.dtype.kind not in "fiu" or cloud.ndim != 2 or cloud.shape[1] != 3
            or not 4 <= len(cloud) <= MAX_REPLAY_POINTS
            or not np.isfinite(cloud).all()):
        raise ValueError("replay needs a finite Nx3 cloud with 4..2000000 points")
    source = source_snapshot(PROJECT_ROOT)
    if not source["complete"]:
        raise ValueError("source identity incomplete")
    sample = _primitive({"pos": np.asarray(pos, float).tolist(),
                         "ego_half_len": float(ego_half_len),
                         "ego_half_width": float(ego_half_width),
                         "heading": payload.heading, "radius": payload.radius,
                         "lua_obstacles": [asdict(ob) for ob in payload.lua_obstacles],
                         "ray_hits": list(payload.ray_hits), "meta": dict(payload.meta)})
    _validate_sample(sample)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "cloud.npy").open("xb") as stream:
        np.save(stream, cloud, allow_pickle=False)
    with (directory / "sample.json").open("x", encoding="utf-8") as stream:
        json.dump(sample, stream, ensure_ascii=False, indent=2)
    manifest = {"schema": SCHEMA, "complete": True,
                "coordinate_frame": "world_xyz_metres", "points": len(cloud),
                "shape": list(cloud.shape), "dtype": str(cloud.dtype),
                "captured_wall_t": time.time(), "source": source,
                "files": {name: {"sha256": _sha(directory / name),
                                 "bytes": (directory / name).stat().st_size}
                          for name in ("cloud.npy", "sample.json")}}
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return {"status": "captured", "directory": str(directory.resolve()),
            "manifest_sha256": _sha(directory / "manifest.json"),
            "captured_wall_t": manifest["captured_wall_t"], "points": len(cloud)}


def load_replay(directory):
    """Reject missing, partial, corrupted or nonfinite evidence; never use pickle."""
    from beamng_autopilot_tech.providers import _RawRange

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or manifest.get("complete") is not True
            or manifest.get("coordinate_frame") != "world_xyz_metres"
            or not (manifest.get("source") or {}).get("complete")
            or not (manifest.get("source") or {}).get("sha256")):
        raise ValueError("incomplete or unsupported range replay")
    for name in ("cloud.npy", "sample.json"):
        record = manifest["files"][name]
        path = directory / name
        if path.stat().st_size != record["bytes"] or _sha(path) != record["sha256"]:
            raise ValueError(f"range replay digest mismatch: {name}")
    cloud = np.load(directory / "cloud.npy", allow_pickle=False)
    if (cloud.dtype.kind not in "fiu" or cloud.shape != tuple(manifest["shape"])
            or str(cloud.dtype) != manifest["dtype"]
            or cloud.ndim != 2 or cloud.shape[1] != 3
            or len(cloud) != manifest["points"] or not 4 <= len(cloud) <= MAX_REPLAY_POINTS
            or not np.isfinite(cloud).all()):
        raise ValueError("invalid range cloud")
    sample = _primitive(json.loads((directory / "sample.json").read_text(encoding="utf-8")))
    _validate_sample(sample)
    obstacles = []
    for fields in sample["lua_obstacles"]:
        fields = dict(fields)
        for name in ("axis", "velocity"):
            if fields.get(name) is not None:
                fields[name] = np.asarray(fields[name], dtype=float)
        obstacles.append(Obstacle(**fields))
    payload = _RawRange(cloud=cloud, lua_obstacles=obstacles,
                        ray_hits=[tuple(point) for point in sample["ray_hits"]],
                        heading=sample["heading"], radius=sample["radius"], meta=sample["meta"])
    return payload, sample, manifest


def _reference_process(path: Path, expected_sha256: str):
    """Load ONLY the verified local reference method, never unpinned discovery."""
    import beamng_autopilot_tech.providers as providers

    if _sha(path) != expected_sha256:
        raise ValueError("reference source hash mismatch")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "TechRangeProvider")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "process")
    namespace = dict(vars(providers))
    # The reference method still uses the unchanged shared math helpers.
    # Freeze the FULL source too; report the scope rather than claiming a
    # whole historical runtime has been reconstructed by this method replay.
    exec("from __future__ import annotations\n" + ast.get_source_segment(source, method), namespace)
    return namespace["process"]


def compare_replay(directory, *, reference_source, reference_sha256,
                   rounds: int = 2, iterations: int = 10, warmup: int = 2) -> dict:
    """Serial CPU rounds; compare every output and retain all elapsed samples.

    Each call starts a fresh tracker. This compares single-frame geometry and
    processing, not multi-frame tracking or closed-loop behaviour.
    """
    from beamng_autopilot_tech.providers import TechRangeProvider

    if rounds < 2 or iterations < 1 or warmup < 0:
        raise ValueError("need >=2 rounds, positive iterations and nonnegative warmup")
    payload, sample, manifest = load_replay(directory)
    before = source_snapshot(PROJECT_ROOT)
    if not before["complete"] or before["sha256"] != manifest["source"]["sha256"]:
        raise ValueError("replay capture/current source differs; use a new explicit experiment")
    legacy = _reference_process(Path(reference_source), reference_sha256)
    expected = None
    results = []
    for round_index in range(rounds):
        record = {"round": round_index, "order": [], "timings_ms": {}, "p95_ms": {}}
        modes = ("reference", "current") if round_index % 2 == 0 else ("current", "reference")
        for mode in modes:
            timings = []
            record["order"].append(mode)
            for i in range(warmup + iterations):
                provider = TechRangeProvider.__new__(TechRangeProvider)
                provider._ego_half_len = sample["ego_half_len"]
                provider._ego_half_w = sample["ego_half_width"]
                provider._lidar_tracker = LidarClusterTracker()
                # Shared registry boxes can be merged/mutated. Start each
                # invocation with identical fresh boxes and metadata.
                local = copy.copy(payload)
                local.lua_obstacles = copy.deepcopy(payload.lua_obstacles)
                local.meta = copy.deepcopy(payload.meta)
                call = legacy if mode == "reference" else TechRangeProvider.process
                started = time.perf_counter()
                output = call(provider, local, sample["pos"])
                elapsed = (time.perf_counter() - started) * 1000.0
                signature = _primitive({"obstacles": [asdict(ob) for ob in output.obstacles],
                                        "ray_hits": output.ray_hits})
                if output.meta.get("source_errors"):
                    raise ValueError(f"replay processing failed: {output.meta['source_errors']}")
                if expected is None:
                    expected = signature
                elif signature != expected:
                    raise ValueError("reference/current single-frame geometry differs")
                if i >= warmup:
                    timings.append(elapsed)
            record["timings_ms"][mode] = timings
            record["p95_ms"][mode] = float(np.percentile(timings, 95))
        results.append(record)
    after = source_snapshot(PROJECT_ROOT)
    if not after["complete"] or before["sha256"] != after["sha256"]:
        raise ValueError("source changed during comparison")
    return {"schema": "tech-range-cpu-comparison-v1", "source": before,
            "source_unchanged": True, "input_manifest_sha256": _sha(Path(directory) / "manifest.json"),
            "input_points": len(payload.cloud), "reference_source": str(Path(reference_source).resolve()),
            "reference_sha256": reference_sha256, "rounds": results,
            "best_p95_ms": {mode: min(r["p95_ms"][mode] for r in results)
                            for mode in ("reference", "current")},
            "outputs_equal": True, "compared_calls": rounds * 2 * (warmup + iterations),
            "n_obstacles": len(expected["obstacles"]), "n_ray_hits": len(expected["ray_hits"]),
            "scope": "same-input, fresh-tracker single-frame CPU processing; no whole-runtime equivalence",
            "independent_safety_truth": "UNKNOWN"}
