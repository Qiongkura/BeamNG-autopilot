"""Bounded automatic truth collection and content-bound training admission.

The existing Tech generator and exporter remain the truth producers. This
module never derives labels from a candidate model or requests human labels.
Admission here is for training; it is not an independent evaluation certificate.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .auto_truth import frame_content_shas
from .credentials import read_dir_credentials


@dataclass(frozen=True)
class AutoTruthSpec:
    map_name: str = "italy"
    anchors: int = 1
    sites: int = 5
    frames: int = 2
    step_m: float = 2.0
    anchor_offset: int = 0
    attempts: int = 2
    line_convention: str = "tiers"
    timeout_s: int = 1800

    def __post_init__(self):
        for key in ("anchors", "sites", "frames", "attempts", "timeout_s"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if (isinstance(self.anchor_offset, bool)
                or not isinstance(self.anchor_offset, int) or self.anchor_offset < 0):
            raise ValueError("anchor_offset must be a non-negative integer")
        if not np.isfinite(self.step_m) or self.step_m <= 0:
            raise ValueError("step_m must be finite and positive")
        if self.line_convention not in ("symmetric", "measured", "relative", "tiers",
                                        "devdist", "pairfar"):
            raise ValueError("unknown line_convention")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def spec_from_config(cfg) -> AutoTruthSpec:
    return AutoTruthSpec(map_name=cfg.collect_map, anchors=cfg.collect_truth_anchors,
                         sites=cfg.collect_truth_sites, frames=cfg.collect_truth_frames,
                         step_m=cfg.collect_step_m,
                         anchor_offset=cfg.collect_truth_anchor_offset,
                         attempts=cfg.collect_truth_attempts,
                         line_convention=cfg.collect_truth_line_convention,
                         timeout_s=cfg.collect_timeout_s)


def scene_command(spec: AutoTruthSpec, directory: Path, *, python: str,
                  scripts: Path, attempt: int) -> list[str]:
    return [python, str(scripts / "m5_controlled_scenes.py"),
            "--map", spec.map_name, "--anchors", str(spec.anchors),
            "--sites", str(spec.sites), "--frames", str(spec.frames),
            "--step-m", str(spec.step_m), "--anchor-offset",
            str(spec.anchor_offset + attempt * spec.anchors),
            "--line-convention", spec.line_convention, "--out", str(directory)]


def export_command(spec: AutoTruthSpec, batch: Path, output: Path, *,
                   python: str, scripts: Path) -> list[str]:
    # Keep the exporter's frozen defaults, including coverage and appearance
    # gates. A failed attempt never lowers these thresholds.
    return [python, str(scripts / "m5_auto_truth_export.py"),
            "--batch", str(batch), "--out", str(output), "--map", spec.map_name]


def _digest(values) -> str:
    return hashlib.sha256("|".join(sorted(values)).encode()).hexdigest()[:16]


def qualify_package(view: Path, *, expected_map: str) -> dict:
    """Recompute every RGB/label hash; a sidecar string alone cannot admit data."""
    out = {"dir": str(view), "qualified": False, "reasons": [], "frames": 0,
           "line_free": False, "channels": {}}
    reasons = out["reasons"]
    try:
        cred = read_dir_credentials(view)
        if (not cred or cred.get("label_source") != "engine_verified"
                or cred.get("truth_verified") is not True):
            raise ValueError("content lacks verified automatic truth credentials")
        blob = json.loads(Path(cred["path"]).read_text(encoding="utf-8"))
        prov = cred["truth_provenance"]
        meta = json.loads((view.parent / "meta.json").read_text(encoding="utf-8"))
        if meta.get("map_name") != expected_map or not meta.get("source_id"):
            raise ValueError("map/source identity missing or mismatched")
        entries = meta.get("frames") or []
        proofs = blob.get("frames") or []
        paths = sorted(view.glob("frame_*.npz"))
        if not paths or len(paths) != len(entries) or len(proofs) != len(paths):
            raise ValueError("frame files, metadata and proofs disagree")
        if len({str(p.get("frame_id")) for p in proofs}) != len(proofs):
            raise ValueError("duplicate proof frame identity")
        source_shas, label_shas = [], []
        line_px = 0
        for i, (path, entry, proof) in enumerate(zip(paths, entries, proofs)):
            if path.name != f"frame_{i:05d}.npz":
                raise ValueError("non-canonical or missing frame index")
            if (view.parent / str(entry.get("path", ""))).resolve() != path.resolve():
                raise ValueError("metadata frame path mismatch")
            with np.load(path, allow_pickle=False) as data:
                rgb, lab = data["colour"], data["label"]
                if rgb.shape[:2] != lab.shape or lab.ndim != 2:
                    raise ValueError("RGB/label dimensions disagree")
                if not np.isin(lab, (0, 1, 2, 255)).all():
                    raise ValueError("unknown label class")
                src, label = frame_content_shas(rgb, lab)
                line_px += int((lab == 2).sum())
            if src != proof.get("source_image_sha") or label != proof.get("label_sha"):
                raise ValueError("frame content no longer matches verified proof")
            source_shas.append(src)
            label_shas.append(label)
        labels = prov.get("labels") or {}
        if (labels.get("source_image_sha") != _digest(source_shas)
                or labels.get("label_sha") != _digest(label_shas)):
            raise ValueError("batch content digest mismatch")
        negative = labels.get("truth_kind") == "line_free"
        if not line_px and not negative:
            raise ValueError("zero line pixels are not proof of a line-free scene")
        if negative and (line_px or not labels.get("line_free_evidence")):
            raise ValueError("line-free declaration contradicts or lacks evidence")
        out.update(frames=len(paths), line_free=negative, line_px=line_px,
                   source_id=meta["source_id"],
                   channels=blob.get("channels") or {},
                   content_digest=_digest(source_shas + label_shas))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        reasons.append(str(exc))
    out["qualified"] = not reasons
    return out


def qualify_export(directory: Path, *, expected_map: str) -> dict:
    """Only explicit exports within this attempt can become training proposals."""
    report = json.loads((directory / "export_report.json").read_text(encoding="utf-8"))
    admitted, isolated = [], list(report.get("isolated") or [])
    seen = set()
    for item in report.get("exported") or []:
        view = Path(item.get("dir") or "").resolve()
        if view in seen:
            isolated.append({"dir": str(view), "reasons": ["duplicate export directory"]})
            continue
        seen.add(view)
        if not view.is_relative_to(directory.resolve()):
            isolated.append({"dir": str(view), "reasons": ["export escapes attempt directory"]})
            continue
        check = qualify_package(view, expected_map=expected_map)
        if item.get("verified") is not True or item.get("n_frames") != check["frames"]:
            check["qualified"] = False
            check["reasons"].append("export summary/proof mismatch")
        (admitted if check["qualified"] else isolated).append(check)
    return {"qualified": bool(admitted), "admitted": admitted, "isolated": isolated,
            "runs": [r["dir"] for r in admitted],
            "frames": sum(r["frames"] for r in admitted),
            "line_free_frames": sum(r["frames"] for r in admitted if r["line_free"]),
            "channels_unknown": sum(not r["channels"] for r in admitted)}


def _save(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
    temp.replace(path)


def run_stage(command, log_path: Path, *, timeout_s: float, heartbeat,
              cwd: Path, process_factory=None) -> dict:
    """Run a single owned child with a live lease and a monotonic wall cap."""
    factory = process_factory or subprocess.Popen
    started = time.monotonic()
    proc = None
    result = {"rc": 125}
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = factory(command, cwd=str(cwd), stdout=log,
                           stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            while proc.poll() is None:
                if not heartbeat():
                    raise RuntimeError("machine lease lost")
                if time.monotonic() - started >= timeout_s:
                    result["rc"] = 124
                    raise TimeoutError("stage wall deadline exceeded")
                time.sleep(0.25)
            result["rc"] = int(proc.returncode)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        result["elapsed_s"] = time.monotonic() - started
    return result


def run_collection(spec: AutoTruthSpec, directory: Path, *, python: str,
                   scripts: Path, runner, identity: str = "") -> dict:
    """Persist attempts before launch; interruption consumes a bounded attempt.

    ``runner(command, log_path, timeout_s, owns_game)`` returns rc/cleanup.
    The caller owns the existing machine lease and performs resource preflight.
    A cleanup failure stops further work, including the offline exporter.
    """
    state_path = directory / "state.json"
    fingerprint = hashlib.sha256((spec.digest() + identity).encode()).hexdigest()
    state = {"schema": "auto-truth-collection-v1", "fingerprint": fingerprint,
             "spec": asdict(spec), "attempts": []}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise ValueError("collection recipe/source changed; use a new run id")
    for old in state["attempts"]:
        if old.get("status") == "admitted":
            try:
                check = qualify_export(Path(old["export_dir"]), expected_map=spec.map_name)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                return {"ok": False, "status": "evidence_changed", "reasons": [str(exc)]}
            if not check["qualified"]:
                return {"ok": False, "status": "evidence_changed", **check}
            return {"ok": True, "status": "reused_verified", **check}

    def certify(rec, export):
        cmd = export_command(spec, Path(rec["raw_dir"]), export,
                             python=python, scripts=scripts)
        rec["export_dir"] = str(export)
        rec["export_command"] = cmd
        _save(state_path, state)
        rec["export"] = runner(cmd, export.parent / (export.name + ".log"), spec.timeout_s, False)
        if rec["export"].get("blocked"):
            rec["status"] = "resource_blocked_before_export"
            _save(state_path, state)
            return {"ok": False, "status": rec["status"], "attempts": state["attempts"]}
        if rec["export"].get("cleanup", {}).get("closed") is not True:
            rec["status"] = "cleanup_unproven"
            _save(state_path, state)
            return {"ok": False, "status": rec["status"], "attempts": state["attempts"]}
        try:
            check = qualify_export(export, expected_map=spec.map_name)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            check = {"qualified": False, "reasons": [str(exc)]}
        rec["qualification"] = check
        if rec["export"].get("rc") == 0 and check["qualified"]:
            rec["status"] = "admitted"
            _save(state_path, state)
            return {"ok": True, "status": "admitted", **check}
        rec["status"] = "isolated"
        _save(state_path, state)
        return None

    # A completed, cleaned collection can resume its offline certification.
    # Use a new export directory, so partial files from the old export remain.
    for old in state["attempts"]:
        collection = old.get("collection") or {}
        if (old.get("status") in ("running", "resource_blocked_before_export")
                and collection.get("rc") == 0
                and collection.get("cleanup", {}).get("closed") is True
                and old.get("raw_dir")):
            old["export_resumes"] = int(old.get("export_resumes", 0)) + 1
            export = Path(old["raw_dir"]).parent / f"qualified_resume_{old['export_resumes']:02d}"
            result = certify(old, export)
            if result is not None:
                return result
    def launched_attempts():
        return sum(not r.get("collection", {}).get("blocked") for r in state["attempts"])

    while launched_attempts() < spec.attempts:
        index = len(state["attempts"])
        collection_index = launched_attempts()
        attempt = directory / f"attempt_{index:02d}"
        attempt.mkdir(parents=True, exist_ok=False)
        batch, export = attempt / "raw", attempt / "qualified"
        rec = {"index": index, "status": "running", "export_dir": str(export), "raw_dir": str(batch)}
        state["attempts"].append(rec)
        _save(state_path, state)
        collect = scene_command(spec, batch, python=python, scripts=scripts, attempt=collection_index)
        rec["collect_command"] = collect
        rec["collection"] = runner(collect, attempt / "collect.log", spec.timeout_s, True)
        if rec["collection"].get("blocked"):
            rec["status"] = "resource_blocked"
            _save(state_path, state)
            return {"ok": False, "status": rec["status"], "attempts": state["attempts"]}
        if rec["collection"].get("cleanup", {}).get("closed") is not True:
            rec["status"] = "cleanup_unproven"
            _save(state_path, state)
            return {"ok": False, "status": rec["status"], "attempts": state["attempts"]}
        if rec["collection"].get("rc") == 0:
            result = certify(rec, export)
            if result is not None:
                return result
        rec["status"] = "isolated"
        _save(state_path, state)
    return {"ok": False, "status": "attempt_budget_exhausted", "attempts": state["attempts"]}
