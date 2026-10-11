"""Archive explicit evidence files with hashes of the bytes actually exported."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def build_archive(output: Path, entries: list[tuple[Path, str]], *, metadata=None,
                  expected_sha256: dict[str, str] | None = None) -> dict:
    """Preserve missing required inputs in the manifest; refuse replacement."""
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "paper-evidence-v2", "metadata": metadata or {},
                "entries": [], "missing": [], "validation_errors": [], "complete": False}
    seen = set()
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in entries:
            if name in seen or name == "MANIFEST.json" or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"invalid/duplicate archive member: {name}")
            seen.add(name)
            try:
                data = path.read_bytes()
            except OSError as exc:
                manifest["missing"].append({"arc": name, "reason": type(exc).__name__})
                continue
            digest = hashlib.sha256(data).hexdigest()
            if expected_sha256 and name in expected_sha256 and digest != expected_sha256[name]:
                manifest["validation_errors"].append({"arc": name, "reason": "frozen input digest differs"})
            archive.writestr(name, data)
            manifest["entries"].append({"arc": name, "bytes": len(data),
                                        "sha256": digest})
        manifest["complete"] = (bool(manifest["entries"]) and not manifest["missing"]
                                and not manifest["validation_errors"])
        archive.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest
