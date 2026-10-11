"""Read explicit, content-pinned paper inputs without discovering live logs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def figure_input_specs(root: Path, manifest: Path, figure: str) -> list[dict]:
    """Validate explicit input locations without discovering or reading logs."""
    root = root.resolve()
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema") != "paper-inputs-v1":
        raise ValueError("unsupported paper input manifest")
    spec = (record.get("figures") or {}).get(figure)
    if not isinstance(spec, dict) or not spec.get("scope"):
        raise ValueError(f"missing figure scope: {figure}")
    entries = spec.get("inputs")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"no frozen inputs: {figure}")
    seen = set()
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("role"):
            raise ValueError("malformed frozen input")
        relative = Path(entry.get("path", ""))
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("paper input escapes project")
        if path in seen:
            raise ValueError("duplicate frozen input")
        seen.add(path)
        digest = entry.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid frozen input digest")
        result.append({**entry, "scope": spec["scope"]})
    return result


def read_figure_inputs(root: Path, manifest: Path, figure: str) -> list[dict]:
    """Refuse missing/changed inputs; adding logs cannot change a frozen figure."""
    result = []
    for entry in figure_input_specs(root, manifest, figure):
        data = (root / entry["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"paper input changed: {entry['path']}")
        result.append({**entry, "data": json.loads(data)})
    return result
