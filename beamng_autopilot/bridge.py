"""GUI <-> autopilot control bridge (JSON command file).

The launcher GUI (scripts/m5_launcher.py) writes one-shot commands to
logs/autopilot_ctl.json; the autopilot loop (m5_autopilot.py) polls the
file and consumes commands exactly like its global hotkeys, so a button
press in the GUI does the same thing as F9 / F10 / F11 in the game.
Parameterized commands such as ``set_speed`` carry a numeric ``value``
field; ``poll`` returns them as ``(cmd, value)`` pairs.

Only the latest command is kept (later commands overwrite earlier ones),
which is fine for the toggle-style commands the GUI sends.  A monotonic
``seq`` watermark lets the autopilot ignore commands that were written
before it started.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from . import config


def ctl_path() -> Path:
    return config.LOGS_DIR / "autopilot_ctl.json"


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class ControlBridge:
    """Small JSON command channel between the GUI and the autopilot."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or ctl_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _write(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)

    def current_seq(self) -> int:
        """Seq of the command currently in the file (0 when empty)."""
        data = _read(self.path)
        if isinstance(data, dict) and isinstance(data.get("seq"), int):
            return data["seq"]
        return 0

    def send(self, cmd: str, value: float | None = None) -> bool:
        """Write one command; returns True on success."""
        with self._lock:
            try:
                seq = self.current_seq() + 1
                data = {"seq": seq, "cmd": str(cmd), "ts": time.time()}
                if value is not None:
                    data["value"] = float(value)
                self._write(data)
                return True
            except Exception:
                return False

    def poll(self, seen: int) -> tuple[list[tuple[str, float | None]], int]:
        """Return (new commands, updated watermark) for the autopilot loop.

        Each command is a ``(cmd, value)`` pair; ``value`` is ``None`` for
        toggle-style commands and numeric for commands such as ``set_speed``.
        """
        with self._lock:
            data = _read(self.path)
            if (isinstance(data, dict)
                    and isinstance(data.get("seq"), int)
                    and data["seq"] > seen
                    and isinstance(data.get("cmd"), str)):
                return [(data["cmd"], data.get("value"))], data["seq"]
            return [], seen

    def clear(self) -> None:
        """Remove the command file (called on autopilot exit)."""
        with self._lock:
            try:
                self.path.unlink(missing_ok=True)
            except Exception:
                pass
