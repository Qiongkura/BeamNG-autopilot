"""T14 事件协议：结构化、可恢复、可重放。

方案要求控制器、训练入口与评估器**都**往
``logs/experiments/<run_id>/events.jsonl`` 追加事件，每条至少带
``schema_version`` / ``run_id`` / ``candidate_id`` / ``dataset_id`` /
``config_hash`` / ``seed`` / ``phase`` / ``status`` / UTC 时间，适用时有
``epoch``/``step``，指标带名/值/单位/分子/分母与**缺失原因**。

两个容易做错的地方，本模块按"能失败的断言"处理：

* **半写行可识别**：JSONL 的最后一行可能是断电时的半条记录。读取时把它
  单独报成 ``unreadable``，而不是跳过或让它污染重放。
* **重启不重复画点**：单调序号 + ``(phase, status, epoch, step)`` 去重键，
  同一 run 重启后重复写入的事件在读端只保留一条。

枚举与方案第 99 行一致；``paused``/``failed`` 是异常态。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1

PHASES = ("queued", "auditing", "needs_review", "training", "evaluating",
          "rejected", "shadow_candidate", "needs_evidence", "final_testing",
          "approved_for_review", "paused", "failed")

#: 允许的状态迁移；未列出的迁移由 ``EventLog.append`` 拒绝并说明原因。
#: 看板与控制器都读同一张表，避免"某个阶段偷偷回退"。
TRANSITIONS = {
    "queued": ("auditing", "paused", "failed"),
    "auditing": ("needs_review", "training", "rejected", "paused", "failed"),
    "needs_review": ("auditing", "paused", "failed"),
    "training": ("evaluating", "failed", "paused"),
    "evaluating": ("rejected", "shadow_candidate", "needs_evidence",
                   "training", "paused", "failed"),
    "needs_evidence": ("evaluating", "training", "paused", "failed"),
    "shadow_candidate": ("final_testing", "rejected", "paused", "failed"),
    "final_testing": ("approved_for_review", "rejected", "paused", "failed"),
    "approved_for_review": ("paused", "failed"),
    "paused": ("queued", "auditing", "training", "evaluating", "failed"),
    "failed": ("queued", "paused"),
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Event:
    """一条事件。``metrics`` 是 ``{name: {value, unit, numerator?,
    denominator?, missing?}}``；缺失原因必须写在 ``missing`` 里，不能把
    缺测写成 0。"""

    run_id: str
    candidate_id: str
    dataset_id: str
    config_hash: str
    seed: int
    phase: str
    status: str
    seq: int = 0
    ts: str = field(default_factory=utc_now)
    epoch: int | None = None
    step: int | None = None
    metrics: dict = field(default_factory=dict)
    note: str = ""
    git_commit: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"unknown phase {self.phase!r}; "
                             f"valid: {list(PHASES)}")
        if not self.run_id:
            raise ValueError("run_id is required")
        if self.seed is None:
            raise ValueError("seed is required (None is not a seed)")

    def to_json(self) -> str:
        return json.dumps({
            "schema_version": self.schema_version,
            "run_id": self.run_id, "candidate_id": self.candidate_id,
            "dataset_id": self.dataset_id, "config_hash": self.config_hash,
            "seed": self.seed, "phase": self.phase, "status": self.status,
            "seq": self.seq, "ts": self.ts, "epoch": self.epoch,
            "step": self.step, "metrics": self.metrics, "note": self.note,
            "git_commit": self.git_commit,
        }, ensure_ascii=False, sort_keys=True)

    @property
    def key(self) -> tuple:
        """去重键：重启后重复画点用这个识别（序号不同也算同一个点）。"""
        return (self.phase, self.status, self.epoch, self.step,
                self.candidate_id, self.seed)


class EventLog:
    """``logs/experiments/<run_id>/events.jsonl`` 的追加与读取。"""

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "events.jsonl"

    # ---- write ---------------------------------------------------------
    def append(self, event: Event, *, enforce_transitions: bool = True
               ) -> Event:
        """追加一条事件，带回单调序号。"""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        last = self.last()
        if enforce_transitions and last is not None \
                and last.phase != event.phase:
            allowed = TRANSITIONS.get(last.phase, ())
            if event.phase not in allowed:
                raise ValueError(
                    f"phase transition {last.phase} -> {event.phase} is not "
                    f"allowed; valid from {last.phase}: {list(allowed)}")
        event.seq = 0 if last is None else last.seq + 1
        line = event.to_json() + "\n"
        # write+flush: a half-written line must be *possible* to observe
        # (the reader tolerates it), not silently buffered away by a crash.
        # If the file ends mid-record (no trailing newline) the next append
        # must first restore the boundary - otherwise this new record is
        # CONCATENATED onto the torn one and lost as well (measured: one
        # torn line swallowed the following epoch event entirely).
        needs_newline = False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                needs_newline = fh.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8") as fh:
            if needs_newline:
                fh.write("\n")
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        return event

    def last(self) -> Event | None:
        events, _ = self.read()
        return events[-1] if events else None

    # ---- read ----------------------------------------------------------
    def read(self) -> tuple[list[Event], list[str]]:
        """``(events, problems)``——问题里既有半写行也有坏 JSON。"""
        if not self.path.exists():
            return [], []
        events: list[Event] = []
        problems: list[str] = []
        raw = self.path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                blob = json.loads(line)
            except Exception as exc:            # noqa: BLE001
                problems.append(f"line {n}: unreadable ({exc})")
                continue
            try:
                events.append(Event(**{k: v for k, v in blob.items()
                                       if k in Event.__dataclass_fields__}))
            except Exception as exc:            # noqa: BLE001
                problems.append(f"line {n}: invalid event ({exc})")
        return events, problems

    def replay(self) -> dict:
        """按序号排序、去掉重启后的重复点，返回可画图的轨迹。

        ``dropped_duplicates`` 计数必须随图一起显示：看板若把重复点画两次，
        曲线会在重启处出现假的跳变。
        """
        events, problems = self.read()
        ordered = sorted(events, key=lambda e: e.seq)
        seen: set = set()
        kept: list[Event] = []
        dropped = 0
        for e in ordered:
            if e.key in seen:
                dropped += 1
                continue
            seen.add(e.key)
            kept.append(e)
        by_phase: dict[str, int] = {}
        for e in kept:
            by_phase[e.phase] = by_phase.get(e.phase, 0) + 1
        return {"events": kept, "problems": problems,
                "n_events": len(kept), "dropped_duplicates": dropped,
                "by_phase": by_phase,
                "last": kept[-1] if kept else None}

    def epochs(self, *, phase: str = "training") -> list[dict]:
        """``[{epoch, metrics, ts}]``，只取带 epoch 的事件。"""
        out = []
        for e in self.replay()["events"]:
            if e.phase != phase or e.epoch is None:
                continue
            out.append({"epoch": int(e.epoch), "metrics": dict(e.metrics),
                        "ts": e.ts, "status": e.status,
                        "step": e.step})
        return out


def metric(value=None, unit: str = "", *, numerator=None, denominator=None,
           missing: str = "") -> dict:
    """构造一条指标记录：缺测要写 ``missing``，不要塞 0。"""
    rec = {"unit": unit}
    if value is None:
        rec["value"] = None
        rec["missing"] = missing or "not measured"
    else:
        rec["value"] = float(value)
        if missing:
            rec["missing"] = missing
    if numerator is not None:
        rec["numerator"] = int(numerator)
    if denominator is not None:
        rec["denominator"] = int(denominator)
    return rec


def experiments_root(logs_dir: Path) -> Path:
    return Path(logs_dir) / "experiments"
