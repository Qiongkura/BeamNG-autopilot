"""后台控制器：配置、资源门、单实例锁、任务状态机（离线可 dry-run）。

方案 §"后台运行规则"要求：任务计划程序只负责在空闲时间启动/恢复，控制器
自己持**单实例锁**，并支持正常退出、断电恢复、磁盘门槛、每日 GPU 时间上限、
最大连续运行时长与温度/显存监测；先 ``--once --dry-run`` 验证将执行的命令与
数据流，再允许常驻。

本模块只做**决策与记录**，不自己训练、不自己采集：``plan_once`` 返回要执行
的命令列表（dry-run 直接打印），``run_once`` 通过注入的 runner 执行。游戏
采集默认关闭（``collect=off``），因为是否允许无人值守启动 Tech 尚待用户
指定。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
@dataclass
class LoopConfig:
    """控制器配置；``config_hash`` 进每条事件，保证候选与配置一一对应。"""

    dataset_manifest: str = ""
    experiments_root: str = "logs/experiments"
    collect: str = "off"                 # off | tech（默认 off，待用户授权）
    daily_gpu_minutes: float = 120.0
    max_wall_minutes: float = 180.0      # 单次连续运行上限
    window_start_hour: int = 0           # 空闲窗口（本地时间，闭区间起点）
    window_end_hour: int = 24
    pause_while_user_active: bool = True
    min_free_vram_mb: float = 2048.0
    min_free_disk_gb: float = 20.0
    max_candidates: int = 6
    max_rounds_without_gain: int = 3
    seeds: tuple = (42, 43, 44)
    dry_run: bool = True
    python: str = ""
    # ---- 数据与训练配方（`run --no-dry-run` 复用 `rounds` 时需要）---------
    # 这些**不是**调度参数而是"这一轮实验的输入"：放在配置里，配置哈希因此
    # 覆盖数据路径与配方，事件里能对齐到具体输入。
    runs: tuple = ()
    baseline_runs: tuple = ()
    eval_runs: tuple = ()
    proposals: str = ""
    rounds: int = 2
    epochs: int = 3
    batch: int = 4
    lr: float = 1e-3
    allow_road_only: bool = False
    equal_steps: bool = True
    trainer_script: str = "m5_train_seg.py"

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @classmethod
    def load(cls, path: Path | str) -> "LoopConfig":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in blob.items()
                 if k in cls.__dataclass_fields__}
        if "seeds" in known:
            known["seeds"] = tuple(known["seeds"])
        return cls(**known)

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False),
                     encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
@dataclass
class ResourceState:
    now_hour: int
    free_vram_mb: float
    free_disk_gb: float
    gpu_minutes_today: float
    user_active: bool = False
    temperature_c: float | None = None


def resource_gate(cfg: LoopConfig, st: ResourceState) -> dict:
    """返回 ``{allowed, reasons, warnings}``；原因必须逐条可读。"""
    reasons: list[str] = []
    warnings: list[str] = []
    if cfg.dry_run:
        warnings.append("dry_run: nothing will be executed")
    if not (cfg.window_start_hour <= st.now_hour < cfg.window_end_hour):
        if cfg.window_start_hour == 0 and cfg.window_end_hour == 24:
            pass
        else:
            reasons.append(f"outside the idle window "
                           f"[{cfg.window_start_hour},{cfg.window_end_hour}) "
                           f"- now {st.now_hour}")
    if st.gpu_minutes_today >= cfg.daily_gpu_minutes:
        reasons.append(f"daily GPU budget exhausted "
                       f"({st.gpu_minutes_today:.0f} >= "
                       f"{cfg.daily_gpu_minutes:.0f} min)")
    if st.free_vram_mb < cfg.min_free_vram_mb:
        reasons.append(f"free VRAM {st.free_vram_mb:.0f} MB < "
                       f"{cfg.min_free_vram_mb:.0f} MB")
    if st.free_disk_gb < cfg.min_free_disk_gb:
        reasons.append(f"free disk {st.free_disk_gb:.1f} GB < "
                       f"{cfg.min_free_disk_gb:.1f} GB")
    if cfg.pause_while_user_active and st.user_active:
        reasons.append("user is using the machine: paused by configuration")
    if cfg.collect == "tech":
        warnings.append("collect=tech requires explicit authorisation; the "
                        "console must be free before starting the game")
    return {"allowed": not reasons, "reasons": reasons,
            "warnings": warnings, "checked_at": _utc()}


def probe_resources(*, vram_probe=None, disk_path: str | Path = ".",
                    gpu_minutes_today: float = 0.0,
                    user_active: bool = False) -> ResourceState:
    """读当前资源。``vram_probe`` 可注入（测试与无 GPU 机器）。"""
    free_vram = 0.0
    temp = None
    if vram_probe is not None:
        free_vram, temp = vram_probe()
    else:
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                free_vram = free / 2 ** 20
        except Exception:                    # noqa: BLE001
            free_vram = 0.0
    try:
        import shutil
        du = shutil.disk_usage(str(disk_path))
        free_disk = du.free / 2 ** 30
    except Exception:                        # noqa: BLE001
        free_disk = 0.0
    return ResourceState(now_hour=time.localtime().tm_hour,
                         free_vram_mb=free_vram, free_disk_gb=free_disk,
                         gpu_minutes_today=gpu_minutes_today,
                         user_active=user_active, temperature_c=temp)


# ---------------------------------------------------------------------------
class InstanceLock:
    """单实例锁：第二个控制器**不能**同时跑（方案 §后台运行规则）。"""

    def __init__(self, path: Path | str, *, stale_s: float = 6 * 3600):
        self.path = Path(path)
        self.stale_s = float(stale_s)

    def acquire(self, *, pid: int | None = None) -> dict:
        pid = os.getpid() if pid is None else int(pid)
        was_present = self.path.exists()
        if was_present:
            try:
                blob = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:                # noqa: BLE001
                blob = {}
            age = time.time() - float(blob.get("t", 0) or 0)
            holder = int(blob.get("pid", -1))
            if age < self.stale_s and holder == pid:
                return {"acquired": True, "already_held": True,
                        "holder": blob}
            if age < self.stale_s and _alive(holder):
                return {"acquired": False, "already_held": False,
                        "reason": f"lock held by pid {holder} "
                                  f"({age:.0f}s old): a second controller must "
                                  f"not run concurrently",
                        "holder": blob}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"pid": pid, "t": time.time(), "host": os.environ.get(
            "COMPUTERNAME", "")}
        self.path.write_text(json.dumps(blob), encoding="utf-8")
        return {"acquired": True, "already_held": False, "holder": blob,
                "recovered_stale": was_present}

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _alive(pid: int) -> bool:
    """进程是否活着；**任何不确定都按"活着"返回**。

    实测坑：中文 Windows 的 ``tasklist`` 输出不是 UTF-8，``text=True`` 会让
    读取线程抛 UnicodeDecodeError，stdout 变空 → pid 判定为"已死" → 锁被
    第二个控制器抢走。所以显式 ``errors="replace"``，并且只有在返回码 0 且
    输出可信时才敢说"死了"。
    """
    if pid <= 0:
        return False
    try:
        import subprocess
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=20)
        if out.returncode != 0:
            return True
        return str(pid) in (out.stdout or "")
    except Exception:                        # noqa: BLE001
        return True


# ---------------------------------------------------------------------------
@dataclass
class PlanAction:
    """一个待执行动作：命令、理由、预算消耗与状态机目标阶段。"""

    name: str
    cmd: list
    reason: str
    phase: str
    gpu_minutes: float = 0.0
    optional: bool = False

    def as_dict(self) -> dict:
        return {"name": self.name, "cmd": [str(c) for c in self.cmd],
                "reason": self.reason, "phase": self.phase,
                "gpu_minutes": self.gpu_minutes, "optional": self.optional}


@dataclass
class LoopPlan:
    run_id: str
    config_hash: str
    actions: list = field(default_factory=list)
    blocked: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"run_id": self.run_id, "config_hash": self.config_hash,
                "actions": [a.as_dict() for a in self.actions],
                "blocked": list(self.blocked), "notes": list(self.notes),
                "dry_run": True}


def plan_once(cfg: LoopConfig, *, run_id: str, candidate_id: str,
              dataset_id: str, python: str, script_dir: Path,
              seed: int, extra_train_args: list | None = None) -> LoopPlan:
    """把"本轮要做什么"渲染成命令清单（dry-run 打印这些，不执行）。"""
    py = python or cfg.python or "python"
    out_dir = Path(cfg.experiments_root) / run_id / "candidates" / candidate_id
    train = [py, str(Path(script_dir) / "m5_train_seg.py"),
             "--dataset-manifest", cfg.dataset_manifest,
             "--dataset-id", dataset_id,
             "--run-id", run_id, "--candidate-id", candidate_id,
             "--events", str(Path(cfg.experiments_root) / run_id
                             / "events.jsonl"),
             "--out", str(out_dir), "--seed", str(seed)]
    if extra_train_args:
        train += list(extra_train_args)
    acts = [PlanAction(name="train_candidate", cmd=train, phase="training",
                       reason="反向传播更新权重（独立目录，checkpoint 可续）",
                       gpu_minutes=2.0)]
    acts.append(PlanAction(
        name="evaluate_candidate", phase="evaluating", gpu_minutes=1.0,
        cmd=[py, str(Path(script_dir) / "m5_seg_autoloop.py"),
             "evaluate", "--run-id", run_id, "--candidate-id", candidate_id,
             "--pairings", str(Path(cfg.experiments_root) / run_id
                               / f"pairings_{candidate_id}.json"),
             "--hard-gate", str(Path(cfg.experiments_root) / run_id
                                / f"hard_gate_{candidate_id}.json")],
        reason=("四层评估：像素/候选/几何(UNKNOWN)/性能；成对输入与硬门槛值由"
                "评估前的产物文件提供，评估本身不猜数")))
    if cfg.collect == "tech":
        acts.append(PlanAction(
            name="collect_tech", phase="auditing", gpu_minutes=5.0,
            optional=True,
            cmd=[py, str(Path(script_dir) / "m5_collect_seg_ring.py"),
                 "--attach", "--runtime", "tech", "--frames", "30",
                 "--step-m", "2.0", "--follow-road",
                 "--out", str(Path("logs/m5_seg") / f"{run_id}_collect")],
            reason="需要用户授权；采集前必须确认没有驾驶会话在控制车辆"))
    return LoopPlan(run_id=run_id, config_hash=cfg.hash(), actions=acts,
                    notes=[f"collect={cfg.collect}",
                           f"dry_run={cfg.dry_run}",
                           f"seeds={list(cfg.seeds)}"])


# ---------------------------------------------------------------------------
def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class RoundRecord:
    """一轮的历史，用于"连续 N 轮无收益就停止"与重放确定性。"""

    round_index: int
    candidate_id: str
    decision: str
    reasons: list = field(default_factory=list)
    ts: str = field(default_factory=_utc)
    gain: float | None = None


def should_stop(cfg: LoopConfig, history: list[RoundRecord], *,
                gpu_minutes_today: float, candidates_used: int) -> dict:
    """停止条件：预算耗尽、连续无收益、或没有可提议的因子。"""
    reasons: list[str] = []
    streak = 0
    for r in reversed(history):
        if r.decision in ("rejected", "needs_evidence") and not r.gain:
            streak += 1
        else:
            break
    if streak >= cfg.max_rounds_without_gain:
        reasons.append(f"{streak} consecutive rounds without a credible gain "
                       f"(limit {cfg.max_rounds_without_gain})")
    if gpu_minutes_today >= cfg.daily_gpu_minutes:
        reasons.append("daily GPU budget exhausted")
    if candidates_used >= cfg.max_candidates:
        reasons.append(f"candidate budget exhausted "
                       f"({candidates_used}/{cfg.max_candidates})")
    return {"stop": bool(reasons), "reasons": reasons,
            "no_gain_streak": streak}


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
