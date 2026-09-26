"""后台控制器：配置、资源门、单实例锁、任务状态机（离线可 dry-run）。

方案 §"后台运行规则"要求：任务计划程序只负责在空闲时间启动/恢复，控制器
自己持**单实例锁**，并支持正常退出、断电恢复、磁盘门槛、每日 GPU 时间上限、
最大连续运行时长与温度/显存监测；先 ``--once --dry-run`` 验证将执行的命令与
数据流，再允许常驻。

本模块只做**决策与记录**，不自己训练、不自己采集：``plan_once`` 返回要执行
的命令列表（dry-run 直接打印），``run_once`` 通过注入的 runner 执行。游戏
采集的**授权与前置检查口径**在 ``experiments/collection.py``：用户 2026-09-25
授权"允许无人值守启动 Tech 采集"，但要过前置检查（游戏已在跑就不抢会话）与
采集后的身份审计（缺 ``map_name``/``source_id`` 一律拒收）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
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
    # 每日 GPU 时长上限（分钟）。**<= 0 = 不设上限**。
    # 用户 2026-09-25 明确："不用限制 120min，只要我没说你就可以用"——
    # 所以工作配置里改成不设上限；上限机制本身留着（要收紧时填数字就行）。
    daily_gpu_minutes: float = 0.0
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
    # 研究臂（弱监督）：漆线真值来源逐 run 指定
    # （"RUN=SOURCE"）。给了就开启 line 通道训练，但判定记
    # research_only、不允许晋级（真值不完整）。
    paint_sources: tuple = ()
    research_arm: bool = False
    # ---- 无人值守采集（用户 2026-09-25 授权；见 experiments/collection.py）---
    # 这些进 config_hash：采集参数一改，事件里的 config_hash 就跟着变，候选与
    # 配置仍然一一对应。
    collect_frames: int = 30
    collect_step_m: float = 2.0
    collect_roles: tuple = ("front_main", "front_fisheye", "pillar_left",
                            "pillar_right")
    collect_follow_road: bool = True
    collect_save_annotation: bool = True
    collect_step: int = 10
    collect_map: str = "italy"
    collect_teleport: tuple = ()        # (x, y, yaw_deg)：换没采过的路段
    collect_runtime: str = "tech"
    collect_timeout_s: int = 1800

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
        if "collect_roles" in known:
            known["collect_roles"] = tuple(known["collect_roles"])
        if "paint_sources" in known:
            known["paint_sources"] = tuple(known["paint_sources"])
        if "collect_teleport" in known:
            known["collect_teleport"] = tuple(known["collect_teleport"])
        return cls(**known)

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False),
                     encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
def _cap_enabled(minutes) -> bool:
    """上限是否生效：``<= 0`` = 不设上限（用户 2026-09-25 的指示）。"""
    try:
        return float(minutes) > 0.0
    except (TypeError, ValueError):
        return False


@dataclass
class ResourceState:
    now_hour: int
    free_vram_mb: float
    free_disk_gb: float
    gpu_minutes_today: float
    #: 三态：True=测到用户在用 / False=测到没人 / **None=信号未接入**（UNKNOWN）。
    #: 方案 §8.2 明令"不能用写死的 user_active=false 冒充检测"，所以"没接入"
    #: 必须是 None，不能落到 False。
    user_active: bool | None = None
    temperature_c: float | None = None
    #: 原始检测值（秒）与信号来源：事后能核对"当时凭什么这么判"
    user_idle_s: float | None = None
    user_activity_source: str = "not connected"


#: 多久没有键鼠输入算"用户在用机器"（秒）。取 2 分钟：比"看一眼屏幕"长，
#: 比"出门一趟"短。
USER_IDLE_ACTIVE_S = 120.0


def _raw_input_idle() -> tuple | None:
    """``(idle_s, tick, dwTime)``；读不到返回 None。32 位 tick 要处理回绕。"""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        info = _LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        tick = int(ctypes.windll.kernel32.GetTickCount())
        idle = ((tick - int(info.dwTime)) & 0xFFFFFFFF) / 1000.0
        return idle, tick, int(info.dwTime)
    except Exception:                          # noqa: BLE001
        return None


def last_input_idle_seconds(*, gap_s: float = 0.6) -> float | None:
    """距上次键盘/鼠标输入的秒数；**信号不可用时返回 None（未接入）**。

    两个坑都踩过（本机实测）：

    * ``GetLastInputInfo`` 给的是 32 位 tick，必须与 ``GetTickCount()`` 配对并
      处理 49.7 天回绕；混用 64 位版本会算出巨大偏差；
    * **接口在无交互桌面的会话里会把 ``dwTime`` 直接返回成当前 tick**（实测
      idle 恒为 0.0s）——看起来像"用户一直在打字"。所以做一次两采样校验：
      若 ``dwTime`` 与 tick 同步推进（差值不变且都等于当前 tick），说明这个
      会话拿不到真实输入时间，返回 None，而不是报一个假的"用户在用"。
    """
    a = _raw_input_idle()
    if a is None:
        return None
    if gap_s > 0:
        time.sleep(gap_s)
        b = _raw_input_idle()
        if b is None:
            return None
        dt_ms = max(1.0, (b[1] - a[1]) * 1000.0 if b[1] >= a[1]
                    else (b[1] + 0x100000000 - a[1]) * 1000.0)
        tracks_clock = abs(dt_ms - gap_s * 1000.0) < 250.0
        if tracks_clock and b[2] == b[1] and a[2] == a[1]:
            return None            # 信号不可用（dwTime 跟着 tick 走）
        a = b
    return max(0.0, float(a[0]))


def user_activity_probe(*, idle_probe=None,
                        idle_threshold_s: float = USER_IDLE_ACTIVE_S) -> dict:
    """真实用户活动信号：``{active, idle_s, source, why}``。

    ``active`` 三态（True/False/None）。``idle_probe`` 可注入（测试/无 GUI）。
    """
    idle_probe = idle_probe or last_input_idle_seconds
    idle = idle_probe()
    if idle is None:
        return {"active": None, "idle_s": None, "source": "not connected",
                "why": ("no input-idle signal available: user activity is "
                        "UNKNOWN, not 'no user'")}
    return {"active": float(idle) < float(idle_threshold_s),
            "idle_s": round(float(idle), 1),
            "source": "GetLastInputInfo (keyboard/mouse idle)",
            "why": (f"idle {float(idle):.0f}s vs active threshold "
                    f"{float(idle_threshold_s):.0f}s")}


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
    _cap = _cap_enabled(cfg.daily_gpu_minutes)
    if _cap and st.gpu_minutes_today >= cfg.daily_gpu_minutes:
        reasons.append(f"daily GPU budget exhausted "
                       f"({st.gpu_minutes_today:.0f} >= "
                       f"{cfg.daily_gpu_minutes:.0f} min)")
    elif not _cap:
        warnings.append(
            f"daily GPU budget: 不设上限（今天已用 "
            f"{st.gpu_minutes_today:.1f} min）"
            "——按用户 2026-09-25 的明确指示：没说停就可以用")
    if st.free_vram_mb < cfg.min_free_vram_mb:
        reasons.append(f"free VRAM {st.free_vram_mb:.0f} MB < "
                       f"{cfg.min_free_vram_mb:.0f} MB")
    if st.free_disk_gb < cfg.min_free_disk_gb:
        reasons.append(f"free disk {st.free_disk_gb:.1f} GB < "
                       f"{cfg.min_free_disk_gb:.1f} GB")
    if cfg.pause_while_user_active and st.user_active:
        reasons.append(f"user is using the machine: paused by configuration "
                       f"({st.user_activity_source}, idle {st.user_idle_s}s)")
    elif cfg.pause_while_user_active and st.user_active is None:
        # 未接入 = UNKNOWN：不据此暂停（没证据），但**必须**说清楚
        # "随用随停尚未实现"，不能让人以为它已经在工作。
        warnings.append(
            f"user activity signal 未接入（{st.user_activity_source}）："
            "无法按用户活动暂停，也不能声称已实现随用随停（方案 §3/§8.2）")
    if cfg.collect == "tech":
        warnings.append("collect=tech requires explicit authorisation; the "
                        "console must be free before starting the game")
    return {"allowed": not reasons, "reasons": reasons,
            "warnings": warnings, "checked_at": _utc()}


def probe_resources(*, vram_probe=None, disk_path: str | Path = ".",
                    gpu_minutes_today: float = 0.0,
                    user_active: bool | None = None,
                    activity_probe=None) -> ResourceState:
    """读当前资源。``vram_probe``/``activity_probe`` 可注入（测试与无 GPU 机器）。

    ``user_active=None`` 表示**去探测**（默认）：拿不到信号就是 None（未接入），
    不是 False。调用方要固定值（测试/显式声明）时才传 True/False。
    """
    if user_active is None:
        act = (activity_probe or user_activity_probe)()
    else:
        act = {"active": bool(user_active), "idle_s": None,
               "source": "caller-declared (not measured)",
               "why": "user_active was passed in, not probed"}
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
                         user_active=act["active"], temperature_c=temp,
                         user_idle_s=act.get("idle_s"),
                         user_activity_source=act.get("source") or "unknown")


# ---------------------------------------------------------------------------
class RunClock:
    """单次**连续运行**的墙钟：起点落在 run 目录，重启不重置。

    为什么起点要落盘：方案 §8.1 要求"单次任务的最长时长"是运行保护，而不是
    "每次启动重新计时"——否则崩溃重启（W3 的故障矩阵）会把上限无限延长。
    停止（墙钟到顶/用户暂停）之后要开新一轮，所以 :meth:`reset` 由停止路径调用。
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _read(self) -> dict:
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
            return blob if isinstance(blob, dict) else {}
        except Exception:                                  # noqa: BLE001
            return {}

    def write_started_at(self, ts: float) -> None:
        blob = self._read()
        blob["started_at"] = float(ts)
        blob["written_at"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(blob), encoding="utf-8")

    def start(self) -> dict:
        """读取起点；没有就**现在**开始计时（返回 ``{started_at, ...}``）。"""
        blob = self._read()
        if not blob.get("started_at"):
            self.write_started_at(time.time())
            blob = self._read()
        return blob

    def minutes(self) -> float:
        blob = self.start()
        return max(0.0, (time.time() - float(blob["started_at"])) / 60.0)

    def reset(self) -> None:
        """开新一轮（停止后调用）：删掉起点，下次从零计。"""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class MachineLease:
    """机器级学习资源租约：同一 GPU 上只允许一个重型实验持有。

    与 run 内 :class:`InstanceLock` 的区别（方案 G02 逐条对应）：

    * **原子创建**：用 ``O_CREAT | O_EXCL`` 拿租约，而不是"先判断存在再写"——
      并发下只有一个人能成功（旧写法两个进程可能都判断为"不存在"然后互相覆盖）；
    * **所有者可验证**：记 pid **和该进程的创建时间**，回收前两者都要对得上。
      只看 pid 会被 PID 复用骗到（旧进程早死了，新进程拿着同一个号的锁）；
    * **心跳**：持有者定期 ``heartbeat()``；过期判据是**心跳旧 + 所有者不存活**
      （或创建时间不符），**不是**"文件多久没动"。旧实现 6 小时无条件接管，
      长任务（8 小时常驻验收）会被第二个进程抢走；
    * 活着但不心跳（疑似挂死）时**仍然拒绝**，只在拒绝理由里写明"心跳过期"，
      把强抢留给显式 ``force``。

    探测函数可注入（``alive_fn`` / ``created_fn`` / ``now_fn``），测试不需要真进程。
    """

    def __init__(self, path: Path | str, *, heartbeat_s: float = 120.0,
                 alive_fn=None, created_fn=None, now_fn=None):
        self.path = Path(path)
        self.heartbeat_s = float(heartbeat_s)
        self._alive_fn = alive_fn or _alive
        self._created_fn = created_fn or pid_created
        self._now = now_fn or time.time

    # ---- 读写 -----------------------------------------------------------
    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:                     # noqa: BLE001
            return {}

    def _write_fresh(self, blob: dict) -> bool:
        """原子创建：已存在则不覆盖。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                         0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(blob, ensure_ascii=False))
        return True

    def _replace(self, blob: dict) -> None:
        """原地更新（心跳/回收）：先写临时文件再 replace，避免半写。"""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(self.path))

    def status(self) -> dict:
        """当前租约（只读；没有则 ``{"held": False}``）。

        ``owner_alive`` 与 ``pid_reused`` 分开报：进程不在了和"PID 被别的进程复用"
        是两件事，回收理由要说清是哪一个。
        """
        blob = self._read()
        if not blob:
            return {"held": False}
        pid = int(blob.get("pid", -1) or -1)
        created = str(blob.get("created") or "")
        now = float(self._now())
        raw_alive = bool(pid > 0 and self._alive_fn(pid))
        got_created = self._created_fn(pid) if raw_alive else None
        pid_reused = bool(raw_alive and created and got_created
                          and str(got_created) != created)
        # 拿不到创建时间时不判定"复用"（证据不足）：按"活着"处理，避免误抢
        owner_alive = bool(raw_alive and not pid_reused)
        return {"held": True, "pid": pid, "created": created,
                "owner_alive": owner_alive, "pid_reused": pid_reused,
                "age_s": round(now - float(blob.get("t") or 0), 1),
                "heartbeat_age_s": round(now - float(blob.get("hb") or 0), 1),
                "heartbeat_stale": (now - float(blob.get("hb") or 0)
                                    > self.heartbeat_s),
                "blob": blob}

    # ---- 生命周期 -------------------------------------------------------
    def acquire(self, *, pid: int | None = None, force: bool = False) -> dict:
        """拿租约。``acquired=False`` 时 ``reason`` 可直接打印给人看。"""
        pid = os.getpid() if pid is None else int(pid)
        created = str(self._created_fn(pid) or "")
        blob = {"pid": pid, "created": created, "t": self._now(),
                "hb": self._now(), "host": os.environ.get("COMPUTERNAME", "")}
        if self._write_fresh(blob):
            return {"acquired": True, "already_held": False, "holder": blob}
        cur = self._read()
        cur_pid = int(cur.get("pid", -1) or -1)
        if cur_pid == pid and str(cur.get("created") or "") == created:
            return {"acquired": True, "already_held": True, "holder": cur}
        st = self.status()
        if st["owner_alive"] and not force:
            why = (f"lease held by pid {cur_pid} (age {st['age_s']:.0f}s, "
                   f"heartbeat {st['heartbeat_age_s']:.0f}s)")
            if st["heartbeat_stale"]:
                why += (" - heartbeat is stale but the owner is still alive: "
                        "not stealing a long task (use force to override)")
            return {"acquired": False, "already_held": False,
                    "reason": why, "holder": cur, "status": st}
        # 回收：所有者不存活/创建时间不符（PID 复用），或显式 force
        if force and st["owner_alive"]:
            reason = "forced by caller"
        elif st.get("pid_reused"):
            reason = "owner pid was reused"
        elif not st["owner_alive"]:
            reason = "owner not alive"
        else:
            reason = "recovered"
        self._replace(blob)
        return {"acquired": True, "already_held": False, "holder": blob,
                "recovered_stale": True, "recovery_reason": reason}

    def heartbeat(self) -> bool:
        """刷新心跳；不是持有者（或被回收）返回 False。"""
        blob = self._read()
        if not blob:
            return False
        if (int(blob.get("pid", -1) or -1) != os.getpid()
                or str(blob.get("created") or "")
                != str(self._created_fn(os.getpid()) or "")):
            return False
        blob["hb"] = self._now()
        self._replace(blob)
        return True

    def release(self) -> None:
        blob = self._read()
        if blob and int(blob.get("pid", -1) or -1) == os.getpid():
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def pid_created(pid: int) -> str | None:
    """进程创建时间（用于区分 PID 复用）；取不到返回 ``None``。

    用 PowerShell 7 的 ``Get-Process.StartTime``——``wmic`` 在本机已不可用（实测），
    ``tasklist`` 不提供创建时间。
    """
    if pid <= 0:
        return None
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        return None
    try:
        r = subprocess.run(
            [shell, "-NoProfile", "-Command",
             f"(Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue)"
             f".StartTime.Ticks"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60)
    except Exception:                         # noqa: BLE001
        return None
    if r.returncode != 0:
        return None
    out = (r.stdout or "").strip()
    return out if out.isdigit() else None


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
                 "--runtime", cfg.collect_runtime,
                 "--map", cfg.collect_map,
                 "--frames", str(int(cfg.collect_frames)),
                 "--step-m", str(float(cfg.collect_step_m)),
                 "--step", str(int(cfg.collect_step)),
                 "--roles", *[str(r) for r in cfg.collect_roles],
                 *(["--teleport"] + [str(float(v)) for v in cfg.collect_teleport]
                   if len(tuple(cfg.collect_teleport or ())) == 3 else []),
                 *(["--follow-road"] if cfg.collect_follow_road else []),
                 *(["--save-annotation"] if cfg.collect_save_annotation else []),
                 "--out", str(Path("logs/m5_seg") / f"collect_{run_id}_<stamp>")],
            reason=("用户 2026-09-25 已授权无人值守采集；启动前过前置检查"
                    "（游戏在跑就不抢会话），采集后过身份审计（缺 "
                    "map_name/source_id 拒收）")))
    return LoopPlan(run_id=run_id, config_hash=cfg.hash(), actions=acts,
                    notes=[f"collect={cfg.collect}",
                           f"collect_roles={list(cfg.collect_roles)}",
                           f"collect_frames={cfg.collect_frames}",
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
    #: 单因归因（``proposer.classify_round_outcome`` 写入判定文件后带回来）。
    #: 只有 ``"meaningful_no_gain"`` 才算"有意义的无收益"；资格失败、缺标注、
    #: 无效因子各自另有原因，不得当成"模型没有提升空间"的证据（方案 v2 §S5）。
    #: 空串 = 旧记录/未归因，停止条件回退旧口径，兼容历史日志。
    outcome: str = ""


def should_stop(cfg: LoopConfig, history: list[RoundRecord], *,
                gpu_minutes_today: float, candidates_used: int,
                wall_minutes: float = 0.0) -> dict:
    """停止条件：预算耗尽、单次墙钟到顶、连续无收益、或候选用尽。

    ``wall_minutes``：本次**连续运行**已进行的分钟数（见 :class:`RunClock`）。
    ``max_wall_minutes <= 0`` 与每日上限同一口径 = 不限制。
    """
    reasons: list[str] = []
    streak = 0
    for r in reversed(history):
        if r.outcome:
            # 有归因就按归因数：无效因子/缺标注/资格失败的"没收益"不是
            # 平台期证据，不该把预算推向停止（方案 v2 §S5）。
            if r.outcome == "meaningful_no_gain":
                streak += 1
                continue
            break
        if r.decision in ("rejected", "needs_evidence") and not r.gain:
            streak += 1
        else:
            break
    if streak >= cfg.max_rounds_without_gain:
        reasons.append(f"{streak} consecutive rounds without a credible gain "
                       f"(limit {cfg.max_rounds_without_gain})")
    if _cap_enabled(cfg.daily_gpu_minutes) and \
            gpu_minutes_today >= cfg.daily_gpu_minutes:
        reasons.append("daily GPU budget exhausted")
    if candidates_used >= cfg.max_candidates:
        reasons.append(f"candidate budget exhausted "
                       f"({candidates_used}/{cfg.max_candidates})")
    if cfg.max_wall_minutes > 0 and wall_minutes >= cfg.max_wall_minutes:
        reasons.append(f"single-run wall clock exhausted "
                       f"({wall_minutes:.1f} >= {cfg.max_wall_minutes:.0f} min)")
    return {"stop": bool(reasons), "reasons": reasons,
            "no_gain_streak": streak, "wall_minutes": round(float(wall_minutes), 1)}


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
