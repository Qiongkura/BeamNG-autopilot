"""无人值守采集：前置检查、命令构造、采集后的身份与内容审计。

**授权记录**：用户 2026-09-25 明确授权"允许无人值守启动 Tech 采集"
（此前 ``LoopConfig.collect`` 长期是 ``off``；方案 §136 规定未授权不得自动
启动采集，见 ``docs/T14_PROGRESS_20260924.md`` 的授权范围与时间）。

授权只放开**自动启动游戏采集**，不放开"把未经人工修订的采集结果当标线真值"：
引擎标注里漆线被画成沥青（见 ``m5_collect_seg_ring.py`` 模块说明），所以采集
回来的数据只能当**路面通道**训练组（``--allow-road-only``）；漆线真值仍然走
人工修订，采集的作用是"提供更多路段 + 排出该修哪几帧"。

三种结果必须分得开（三态口径，不把 UNKNOWN 写成 PASS）：

* ``ok=False``：硬性失败，这采集**不能**进训练——meta 缺失/不可解析、
  ``map_name`` 或 ``source_id`` 为空、请求的视角一帧都没有。
* ``ok=True`` 但有 ``warnings``：能进训练，但要知道——身份来自命令行回退、
  某视角没有任何漆线像素（线通道零信息）、实际帧数少于请求。
* 通过且无警告。

本模块只做**决策与校验**，自己不启动游戏；启动在
``scripts/m5_collect_seg_ring.py``（无人值守用 ``--runtime tech``，不带
``--attach``）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


#: 默认视角：前向两档 + 左右侧柱。左右侧是"标线两侧"证据的来源，
#: 后向视角对压线判断没有贡献，省下来的时间给 step-m 走更远。
DEFAULT_ROLES: tuple = ("front_main", "front_fisheye", "pillar_left",
                        "pillar_right")

#: 判"有人在开"的进程名。**Tech 与 Steam 版的进程名不一样**（实测：
#: G:\BeamNG.tech.v0.38.5.0\Bin64\BeamNG.tech.x64.exe vs
#: BeamNG.drive.x64.exe）——只查 drive 的名字会在别人开着 Tech 时报"没在跑"，
#: 于是采集去抢一个已经在用的端口。
GAME_IMAGES: tuple = ("BeamNG.tech.x64.exe", "BeamNG.drive.x64.exe",
                      "BeamNG.drive.exe", "BeamNG.tech.exe")

UNATTENDED_FORBIDDEN = "attach"


@dataclass(frozen=True)
class CollectSpec:
    """一次采集的输入参数（全部进事件与产物，便于复现）。"""

    frames: int = 30
    step_m: float = 2.0
    roles: tuple = DEFAULT_ROLES
    follow_road: bool = True
    runtime: str = "tech"
    save_annotation: bool = True
    step: int = 10
    map_name: str = "italy"
    attach: bool = False
    timeout_s: int = 1800
    #: ``(x, y, yaw_deg)`` 或空。集合采过的地方要**换起点**：同一个 spawn 连采
    #: 两次会得到同一段路，第二份就是重复样本（不是新数据）。
    teleport: tuple = ()

    def as_dict(self) -> dict:
        blob = asdict(self)
        blob["roles"] = list(self.roles)
        blob["teleport"] = list(self.teleport)
        return blob


def game_running_probe(images: tuple = GAME_IMAGES) -> bool | None:
    """游戏是否在运行：``True`` / ``False`` / ``None``（探测不确定）。

    ``None`` 不是 ``False``：中文 Windows 的 ``tasklist`` 输出不是 UTF-8，
    读取失败会让输出变空，把"输出为空"当成"没在跑"就会去踩正在驾驶的会话
    （``controller._alive`` 上踩过同一个坑）。所以显式 ``errors="replace"``，
    并且**只有**返回码 0 且能看清输出时才敢说 ``False``。
    """
    for img in images:
        try:
            out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {img}",
                                  "/NH"],
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace",
                                 timeout=20)
        except Exception:                        # noqa: BLE001
            return None
        if out.returncode != 0:
            return None
        stdout = (out.stdout or "")
        if img.lower() in stdout.lower():
            return True
    return False


def collector_python(root: str | Path, *, configured: str = "") -> tuple:
    """采集该用哪个解释器：``(路径, 来源)``。

    实测踩到：无人值守入口拿 ``sys.executable``（系统 Python）去起采集，那个
    解释器**没有 beamngpy**，于是白起一局、rc=1、身份审计拒收——闸门拦住了脏
    数据，但这一轮的时间是白花的。项目自己的 venv 才是装了 beamngpy 的那个。
    """
    if configured:
        return str(configured), "config"
    venv = Path(root) / ".venv" / "Scripts" / "python.exe"
    if venv.exists():
        return str(venv), "project-venv"
    return str(sys.executable), "sys.executable"


def python_can_import(python: str, module: str = "beamngpy",
                      *, timeout_s: int = 90) -> bool:
    """那个解释器能不能 ``import <module>``：启动前先问，别让子进程替我们问。"""
    try:
        out = subprocess.run([str(python), "-c", f"import {module}"],
                             capture_output=True, text=True, errors="replace",
                             timeout=int(timeout_s))
        return out.returncode == 0
    except Exception:                            # noqa: BLE001
        return False


def collect_command(*, python: str, script: str | Path, spec: CollectSpec,
                    out_dir: str | Path) -> list:
    """采集命令行；``--out`` 由调用方给死，采集目录因此可写进事件与产物。"""
    cmd = [str(python), str(script),
           "--runtime", str(spec.runtime),
           "--map", str(spec.map_name),
           "--frames", str(int(spec.frames)),
           "--step-m", str(float(spec.step_m)),
           "--step", str(int(spec.step)),
           "--roles", *[str(r) for r in spec.roles],
           "--out", str(out_dir)]
    if len(tuple(spec.teleport or ())) == 3:
        x, y, yaw = spec.teleport
        cmd += ["--teleport", str(float(x)), str(float(y)), str(float(yaw))]
    if spec.follow_road:
        cmd.append("--follow-road")
    if spec.save_annotation:
        cmd.append("--save-annotation")
    if spec.attach:
        cmd.append("--attach")
    return cmd


def _pid_created(pid: int) -> str | None:
    """进程创建时间（.NET ticks 字符串）；取不到返回 ``None``。

    与 ``controller.pid_created`` 同源（PowerShell 7 的 ``Get-Process.StartTime``）：
    **PID 会被复用**，只看 pid 号无法证明“这个进程是我们起的”。
    """
    from .controller import pid_created
    return pid_created(pid)


def game_procs(images: tuple = GAME_IMAGES) -> dict | None:
    """``{pid: created_ticks}``；探测失败返回 ``None``（不确定就不敢动）。

    ``created`` 可能是 ``None``（拿不到创建时间）——那种进程**不能**被当作
    “我们起的”来关闭（方案 G06：按真实所有权关闭，而不是凭名字或 pid 号）。
    """
    pids = game_pids(images)
    if pids is None:
        return None
    return {int(p): _pid_created(int(p)) for p in pids}


def _created_unix(ticks) -> float | None:
    """``.NET ticks`` -> unix 秒；解析不了返回 ``None``。"""
    try:
        # .NET ticks = 自 0001-01-01 起的 100ns；到 1970-01-01 的秒数是固定常数。
        # 实测踩到：用 datetime 相减取偏移会把符号弄反（那个差是负数），
        # 于是"创建时间早于启动时刻"永远判不出来 -> 反而把所有新 pid 都当自己的关掉。
        TICKS_TO_UNIX_S = 62135596800.0
        return float(ticks) / 1e7 - TICKS_TO_UNIX_S
    except Exception:                                     # noqa: BLE001
        return None


def game_pids(images: tuple = GAME_IMAGES) -> set | None:
    out: set = set()
    for img in images:
        try:
            r = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {img}",
                                "/NH"],
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=20)
        except Exception:                            # noqa: BLE001
            return None
        if r.returncode != 0:
            return None
        for line in (r.stdout or "").splitlines():
            low = line.lower()
            if img.lower() not in low:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                out.add(int(parts[1]))
    return out


def close_started_game(pids_before, *, images: tuple = GAME_IMAGES,
                       timeout_s: int = 30,
                       launched_after: float | None = None) -> dict:
    """结束**本次采集期间新出现**的游戏进程；已存在的会话一律不碰。

    实测缺陷（2026-09-25）：无人值守采集结束后游戏还在跑——采集器只关了连接，
    进程留着。后果两条：① 它一直占 GPU，而"游戏时间"只在采集那几分钟记账，
    后面的实验都在被污染的机器上跑；② 第 3 轮那批推理 p95（10.6–80.2 ms，
    7.6 倍漂移）就是它在跑的时候测的。所以采集收尾必须把**自己起的**游戏关掉。
    只杀 ``after - before`` 的差集：用户自己开着的会话不在差集里，永远不会被误杀。
    """
    if pids_before is None:
        return {"closed": False, "killed": [],
                "reason": "采集前没能可靠列出游戏进程：不猜、不乱杀"}
    after = game_pids(images)
    if after is None:
        return {"closed": False, "killed": [],
                "reason": "采集后没能可靠列出游戏进程：不猜、不乱杀"}
    started = sorted(after - set(int(p) for p in pids_before))
    # 所有权校验（方案 G06）：新出现的 pid 还要能证明“是这次任务起的”——
    # 创建时间晚于我们启动采集的时刻。拿不到创建时间、或创建时间早于启动时刻
    # （PID 复用），一律**不关**，只报 unverified 让人看清。
    unverified: list = []
    if started and launched_after is not None:
        procs = game_procs(images)
        verified: list = []
        for pid in started:
            created = (procs or {}).get(pid)
            cu = _created_unix(created) if created is not None else None
            if cu is None:
                unverified.append({"pid": pid,
                                   "why": "no usable creation time: "
                                          "ownership unproven"})
                continue
            if cu + 5.0 < float(launched_after):
                unverified.append({"pid": pid,
                                   "why": "created before this task started "
                                          "(pid reuse)"})
                continue
            verified.append(pid)
        started = verified
    if not started:
        return {"closed": True, "killed": [],
                "unverified": unverified,
                "note": ("采集没有留下新的游戏进程"
                         + ("（有进程无法证明归属，未动）"
                            if unverified else ""))}
    killed, failed = [], []
    for pid in started:
        try:
            r = subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=int(timeout_s))
            (killed if r.returncode == 0 else failed).append(pid)
        except Exception as exc:                     # noqa: BLE001
            failed.append(pid)
            _ = exc
    left = game_pids(images)
    still = sorted((set(int(p) for p in left) if left is not None else set())
                   & set(started))
    return {"closed": not still, "killed": killed, "failed": failed,
            "unverified": unverified, "still_running": still,
            "reason": ("" if not still else
                       f"仍在运行：{still}（GPU 仍在被占用，后续计时会被污染）")}


def preflight(*, spec: CollectSpec, game_running: bool | None,
              free_vram_mb: float, free_disk_gb: float,
              min_free_vram_mb: float = 2048.0,
              min_free_disk_gb: float = 20.0,
              force: bool = False,
              collector_python_ok: bool | None = None,
              collector_python: str = "") -> dict:
    """启动前的硬检查；``force`` 只降级"人在不在开"这一类，不降级资源门。"""
    reasons: list = []
    warnings: list = []
    if int(spec.frames) <= 0:
        reasons.append(f"frames={spec.frames} <= 0：采不到东西")
    if float(spec.step_m) <= 0:
        warnings.append("step_m<=0：静止取帧会得到重复样本（实测过首末帧"
                        "姿态完全相同），只能当单帧样例，不是序列")
    if game_running is None:
        msg = ("无法确认游戏是否在运行（tasklist 探测失败）：无人值守不猜，"
               "要么修探测要么显式 --collect-force")
        (warnings if force else reasons).append(msg)
    elif game_running and not spec.attach:
        msg = ("游戏已在运行：可能有人在开，采集不抢会话（要么等它退出，"
               "要么显式 --collect-force）")
        (warnings if force else reasons).append(msg)
    if spec.attach:
        warnings.append("attach 会进入一个已经存在的会话：无人值守不该用"
                        "（它可能正在被驾驶）")
    if float(free_vram_mb) < float(min_free_vram_mb):
        reasons.append(f"free VRAM {free_vram_mb:.0f} MB < "
                       f"{min_free_vram_mb:.0f} MB")
    _has_teleport = len(tuple(spec.teleport or ())) == 3
    if float(spec.step_m) > 0 and not spec.follow_road and _has_teleport:
        warnings.append("给了 teleport 但没开 follow-road：直线推算会在几米内"
                        "离开铺装（实测过），只适合单帧样例")
    if collector_python_ok is False:
        reasons.append(f"采集解释器 {collector_python or '(unknown)'} 里没有 "
                       f"beamngpy：起了也是白起（无人值守先问再启动）")
    elif collector_python_ok is None:
        warnings.append("没有检查采集解释器能否 import beamngpy：出了错要等"
                        "子进程报")
    if float(free_disk_gb) < float(min_free_disk_gb):
        reasons.append(f"free disk {free_disk_gb:.1f} GB < "
                       f"{min_free_disk_gb:.1f} GB")
    return {"ok": not reasons, "reasons": reasons, "warnings": warnings,
            "spec": spec.as_dict(), "forced": bool(force),
            "collector_python": str(collector_python)}


def _roles_from_meta(meta: dict, out_dir: Path) -> dict:
    roles = meta.get("roles")
    if isinstance(roles, dict) and roles:
        return {str(k): int(v or 0) for k, v in roles.items()}
    found: dict = {}
    if out_dir.exists():
        for d in sorted(p for p in out_dir.iterdir() if p.is_dir()):
            found[d.name] = len(list(d.glob("*.npz")))
    return found


def verify_collection(out_dir: str | Path, *,
                      expected_roles=None,
                      expected_frames: int | None = None) -> dict:
    """采集后审计：这份采集能不能进训练，身份是否**有来源**。

    这是"采集 → 训练"之间唯一的闸门；缺身份或空视角一律拒收，绝不静默接受
    （3h 轮 48 帧人工修订因身份丢失被判死的教训：不补票，只止损）。
    """
    out = Path(out_dir)
    reasons: list = []
    warnings: list = []
    meta: dict = {}
    meta_path = out / "meta.json"
    if not out.exists():
        reasons.append(f"采集目录不存在：{out}")
    elif not meta_path.exists():
        reasons.append(f"缺 meta.json（身份无处可查）：{meta_path}")
    else:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:                 # noqa: BLE001
            reasons.append(f"meta.json 不可解析（{type(exc).__name__}）："
                           f"{meta_path}")

    roles: dict = _roles_from_meta(meta, out) if meta else {}
    map_name = str(meta.get("map_name") or "").strip()
    source_id = str(meta.get("source_id") or "").strip()
    map_source = meta.get("map_name_source")
    if meta and not map_name:
        reasons.append("meta 里 map_name 为空：训练准入门会以"
                       "「no map identity」拒收，别把这类目录喂给 rounds")
    if meta and not source_id:
        reasons.append("meta 里 source_id 为空：同图不同次采集无法区分，"
                       "整组隔离会失效")
    if meta and not map_source:
        warnings.append("map_name_source 缺失：身份没有来源（manifest 会记"
                        "一条 note），这份采集的身份可信度按「来源未知」记")
    elif str(map_source) == "argument-fallback":
        warnings.append("map_name_source=argument-fallback：身份是命令行回退"
                        "值，不是从运行中的会话读出来的")

    frames_total = int(sum(roles.values()))
    if meta and frames_total <= 0:
        reasons.append("一个帧都没有：采集启动过但没有产出")
    exp = tuple(expected_roles or ())
    missing = [r for r in exp if int(roles.get(r, 0)) <= 0]
    if missing:
        reasons.append(f"请求的视角没有任何帧：{missing}")
    if expected_frames is not None and 0 < frames_total < int(expected_frames):
        warnings.append(f"实际 {frames_total} 帧 < 请求 {expected_frames} 帧"
                        f"（可能提前结束，按实测帧数记账）")

    paint_by_role: dict = {}
    recs = meta.get("frames") if isinstance(meta.get("frames"), list) else []
    for rec in recs:
        view = str(rec.get("view") or "")
        if int(rec.get("line_pixels") or 0) > 0:
            paint_by_role[view] = paint_by_role.get(view, 0) + 1
    if recs:
        for role, n in sorted(roles.items()):
            if n > 0 and not paint_by_role.get(role):
                warnings.append(f"{role} 全程没有漆线像素：线通道零信息，"
                                f"人工修订用不上这个视角")

    return {"ok": not reasons, "reasons": reasons, "warnings": warnings,
            "map_name": map_name, "map_name_source": map_source,
            "source_id": source_id, "roles": roles,
            "frames_total": frames_total,
            "paint_frames_by_role": paint_by_role,
            "meta_path": str(meta_path), "out_dir": str(out)}


def paint_frame_priority(meta: dict, *, top: int = 20,
                         view: str | None = None) -> list:
    """按漆线像素排序的帧清单——人工修订该从这几帧开始。

    线通道的缺口不在"标签写错了"，而在"没有可信的漆线真值"；采集能提供的
    最有用的东西就是**哪里看得见漆线**（引擎不给 line 类，但像素在那里）。
    """
    recs = meta.get("frames") if isinstance(meta.get("frames"), list) else []
    rows = [r for r in recs if (view is None or r.get("view") == view)]
    rows.sort(key=lambda r: -int(r.get("line_pixels") or 0))
    out = []
    for r in rows[:max(0, int(top))]:
        out.append({"view": r.get("view"), "path": r.get("path"),
                    "line_pixels": int(r.get("line_pixels") or 0),
                    "pos": r.get("pos"), "heading": r.get("heading"),
                    "exposure": r.get("exposure")})
    return out


def collection_proposal(*, role_dirs, candidate_id: str | None = None,
                        stamp: str = "", note: str = "",
                        selection: dict | None = None) -> dict:
    """把一次采集变成 ``add_runs`` 数据因子提议（下一轮训练直接消费）。

    ``selection``：:func:`beamng_autopilot.experiments.selection.pick_batch` 的
    产出（方案 §7.6「每次只增加预设小批场景，经审计后才训练」）。带上它，
    提议里就能看到"这一批选了多少帧、多少只能进复核队列、配额丢了什么"，
    而不是把整次采集无声地倒进训练集。
    """
    dirs = [str(Path(d)) for d in role_dirs]
    cid = candidate_id or (f"collect-{stamp}" if stamp else "collect")
    base_note = note or ("无人值守采集产出的数据因子：把本次采集的视角目录"
                         "追加进训练集（路面通道；漆线真值仍待人工修订）")
    out = {"note": base_note,
           "proposals": [{"candidate_id": cid,
                          "family": "data_composition",
                          "factor": {"add_runs": dirs}}]}
    if selection:
        n = int(selection.get("n_items") or 0)
        rev = int(selection.get("n_review_only") or 0)
        out["selection"] = selection
        out["note"] = (f"{base_note} | 选样：本批 {n} 帧，其中 {rev} 帧"
                       f"只能进复核队列（标签未复核）")
        out["proposals"][0]["selection"] = {
            "n_items": n, "n_review_only": rev,
            "n_trainable": int(selection.get("n_trainable") or 0),
            "reasons": list(selection.get("reasons") or [])}
    return out


def merge_proposals(*blobs) -> dict:
    """合并多个提议文件：按 ``candidate_id`` 去重，先出现的顺序保留。"""
    merged: list = []
    seen: set = set()
    notes: list = []
    for blob in blobs:
        if not blob:
            continue
        if blob.get("note"):
            notes.append(str(blob["note"]))
        for p in blob.get("proposals") or []:
            cid = str(p.get("candidate_id") or "")
            if cid and cid in seen:
                continue
            seen.add(cid)
            merged.append(p)
    return {"note": " | ".join(notes), "proposals": merged}
