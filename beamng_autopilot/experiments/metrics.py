"""训练监控的指标存储：逐 step 训练指标 + 定时硬件采样 + 任务状态。

三类记录写进同一个 JSONL（``logs/experiments/<run_id>/metrics.jsonl``），
每条带单调 ``seq``，所以前端可以"按序号增量拉取 + 去重"，刷新/断线/异常
退出后都能接着看：

* ``task``   —— 任务信息：run_id / 名称 / 状态 / 开始时间 / 总 step / 当前
  step / epoch。状态取值见 :data:`STATUSES`。
* ``train``  —— 每个优化步一行：step / epoch / loss / acc / grad_norm /
  lr / step_s。
* ``system`` —— 定时（默认 2 s）采样：GPU 显存/功耗/利用率、CPU 利用率、
  系统内存；**按设备逐条记录**，另有一条 ``device=None`` 的汇总，
  ``aggregate`` 字段写明汇总口径。

三条纪律（方案 §3 明写）：

1. **不假装有数据**：读不到的指标写进 ``unavailable: {name: 原因}``，
   数值字段为 ``None``；"任务不适用"（如没有 step 级 accuracy）用
   ``not_applicable`` 说明；**0 就是 0**，与两者都不混。
2. **原始记录不删**：抽稀只发生在绘图点（:func:`downsample`），
   :func:`stats_of` 永远基于原始值计算。
3. **时间轴与 step 轴不混用**：训练曲线用 ``step``，硬件曲线用
   ``t``（相对开始时间的秒数）。
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

STATUSES = ("waiting", "running", "paused", "completed", "failed")

#: 硬件采样周期（秒）：方案建议先用 2 s
DEFAULT_SAMPLE_S = 2.0

#: 汇总口径：显存/功耗求和，利用率取最大。多 GPU 时前端按设备切换，
#: 并把这段说明显示在图例里，避免把"求和"误读成"单卡"。
AGGREGATE_NOTE = "mem/power = sum over devices, util = max over devices"


# ---------------------------------------------------------------------------
# JSON 安全：非有限值不能进数据流
def sanitize_numbers(obj) -> tuple:
    """``(clean, bad_fields)``：把 NaN/±Inf 换成 None 并记下字段名。

    实测教训：梯度范数在 AMP 下可能溢出成 ``inf``，而 ``json.dumps`` 默认会
    写成裸 ``Infinity`` —— **这不是合法 JSON**，浏览器 ``fetch().json()`` 直接
    抛 SyntaxError，于是整页一条数据都读不到。所以非有限值一律换成 ``None``
    并单独记录，前端看到的是"这个点没有有效值"而不是一个假数字。
    """
    bad: list = []

    def walk(v, path):
        if isinstance(v, float):
            if math.isfinite(v):
                return v
            bad.append(path)
            return None
        if isinstance(v, dict):
            return {k: walk(x, f"{path}.{k}" if path else str(k))
                    for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [walk(x, f"{path}[{i}]") for i, x in enumerate(v)]
        return v

    return walk(obj, ""), bad


# ---------------------------------------------------------------------------
# 存储
@dataclass
class MetricsStore:
    """``<run_dir>/metrics.jsonl`` 的追加与读取（半写行可识别）。"""

    run_dir: Path

    @property
    def path(self) -> Path:
        return Path(self.run_dir) / "metrics.jsonl"

    def append(self, record: dict) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = dict(record)
        rec.setdefault("t", time.time())
        rec.setdefault("kind", "train")
        rec["seq"] = self.next_seq()
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        # 半写行要能被观察到（读取端会报出来），所以不吞异常、不缓存
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
        return rec

    def next_seq(self) -> int:
        last = self.last_seq()
        return last + 1

    def last_seq(self) -> int:
        if not self.path.exists():
            return 0
        seq = 0
        with self.path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            tail = fh.read().decode("utf-8", errors="replace")
        for line in tail.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                seq = max(seq, int(json.loads(line).get("seq") or 0))
            except Exception:                # noqa: BLE001
                continue
        return seq

    def read(self) -> tuple[list[dict], list[str]]:
        """``(records, problems)``；问题里含半写行与坏 JSON。"""
        if not self.path.exists():
            return [], []
        out: list[dict] = []
        problems: list[str] = []
        raw = self.path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:         # noqa: BLE001
                problems.append(f"line {n}: unreadable ({exc})")
                continue
            rec, bad = sanitize_numbers(rec)
            if bad:
                problems.append(
                    f"line {n}: non-finite value in {sorted(set(bad))[:3]} "
                    f"-> null（非有限值不当数字用）")
            out.append(rec)
        out.sort(key=lambda r: int(r.get("seq") or 0))
        return out, problems

    def read_since(self, since_seq: int) -> dict:
        """增量拉取：只返回 ``seq > since_seq`` 的记录，并带回最大 seq。

        前端按 ``next_since`` 继续拉，重启后重复写入的记录由 seq 天然去重
        （同一 seq 不会出现两条，因为序号在写入时分配）。
        """
        recs, problems = self.read()
        fresh = [r for r in recs if int(r.get("seq") or 0) > int(since_seq)]
        return {"records": fresh, "problems": problems,
                "next_since": (int(fresh[-1]["seq"]) if fresh
                               else int(since_seq)),
                "total": len(recs)}

    # ---- 便捷查询 -----------------------------------------------------
    def task(self) -> dict | None:
        recs, _ = self.read()
        tasks = [r for r in recs if r.get("kind") == "task"]
        return tasks[-1] if tasks else None

    def train_records(self) -> list[dict]:
        recs, _ = self.read()
        return [r for r in recs if r.get("kind") == "train"]

    def system_records(self) -> list[dict]:
        recs, _ = self.read()
        return [r for r in recs if r.get("kind") == "system"]

    def epoch_records(self) -> list[dict]:
        recs, _ = self.read()
        return [r for r in recs if r.get("kind") == "epoch"]


# ---------------------------------------------------------------------------
# 统计与抽稀
def stats_of(values) -> dict:
    """均值/中位数/极值/p95，**忽略非有限值**并报告丢了多少。

    空输入或全是非有限值时，统计量返回 ``None`` 并给出 ``missing`` 原因——
    不能返回 0，那会把"没测到"画成"测到 0"。
    """
    finite, dropped = [], 0
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if math.isfinite(f):
            finite.append(f)
        else:
            dropped += 1
    if not finite:
        return {"n": 0, "dropped_non_finite": dropped,
                "missing": "no finite values recorded"}
    finite.sort()
    n = len(finite)
    out = {
        "n": n, "dropped_non_finite": dropped,
        "mean": sum(finite) / n,
        "median": (finite[n // 2] if n % 2
                   else 0.5 * (finite[n // 2 - 1] + finite[n // 2])),
        "min": finite[0], "max": finite[-1],
        "p95": finite[min(n - 1, int(round(0.95 * (n - 1))))],
    }
    return out


def moving_average(values, window: int) -> list:
    """滑动平均；``window <= 1`` 时原样返回。空值跳过而不是当 0。"""
    w = int(window)
    if w <= 1:
        return [None if v is None else float(v) for v in values]
    out: list = []
    buf: list = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        buf.append(float(v))
        if len(buf) > w:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def downsample(points: list, max_points: int) -> dict:
    """等步长抽稀**绘图点**；统计仍用原始值（方案 §4）。

    返回 ``{points, stride, note, n_raw}``。抽稀时保留首末点，避免曲线末端
    看起来提前结束。
    """
    pts = list(points)
    n = len(pts)
    max_points = max(2, int(max_points))
    if n <= max_points:
        return {"points": pts, "stride": 1, "n_raw": n, "note": ""}
    stride = int(math.ceil(n / max_points))
    kept = pts[::stride]
    if kept[-1] != pts[-1]:
        kept.append(pts[-1])
    return {"points": kept, "stride": stride, "n_raw": n,
            "note": f"绘图抽稀 1/{stride}（原始 {n} 点全量保留，统计基于原始值）"}


# ---------------------------------------------------------------------------
# 硬件/系统采样
def gpu_probe(devices: list | None = None) -> tuple[list, dict]:
    """``(per_device, unavailable)``；每项含 mem_gib/power_w/util_pct。

    优先 pynvml（能读功耗与利用率），失败退回 ``nvidia-smi``，两者都不可用
    时返回 ``unavailable`` 原因而不是 0。
    """
    unavailable: dict = {}
    per_device: list = []
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        idxs = list(range(count)) if not devices else [int(d) for d in devices]
        for i in idxs:
            if i >= count:
                unavailable[f"gpu{i}"] = f"device {i} not present (found {count})"
                continue
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            item = {"device": i}
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                item["mem_gib"] = mem.used / 2 ** 30
            except Exception as exc:         # noqa: BLE001
                item["mem_gib"] = None
                unavailable[f"gpu{i}.mem_gib"] = f"pynvml memory: {exc}"
            try:
                item["power_w"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception as exc:         # noqa: BLE001
                item["power_w"] = None
                unavailable[f"gpu{i}.power_w"] = (
                    f"this device does not provide power readings ({exc})")
            try:
                item["util_pct"] = float(
                    pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception as exc:         # noqa: BLE001
                item["util_pct"] = None
                unavailable[f"gpu{i}.util_pct"] = f"pynvml utilization: {exc}"
            per_device.append(item)
        try:
            pynvml.nvmlShutdown()
        except Exception:                    # noqa: BLE001
            pass
        return per_device, unavailable
    except Exception as exc:                 # noqa: BLE001
        unavailable["gpu"] = f"pynvml unavailable: {exc}"
    # 退路：nvidia-smi（一次调用读全部设备）
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=20)
        if out.returncode != 0:
            unavailable["gpu"] = f"nvidia-smi rc={out.returncode}"
            return per_device, unavailable
        for line in (out.stdout or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                per_device.append({
                    "device": int(parts[0]),
                    "mem_gib": float(parts[1]) / 1024.0,
                    "power_w": float(parts[2]),
                    "util_pct": float(parts[3])})
            except ValueError:
                unavailable.setdefault("gpu_parse", f"unparsable row: {line}")
        unavailable.pop("gpu", None)
    except Exception as exc:                 # noqa: BLE001
        unavailable["gpu"] = f"nvidia-smi unavailable: {exc}"
    return per_device, unavailable


_CPU_PRIMED = {"done": False}


def cpu_probe() -> tuple[dict, dict]:
    """``(values, unavailable)``；CPU 利用率与系统内存。

    首次调用必须带一小段阻塞间隔：``psutil.cpu_percent(interval=None)`` 在没有
    上一次基线时返回 0.0，会把"还没测到"画成"CPU 空闲 0%"——这正是方案禁止的
    那种混淆。
    """
    values: dict = {}
    unavailable: dict = {}
    try:
        import psutil
        if not _CPU_PRIMED["done"]:
            values["cpu_util_pct"] = float(
                psutil.cpu_percent(interval=0.1))
            _CPU_PRIMED["done"] = True
        else:
            values["cpu_util_pct"] = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        values["sys_mem_gib"] = (vm.total - vm.available) / 2 ** 30
        values["sys_mem_total_gib"] = vm.total / 2 ** 30
    except Exception as exc:                 # noqa: BLE001
        unavailable["cpu_util_pct"] = f"psutil unavailable: {exc}"
        unavailable["sys_mem_gib"] = f"psutil unavailable: {exc}"
    return values, unavailable


def sample_system(devices: list | None = None) -> dict:
    """一次硬件采样记录（含设备汇总与 ``unavailable`` 原因）。"""
    per_device, un_gpu = gpu_probe(devices)
    cpu, un_cpu = cpu_probe()
    unavailable = {**un_gpu, **un_cpu}
    mems = [d["mem_gib"] for d in per_device if d.get("mem_gib") is not None]
    pows = [d["power_w"] for d in per_device if d.get("power_w") is not None]
    utils = [d["util_pct"] for d in per_device if d.get("util_pct") is not None]
    rec = {
        "kind": "system", "t": time.time(),
        "devices": per_device,
        "gpu_mem_gib": sum(mems) if mems else None,
        "gpu_mem_max_gib": max(mems) if mems else None,
        "gpu_power_w": sum(pows) if pows else None,
        "gpu_util_pct": max(utils) if utils else None,
        "cpu_util_pct": cpu.get("cpu_util_pct"),
        "sys_mem_gib": cpu.get("sys_mem_gib"),
        "sys_mem_total_gib": cpu.get("sys_mem_total_gib"),
        "aggregate": AGGREGATE_NOTE,
    }
    if unavailable:
        rec["unavailable"] = unavailable
    return rec


class SystemSampler:
    """后台定时采样线程；``stop()`` 幂等，异常不影响训练。

    ``probe`` 可注入（测试与无 GPU 机器）：返回一条 system 记录。
    """

    def __init__(self, store: MetricsStore, *, interval_s: float = DEFAULT_SAMPLE_S,
                 devices: list | None = None, probe=None, run_id: str = ""):
        self.store = store
        self.interval_s = max(0.2, float(interval_s))
        self.devices = devices
        self.probe = probe or (lambda: sample_system(devices))
        self.run_id = run_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n_samples = 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                rec = dict(self.probe())
                rec.setdefault("kind", "system")
                if self.run_id:
                    rec["run_id"] = self.run_id
                self.store.append(rec)
                self.n_samples += 1
            except Exception as exc:         # noqa: BLE001
                # 采样失败不能打断训练，但要留痕（前端显示"采样异常"）
                try:
                    self.store.append({"kind": "system", "t": time.time(),
                                       "unavailable": {
                                           "sampler": f"{type(exc).__name__}: "
                                                      f"{exc}"}})
                except Exception:            # noqa: BLE001
                    pass
            self._stop.wait(self.interval_s)

    def start(self) -> "SystemSampler":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="metric-sampler")
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))
            self._thread = None


# ---------------------------------------------------------------------------
# 任务状态与演示数据
def task_record(run_id: str, name: str, status: str, *, total_steps: int | None,
                current_step: int | None = None, epoch: int | None = None,
                started_at: float | None = None, error: str | None = None,
                extra: dict | None = None) -> dict:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; valid: {list(STATUSES)}")
    rec = {"kind": "task", "t": time.time(), "run_id": run_id, "name": name,
           "status": status, "total_steps": total_steps,
           "current_step": current_step, "epoch": epoch,
           "started_at": started_at}
    if error:
        rec["error"] = error
    if extra:
        rec.update(extra)
    return rec


def demo_records(run_id: str, *, steps: int = 300, seed: int = 7,
                 with_hardware: bool = True, interval_s: float = 1.0,
                 unavailable: dict | None = None) -> list:
    """**演示数据**：每条都带 ``demo: True``，前端会显示 DEMO 横幅。

    只用于界面联调/截图；与真实模式的唯一区别就是这条标记，绝不用它冒充
    真实训练结果。
    """
    import random
    rnd = random.Random(seed)
    recs = [dict(task_record(run_id, "DEMO 训练任务", "running",
                             total_steps=steps, current_step=0, epoch=0,
                             started_at=time.time()), demo=True)]
    loss = 2.4
    for s in range(1, steps + 1):
        loss = max(0.18, loss * (0.992 + rnd.uniform(-0.01, 0.01)))
        grad = max(0.05, 3.2 * (0.985 ** s) + rnd.uniform(0, 0.35))
        recs.append({"kind": "train", "t": time.time() + s * 0.05, "seq": 0,
                     "step": s, "epoch": s // 60, "loss": loss,
                     "acc": min(0.995, 0.55 + s * 0.0012
                                + rnd.uniform(-0.01, 0.01)),
                     "grad_norm": grad,
                     "lr": 3e-4 * (0.995 ** s),
                     "step_s": 0.05 + rnd.uniform(-0.008, 0.012),
                     "demo": True})
        if with_hardware and s % 4 == 0:
            recs.append({
                "kind": "system", "t": time.time() + s * 0.05, "seq": 0,
                "devices": [{"device": 0,
                             "mem_gib": 3.2 + 0.9 * rnd.random(),
                             "power_w": 45 + 25 * rnd.random(),
                             "util_pct": 30 + 60 * rnd.random()}],
                "gpu_mem_gib": 3.2 + 0.9 * rnd.random(),
                "gpu_power_w": 45 + 25 * rnd.random(),
                "gpu_util_pct": 30 + 60 * rnd.random(),
                "cpu_util_pct": 12 + 25 * rnd.random(),
                "sys_mem_gib": 9.4 + 0.6 * rnd.random(),
                "sys_mem_total_gib": 31.9,
                "aggregate": AGGREGATE_NOTE,
                **({"unavailable": unavailable} if unavailable else {}),
                "demo": True})
    recs.append(dict(task_record(run_id, "DEMO 训练任务", "completed",
                                 total_steps=steps, current_step=steps,
                                 epoch=max(1, steps // 60)), demo=True))
    return recs
