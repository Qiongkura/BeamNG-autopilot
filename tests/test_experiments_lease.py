"""机器级学习资源租约（方案 G02）：原子、可验证、长任务不被抢。

方案点名的旧事实：`controller.py` 用"先判断存在再写文件"，且存活进程的锁超过 6 小时
可能被接管。本文件钉四个反例/正例：

1. **并发只有一个成功**（原子创建）；
2. **长任务不被抢**：持有者仍存活、只是心跳旧 → 拒绝接管（旧实现会在 6 小时后抢走）；
3. **死掉的持有者可以回收**，且理由写清楚；
4. **PID 复用**（同号不同创建时间）按"不是本人"处理，不能靠 pid 蒙混。

探测函数全部注入，不需要真进程。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.controller import MachineLease  # noqa: E402


def _lease(tmp_path: Path, *, alive, created, now=None) -> MachineLease:
    return MachineLease(tmp_path / "machine_lease.json",
                        alive_fn=alive, created_fn=created,
                        now_fn=now or time.time)


def _plant(path: Path, *, pid: int, created: str, age_s: float,
           hb_age_s: float) -> None:
    """预置一份租约文件（模拟别人持有）。"""
    now = time.time()
    path.write_text(json.dumps({"pid": pid, "created": created,
                                "t": now - age_s, "hb": now - hb_age_s,
                                "host": "TEST"}), encoding="utf-8")


def test_only_one_owner_can_hold_the_lease(tmp_path):
    created = {"111": "t-111", "222": "t-222"}
    L = _lease(tmp_path, alive=lambda p: True, created=lambda p: created.get(str(p)))
    first = L.acquire(pid=111)
    assert first["acquired"] is True and first["already_held"] is False
    second = L.acquire(pid=222)
    assert second["acquired"] is False, "第二个持有者必须被拒"
    assert "pid 111" in second["reason"], second
    # 同一所有者重复获取是安全的（run -> rounds 不会自锁）
    again = L.acquire(pid=111)
    assert again["acquired"] is True and again["already_held"] is True


def test_a_live_holder_is_not_stolen_even_with_a_stale_heartbeat(tmp_path):
    """旧实现 6 小时无条件接管：长任务（8 小时常驻）会被抢走。"""
    L = _lease(tmp_path, alive=lambda p: True, created=lambda p: "t-111")
    _plant(tmp_path / "machine_lease.json", pid=111, created="t-111",
           age_s=10 * 3600, hb_age_s=9 * 3600)          # 10 小时前拿到，9 小时没心跳
    got = L.acquire(pid=222)
    assert got["acquired"] is False, "活着就不许抢"
    assert "not stealing a long task" in got["reason"], got
    st = got["status"]
    assert st["owner_alive"] is True and st["heartbeat_stale"] is True
    # 显式 force 才允许抢，并写明理由
    forced = L.acquire(pid=222, force=True)
    assert forced["acquired"] is True
    assert forced["recovery_reason"] == "forced by caller", forced


def test_a_dead_owner_is_recovered(tmp_path):
    L = _lease(tmp_path, alive=lambda p: False, created=lambda p: "t-111")
    _plant(tmp_path / "machine_lease.json", pid=111, created="t-111",
           age_s=3600, hb_age_s=3600)
    got = L.acquire(pid=222)
    assert got["acquired"] is True and got["recovered_stale"] is True
    assert got["recovery_reason"] == "owner not alive", got


def test_pid_reuse_is_not_treated_as_the_same_owner(tmp_path):
    """同号不同创建时间 = 另一个进程（PID 复用）：不能靠 pid 蒙混。"""
    L = _lease(tmp_path, alive=lambda p: True, created=lambda p: "t-NEW")
    _plant(tmp_path / "machine_lease.json", pid=111, created="t-OLD",
           age_s=60, hb_age_s=1)
    got = L.acquire(pid=111)          # 同一个 pid 号，但创建时间变了
    assert got["acquired"] is True and got.get("recovered_stale") is True
    assert got["recovery_reason"] == "owner pid was reused", got


def test_heartbeat_and_release_respect_ownership(tmp_path):
    created = {"111": "t-111", "222": "t-222"}
    L = _lease(tmp_path, alive=lambda p: True, created=lambda p: created.get(str(p)))
    L.acquire(pid=111)
    before = json.loads((tmp_path / "machine_lease.json").read_text(encoding="utf-8"))
    assert L.heartbeat() is False, "不是持有者（本进程不是 111）不能刷心跳"
    after = json.loads((tmp_path / "machine_lease.json").read_text(encoding="utf-8"))
    assert after["hb"] == before["hb"], "非持有者不许改动租约文件"
    L.release()
    assert (tmp_path / "machine_lease.json").exists(), "非持有者不许删除租约"
