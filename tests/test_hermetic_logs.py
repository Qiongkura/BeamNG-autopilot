"""隔离基线本身也要被钉住：测试不得读写仓库真实的机器级状态。

``tests/conftest.py`` 把整个会话的 ``config.LOGS_DIR`` 指到临时目录，理由是
``logs/experiments/`` 下有两个**机器级共享文件**：

* ``machine_lease.json`` —— 同一 GPU 只允许一个重型实验，无人值守循环持有它；
  测试去抢它，会造成两种情况，两种都坏：循环在跑 → 测试假红；循环没跑 →
  测试抢到真实租约，**跑一次回归门就能挡住真实实验**。
* ``gpu_ledger_machine.json`` —— 全机累计 GPU 分钟数（每日预算），测试写进去
  会污染真实预算。

这里的断言不依赖 ``conftest`` 的实现细节，而是直接问代码"你会去哪里拿租约"，
再要求那个位置不在仓库 ``logs/`` 下；``conftest`` 被误删/被绕过时，这个文件
会立刻变红。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402


def _load_autoloop():
    spec = importlib.util.spec_from_file_location(
        "_m5_seg_autoloop_isolation", ROOT / "scripts" / "m5_seg_autoloop.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _real_logs() -> Path:
    return (ROOT / "logs").resolve()


def test_session_logs_dir_is_not_the_repo_logs():
    got = Path(config.LOGS_DIR).resolve()
    assert got != _real_logs(), (
        f"测试会话的 LOGS_DIR 指向真实产物目录 {got}："
        "测试会与正在跑的无人值守循环互相干扰")
    assert _real_logs() not in got.parents


def test_machine_lease_and_gpu_ledger_are_isolated_together():
    """租约是真实函数算出来的路径；GPU 台账与它同目录，一起被隔离。"""
    loop = _load_autoloop()
    lease = Path(loop.machine_lease().path).resolve()
    assert lease.name == "machine_lease.json"
    assert lease.parent.name == "experiments"
    assert lease != _real_logs() and _real_logs() not in lease.parents
    # 全机 GPU 台账按同一目录约定落盘（scripts/m5_seg_autoloop.py 的
    # gpu_ledger 路径就是 lease.parent / "gpu_ledger_machine.json"）
    assert lease.parent != _real_logs() / "experiments"
