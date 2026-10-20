"""测试会话的隔离基线：不碰仓库里真实的 ``logs/``。

为什么必须这么做（2026-09-26 实测）：``logs/experiments/`` 下有**机器级共享
状态**，不是测试的私有产物——

* ``machine_lease.json``：同一 GPU 只允许一个重型实验，无人值守循环持有它；
* ``gpu_ledger_machine.json``：全机累计 GPU 分钟数，用来卡每日预算。

任何在**进程内**调 ``loop.main()`` 的测试都会去抢那份真实租约：

* 循环正在跑 → 租约被活着的 pid 持有（拒绝抢占）→ 测试拿到 exit 4 而期望
  5/0，**假红**；实测同一批测试在租约空闲时全绿、循环起来后全红；
* 循环没跑 → 测试抢到并改写真实租约 → **跑一次回归门就可能挡住真实实验**，
  还会把 GPU 分钟数记进真实台账。

``beamng_autopilot/config.py`` 早就留了开关（``BEAMNG_LOGS_DIR``，注释写明
"测试、CI 与沙箱用"），只是没有 conftest 去接。这里在 pytest 导入任何测试
模块**之前**把它指到会话级临时目录，于是：

* 单个测试自己的 ``monkeypatch.setattr(config, "LOGS_DIR", tmp_path)`` 仍然
  优先（它们本来就自己指到 tmp）；
* 子进程测试各自设 ``env["BEAMNG_LOGS_DIR"]``，不受影响；
* 显式用 ``ROOT / "logs" / ...`` 的测试（如 ``test_phase0_contracts`` 用真实
  权重文件）不走 ``config``，也不受影响。

守门的第二半是 ``_hermetic_logs_dir`` 那个 session 夹具：如果 ``LOGS_DIR``
最终仍指向仓库真实 ``logs/``（例如有人在 shell 里设了 ``BEAMNG_LOGS_DIR``），
就直接中止会话并说清原因，而不是静默跑出一堆假红/假绿。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 本会话自建的沙箱目录（只有自己建的那个才在收尾时删除，避免误删别人指定的）。
_CREATED_SANDBOX: str | None = None

if not os.environ.get("BEAMNG_LOGS_DIR"):
    _CREATED_SANDBOX = tempfile.mkdtemp(prefix="beamng-pytest-logs-")
    os.environ["BEAMNG_LOGS_DIR"] = _CREATED_SANDBOX


@pytest.fixture(scope="session", autouse=True)
def _hermetic_logs_dir():
    from beamng_autopilot import config

    real = (ROOT / "logs").resolve()
    got = Path(config.LOGS_DIR).resolve()
    if got == real or real in got.parents:
        # 中止而不是 pytest.fail：这是配置错误，不是某个用例的失败；让人一眼
        # 看到原因，而不是几千条 error。
        pytest.exit(
            f"测试会话的 LOGS_DIR 指向真实产物目录 {got}。logs/experiments/ 下"
            f"有机器级租约与全机 GPU 台账：测试会与正在跑的无人值守循环互相"
            f"干扰（它让你假红，你让它停下来）。请 unset BEAMNG_LOGS_DIR（或"
            f"指向别的目录）后重跑。", returncode=3)
    yield
    if _CREATED_SANDBOX:
        shutil.rmtree(_CREATED_SANDBOX, ignore_errors=True)


# ---------------------------------------------------------------------------
# 子进程统一 UTF-8（2026-10-20 实测）
#
# 现象：同一份代码，``pytest tests/`` 在普通 PowerShell 里 7 个失败、
# 加 ``PYTHONUTF8=1`` 后 0 个失败。根因不是逻辑，是**控制台代码页**：
#
# * 子进程（``sys.executable ...``）继承中文控制台 GBK，打印 "↔"（U+2194）
#   或其它非 GBK 字符时直接 ``UnicodeEncodeError``，测试抓到的 stdout
#   是半截/空，于是"命令没输出"这类断言假红；
# * 父进程 ``subprocess.run(..., text=True)`` 又按 locale（GBK）解码子进程
#   写出的 UTF-8 字节，中文断言变成乱码。
#
# 两边都要钉死：子进程通过 ``PYTHONUTF8``（会被继承）强制 UTF-8 模式，
# 父进程的文本模式默认按 UTF-8 解码。显式传 ``encoding=`` 的调用点不受影响
# （它们本来就是对的那部分）。

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

if os.name == "nt":
    import subprocess

    _popen_init = subprocess.Popen.__init__

    def _utf8_text_mode(self, *args, **kwargs):
        if (kwargs.get("text") or kwargs.get("universal_newlines")) \
                and not kwargs.get("encoding"):
            kwargs["encoding"] = "utf-8"
        _popen_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _utf8_text_mode
