"""提交纪律检查器：中文提交标题下也必须能跑。

它坏了就等于推送前的门不存在（AGENTS.md 要求推送前自查）。2026-10-20
实测：``text=True`` 按 GBK 解码，遇到中文主题抛 ``UnicodeDecodeError``，
stdout 变 None，``check_commit`` 崩在 ``.strip()``——检查器从来没能跑完过。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_commit_scope", ROOT / "scripts" / "check_commit_scope.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_commit_scope"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_git_output_is_decoded_as_utf8() -> None:
    """真实 git 调用必须能解出中文主题（不是乱码、不是 None）。"""
    mod = _load()
    subject = mod._git("log", "-1", "--format=%s").strip()
    assert subject, "读不到最近一个提交的标题"
    assert "\ufffd" not in subject, f"主题被解码坏了：{subject!r}"
    assert any(ord(ch) > 127 for ch in subject), (
        "本仓库提交标题都是中文；解不出非 ASCII 说明解码路径又退回 locale 了")


def test_module_boundaries() -> None:
    mod = _load()
    assert mod._module_of("beamng_autopilot/lane/reference.py") == \
        "beamng_autopilot/lane"
    assert mod._module_of("beamng_autopilot/config.py") == \
        "beamng_autopilot/config.py"
    assert mod._module_of("scripts/a.py") == "scripts"
    assert mod._module_of("tests/test_x.py") == "tests"
    assert mod._module_of("docs/README.md") == "docs"


def test_a_commit_touching_two_modules_is_reported(monkeypatch) -> None:
    mod = _load()

    def fake_git(*args):
        if args[0] == "log":
            return "feat: 混模块\n"
        return "beamng_autopilot/safety_monitor.py\nbeamng_autopilot/fsd_drive.py\n"

    monkeypatch.setattr(mod, "_git", fake_git)
    bad = mod.check_commit("deadbeef")
    assert bad and "deadbee" in bad[0]


def test_tests_travel_with_the_module(monkeypatch) -> None:
    """``tests/`` 跟着被改模块走，不算第二个模块。"""
    mod = _load()

    def fake_git(*args):
        if args[0] == "log":
            return "feat: 一个模块加它的测试\n"
        return ("beamng_autopilot/experiments/controller.py\n"
                "tests/test_experiments.py\n")

    monkeypatch.setattr(mod, "_git", fake_git)
    assert mod.check_commit("abc1234") == []


def test_the_checker_runs_end_to_end_without_crashing() -> None:
    """端到端：脚本本身能跑完并给出 RESULT（回归的是"崩在 .strip()"）。"""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_commit_scope.py"),
         "--last", "1"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120)
    combined = (r.stdout or "") + (r.stderr or "")
    assert "Traceback" not in combined, combined[-800:]
    assert "检查了" in r.stdout, combined[-800:]
    assert r.returncode in (0, 1), f"rc={r.returncode}"
