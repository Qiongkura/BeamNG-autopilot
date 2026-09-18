"""提交纪律检查器：一个提交只能动一个模块。

用户多次强调「按模块提交」，但靠人记会漏（2026-09-18 就出现过
``safety_monitor.py`` + ``fsd_drive.py`` 混在同一提交里）。这个脚本把
规则变成可执行的检查，推送前跑一次：

    .venv\\Scripts\\python.exe scripts\\check_commit_scope.py           # origin/main..HEAD
    .venv\\Scripts\\python.exe scripts\\check_commit_scope.py HEAD~5..HEAD
    .venv\\Scripts\\python.exe scripts\\check_commit_scope.py --last 8

规则（见 AGENTS.md「提交纪律」）：

* 模块边界 = ``beamng_autopilot/<子包或顶层模块>`` / ``scripts`` /
  ``tests`` / ``docs`` / 顶层文件（如 ``AGENTS.md``）。
* 一个提交只允许**一个**模块，外加它自己的测试（``tests/`` 随被改模块走）。
* 违反时打印违规文件清单并返回 1，便于 CI / 推送前拦截。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 测试可以跟随它验证的模块，不算第二个模块。
_TEST_DIR = "tests"


def _module_of(path: str) -> str:
    """把仓库内路径归到一个模块名。"""
    p = path.replace("\\", "/").strip()
    if not p:
        return ""
    parts = p.split("/")
    if parts[0] == "beamng_autopilot":
        if len(parts) >= 3 and parts[1] not in ("__pycache__",):
            # 子包（lane/、vision/、planning/ ...）算一个模块
            return f"beamng_autopilot/{parts[1]}"
        return f"beamng_autopilot/{parts[1]}"
    if len(parts) == 1:
        return parts[0]
    return parts[0]


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                         text=True, check=True)
    return out.stdout


def commits_in_range(rev_range: str | None, last: int | None) -> list[str]:
    if last is not None:
        rev_range = f"HEAD~{int(last)}..HEAD" if last > 0 else "HEAD"
    if not rev_range:
        # 默认：上游..HEAD（上游不存在时用最近 1 个提交）
        try:
            _git("rev-parse", "--verify", "@{u}")
            rev_range = "@{u}..HEAD"
        except subprocess.CalledProcessError:
            rev_range = "HEAD"
    out = _git("log", "--format=%H", rev_range).split()
    return list(reversed(out))          # 时间正序


def check_commit(sha: str) -> list[str]:
    """返回该提交的违规说明（空列表 = 合规）。"""
    subject = _git("log", "-1", "--format=%s", sha).strip()
    files = [f for f in _git("show", "--name-only", "--format=",
                             sha).splitlines() if f.strip()]
    modules = sorted({m for m in (_module_of(f) for f in files)
                      if m and m != _TEST_DIR})
    if len(modules) <= 1:
        return []
    return [f"{sha[:7]} {subject!r} 动了 {len(modules)} 个模块: "
            f"{', '.join(modules)}\n      files: "
            + ", ".join(sorted(files))]


def main() -> int:
    ap = argparse.ArgumentParser(description="一个提交只能动一个模块")
    ap.add_argument("range", nargs="?", default=None,
                    help="git 范围，默认 @{u}..HEAD（无上游则最近一个提交）")
    ap.add_argument("--last", type=int, default=None,
                    help="检查最近 N 个提交")
    args = ap.parse_args()

    shas = commits_in_range(args.range, args.last)
    if not shas:
        print("没有需要检查的提交（范围为空）")
        return 0
    bad: list[str] = []
    for sha in shas:
        bad.extend(check_commit(sha))
    print(f"检查了 {len(shas)} 个提交")
    if not bad:
        print("RESULT: OK - 每个提交只动一个模块")
        return 0
    print(f"RESULT: {len(bad)} 个提交跨模块（需要拆分）")
    for b in bad:
        print("  - " + b)
    return 1


if __name__ == "__main__":
    sys.exit(main())
