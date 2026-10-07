"""驾驶栈的整轮墙钟上限（--max-wall-s）必须有：E5 的孤儿实例就是这么来的。

背景（2026-10-07 实测）：strict + sensor 的放置循环按设计不会退出（"停住、恢复、继续"契约），
我用外部 `timeout` 去砍 → 驱动的 `finally` 没执行 → 游戏实例变孤儿 → 端口/CEF 冲突，
后续启动全部失败。因此上限必须在**驱动内部**，且 CLI 要暴露它。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "beamng_autopilot" / "fsd_drive.py"
CLI = ROOT / "scripts" / "m5_fsd_drive.py"


def test_cli_exposes_max_wall_s() -> None:
    s = CLI.read_text(encoding="utf-8")
    assert "--max-wall-s" in s, "CLI 必须暴露整轮墙钟上限"
    # argparse 由 --max-wall-s 派生 args.max_wall_s，源码里不必写下划线形式


def test_module_enforces_wall_cap_in_both_loops() -> None:
    s = MOD.read_text(encoding="utf-8")
    assert "_wall_hit = lambda" in s, "run() 入口要定义墙钟判定"
    # 放置循环与行驶循环各有一处检查（放置超限 return 3 = 未获得放置）
    assert s.count("if _wall_hit():") == 2, "放置与行驶两处都要检查"
    assert "return 3" in s, "放置阶段超限要有区别于 0/1/2 的返回码"


def test_wall_cap_is_opt_in() -> None:
    """默认 None：不设上限时行为与既有运行一致（无人值守长跑不能被悄悄截断）。"""
    s = MOD.read_text(encoding="utf-8")
    assert 'getattr(args, "max_wall_s", None)' in s
    assert "bool(_wall_max)" in s
