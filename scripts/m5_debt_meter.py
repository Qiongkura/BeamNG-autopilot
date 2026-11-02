"""技术债计量与**棘轮**：只许减少，不许增加。

为什么要它（2026-10-20 评审实测）：库里 165 个文件有 **351 处
``except Exception``，其中 79 处是静默吞掉**（``except Exception: pass``），
另有 163 处 ``print``。这些数字不是"风格问题"：静默吞异常会把"传感器没
数据""文件没写成功"变成一条正常的空结果，评审里已经踩过（`tasklist`
被权限拒绝却当成"没在跑"）。

一次性重写 351 处不现实，且风险全在行为变更上。所以先立**棘轮**：
把当前状态记成基线，之后任何提交若让数字变大就红；想还债就逐模块降。

计量口径（只读、AST、不执行代码）：

* 范围默认 ``beamng_autopilot/`` + ``beamng_autopilot_tech/``（库）。
  ``scripts/`` 是 CLI，``print`` 属正常输出，默认不计入棘轮。
* ``except_exception``：``except Exception``（含 ``as e``）。
* ``silent_except``：处理器体只有 ``pass``/``continue``——真正把错误吞掉的那类。
* ``prints``：``print(...)`` 调用数。
* ``max_func_lines``：最长函数/方法的行数（god-function 的度量）。
* 读文件用 ``utf-8-sig``：仓库里有带 BOM 的源文件，``ast.parse`` 会直接
  抛 ``SyntaxError: invalid non-printable character U+FEFF``。

用法::

    # 看当前数字与最脏的文件
    .venv\\Scripts\\python.exe scripts\\m5_debt_meter.py
    # 与基线比对（CI / 提交前；有增长返回 1）
    .venv\\Scripts\\python.exe scripts\\m5_debt_meter.py --check
    # 还债之后重写基线（必须显式，避免顺手放宽）
    .venv\\Scripts\\python.exe scripts\\m5_debt_meter.py --write
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCOPE = ("beamng_autopilot", "beamng_autopilot_tech")
BASELINE = Path(__file__).resolve().parent / "debt_baseline.json"
METRICS = ("except_exception", "silent_except", "prints", "max_func_lines")
#: 参与**总量**比较的指标。``max_func_lines`` 不进总量：它是"每个文件最长函数"
#: 的和，正常新增一个小函数也会把它推高；它只做逐文件棘轮（见下）。
TOTAL_METRICS = ("except_exception", "silent_except", "prints")
#: 逐文件棘轮：静默吞异常，以及"单个函数又变长了"（新文件从 0 起算）。
PER_FILE_METRICS = ("silent_except", "max_func_lines")


def _handler_is_silent(node: ast.ExceptHandler) -> bool:
    return (len(node.body) == 1
            and isinstance(node.body[0], (ast.Pass, ast.Continue)))


def _handler_is_exception(node: ast.ExceptHandler) -> bool:
    """``except Exception`` / ``except Exception as e``；元组里含它也认。"""
    kind = node.type
    names = []
    if isinstance(kind, ast.Name):
        names = [kind.id]
    elif isinstance(kind, ast.Tuple):
        names = [e.id for e in kind.elts if isinstance(e, ast.Name)]
    return "Exception" in names


def measure_file(path: Path) -> dict:
    """单个文件的指标；解析失败时抛 ``SyntaxError``（不静默跳过）。"""
    source = path.read_text(encoding="utf-8-sig", errors="replace")
    tree = ast.parse(source)
    handlers = [n for n in ast.walk(tree)
                if isinstance(n, ast.ExceptHandler) and _handler_is_exception(n)]
    prints = sum(1 for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "print")
    longest = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None) or node.lineno
            longest = max(longest, int(end) - int(node.lineno) + 1)
    return {
        "lines": len(source.splitlines()),
        "except_exception": len(handlers),
        "silent_except": sum(1 for h in handlers if _handler_is_silent(h)),
        "prints": prints,
        "max_func_lines": longest,
    }


def scan(roots: tuple[str, ...] = DEFAULT_SCOPE) -> dict:
    """``{relpath: metrics}``；按路径排序，保证输出确定。"""
    out: dict = {}
    for name in roots:
        base = Path(name)
        if not base.is_absolute():
            base = ROOT / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                rel = path.relative_to(ROOT).as_posix()
            except ValueError:                  # --scope 指到仓库外（测试用）
                rel = path.resolve().as_posix()
            out[rel] = measure_file(path)
    return out


def totals(files: dict) -> dict:
    """总量指标；``max_func_lines`` 另给"全库最长"，不参与求和比较。"""
    out = {m: sum(int(v.get(m, 0)) for v in files.values()) for m in METRICS}
    out["max_func_lines"] = max(
        (int(v.get("max_func_lines", 0)) for v in files.values()), default=0)
    return out


def load_baseline(path: Path = BASELINE) -> dict:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"scope": tuple(blob.get("scope", DEFAULT_SCOPE)),
            "totals": blob.get("totals", {}),
            "files": blob.get("files", {})}


def compare(files: dict, base: dict) -> tuple[list[str], list[str]]:
    """返回 ``(failures, improvements)``。

    * 任何指标总量变大 → failure（"这边加、那边减"也不允许，因为总量才是债）；
    * ``silent_except`` 在**逐文件**层面变大 → failure（哪怕总量持平）；
    * 变小 → 记 improvement（提示可以重写基线）。
    """
    failures: list[str] = []
    improvements: list[str] = []
    now, before = totals(files), base.get("totals", {})
    for metric in TOTAL_METRICS:
        delta = int(now[metric]) - int(before.get(metric, 0))
        if delta > 0:
            failures.append(f"总量 {metric} 增加 {delta}（{before.get(metric, 0)} → {now[metric]}）")
        elif delta < 0:
            improvements.append(f"总量 {metric} 减少 {-delta}（{before.get(metric, 0)} → {now[metric]}）")
    base_files = base.get("files", {})
    for rel, metrics in files.items():
        old = base_files.get(rel)
        for metric in PER_FILE_METRICS:
            was = int((old or {}).get(metric, 0))
            has = int(metrics.get(metric, 0))
            if has > was:
                failures.append(f"{rel}: {metric} {was} → {has}")
    for rel in base_files:
        if rel not in files:
            improvements.append(f"{rel} 已不存在（基线可收紧）")
    return failures, improvements


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="技术债计量与棘轮（只读 AST）")
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--write", action="store_true", help="用当前状态重写基线")
    ap.add_argument("--check", action="store_true", help="与基线比对，有增长返回 1")
    ap.add_argument("--scope", nargs="+", default=list(DEFAULT_SCOPE),
                    help=f"相对仓库根的扫描目录（默认 {list(DEFAULT_SCOPE)}）")
    ap.add_argument("--json", default=None, help="把当前计量写成 JSON（审计用）")
    args = ap.parse_args(argv)

    files = scan(tuple(args.scope))
    now = totals(files)
    if args.json:
        Path(args.json).write_text(
            json.dumps({"version": 1, "scope": list(args.scope),
                        "totals": now, "files": files},
                       ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[debt] {len(files)} 个文件："
          + " / ".join(f"{k} {now[k]}" for k in TOTAL_METRICS)
          + f" / 最长函数 {now['max_func_lines']} 行")

    if args.write:
        baseline = Path(args.baseline)
        baseline.write_text(
            json.dumps({"version": 1, "scope": list(args.scope),
                        "generated": "m5_debt_meter.py --write",
                        "totals": now, "files": files},
                       ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8")
        print(f"[debt] 基线已写：{baseline}")
        return 0

    if args.check:
        baseline = Path(args.baseline)
        if not baseline.is_file():
            print(f"[debt] 缺基线：{baseline}（先跑 --write 并提交）",
                  file=sys.stderr)
            return 2
        base = load_baseline(baseline)
        failures, improvements = compare(files, base)
        for line in improvements:
            print(f"[debt] 改善：{line}")
        if failures:
            print(f"[debt] RESULT: FAIL - {len(failures)} 项债务增长", file=sys.stderr)
            for line in failures:
                print(f"[debt]   {line}", file=sys.stderr)
            return 1
        print("[debt] RESULT: OK - 没有新增债务")
        return 0

    worst = sorted(files.items(),
                   key=lambda kv: (-kv[1]["silent_except"],
                                   -kv[1]["max_func_lines"]))[:8]
    for rel, m in worst:
        print(f"[debt]   {m['silent_except']:>3} 静默异常 / "
              f"{m['max_func_lines']:>5} 最长函数行 / "
              f"{m['prints']:>4} print  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
