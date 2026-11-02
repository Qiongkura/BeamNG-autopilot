"""技术债棘轮：计量口径要稳，增长必须红，减少必须能被看见。

棘轮的价值全在"它判的增长是真的增长"：口径一抖（例如把 ``except (A, Exception)``
漏掉、或 BOM 文件解析失败被静默跳过），它就会既漏报又把基线写坏。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_debt_meter", ROOT / "scripts" / "m5_debt_meter.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_debt_meter"] = mod
    spec.loader.exec_module(mod)
    return mod


def _file(tmp_path: Path, name: str, body: str, *, bom: bool = False) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    data = ("\ufeff" if bom else "") + body
    p.write_text(data, encoding="utf-8")
    return p


def test_measure_counts_only_the_intended_patterns(tmp_path: Path) -> None:
    mod = _load()
    p = _file(tmp_path, "a.py", (
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception:\n"          # 1：静默
        "        pass\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as e:\n"      # 2：不静默（有语句）
        "        print(e)\n"
        "    try:\n"
        "        pass\n"
        "    except (ValueError, Exception):\n"   # 3：元组里含 Exception
        "        continue\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError:\n"          # 不算
        "        pass\n"
        "    print('x')\n"                  # prints=2（含上面一处）
        "\n"
        "def g():\n"
        "    return 1\n"
    ))
    m = mod.measure_file(p)
    assert m["except_exception"] == 3
    assert m["silent_except"] == 2, m
    assert m["prints"] == 2
    assert m["max_func_lines"] == 18, m


def test_a_utf8_bom_does_not_break_the_parser(tmp_path: Path) -> None:
    """仓库里真有带 BOM 的源文件；ast.parse(str) 会直接 SyntaxError。"""
    mod = _load()
    p = _file(tmp_path, "bom.py", '"""doc"""\n\ndef f():\n    return 1\n',
              bom=True)
    m = mod.measure_file(p)
    assert m["lines"] >= 4


def test_growth_fails_and_reduction_is_reported() -> None:
    mod = _load()
    base = {"totals": {"except_exception": 2, "silent_except": 1,
                       "prints": 5, "max_func_lines": 10},
            "files": {"a.py": {"silent_except": 1, "max_func_lines": 10}}}
    grown = {"a.py": {"except_exception": 3, "silent_except": 1,
                      "prints": 5, "max_func_lines": 10}}
    fails, _ = mod.compare(grown, base)
    assert any("except_exception" in f for f in fails)

    smaller = {"a.py": {"except_exception": 1, "silent_except": 0,
                        "prints": 4, "max_func_lines": 9}}
    fails, imp = mod.compare(smaller, base)
    assert not fails
    assert any("减少" in i for i in imp)


def test_silent_except_is_ratcheted_per_file() -> None:
    """总量不变也不行：把静默异常从干净的模块挪到另一个也算净增。"""
    mod = _load()
    base = {"totals": {"except_exception": 2, "silent_except": 1,
                       "prints": 0, "max_func_lines": 10},
            "files": {"a.py": {"silent_except": 1, "max_func_lines": 5},
                      "b.py": {"silent_except": 0, "max_func_lines": 5}}}
    moved = {"a.py": {"silent_except": 0, "except_exception": 1,
                      "prints": 0, "max_func_lines": 5},
             "b.py": {"silent_except": 1, "except_exception": 1,
                      "prints": 0, "max_func_lines": 5}}
    fails, _ = mod.compare(moved, base)
    assert any(f.startswith("b.py") for f in fails), fails


def test_write_then_check_round_trips_on_a_private_scope(tmp_path: Path) -> None:
    mod = _load()
    _file(tmp_path / "lib", "mod.py", "try:\n    pass\nexcept Exception:\n"
                                      "    pass\n")
    baseline = tmp_path / "base.json"
    assert mod.main(["--scope", str(tmp_path / "lib"),
                     "--baseline", str(baseline), "--write"]) == 0
    blob = json.loads(baseline.read_text(encoding="utf-8"))
    assert blob["totals"]["silent_except"] == 1
    assert mod.main(["--scope", str(tmp_path / "lib"),
                     "--baseline", str(baseline), "--check"]) == 0
    _file(tmp_path / "lib", "mod2.py",
          "try:\n    pass\nexcept Exception:\n    pass\n")
    assert mod.main(["--scope", str(tmp_path / "lib"),
                     "--baseline", str(baseline), "--check"]) == 1


def test_the_repo_baseline_is_current() -> None:
    """真实基线必须与当前库一致：有人还了债没重写基线，这里会提示。"""
    mod = _load()
    assert mod.main(["--check"]) == 0
