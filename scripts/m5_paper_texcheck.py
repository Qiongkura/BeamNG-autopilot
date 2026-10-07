"""投稿源包静态检查：没有 LaTeX 引擎时能查的部分（不声称编译通过）。

检查项：
1. 环境配平（``\\begin{X}`` / ``\\end{X}`` 计数一致，按出现顺序配对）；
2. 每个 ``\\includegraphics{dir/fig.png}`` 的文件存在；
3. 每个 ``\\cite{k}`` 都能在 ``\\bibitem{k}`` 里找到；
4. 未替换的模板残留：``{@fig:``、``[@``、``Table name to be confirmed``、``(Generated from``；
5. 花括号配平（忽略转义 ``\\{`` ``\\}``）；
6. 摘要环境内出现且仅出现一次。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_paper_texcheck.py docs\\paper\\submission\\main.tex
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def check(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8")
    root = path.parent
    bad: list[str] = []

    # 1) 环境配平
    stack: list[str] = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", txt):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack:
                bad.append(f"\\end{{{env}}} 没有对应的 \\begin")
            elif stack[-1] != env:
                bad.append(f"\\end{{{env}}} 与 \\begin{{{stack[-1]}}} 不配对")
                stack.pop()
            else:
                stack.pop()
    for env in stack:
        bad.append(f"\\begin{{{env}}} 没有 \\end")

    # 2) 图存在
    for m in re.finditer(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", txt):
        p = root / m.group(1)
        if not p.is_file():
            bad.append(f"缺图: {m.group(1)}")

    # 3) 引文闭环
    cites = set(re.findall(r"\\cite\{([^}]+)\}", txt))
    items = set(re.findall(r"\\bibitem\{([^}]+)\}", txt))
    for c in sorted(cites - items):
        bad.append(f"\\cite{{{c}}} 没有对应 \\bibitem")
    for k in sorted(items - cites):
        bad.append(f"\\bibitem{{{k}}} 未被引用（IEEE 要求只列引用过的）")

    # 4) 模板残留
    for pat, what in ((r"\{@fig:", "未替换的图引用 {@fig:...}"),
                      (r"\[@", "未替换的引文 [@...]"),
                      ("Table name to be confirmed", "题录占位（期刊名待核对）"),
                      (r"\(Generated from", "参考文献占位行"),
                      ("\\todo", "遗留 \\todo")):
        if re.search(pat, txt):
            bad.append(f"残留: {what}")

    # 5) 花括号配平（忽略 \{ \}）
    stripped = re.sub(r"\\[{}]", "", txt)
    if stripped.count("{") != stripped.count("}"):
        bad.append(f"花括号不配平: {{={stripped.count('{')} }}={stripped.count('}')}")

    # 6) 摘要只出现一次
    n_abs = len(re.findall(r"\\begin\{abstract\}", txt))
    if n_abs != 1:
        bad.append(f"abstract 环境出现 {n_abs} 次")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tex", nargs="+")
    args = ap.parse_args()
    total = 0
    for t in args.tex:
        p = Path(t)
        bad = check(p)
        total += len(bad)
        print(f"[texcheck] {p}: {'OK' if not bad else f'{len(bad)} 个问题'}")
        for b in bad:
            print("   -", b)
    print(f"\n=== 静态检查合计 {total} 个问题（未编译 LaTeX：本机无引擎）===")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
