"""docs/ 索引生成器：224 篇报告按系列、类型、配对与引用关系摊开。

为什么需要它（2026-10-20 复核）：``docs/`` 有 224 个文件、205 个 md，其中
README 只链到 9 个、约 196 个没有任何入口；实验报告靠文件名约定
（``T16_<主题>_PREREG_<日期>`` / ``..._RESULT_<日期>``）自行配对，日期还会
漂移（PREREG 20261012 配 RESULT 20261013）。人肉维护必然漏，所以索引由
脚本生成：**机器能算的（类型/日期/配对/被引用）不写进文档**。

用法::

    # 看统计，不写文件
    .venv\\Scripts\\python.exe scripts\\m5_docs_index.py
    # 写出 docs/INDEX.md
    .venv\\Scripts\\python.exe scripts\\m5_docs_index.py --write
    # 门禁：索引与现状不一致时返回 1（README 与 CI 都可以用）
    .venv\\Scripts\\python.exe scripts\\m5_docs_index.py --check

只读 docs/ 与仓库根的 README.md / AGENTS.md，不碰 ``logs/``。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX_NAME = "INDEX.md"

#: 文件名里的类型标记（按优先级匹配；写这里而不是散在代码里）。
KIND_RULES = (
    ("PREREG", "预注册"),
    ("RESULT", "结果"),
    ("ATTRIB", "定因"),
    ("PROTOCOL", "协议"),
    ("REVIEW", "评审"),
    ("REPORT", "报告"),
    ("STATUS", "状态"),
    ("HANDOFF", "交接"),
    ("EVIDENCE", "证据"),
    ("AUDIT", "审计"),
    ("DESIGN", "设计"),
    ("PLAN", "计划"),
    ("INDEX", "索引"),
)

_DATE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")


def parse_name(name: str) -> dict:
    """从文件名里抽系列/类型/日期/配对键。解析不出来就给保守默认值。"""
    stem = Path(name).stem
    date = ""
    m = _DATE_RE.search(stem)
    if m:
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    kind, token = "其它", ""
    upper = stem.upper()
    for marker, label in KIND_RULES:
        if marker in upper:
            kind, token = label, marker
            break
    series = stem.split("_")[0] if "_" in stem else "其它"
    if series.upper() != series and not series.upper().startswith("T"):
        series = "其它"
    # 配对键 = 去掉日期与类型词后的主题；日期漂移的 PRE/RE 也能配上。
    key = stem
    if m:
        key = key.replace(m.group(0), "")
    key = re.sub(rf"(?i)_{token}_|{token}", "", key) if token else key
    key = re.sub(r"_+", "_", key).strip("_").lower()
    return {"stem": stem, "date": date, "kind": kind, "series": series,
            "key": key, "token": token}


def _title_of(path: Path) -> str:
    """第一条一级标题；没有就取第一行非空文本。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines()[:40]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:80]
    return ""


def collect(docs_dir: Path = DOCS) -> list[dict]:
    """列出 docs 下的 md（含子目录，跳过 INDEX.md 自己）。"""
    out: list[dict] = []
    for path in sorted(docs_dir.rglob("*.md")):
        if path.name == INDEX_NAME:
            continue
        rel = path.relative_to(docs_dir).as_posix()
        out.append({**parse_name(path.name), "rel": rel,
                    "title": _title_of(path)})
    return out


def _citation_blob(docs_dir: Path) -> list[tuple[str, str]]:
    """所有参与"被引用"统计的文本：docs/**/*.md + README + AGENTS。"""
    blobs: list[tuple[str, str]] = []
    for path in sorted(docs_dir.rglob("*.md")):
        if path.name == INDEX_NAME:
            continue
        blobs.append((path.relative_to(docs_dir).as_posix(),
                      path.read_text(encoding="utf-8", errors="replace")))
    for name in ("README.md", "README.en.md", "AGENTS.md"):
        path = ROOT / name
        if path.is_file():
            blobs.append((name, path.read_text(encoding="utf-8",
                                               errors="replace")))
    return blobs


def annotate(entries: list[dict], blob: list[tuple[str, str]]) -> None:
    """就地写入 ``cited``（被其它文件提到几次）与 ``pair``（配对文件）。"""
    keys: dict[str, dict] = {}
    for e in entries:
        keys.setdefault(e["key"], {"预注册": None, "结果": None})
        if e["kind"] == "预注册":
            keys[e["key"]]["预注册"] = e
        elif e["kind"] == "结果":
            keys[e["key"]]["结果"] = e
    for e in entries:
        e["cited"] = sum(
            1 for rel, text in blob
            if rel != e["rel"] and e["stem"] in text)
        pair = keys.get(e["key"], {})
        if e["kind"] == "预注册":
            e["pair"] = pair.get("结果")
        elif e["kind"] == "结果":
            e["pair"] = pair.get("预注册")
        else:
            e["pair"] = None


def _sort_key(e: dict):
    return (e["series"], "".join(chr(0x10FFFF - ord(c)) for c in e["date"])
            if e["date"] else "", e["rel"])


def render(entries: list[dict]) -> str:
    """生成 INDEX.md 全文（确定性输出：同样输入 → 同样字节）。"""
    kinds: dict[str, int] = {}
    for e in entries:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    orphan = sum(1 for e in entries if not e["cited"])
    paired = sum(1 for e in entries if e["kind"] in ("预注册", "结果")
                 and e["pair"] is not None)
    lines = [
        "# 文档索引（脚本生成）",
        "",
        "> 生成：`.venv\\Scripts\\python.exe scripts\\m5_docs_index.py --write`；"
        "校验：`--check`（过期返回 1）。",
        "> 手改本文件会被 `--check` 判为过期——要加说明请写进对应文档或生成器。",
        "",
        f"共 **{len(entries)}** 篇："
        + " / ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
        + f"；预注册↔结果已配对 **{paired}** 篇；**{orphan}** 篇没有被任何文档"
        "（含 README/AGENTS）引用，集中列在文末「无正文入口」。",
        "",
        "「被引用」= 其它文档正文里出现该文件名（不含扩展名）的次数；"
        "0 表示没有正文入口，需要补链或归档。",
        "",
    ]
    series: dict[str, list[dict]] = {}
    for e in sorted(entries, key=_sort_key):
        series.setdefault(e["series"], []).append(e)
    for name in sorted(series, key=lambda s: (-len(series[s]), s)):
        rows = series[name]
        lines.append(f"## {name}（{len(rows)}）")
        lines.append("")
        lines.append("| 日期 | 类型 | 文件 | 标题 | 配对 | 被引用 |")
        lines.append("| --- | --- | --- | --- | --- | ---: |")
        for e in rows:
            pair = f"↔ {e['pair']['stem']}" if e.get("pair") else ""
            title = (e["title"] or "").replace("|", "\\|")
            lines.append(
                f"| {e['date'] or '—'} | {e['kind']} | [{e['rel']}]({e['rel']}) "
                f"| {title} | {pair} | {e['cited']} |")
        lines.append("")

    # 无正文入口的文档：索引本身要保证它们**至少有一条可达路径**，否则
    # "有索引"仍等于"找不到"（2026-10-20 复核：205 个 md 里 28 个零引用）。
    orphans = [e for e in entries if not e["cited"]]
    lines.append(f"## 无正文入口（{len(orphans)}）")
    lines.append("")
    lines.append("这些文档没有任何正文引用（不含本索引）。要么在相关报告里补链，"
                 "要么确认为历史归档；两种处理都不改变文件本身。")
    lines.append("")
    if orphans:
        lines.append("| 日期 | 类型 | 文件 | 标题 |")
        lines.append("| --- | --- | --- | --- |")
        for e in sorted(orphans, key=_sort_key):
            title = (e["title"] or "").replace("|", "\\|")
            lines.append(f"| {e['date'] or '—'} | {e['kind']} | "
                         f"[{e['rel']}]({e['rel']}) | {title} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="docs/ 索引生成器")
    ap.add_argument("--docs-dir", default=str(DOCS))
    ap.add_argument("--write", action="store_true", help="写出 docs/INDEX.md")
    ap.add_argument("--check", action="store_true",
                    help="索引与现状不一致时返回 1（不写文件）")
    args = ap.parse_args(argv)

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        print(f"[index] docs 目录不存在：{docs_dir}", file=sys.stderr)
        return 2
    entries = collect(docs_dir)
    annotate(entries, _citation_blob(docs_dir))
    text = render(entries)
    target = docs_dir / INDEX_NAME
    stats = (f"{len(entries)} 篇 / "
             f"无引用 {sum(1 for e in entries if not e['cited'])} 篇 / "
             f"已配对 {sum(1 for e in entries if e.get('pair'))} 篇")
    if args.check:
        if not target.is_file():
            print(f"[index] 缺少 {target}（先跑 --write）", file=sys.stderr)
            return 1
        if target.read_text(encoding="utf-8") != text:
            print("[index] 索引已过期：重跑 --write 并提交", file=sys.stderr)
            return 1
        print(f"[index] OK：{stats}")
        return 0
    if args.write:
        target.write_text(text, encoding="utf-8")
        print(f"[index] 已写 {target}：{stats}")
        return 0
    print(f"[index] {stats}（dry-run：加 --write 落盘）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
