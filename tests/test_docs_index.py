"""docs 索引生成器：类型/日期/配对/引用统计必须**机器可算且可复现**。

索引的价值全在"它是不是真的对得上"：配对错了会让人误读实验结果，
漏算引用会把有入口的文档判成孤儿。这里用临时 docs 树钉住语义。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_docs_index", ROOT / "scripts" / "m5_docs_index.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_docs_index"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    _write(docs / "ZZ_FIXTURE_PREREG_20261012.md", "# 预注册：剂量\n\n正文\n")
    _write(docs / "ZZ_FIXTURE_RESULT_20261013.md", "# 结果：剂量\n\n结论\n")
    _write(docs / "ZZ_ORPHAN_REPORT_20261001.md", "# 孤儿报告\n\n没人提\n")
    _write(docs / "ZZ_LINKER_REVIEW_20261002.md",
           "# 链接者\n\n见 ZZ_FIXTURE_RESULT_20261013 的结论。\n")
    return docs


def test_parse_name_reads_the_conventions() -> None:
    mod = _load()
    got = mod.parse_name("T16_ECA3_CROSSPOOL_PREREG_20261018.md")
    assert got["date"] == "2026-10-18"
    assert got["kind"] == "预注册"
    assert got["series"] == "T16"
    assert got["stem"] == "T16_ECA3_CROSSPOOL_PREREG_20261018"


def test_pairing_survives_date_drift() -> None:
    """PREREG 20261012 与 RESULT 20261013 必须配成一对（实测就这么漂的）。"""
    mod = _load()
    a = mod.parse_name("T16_X_PREREG_20261012.md")
    b = mod.parse_name("T16_X_RESULT_20261013.md")
    assert a["key"] == b["key"], (a["key"], b["key"])


def test_index_pairs_counts_citations_and_marks_orphans(tmp_path: Path) -> None:
    mod = _load()
    docs = _tree(tmp_path)
    entries = mod.collect(docs)
    mod.annotate(entries, mod._citation_blob(docs))
    by_stem = {e["stem"]: e for e in entries}

    pre = by_stem["ZZ_FIXTURE_PREREG_20261012"]
    res = by_stem["ZZ_FIXTURE_RESULT_20261013"]
    assert pre["pair"] is res and res["pair"] is pre
    assert res["cited"] >= 1, "被 ZZ_LINKER 提到过"
    assert by_stem["ZZ_ORPHAN_REPORT_20261001"]["cited"] == 0
    assert pre["title"] == "预注册：剂量"

    text = mod.render(entries)
    assert ("| 2026-10-13 | 结果 | "
            "[ZZ_FIXTURE_RESULT_20261013.md](ZZ_FIXTURE_RESULT_20261013.md)") in text
    assert "↔ ZZ_FIXTURE_PREREG_20261012" in text
    assert "## ZZ（4）" in text


def test_render_is_deterministic(tmp_path: Path) -> None:
    mod = _load()
    docs = _tree(tmp_path)
    entries = mod.collect(docs)
    mod.annotate(entries, mod._citation_blob(docs))
    assert mod.render(entries) == mod.render(entries)


def test_check_mode_reports_a_missing_or_stale_index(tmp_path: Path) -> None:
    mod = _load()
    docs = _tree(tmp_path)
    assert mod.main(["--docs-dir", str(docs), "--check"]) == 1, "缺文件要报 1"
    assert mod.main(["--docs-dir", str(docs), "--write"]) == 0
    assert (docs / mod.INDEX_NAME).is_file()
    assert mod.main(["--docs-dir", str(docs), "--check"]) == 0
    _write(docs / "ZZ_NEW_RESULT_20261020.md", "# 新结果\n\n正文\n")
    assert mod.main(["--docs-dir", str(docs), "--check"]) == 1, "有新文档就要重生成"
