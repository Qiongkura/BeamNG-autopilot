"""主张清单与正文的**结构性**守卫：编号、列数、状态口径、两份副本一致。

CLAIMS 是论文的"主张 → 数值 → 冻结输入"账本，靠人手工维护。2026-10-20
补录 C19–C39 时发现它已经停在 10-09 而正文更新到 10-18——账本漂移没有
任何机器检查。这里钉住四件容易坏、坏了也没人发现的事：

* 编号连续且不重复（C1..Cn，不能跳号/重号）；
* 每行六列（补一行少一个分隔符，渲染出来就错位）；
* 状态列的两种口径（历史条目「已复算」、2026-10-20 补录「未复算」）；
* `docs/paper/submission/CLAIMS_*.md` 与主副本逐字节相同（它是硬拷贝，
  之前靠人记得同步）。

另外把 2026-10-20 修掉的**符号错**做成回归：正文不许再出现
「道路链高于地形」（实测链在地面下方 ~69 m）。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "docs" / "paper" / "CLAIMS_20261007.md"
SUBMISSION_CLAIMS = ROOT / "docs" / "paper" / "submission" / "CLAIMS_20261007.md"

_ROW = re.compile(r"^\|\s*(C\d+)\s*\|(.*)$")


def _rows() -> list[tuple[str, list[str]]]:
    """返回 ``[(id, [6 列文本])]``；只认以 ``| C<n> |`` 开头的表行。"""
    out: list[tuple[str, list[str]]] = []
    for line in CLAIMS.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if not m:
            continue
        cells = [m.group(1)] + [c.strip()
                                for c in m.group(2).rstrip("|").split("|")]
        out.append((m.group(1), cells))
    return out


def test_claim_ids_are_contiguous_without_duplicates() -> None:
    ids = [cid for cid, _ in _rows()]
    assert ids, "CLAIMS 里一行主张都没解析到"
    nums = [int(cid[1:]) for cid in ids]
    assert nums == list(range(1, len(nums) + 1)), f"编号不连续/有重复：{ids}"
    assert len(set(ids)) == len(ids)


def test_every_claim_row_has_six_columns() -> None:
    bad = [(cid, len(cells)) for cid, cells in _rows() if len(cells) != 6]
    assert not bad, f"这些主张行不是六列（# 主张 数值 来源 配置 状态）：{bad}"


def test_status_column_keeps_the_two_conventions() -> None:
    """C1–C18 是历史条目（已复算），C19 起是 2026-10-20 补录（未复算）。

    补录条目**不得**写「已复算」：本轮只读了文档与产物路径，没有重算指标。
    """
    rows = {cid: cells for cid, cells in _rows()}
    old = [cid for cid in rows if int(cid[1:]) <= 18]
    new = [cid for cid in rows if int(cid[1:]) >= 19]
    assert old and new
    for cid in old:
        assert "已复算" in rows[cid][5], (cid, rows[cid][5])
    for cid in new:
        assert rows[cid][5].startswith("未复算"), (cid, rows[cid][5])


def test_submission_copy_is_byte_identical() -> None:
    """投稿源包是硬拷贝：不同步就会交出一份旧账本。"""
    assert SUBMISSION_CLAIMS.is_file(), "submission/ 里缺 CLAIMS 副本"
    a = hashlib.sha256(CLAIMS.read_bytes()).hexdigest()
    b = hashlib.sha256(SUBMISSION_CLAIMS.read_bytes()).hexdigest()
    assert a == b, ("submission/CLAIMS 与主副本不一致："
                    f"{a[:16]} vs {b[:16]}；同步命令："
                    "Copy-Item docs/paper/CLAIMS_20261007.md "
                    "docs/paper/submission/ -Force")


def test_the_69_m_sign_error_cannot_come_back() -> None:
    """实测链在地面**下方** ~69 m（`mid.z` 144.7 对 213.6）。

    正文原写「above the terrain / 高于地形」，四份文件同错；这里同时钉住
    中英文源，避免只改一边。
    """
    en = (ROOT / "docs" / "paper" / "paper.md").read_text(encoding="utf-8")
    zh = (ROOT / "docs" / "paper" / "paper_zh.md").read_text(encoding="utf-8")
    assert "above the terrain" not in en
    assert "高于地形" not in zh
    assert "below the terrain" in en
    assert "低于地形" in zh
