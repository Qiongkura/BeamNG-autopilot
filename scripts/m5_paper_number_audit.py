"""论文数字审计：把正文表格里的每个数值回查到已记录产物。

用途：E0 纪律的可执行版本——「主张 → 数值 → 来源」不能靠人记。本脚本只读，
不改任何文件；不一致时打印差异并以非零码退出。

覆盖：
* 表 I（两套定义的五个门，来自 v7/v8 判定 JSON）
* 表 II（后处理 × 范围的召回 2×2，来自同一批判定的 pixel/paint_recall 块）
* 表 V（后处理单因子，来自 e1_c{0..4}_*）
* 表 VI（等种子剂量，来自 e2_dose0x/4x 与 e1_c4_v8）
* 正文关键数字：确认集总体、a27、仲裁原因 1383/512、2443 结算帧、范围保留 69%/89%

用法::

    .venv\\Scripts\\python.exe scripts\\m5_paper_number_audit.py
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "logs" / "experiments"
PAPER = ROOT / "docs" / "paper"


def _load(name: str) -> dict:
    return json.loads((E / name).read_text(encoding="utf-8"))


def _mean(vals) -> float | None:
    v = [x for x in vals if isinstance(x, (int, float))]
    return st.mean(v) if v else None


def _dev_means(name: str, pool: str = "dev") -> dict:
    b = _load(name)[pool]
    c, p, pr = b["candidate"], b["pixel"], b["paint_recall"]
    s = sorted(c)
    return {"coverage": _mean([c[x]["candidate_reference_coverage"] for x in s]),
            "identity": _mean([c[x]["candidate_identity_rate"] for x in s]),
            "role": _mean([c[x]["left_right_role_agreement"] for x in s]),
            "precision": _mean([p[x]["line_precision"] for x in s]),
            "recall_label": _mean([p[x]["line_recall"] for x in s]),
            "recall_paint": _mean([pr[x]["recall_paint"] for x in s]),
            "gt_keep": sum(pr[x]["truth_paint_px"] for x in s) / sum(pr[x]["truth_px"] for x in s)}


def _table_rows(md: str, caption: str) -> list[list[str]]:
    """取题注后面的第一张表（按 ``|`` 行解析）。"""
    i = md.index(caption)
    j = md.index("\n\n", i)
    rows = []
    for line in md[j:].splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def _num(s: str) -> float | None:
    m = re.match(r"^\**([0-9]*\.?[0-9]+)\**$", s.replace("**", ""))
    return float(m.group(1)) if m else None


def main() -> int:
    bad: list[str] = []
    checked = 0

    def expect(what: str, got, want, tol: float = 5e-4) -> None:
        nonlocal checked
        checked += 1
        if got is None or want is None:
            bad.append(f"{what}: 缺值 got={got} want={want}")
        elif abs(got - want) > tol:
            bad.append(f"{what}: 稿件 {want} 对来源 {got}（差 {abs(got - want):.4f}）")

    # ---- 表 I：两套定义的五个门（v7 标签范围 / v8 漆范围）
    v7d, v7r = _dev_means("r2_verdict_v7_20261005.json"), _dev_means("r2_verdict_v7_20261005.json", "r3")
    v8d, v8r = _dev_means("r2_verdict_v8_ADOPTED_20261005.json"), _dev_means("r2_verdict_v8_ADOPTED_20261005.json", "r3")
    for path in (PAPER / "paper.md", PAPER / "paper_zh.md"):
        md = path.read_text(encoding="utf-8")
        cap = "TABLE: Table I." if path.name == "paper.md" else "TABLE: 表 I."
        rows = _table_rows(md, cap)
        # 行序：dev baseline / dev adopted / r3 baseline / r3 adopted
        wants = [(v7d, ["coverage", "identity", "precision", "recall_label", "role"]),
                 (v8d, ["coverage", "identity", "precision", "recall_paint", "role"]),
                 (v7r, ["coverage", "identity", "precision", "recall_label", "role"]),
                 (v8r, ["coverage", "identity", "precision", "recall_paint", "role"])]
        for r, (src, keys) in zip(rows[1:], wants):
            for cell, k in zip(r[2:], keys):
                expect(f"{path.name} 表I[{r[0]}/{r[1]}/{k}]", round(src[k], 3), _num(cell), 6e-4)

        # ---- 表 V：后处理单因子
        cap5 = "TABLE: Table V." if path.name == "paper.md" else "TABLE: 表 V."
        srcs5 = [_dev_means(f) for f in ("e1_c0_v7_20261007.json", "e1_c1_lat_20261007.json",
                                         "e1_c2_merge_20261007.json", "e1_c3_appgate_20261007.json",
                                         "e1_c4_v8_20261007.json")]
        for r, src in zip(_table_rows(md, cap5)[1:], srcs5):
            for cell, k in zip(r[1:], ["precision", "recall_label", "recall_paint", "gt_keep", "identity"]):
                expect(f"{path.name} 表V[{r[0]}/{k}]", round(src[k], 3), _num(cell), 6e-4)

        # ---- 表 VI：等种子剂量
        cap6 = "TABLE: Table VI." if path.name == "paper.md" else "TABLE: 表 VI."
        e2 = [_dev_means("e2_dose0x_20261007.json"), _dev_means("e2_dose4x_20261007.json"),
              _dev_means("e1_c4_v8_20261007.json")]
        for r, src in zip(_table_rows(md, cap6)[1:4], e2):
            expect(f"{path.name} 表VI[{r[0]}/identity]", round(src["identity"], 3), _num(r[1]), 6e-4)
            expect(f"{path.name} 表VI[{r[0]}/role]", round(src["role"], 3), _num(r[4]), 6e-4)

    # ---- 正文关键数字
    conf = _load("final_set_v5_20261005/seal/confirmation_base6x-seed42.json")["results"]
    ov = conf["overall"]
    expect("确认集 总体召回", round(ov["line_recall"], 3), 0.460)
    expect("确认集 总体精度", round(ov["line_precision"], 3), 0.763)
    expect("确认集 线 IoU", round(ov["line_iou"], 3), 0.403)
    expect("确认集 道路 IoU", round(ov["road_iou"], 3), 0.633)
    a27 = conf["per_group"]["italy/m5auto_a27"]
    expect("a27 精度", round(a27["line_precision"], 3), 0.169)
    expect("a27 召回", round(a27["line_recall"], 3), 0.190)
    expect("确认集 负例合格帧", ov["negative_line"]["eligible_frames"], 105)
    expect("确认集 假阳性帧", ov["negative_line"]["false_positive_frames"], 22)

    # 仲裁原因与结算帧（与 fig23 同源：固定 12 个城镇运行、t>=8 s）
    import collections
    ts = ["1791221139", "1791221359", "1791221561", "1791221776", "1791259365", "1791259786",
          "1791260204", "1791260603", "1791259583", "1791259992", "1791260405", "1791260800"]
    c = collections.Counter()
    for t in ts:
        p = ROOT / "logs" / "fsd_benchmark" / f"town_{t}.json"
        if p.is_file():
            for h in json.loads(p.read_text(encoding="utf-8")):
                if float(h.get("t") or 0) >= 8.0:
                    c[str(h.get("reason") or "(none)")] += 1
    expect("结算帧总数", sum(c.values()), 2443)
    expect("无可用路径", c.get("no drivable path"), 1383)
    expect("规划车体穿越", c.get("planned vehicle body crosses lane boundary"), 512)

    # 范围保留比例（69% / 89%）
    expect("开发池真值保留", round(v7d["gt_keep"], 2), 0.69, 5e-3)
    expect("有限类别池真值保留", round(v7r["gt_keep"], 2), 0.89, 5e-3)

    print(f"=== 数字审计：检查 {checked} 项 ===")
    for b in bad:
        print("  ✗", b)
    print("RESULT:", "OK - 稿件数字与来源一致" if not bad else f"{len(bad)} 项不一致")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
